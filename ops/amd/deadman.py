"""The laptop-side dead-man switch: destroys the tagged droplet at a wall-clock deadline, when the
accrued cost reaches a cap, or when the droplet turns up powered off.

    setsid nohup python ops/amd/deadman.py --deadline-minutes 240 --budget 35 \\
        >> logs/deadman.log 2>&1 < /dev/null &
    python ops/amd/deadman.py --once --dry-run            one decision, destroys nothing

A droplet cannot destroy itself, and a powered-off GPU droplet still bills, so the switch that
actually stops the meter has to live outside the droplet. This one is independent of it in every
direction:

  * it runs on the laptop, so a wedged, rebooting or reclaimed droplet cannot stop it;
  * it finds the droplet by TAG, not by a remembered id, so a create the driver never recorded is
    still found, and it deletes by tag in one call;
  * cost is the larger of the LEDGER's accrual and the API's own (age x hourly price of every live
    tagged droplet), so an unrecorded create is still priced;
  * the wall-clock deadline fires even when both of those are wrong.

`decide()` is a pure function of its inputs; `tick()` takes the API client and a clock, so the
tests drive the real logic with stubs. Every decision that destroys is appended to the ledger and
printed.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ops.amd import cloud  # noqa: E402
from ops.amd import ledger as L  # noqa: E402
from ops.amd.doapi import DoApi, load_dotenv, token_from_env  # noqa: E402
from ops.amd.plan import (DEFAULT_BUDGET, DESTROY_MARGIN, HARD_TOTAL_LIMIT,  # noqa: E402
                          PRICE_MI350X, TAG, TOTAL_CAP, effective_total_cap)

POLL_SECONDS = 30
# After the deadline with nothing left to destroy, keep looking this long (a create may still be
# landing), then exit rather than loop forever.
GRACE_SECONDS = 300
MAX_BACKOFF_SECONDS = 300.0


@dataclass(frozen=True)
class Decision:
    destroy: bool
    reason: str
    verified_gone: bool | None = None   # set by tick() after a destroy: did the GETs confirm it?

    def as_json(self) -> dict:
        return {"destroy": self.destroy, "reason": self.reason, "verified_gone": self.verified_gone}


def _dur(seconds: float) -> str:
    return f"{seconds / 3600.0:.1f} h" if abs(seconds) >= 5400 else f"{seconds / 60.0:.1f} min"


def decide(now: float, deadline: float, session_dollars: float, total_dollars: float,
           session_cap: float, total_cap: float, droplets: list[dict]) -> Decision:
    """Pure. Destroy when anything tagged exists AND any of: deadline passed, session cap reached,
    total cap reached, a droplet is powered off. With nothing tagged there is nothing to destroy,
    and claiming otherwise would put a lie in a log that exists to be trusted.

    Order is deliberate: the deadline first, because it is the only condition that survives an
    empty or wrong ledger; then the money.
    """
    if not droplets:
        return Decision(False, "no tagged droplet exists: nothing to destroy")
    names = [d.get("name") for d in droplets]
    if now >= deadline:
        return Decision(True, f"wall-clock deadline passed {_dur(now - deadline)} ago; "
                              f"{names} still exist")
    if session_dollars >= session_cap:
        return Decision(True, f"session cost ${session_dollars:.2f} reached the "
                              f"${session_cap:.2f} cap; {names} still exist")
    # The hard limit is applied HERE, whatever total_cap was passed in, and the destroy's own cost
    # is held back: the destroy has to finish before the limit, not start at it.
    limit = effective_total_cap(total_cap) - DESTROY_MARGIN
    if total_dollars >= limit:
        which = "the HARD limit" if total_cap >= HARD_TOTAL_LIMIT else "the total cap"
        return Decision(True, f"total cost ${total_dollars:.2f} reached ${limit:.2f} (= {which} "
                              f"${effective_total_cap(total_cap):.2f} minus ${DESTROY_MARGIN:.2f} "
                              f"for the destroy itself); {names} still exist")
    off = [d.get("name") for d in droplets if d.get("status") == "off"]
    if off:
        return Decision(True, f"{off} powered off but still billing (power-off does not stop "
                              "billing); destroying")
    return Decision(False, f"ok: {_dur(deadline - now)} to the deadline, session "
                           f"${session_dollars:.2f}/${session_cap:.2f}, total "
                           f"${total_dollars:.2f}/${total_cap:.2f}")


def tick(api, tag: str, ledger_path: Path, now: float, deadline: float, session_cap: float,
         total_cap: float, price: float, dry_run: bool, sleep=time.sleep, clock=None) -> Decision:
    """One poll: list by tag, reconcile the ledger, decide, destroy if needed, log.

    `now` is when the decision was taken; `clock` (the real clock in production) stamps the events
    the destroy writes. Re-using `now` for them would close the ledger's interval at the moment of
    the decision and drop the minute or two the destroy and its verification actually took."""
    droplets = cloud.tagged(api, tag)
    cloud.reconcile(droplets, ledger_path, now)
    spent = L.spend(L.read(ledger_path), now, price)
    # The API's age-based figure can only ADD to what the ledger knows: closed intervals the
    # ledger recorded stay, and the open one is priced at the larger of the two views.
    open_part = spent.open_seconds / 3600.0 * spent.rate if spent.open else 0.0
    api_cost = cloud.api_dollars(droplets, now, price)
    extra = max(0.0, (spent.session - open_part) + api_cost - spent.session)
    session, total = spent.session + extra, spent.total + extra
    decision = decide(now, deadline, session, total, session_cap, total_cap, droplets)
    if decision.destroy:
        if dry_run:
            print(f"  DRY RUN would destroy by tag: {decision.reason}")
        else:
            stamp = clock or (lambda: now)
            ok = cloud.destroy(api, tag, ledger_path, sleep=sleep, now=stamp,
                               reason="deadman: " + decision.reason)
            L.append(ledger_path, L.NOTE, now=stamp(),
                     text="deadman destroyed: " + decision.reason, verified_gone=ok)
            decision = replace(decision, verified_gone=ok)
    return decision


def watch(api, *, tag: str, ledger_path: Path, deadline: float, session_cap: float,
          total_cap: float, price: float, poll: float = POLL_SECONDS, once: bool = False,
          dry_run: bool = False, as_json: bool = False, sleep=time.sleep, clock=time.time,
          max_backoff: float = MAX_BACKOFF_SECONDS, out=None) -> int:
    """The loop. It returns an exit code, and it returns only when it is safe to:

      * after a destroy that the GETs VERIFIED (tag listing empty, every recorded id 404), or
      * past the deadline with a successful listing that shows nothing tagged, or
      * `once`, after one decision.

    A tick that raises ANYTHING (a read timeout, an IncompleteRead, a non-JSON 200, a bug) is
    logged and retried with exponential backoff; it never ends the watcher. A destroy that could
    not be verified is retried every poll, forever: while a tagged droplet may exist, this process
    does not exit. A heartbeat is written every iteration, so a watcher that died, or that is alive
    but blind, is visible to the driver's gate."""
    say = out or (lambda s: print(s, flush=True))
    failures = 0
    last_ok: float | None = None

    def beat(state: str) -> None:
        try:
            L.heartbeat_write(ledger_path, clock(), last_ok=last_ok, poll_seconds=poll, tag=tag,
                              deadline=deadline, budget=session_cap, total_cap=total_cap,
                              state=state, pid=os.getpid())
        except OSError as exc:       # a full disk must not stop the watcher; the gate will notice
            say(f"  could not write the heartbeat: {exc}")

    while True:
        now = clock()
        beat("tick")
        try:
            d = tick(api, tag, ledger_path, now, deadline, session_cap, total_cap, price, dry_run,
                     sleep=sleep, clock=clock)
        except (Exception, SystemExit) as exc:     # noqa: BLE001 - nothing may kill the watcher
            failures += 1
            wait = min(poll * 2 ** min(failures, 8), max_backoff)
            say(f"[{time.strftime('%H:%M:%S')}] tick failed ({failures} in a row): "
                f"{type(exc).__name__}: {exc}; retrying in {wait:.0f}s")
            beat("error")
            if once:
                return 1
            sleep(wait)
            continue
        failures = 0
        last_ok = clock()
        beat("ok")
        say(json.dumps(d.as_json()) if as_json else f"[{time.strftime('%H:%M:%S')}] {d.reason}")
        if d.destroy and dry_run:
            return 0
        if d.destroy and d.verified_gone:
            say("destroy verified (tag listing empty, every recorded droplet id 404). "
                "Exiting: re-arm a NEW deadman before any further create.")
            return 0
        if d.destroy:
            say("destroy NOT verified: a tagged droplet may still exist and be billing. "
                "NOT exiting; trying again.")
            if once:
                return 1
            sleep(poll)
            continue
        if once:
            return 0
        if now > deadline + GRACE_SECONDS and d.reason.startswith("no tagged droplet"):
            say("past the deadline and a successful listing shows nothing tagged; exiting")
            return 0
        sleep(poll)


def total_cap_arg(text: str) -> float:
    value = float(text)
    if value > HARD_TOTAL_LIMIT:
        raise argparse.ArgumentTypeError(
            f"${value:g} is above the ${HARD_TOTAL_LIMIT:g} HARD total limit, which no flag or "
            "environment variable can raise")
    return value


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--deadline-minutes", type=float, required=True,
                    help="wall clock from now (or from --start) after which the droplet is destroyed")
    ap.add_argument("--start", type=float, default=None, help="epoch the deadline counts from")
    ap.add_argument("--budget", type=float, default=DEFAULT_BUDGET, help="session cap in dollars")
    ap.add_argument("--total-cap", type=total_cap_arg, default=TOTAL_CAP,
                    help=f"working total cap (never above the ${HARD_TOTAL_LIMIT:g} hard limit)")
    ap.add_argument("--price", type=float, default=PRICE_MI350X)
    ap.add_argument("--ledger", default=str(Path(__file__).resolve().parent / "ledger.jsonl"))
    ap.add_argument("--tag", default=TAG)
    ap.add_argument("--poll-seconds", type=float, default=POLL_SECONDS)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    if not os.environ.get("AMD_OFFLINE"):
        load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")
    if not token_from_env():
        # Exiting here is silent to anyone who is not watching this terminal, so the driver does not
        # rely on anyone watching: no heartbeat is written, and `create` is refused without one.
        sys.exit("deadman NOT ARMED: no DigitalOcean token (AMD_CLOUD_API_TOKEN or "
                 "DIGITALOCEAN_ACCESS_TOKEN in the repo .env). The driver will refuse to create "
                 "or bill anything until a deadman heartbeat exists.")
    signal.signal(signal.SIGHUP, signal.SIG_IGN)   # a closed terminal must not disarm it
    api = DoApi(token_from_env())
    start = args.start if args.start is not None else time.time()
    deadline = start + args.deadline_minutes * 60
    ledger_path = Path(args.ledger)
    L.append(ledger_path, L.NOTE, text=f"deadman armed: deadline in {args.deadline_minutes:g} min, "
                                       f"session cap ${args.budget:g}, total cap ${args.total_cap:g} "
                                       f"(hard limit ${HARD_TOTAL_LIMIT:g})")
    print(f"deadman armed: deadline {time.strftime('%H:%M:%S', time.localtime(deadline))} "
          f"({args.deadline_minutes:g} min), session cap ${args.budget:g}, total cap "
          f"${args.total_cap:g} (hard limit ${HARD_TOTAL_LIMIT:g}), tag '{args.tag}', heartbeat "
          f"{L.heartbeat_path(ledger_path)}", flush=True)
    sys.exit(watch(api, tag=args.tag, ledger_path=ledger_path, deadline=deadline,
                   session_cap=args.budget, total_cap=args.total_cap, price=args.price,
                   poll=args.poll_seconds, once=args.once, dry_run=args.dry_run,
                   as_json=args.json))


if __name__ == "__main__":
    main()
