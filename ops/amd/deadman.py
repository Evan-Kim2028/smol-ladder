"""The laptop-side dead-man switch: destroys the tagged droplet at a wall-clock deadline, when the
accrued cost reaches a cap, or when the droplet turns up powered off.

    python ops/amd/deadman.py --deadline-minutes 240 --budget 35 &
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
import sys
import time
from dataclasses import dataclass
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


@dataclass(frozen=True)
class Decision:
    destroy: bool
    reason: str

    def as_json(self) -> dict:
        return {"destroy": self.destroy, "reason": self.reason}


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
         total_cap: float, price: float, dry_run: bool, sleep=time.sleep) -> Decision:
    """One poll: list by tag, reconcile the ledger, decide, destroy if needed, log."""
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
            ok = cloud.destroy(api, tag, ledger_path, sleep=sleep, now=lambda: now,
                               reason="deadman: " + decision.reason)
            L.append(ledger_path, L.NOTE, now=now, text="deadman destroyed: " + decision.reason,
                     verified_gone=ok)
    return decision


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

    import os
    if not os.environ.get("AMD_OFFLINE"):
        load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")
    api = DoApi(token_from_env())
    start = args.start if args.start is not None else time.time()
    deadline = start + args.deadline_minutes * 60
    ledger_path = Path(args.ledger)
    L.append(ledger_path, L.NOTE, text=f"deadman armed: deadline in {args.deadline_minutes:g} min, "
                                       f"session cap ${args.budget:g}, total cap ${args.total_cap:g}")
    print(f"deadman armed: deadline {time.strftime('%H:%M:%S', time.localtime(deadline))} "
          f"({args.deadline_minutes:g} min), session cap ${args.budget:g}, total cap "
          f"${args.total_cap:g}, tag '{args.tag}'", flush=True)

    while True:
        now = time.time()
        try:
            d = tick(api, args.tag, ledger_path, now, deadline, args.budget, args.total_cap, args.price,
                     args.dry_run)
        except SystemExit as exc:  # a failed GET: say so and try again; never die silently
            print(f"  could not list droplets: {exc}", file=sys.stderr, flush=True)
            d = Decision(False, "listing failed; will retry")
        print(json.dumps(d.as_json()) if args.json else f"[{time.strftime('%H:%M:%S')}] {d.reason}",
              flush=True)
        if d.destroy or args.once:
            sys.exit(0)
        if now > deadline + GRACE_SECONDS:
            print("past the deadline and nothing tagged remains; exiting", flush=True)
            sys.exit(0)
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
