"""The spend ledger: every lifecycle event, timestamped, with the rate it was billed at.

A JSONL file, one object per line, appended and never rewritten (git-ignored: ops/amd/ledger.jsonl).
It is the only local record of what this session cost, and it is the input to every budget guard,
so it has two properties that matter more than elegance:

  * it is append-only, and
  * spend is RECOMPUTED from the events, never incremented.

The second is what makes it survive the failure it exists for. A spot reclaim is not a clean event:
the droplet stops existing without telling us, and whatever ran until that moment still billed. A
ledger that added up *closed* intervals would miss the one that matters. Here an interval with no
closing event is still counted, up to `now`.

Money-moving events:  created -> (ready) -> destroyed | reclaimed.  A `created` is recorded BEFORE
the create request is sent, because billing starts at creation and a crash between the request and
the bookkeeping must err on the side of over-counting.

Two caps, because they answer different questions:
  * the SESSION cap  (--budget, default $35): what this one visit may spend
  * the TOTAL cap    (default $90): everything the account may ever spend, including the
                     `prior` events (money spent before this ledger existed).
The rate is stored per `created` event, so a mid-session fallback from MI350X spot ($2.46) to
MI325X on-demand ($3.80) keeps pricing the earlier hours correctly.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

from ops.amd.plan import HARD_TOTAL_LIMIT, effective_total_cap

CREATED = "created"
READY = "ready"
DESTROYED = "destroyed"
RECLAIMED = "reclaimed"
SESSION = "session"
PRIOR = "prior"
STEP_START = "step_start"
STEP_END = "step_end"
MEASURED = "measured"
NOTE = "note"

CLOSING = (DESTROYED, RECLAIMED)


def append(path: Path, event: str, now: float | None = None, **fields) -> dict:
    """Append one event as one flushed line. Returns the record."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"ts": time.time() if now is None else now, "event": event, **fields}
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")
        fh.flush()
    return record


def read(path: Path) -> list[dict]:
    """Every event in order. A corrupt line is skipped: a truncated last line is the expected
    shape of "the process died during teardown", and losing the whole ledger then would be worse
    than losing its final entry."""
    path = Path(path)
    if not path.exists():
        return []
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict) and isinstance(record.get("ts"), (int, float)):
            events.append(record)
    return events


@dataclass(frozen=True)
class Interval:
    start: float
    end: float
    price: float
    open: bool
    droplet_id: object = None

    @property
    def seconds(self) -> float:
        return max(0.0, self.end - self.start)


def intervals(events: list[dict], now: float) -> list[Interval]:
    """Billing intervals. A `created` ends at the next closing event, or at the next `created`
    (a create that was never closed must not silently stop billing), or at `now`."""
    out: list[Interval] = []
    live: dict | None = None

    def close(end: float, still_open: bool = False) -> None:
        nonlocal live
        if live is not None:
            out.append(Interval(float(live["ts"]), end, float(live.get("price_per_hour") or 0.0),
                                still_open, live.get("droplet_id")))
            live = None

    for event in events:
        kind = event.get("event")
        if kind == CREATED:
            close(float(event["ts"]))
            live = event
        elif kind in CLOSING:
            close(float(event["ts"]))
    close(now, still_open=True)
    return out


def open_interval(events: list[dict]) -> dict | None:
    """The `created` event with no closing event after it: the droplet that is billing now."""
    live = None
    for event in events:
        if event.get("event") == CREATED:
            live = event
        elif event.get("event") in CLOSING:
            live = None
    return live


def droplet_ids(events: list[dict]) -> set:
    """Every droplet id this ledger ever recorded: what a destroy must see answer 404."""
    return {e["droplet_id"] for e in events
            if e.get("event") in (CREATED, READY) and e.get("droplet_id") is not None}


def session_start(events: list[dict]) -> float:
    starts = [float(e["ts"]) for e in events if e.get("event") == SESSION]
    return starts[-1] if starts else 0.0


@dataclass(frozen=True)
class Spend:
    session: float
    total: float
    open: bool
    rate: float          # dollars/hour the open interval (else the last one) bills at
    open_seconds: float

    def as_json(self) -> dict:
        return {"session_dollars": round(self.session, 4), "total_dollars": round(self.total, 4),
                "billing": self.open, "price_per_hour": self.rate,
                "open_seconds": round(self.open_seconds, 1)}


def spend(events: list[dict], now: float, default_rate: float = 0.0) -> Spend:
    """Session and total dollars at `now`. An interval straddling the session start is pro-rated
    so that only the part inside the session counts toward the session cap."""
    since = session_start(events)
    session = total = 0.0
    rate = default_rate
    open_seconds = 0.0
    is_open = False
    for iv in intervals(events, now):
        total += iv.seconds / 3600.0 * iv.price
        inside = max(0.0, iv.end - max(iv.start, since))
        session += inside / 3600.0 * iv.price
        rate = iv.price or rate
        if iv.open:
            is_open, open_seconds = True, iv.seconds
    prior = sum(float(e.get("dollars") or 0.0) for e in events if e.get("event") == PRIOR)
    return Spend(session=session, total=total + prior, open=is_open, rate=rate,
                 open_seconds=open_seconds)


@dataclass(frozen=True)
class Verdict:
    allowed: bool
    reason: str
    session_after: float
    total_after: float


def verdict(events: list[dict], now: float, step_seconds: float, rate: float,
            session_cap: float, total_cap: float, reserve_seconds: float = 0.0) -> Verdict:
    """The gate every billed step passes through before it starts.

    Projected = what is already accrued + this step at `rate` + a reserve that keeps the final
    sync and destroy affordable. A projection that lands exactly on a cap is allowed; one cent
    over is refused. The reason is written to be pasted into a log, because the one time it
    matters is the moment a step is refused and nobody is watching.
    """
    cur = spend(events, now, rate)
    hard = total_cap >= HARD_TOTAL_LIMIT
    total_cap = effective_total_cap(total_cap)    # the hard limit is applied here, not by the caller
    extra = (max(step_seconds, 0.0) + max(reserve_seconds, 0.0)) / 3600.0 * rate
    session_after = cur.session + extra
    total_after = cur.total + extra
    eps = 1e-9
    if session_after > session_cap + eps:
        return Verdict(False, (
            f"REFUSING: session projected ${session_after:.2f} (${cur.session:.2f} accrued + "
            f"${extra:.2f} for this step and the sync/destroy reserve) exceeds the "
            f"${session_cap:.2f} session budget by ${session_after - session_cap:.2f}."),
            session_after, total_after)
    if total_after > total_cap + eps:
        return Verdict(False, (
            f"REFUSING: total projected ${total_after:.2f} (${cur.total:.2f} accrued + "
            f"${extra:.2f}) exceeds the ${total_cap:.2f} "
            f"{'HARD total limit, which nothing may override,' if hard else 'total cap'} by "
            f"${total_after - total_cap:.2f}."), session_after, total_after)
    return Verdict(True, (
        f"ok: session ${cur.session:.2f} + ${extra:.2f} = ${session_after:.2f} of "
        f"${session_cap:.2f}; total ${total_after:.2f} of ${total_cap:.2f}"),
        session_after, total_after)


def latest(events: list[dict], kind: str) -> dict | None:
    for event in reversed(events):
        if event.get("event") == kind:
            return event
    return None


def summarise(events: list[dict], now: float, default_rate: float = 0.0) -> dict:
    """The `--status` view: accrued cost, uptime, and whether anything is still billing."""
    cur = spend(events, now, default_rate)
    live = open_interval(events)
    out = cur.as_json()
    out["events"] = len(events)
    if live is not None:
        out["uptime_seconds"] = round(now - float(live["ts"]), 1)
        out["droplet_id"] = live.get("droplet_id")
        out["ip"] = live.get("ip") or (latest(events, READY) or {}).get("ip")
        out["state"] = "RUNNING: billing until it is DESTROYED (powering off does not stop it)"
    else:
        out["uptime_seconds"] = 0.0
        out["state"] = "no open interval in the ledger: nothing should be billing"
    return out


# ── the dead-man switch's heartbeat ───────────────────────────────────────────────
#
# A watcher that died (a missing token, a crashed API call, a closed terminal) looks exactly like a
# watcher with nothing to do. So the watcher writes a timestamp every poll to a small file next to
# the ledger, and the driver refuses to create a droplet, or start any billed step, unless that
# timestamp is fresh: an unarmed session cannot start billing.

HEARTBEAT_POLLS = 3          # fresh = written within this many poll intervals
DEFAULT_POLL_SECONDS = 30.0


def heartbeat_path(ledger_path) -> Path:
    return Path(str(ledger_path) + ".heartbeat")


def heartbeat_write(ledger_path, now: float, *, last_ok: float | None, poll_seconds: float,
                    tag: str, deadline: float, budget: float, total_cap: float,
                    state: str = "ok", pid: int = 0) -> None:
    """Atomic replace, so a reader never sees half a file. `last_ok` is the last tick that
    actually listed the droplets: a watcher that is alive but blind (every call failing) is
    reported as such, because it is not protecting anything."""
    path = heartbeat_path(ledger_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps({"ts": now, "last_ok": last_ok, "poll_seconds": poll_seconds,
                               "tag": tag, "deadline": deadline, "budget": budget,
                               "total_cap": total_cap, "state": state, "pid": pid}, sort_keys=True))
    tmp.replace(path)


def heartbeat_read(ledger_path) -> dict | None:
    try:
        data = json.loads(heartbeat_path(ledger_path).read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) and isinstance(data.get("ts"), (int, float)) else None


def heartbeat_status(ledger_path, now: float, tag: str | None = None,
                     budget: float | None = None, total_cap: float | None = None) -> tuple[bool, str]:
    """(fresh, why). Fresh means: written within HEARTBEAT_POLLS poll intervals, listed the droplets
    successfully within that window, watches the same tag, was armed with caps no looser than
    this session's, and its deadline has not passed."""
    hb = heartbeat_read(ledger_path)
    if hb is None:
        return False, f"no heartbeat file at {heartbeat_path(ledger_path)}: the deadman is not running"
    window = HEARTBEAT_POLLS * float(hb.get("poll_seconds") or DEFAULT_POLL_SECONDS)
    age = now - float(hb["ts"])
    if age > window:
        return False, f"last heartbeat {age:.0f}s ago (limit {window:.0f}s): the deadman stopped"
    if age < -window:
        return False, f"heartbeat is {-age:.0f}s in the future: clock skew or a stale file"
    ok = hb.get("last_ok")
    if not isinstance(ok, (int, float)) or now - float(ok) > window:
        return False, ("the deadman is running but cannot list droplets (every call failing), so "
                       "it is not protecting anything")
    if tag is not None and hb.get("tag") != tag:
        return False, f"the deadman watches tag {hb.get('tag')!r}, not {tag!r}"
    eps = 1e-9
    if budget is not None and float(hb.get("budget") or 0) > budget + eps:
        return False, f"the deadman was armed with a looser session cap (${hb.get('budget')}) than ${budget:g}"
    if total_cap is not None and float(hb.get("total_cap") or 0) > total_cap + eps:
        return False, f"the deadman was armed with a looser total cap (${hb.get('total_cap')}) than ${total_cap:g}"
    if float(hb.get("deadline") or 0) <= now:
        return False, "the deadman's wall-clock deadline has passed"
    return True, f"fresh: {age:.0f}s old, state {hb.get('state')}"
