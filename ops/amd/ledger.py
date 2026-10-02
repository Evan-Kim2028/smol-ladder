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
