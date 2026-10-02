"""Droplet lifecycle against an injected API client: preflight, create, status, destroy, reconcile.

Every function takes the client (`DoApi` in production, a fake in tests) and a `sleep`, so the
tests exercise the real logic with no network and no waiting. Nothing here runs unless the
reviewer's command (`driver.py create --yes`, `destroy --yes`, the dead-man switch) calls it.
"""

from __future__ import annotations

import calendar
import time
from typing import Callable

from ops.amd import ledger as L
from ops.amd.plan import Config, create_body

Sleep = Callable[[float], None]


def tagged(api, tag: str) -> list[dict]:
    status, body = api.get(f"/droplets?tag_name={tag}&per_page=200")
    if status != 200 or not isinstance(body.get("droplets"), list):
        raise SystemExit(f"listing droplets tagged {tag!r} failed ({status}): {body}")
    return body["droplets"]


def public_ip(droplet: dict) -> str:
    for net in (droplet.get("networks") or {}).get("v4", []):
        if net.get("type") == "public":
            return net["ip_address"]
    return ""


def preflight(api, cfg: Config) -> list[tuple[str, bool, str]]:
    """Read-only checks. Every line is (check, ok, detail); the create refuses on any failure."""
    out: list[tuple[str, bool, str]] = []
    status, body = api.get("/sizes?per_page=200")
    size = next((s for s in body.get("sizes", []) if s.get("slug") == cfg.size), None) \
        if status == 200 else None
    if size is None:
        out.append(("size exists", False, f"{cfg.size} not returned by GET /sizes ({status})"))
    else:
        out.append(("size exists", True, f"{cfg.size} ${size.get('price_hourly')}/h"))
        out.append(("size offered in region", cfg.region in size.get("regions", []),
                    f"{cfg.region} in {size.get('regions')}"))
        rate = float(size.get("price_hourly") or 0.0)
        out.append(("price matches the plan", abs(rate - cfg.price) < 0.005,
                    f"account says ${rate}/h, plan uses ${cfg.price}/h"))
    status, body = api.get("/images?type=application&per_page=200")
    image = next((i for i in body.get("images", []) if i.get("slug") == cfg.image), None) \
        if status == 200 else None
    out.append(("image slug exists", image is not None,
                f"{cfg.image}" + (f" regions {image.get('regions')}" if image else "")))
    if image is not None:
        out.append(("image offered in region", cfg.region in image.get("regions", []),
                    f"{cfg.region}"))
    status, body = api.get("/account/keys?per_page=200")
    fps = [k.get("fingerprint") for k in body.get("ssh_keys", [])] if status == 200 else []
    out.append(("ssh key fingerprint registered",
                bool(cfg.fingerprint) and cfg.fingerprint in fps,
                f"{cfg.fingerprint or '<none given>'} among {len(fps)} registered key(s)"))
    try:
        existing = tagged(api, cfg.tag)
        out.append(("no droplet already tagged", not existing,
                    f"{len(existing)} existing: {[d.get('name') for d in existing]}"))
    except SystemExit as exc:
        out.append(("no droplet already tagged", False, str(exc)))
    return out


def create(api, cfg: Config, ledger_path, sleep: Sleep = time.sleep, now=time.time,
           wait_s: float = 600.0, new_session: bool = False) -> dict:
    """Record, send, wait for active, record the IP. The `created` event is written BEFORE the
    request: billing starts at creation, and a crash in between must over-count, not under-count.

    A SESSION marker starts the session budget over. It is written for the first create, or when
    asked (`--new-session`): re-creating after a spot reclaim is the SAME session, so the
    $35 already spent must still count against it."""
    if new_session or L.session_start(L.read(ledger_path)) == 0.0:
        L.append(ledger_path, L.SESSION, now=now(), budget=cfg.budget, total_cap=cfg.total_cap)
    L.append(ledger_path, L.CREATED, now=now(), price_per_hour=cfg.price, size=cfg.size,
             region=cfg.region, image=cfg.image, droplet_id=None, pending=True)
    status, body = api.post("/droplets", create_body(cfg))
    droplet = body.get("droplet") if isinstance(body, dict) else None
    if status not in (200, 201, 202) or not droplet:
        L.append(ledger_path, L.DESTROYED, now=now(), reason="create failed", http=status)
        raise SystemExit(f"create failed ({status}): {body}\n(ledger interval closed: nothing "
                         "was created, so nothing is billing)")
    did = droplet["id"]
    L.append(ledger_path, L.CREATED, now=now(), price_per_hour=cfg.price, size=cfg.size,
             region=cfg.region, image=cfg.image, droplet_id=did)
    waited = 0.0
    ip = ""
    while waited <= wait_s:
        status, body = api.get(f"/droplets/{did}")
        d = body.get("droplet", {}) if status == 200 else {}
        ip = public_ip(d)
        if d.get("status") == "active" and ip:
            L.append(ledger_path, L.READY, now=now(), droplet_id=did, ip=ip)
            return {"droplet_id": did, "ip": ip}
        sleep(5)
        waited += 5
    raise SystemExit(f"droplet {did} did not become active within {wait_s:.0f}s. It IS billing: "
                     "run `driver.py destroy --yes` or look at the console.")


def destroy(api, tag: str, ledger_path, sleep: Sleep = time.sleep, now=time.time,
            checks: int = 12, reason: str = "destroy") -> bool:
    """DELETE by tag, then GET until nothing with the tag remains. The interval is closed in the
    ledger only when the GET confirms, because a 204 is a request and the GET is the outcome."""
    status, body = api.delete(f"/droplets?tag_name={tag}")
    if status not in (200, 202, 204, 404):
        print(f"  DELETE by tag returned {status}: {body}")
    for attempt in range(checks):
        remaining = tagged(api, tag)
        if not remaining:
            L.append(ledger_path, L.DESTROYED, now=now(), tag=tag, verified_gone=True,
                     reason=reason)
            return True
        print(f"  still present ({[d.get('name') for d in remaining]}), check {attempt + 1}/{checks}")
        sleep(5)
    L.append(ledger_path, L.NOTE, now=now(), text=f"destroy NOT verified for tag {tag}")
    return False


def parse_created_at(text: str) -> float:
    return float(calendar.timegm(time.strptime(text, "%Y-%m-%dT%H:%M:%SZ")))


def api_dollars(droplets: list[dict], now: float, default_rate: float) -> float:
    """Spend implied by the API alone: each live droplet's age times its size's hourly price.
    Independent of the ledger, so a create the ledger never recorded is still counted."""
    total = 0.0
    for d in droplets:
        try:
            age = max(0.0, now - parse_created_at(d["created_at"]))
        except (KeyError, ValueError):
            continue
        rate = float((d.get("size") or {}).get("price_hourly") or default_rate)
        total += age / 3600.0 * rate
    return total


def reconcile(droplets: list[dict], ledger_path, now: float, grace: float = 120.0) -> bool:
    """If the ledger says a droplet is billing but the API shows none with the tag, it was
    reclaimed (spot) or destroyed elsewhere: close the interval so the ledger stops accruing.
    Only ever called with a listing that succeeded; returns True if it closed one. An interval
    younger than `grace` seconds is left alone: a droplet that was only just created may not be
    in the tag listing yet, and closing its interval then would stop counting real spend."""
    events = L.read(ledger_path)
    live = L.open_interval(events)
    if live is not None and now - float(live["ts"]) < grace:
        return False
    if live is not None and not droplets:
        L.append(ledger_path, L.RECLAIMED, now=now,
                 reason="ledger had an open interval but no tagged droplet exists")
        return True
    return False
