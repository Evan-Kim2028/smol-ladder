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


CREATE_WAIT_S = 600.0
RESOLVE_S = 90.0             # how long an ambiguous create is chased through the tag listing
PENDING_GRACE_S = 600.0      # an unresolved create is not written off as "nothing happened" sooner


def outcome_unknown(status: int) -> bool:
    """A create that may or may not have happened. 0 is a timeout or a dropped connection (the
    request can have been processed before the answer was lost); 429 and 5xx are an overloaded API
    that may have queued it. Only an unambiguous refusal (422 no capacity, 401, 403, ...) means
    nothing was created."""
    return status == 0 or status == 408 or status == 429 or status >= 500


def resolve_unknown_create(api, tag: str, ledger_path, status: int, body, sleep: Sleep, now,
                           resolve_s: float = RESOLVE_S):
    """Chase an ambiguous create through the tag listing for ~resolve_s. Returns the droplet's id if
    one shows up. If none does, the ledger interval stays OPEN and this raises: a droplet that
    appears a minute later is billing, and a closed interval would hide it from every guard."""
    waited = 0.0
    last_error = ""
    while waited <= resolve_s:
        try:
            found = tagged(api, tag)
        except SystemExit as exc:
            found, last_error = None, str(exc)
        if found:
            if len(found) > 1:
                L.append(ledger_path, L.NOTE, now=now(), text=f"TWO droplets tagged {tag}: "
                         f"{[d.get('id') for d in found]}")
                raise SystemExit(f"{len(found)} droplets are tagged {tag!r} after an ambiguous "
                                 "create. Run `driver.py destroy --yes` now.")
            print(f"  create answered {status}, but droplet {found[0].get('id')} exists: adopting it")
            return found[0]["id"]
        sleep(5)
        waited += 5
    L.append(ledger_path, L.NOTE, now=now(), text=f"create outcome UNKNOWN (http {status}); no tagged "
             f"droplet seen for {resolve_s:.0f}s; interval left open", http=status)
    raise SystemExit(
        f"create outcome UNKNOWN (http {status}: {str(body)[:200]}). No droplet carrying tag {tag!r} "
        f"was visible for {resolve_s:.0f}s{' (listing errors: ' + last_error + ')' if last_error else ''}, "
        "but the request may still land. The ledger interval stays OPEN so every guard keeps "
        "counting. Do NOT create again: re-creation is refused until this resolves. Run "
        "`driver.py status` in a few minutes (it adopts or closes the pending create), check the "
        "console, or `driver.py destroy --yes`.")


def create(api, cfg: Config, ledger_path, sleep: Sleep = time.sleep, now=time.time,
           wait_s: float = CREATE_WAIT_S, new_session: bool = False,
           resolve_s: float = RESOLVE_S) -> dict:
    """Record, send, wait for active, record the IP. The `created` event is written BEFORE the
    request: billing starts at creation, and a crash in between must over-count, not under-count.

    Never two droplets: refused if the ledger has an open interval (a droplet, or an unresolved
    create) or if anything is already tagged. An answer that does not say whether the droplet was
    made (timeout, 429, 5xx) is resolved against the tag listing and never closes the interval.

    A SESSION marker starts the session budget over. It is written for the first create, or when
    asked (`--new-session`): re-creating after a spot reclaim is the SAME session, so the
    $35 already spent must still count against it."""
    events = L.read(ledger_path)
    live = L.open_interval(events)
    if live is not None:
        raise SystemExit(
            f"refusing to create: the ledger has an open interval (droplet {live.get('droplet_id')}"
            f"{', create still unresolved' if live.get('droplet_id') is None else ''}). Run "
            "`driver.py status` (reconciles it against the API) or `destroy --yes` first.")
    existing = tagged(api, cfg.tag)       # a failed listing raises: no listing, no create
    if existing:
        raise SystemExit(f"refusing to create: {len(existing)} droplet(s) already tagged "
                         f"{cfg.tag!r} ({[d.get('name') for d in existing]}). Never two droplets.")
    if new_session or L.session_start(events) == 0.0:
        L.append(ledger_path, L.SESSION, now=now(), budget=cfg.budget, total_cap=cfg.total_cap)
    L.append(ledger_path, L.CREATED, now=now(), price_per_hour=cfg.price, size=cfg.size,
             region=cfg.region, image=cfg.image, droplet_id=None, pending=True)
    status, body = api.post("/droplets", create_body(cfg))
    droplet = body.get("droplet") if isinstance(body, dict) else None
    made = status in (200, 201, 202) and isinstance(droplet, dict) and droplet.get("id") is not None
    if made:
        did = droplet["id"]
    elif outcome_unknown(status) or status in (200, 201, 202):
        did = resolve_unknown_create(api, cfg.tag, ledger_path, status, body, sleep, now, resolve_s)
    else:
        L.append(ledger_path, L.DESTROYED, now=now(), reason="create refused", http=status)
        raise SystemExit(f"create refused ({status}): {body}\n(ledger interval closed: the API "
                         "said no, so nothing was created and nothing is billing)")
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


def is_gpu(droplet: dict) -> bool:
    slug = droplet.get("size_slug") or (droplet.get("size") or {}).get("slug") or ""
    return str(slug).startswith("gpu-")


def audit_account(api, ours: set, tag: str) -> list[dict]:
    """Every droplet on the account that is not one of ours, for the destroy's report. READ-ONLY:
    a droplet that is not ours is never deleted here, whatever it is. Raises SystemExit if the
    listing fails, so the caller can say the audit did not happen."""
    status, body = api.get("/droplets?per_page=200")
    if status != 200 or not isinstance(body.get("droplets"), list):
        raise SystemExit(f"account listing failed ({status}): {body}")
    droplets = body["droplets"]
    total = (body.get("meta") or {}).get("total")
    if isinstance(total, int) and total > len(droplets):
        print(f"  !! account audit incomplete: {total} droplets exist, the first {len(droplets)} were listed")
    return [d for d in droplets if d.get("id") not in ours or tag in (d.get("tags") or [])]


def destroy(api, tag: str, ledger_path, sleep: Sleep = time.sleep, now=time.time,
            checks: int = 12, reason: str = "destroy") -> bool:
    """DELETE by tag, then verify. The interval is closed in the ledger only when verification
    passes, because a 204 is a request and the GETs are the outcome. Verified means ALL of:

      * the tag listing is empty,
      * every droplet id the ledger ever recorded answers GET with 404 (a droplet that lost its tag
        is invisible to the listing and still bills; one that still exists is deleted by id: it is
        ours, the ledger recorded it),
      * then the whole account is listed and anything unexpected is REPORTED LOUDLY and written to
        the ledger, never deleted: an untagged GPU droplet that is not ours is the owner's call.
    """
    ours = L.droplet_ids(L.read(ledger_path))
    status, body = api.delete(f"/droplets?tag_name={tag}")
    if status not in (200, 202, 204, 404):
        print(f"  DELETE by tag returned {status}: {body}")
    gone: set = set()
    for attempt in range(checks):
        problems: list[str] = []
        try:
            remaining = tagged(api, tag)
            if remaining:
                problems.append(f"still tagged: {[d.get('name') for d in remaining]}")
        except SystemExit as exc:
            problems.append(f"tag listing failed: {exc}")
        for did in sorted(ours - gone, key=str):
            try:
                st, _ = api.get(f"/droplets/{did}")
            except Exception as exc:  # noqa: BLE001 - an unreadable answer is "not verified"
                problems.append(f"droplet {did}: GET raised {type(exc).__name__}")
                continue
            if st == 404:
                gone.add(did)
            elif st == 200:
                problems.append(f"droplet {did} still exists (GET 200); deleting it by id")
                api.delete(f"/droplets/{did}")
            else:
                problems.append(f"droplet {did}: GET returned {st}, so it is not verified gone")
        if not problems:
            break
        print(f"  not yet verified ({'; '.join(problems)}), check {attempt + 1}/{checks}")
        sleep(5)
    else:
        L.append(ledger_path, L.NOTE, now=now(), text=f"destroy NOT verified for tag {tag}")
        return False
    try:
        unexpected = audit_account(api, ours, tag)
    except SystemExit as exc:
        print(f"  !! could not audit the account for stray droplets: {exc}")
        unexpected = []
    for d in unexpected:
        kind = "UNTAGGED GPU DROPLET" if is_gpu(d) else "other droplet"
        print(f"  !!!! {kind} on the account, NOT ours and NOT touched: id {d.get('id')} "
              f"name {d.get('name')!r} size {d.get('size_slug')} status {d.get('status')}"
              + (" -- a GPU droplet bills whether or not it is tagged; check the console" if is_gpu(d) else ""))
    if unexpected:
        L.append(ledger_path, L.NOTE, now=now(), text="UNEXPECTED droplets on the account after "
                 "destroy (not ours, not deleted)",
                 droplets=[{"id": d.get("id"), "name": d.get("name"), "size": d.get("size_slug"),
                            "gpu": is_gpu(d)} for d in unexpected])
    L.append(ledger_path, L.DESTROYED, now=now(), tag=tag, verified_gone=True, reason=reason,
             checked_ids=sorted(ours, key=str))
    return True


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


def reconcile(droplets: list[dict], ledger_path, now: float, grace: float = 120.0,
              pending_grace: float = PENDING_GRACE_S) -> bool:
    """If the ledger says a droplet is billing but the API shows none with the tag, it was
    reclaimed (spot) or destroyed elsewhere: close the interval so the ledger stops accruing.
    Only ever called with a listing that succeeded; returns True if it closed one. An interval
    younger than `grace` seconds is left alone: a droplet that was only just created may not be
    in the tag listing yet, and closing its interval then would stop counting real spend."""
    events = L.read(ledger_path)
    live = L.open_interval(events)
    if live is not None:
        # A create whose answer was lost has no droplet id: it may still land, so it is given longer.
        limit = pending_grace if live.get("droplet_id") is None else grace
        if now - float(live["ts"]) < limit:
            return False
    if live is not None and not droplets:
        L.append(ledger_path, L.RECLAIMED, now=now,
                 reason="ledger had an open interval but no tagged droplet exists")
        return True
    return False
