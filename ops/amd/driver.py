"""Run the session: three SFT arms, then one four-model ladder evaluation, on one AMD droplet.

    python ops/amd/driver.py dry-run            every command in order and the costed table; touches nothing
    python ops/amd/driver.py plan [--json]      the costed table only
    python ops/amd/driver.py preflight ...      read-only GETs: is this create going to work
    python ops/amd/driver.py create ... --yes   POST the droplet, wait for the IP, record it
    python ops/amd/driver.py bootstrap          upload the stage dir, run the ONE entry script
    python ops/amd/driver.py smoke              checklist, throughput bench, kill/resume, probe, go/no-go
    python ops/amd/driver.py project            re-run the go/no-go with new flags (no spend)
    python ops/amd/driver.py train              SFT A, B, A+B (resumable, skips finished arms)
    python ops/amd/driver.py serve              ONE vLLM: base + all adapters
    python ops/amd/driver.py tunnel [up|down]   ssh -L to the droplet (loopback)
    python ops/amd/driver.py eval               ladder on THIS laptop: L1 for all four, then L2..L4, control
    python ops/amd/driver.py sync               adapters + logs off the droplet, then verify them
    python ops/amd/driver.py destroy --yes      DELETE by tag, then GET until none remains
    python ops/amd/driver.py go                 train -> serve -> tunnel -> eval -> sync -> destroy
    python ops/amd/driver.py status             uptime and accrued cost, from the ledger

Nothing mutates the cloud without `--yes`, and `--yes` is only ever typed by the reviewer. Every
billed step passes a budget gate first (ledger accrual + the step + a reserve for sync/destroy
must stay under --budget and under the total cap), and every lifecycle event is appended to
ops/amd/ledger.jsonl. See docs/AMD_RUNBOOK.md.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

OPS = Path(__file__).resolve().parent
REPO_ROOT = OPS.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ops.amd import cloud  # noqa: E402
from ops.amd import ledger as L  # noqa: E402
from ops.amd import plan as P  # noqa: E402
from ops.amd.doapi import DoApi, load_dotenv, token_from_env  # noqa: E402

TRIAL_LINE = re.compile(r"^\[\d+/\d+\] .* reward=", re.M)


# ── configuration ─────────────────────────────────────────────────────────────────

def local_fingerprint(pub: Path) -> str:
    """MD5 fingerprint of a public key, the form DigitalOcean wants. Empty if unavailable."""
    try:
        out = subprocess.check_output(["ssh-keygen", "-E", "md5", "-lf", str(pub)], text=True)
        return out.split()[1].removeprefix("MD5:")
    except (OSError, subprocess.CalledProcessError, IndexError):
        return ""


def add_common(ap: argparse.ArgumentParser) -> None:
    g = ap.add_argument_group("hardware")
    g.add_argument("--fallback", action="store_true",
                   help=f"MI325X on-demand ({P.REGION_MI325X}, ${P.PRICE_MI325X}/h) instead of MI350X spot")
    g.add_argument("--size", default="")
    g.add_argument("--region", default="")
    g.add_argument("--price", type=float, default=0.0, help="override the hourly rate")
    g.add_argument("--image", default=P.IMAGE_DEFAULT)
    g.add_argument("--ssh-key-fingerprint", default="")
    g.add_argument("--ssh-pubkey", default=str(Path.home() / ".ssh" / "id_ed25519.pub"))
    g.add_argument("--tag", default=P.TAG)
    g = ap.add_argument_group("money")
    g.add_argument("--budget", type=float, default=P.DEFAULT_BUDGET, help="session cap, dollars")
    g.add_argument("--total-cap", type=float, default=P.TOTAL_CAP)
    g.add_argument("--deadline-minutes", type=float, default=0.0)
    g = ap.add_argument_group("work")
    g.add_argument("--arms", default=",".join(P.ARMS))
    g.add_argument("--max-length", type=int, default=8192)
    g.add_argument("--limit", type=int, default=250, help="tasks per rung per model")
    g.add_argument("--samples", type=int, default=2, help="samples at L1")
    g.add_argument("--late-samples", type=int, default=1, help="samples at L2..L4")
    g.add_argument("--rungs", default=",".join(P.RUNGS))
    g.add_argument("--workers", type=int, default=12)
    g.add_argument("--no-base", action="store_true", help="skip the base-model control")
    g.add_argument("--no-program-control", action="store_true")
    g.add_argument("--serve-mode", choices=["lora", "merged"], default="lora")
    g = ap.add_argument_group("where")
    g.add_argument("--host", default="", help="droplet IP (default: from the ledger)")
    g.add_argument("--identity", default="")
    g.add_argument("--commit", default="HEAD")
    g.add_argument("--stage-dir", default="/tmp/smol-ladder-stage")
    g.add_argument("--ledger", default=str(OPS / "ledger.jsonl"))


def build_config(args: argparse.Namespace) -> P.Config:
    arms = tuple(a.strip().upper() for a in args.arms.split(",") if a.strip())
    bad = [a for a in arms if a not in P.ARMS]
    if bad:
        raise SystemExit(f"unknown arm(s) {bad}; choose from {list(P.ARMS)}")
    rungs = tuple(r.strip() for r in args.rungs.split(",") if r.strip())
    if not rungs or any(r not in P.RUNGS for r in rungs):
        raise SystemExit(f"--rungs must be a subset of {list(P.RUNGS)}")
    fb = args.fallback
    fp = args.ssh_key_fingerprint or local_fingerprint(Path(args.ssh_pubkey))
    return P.Config(
        size=args.size or (P.SIZE_MI325X if fb else P.SIZE_MI350X),
        region=args.region or (P.REGION_MI325X if fb else P.REGION_MI350X),
        price=args.price or (P.PRICE_MI325X if fb else P.PRICE_MI350X),
        image=args.image, fingerprint=fp, tag=args.tag, budget=args.budget,
        total_cap=args.total_cap, deadline_minutes=args.deadline_minutes, arms=arms,
        max_length=args.max_length, limit=args.limit, samples=args.samples,
        late_samples=args.late_samples, rungs=rungs, workers=args.workers,
        include_base=not args.no_base, program_control=not args.no_program_control,
        serve_mode=args.serve_mode, host=args.host, identity=args.identity, commit=args.commit,
        stage_dir=args.stage_dir, ledger=args.ledger)


def tokens_for(cfg: P.Config) -> dict[str, P.SetTokens]:
    staged = P.load_tokens(Path(cfg.stage_dir) / "tokens.json", cfg.max_length)
    return staged or P.heuristic_tokens(REPO_ROOT / "data", cfg.max_length)


def with_host(cfg: P.Config, events: list[dict]) -> P.Config:
    """The droplet's IP comes from the ledger unless --host was given."""
    if not cfg.host:
        ready = L.latest(events, L.READY)
        live = L.open_interval(events)
        if ready and live and ready.get("droplet_id") == live.get("droplet_id"):
            cfg.host = ready.get("ip", "")
    return cfg


def make_plan(cfg: P.Config, events: list[dict]) -> tuple[list[P.Step], list[P.Row]]:
    meas = P.measured_from_ledger(events)
    tokens = tokens_for(cfg)
    rows = P.projection(cfg, tokens, meas)
    cfg.deadline_minutes = cfg.deadline_minutes or P.default_deadline_minutes(cfg, rows)
    return P.build_plan(cfg, tokens, meas), rows


# ── printing ──────────────────────────────────────────────────────────────────────

def print_table(cfg: P.Config, rows: list[P.Row], spent_session: float = 0.0) -> dict:
    total = P.total_dollars(rows, cfg.price)
    hours = sum(r.seconds for r in rows) / 3600.0
    kind = "spot" if cfg.spot else "on-demand"
    print(f"## costed plan: {cfg.size} in {cfg.region} ({kind}) at ${cfg.price}/h, "
          f"session budget ${cfg.budget:.2f}, total cap ${cfg.total_cap:.2f}, image {cfg.image}")
    print(f"  {'stage':<30} {'hours':>6} {'dollars':>8}  basis")
    for r in rows:
        print(f"  {r.stage:<30} {r.seconds / 3600.0:>6.2f} {r.dollars(cfg.price):>8.2f}  {r.basis}")
    print(f"  {'TOTAL':<30} {hours:>6.2f} {total:>8.2f}  "
          f"(session headroom ${cfg.budget - total:.2f}, credit headroom ${P.CREDIT - total:.2f})")
    if total > cfg.budget:
        print(f"  !! OVER THE SESSION BUDGET by ${total - cfg.budget:.2f}")
    if any(not r.measured and "UNMEASURED" in r.basis for r in rows):
        print("  !! rows marked UNMEASURED are placeholders; the smoke measures them and `project` "
              "re-renders this table before any arm is trained")
    return {"rows": [{"stage": r.stage, "seconds": round(r.seconds), "dollars":
                      round(r.dollars(cfg.price), 2), "basis": r.basis} for r in rows],
            "total_hours": round(hours, 2), "total_dollars": round(total, 2),
            "price_per_hour": cfg.price, "budget": cfg.budget, "within_budget": total <= cfg.budget}


def show_step(step: P.Step) -> None:
    where = {"laptop": "on this laptop", "droplet": "on the droplet (via ssh)", "api": "DigitalOcean API"}
    billed = f"{step.seconds / 60.0:.0f} min billed" if step.billed and step.seconds else "not billed"
    print(f"\n# [{step.phase}] {step.name} ({where[step.where]}; {billed})")
    print(f"#   {step.note}")
    if step.api:
        print(f"#   {step.api}")
    for i, cmd in enumerate(step.cmds):
        print(f"  $ {cmd.shell()}" + (" &" if len(step.cmds) > 1 else ""))
    if len(step.cmds) > 1:
        print("  $ wait")


def dry_run(cfg: P.Config, events: list[dict]) -> None:
    steps, rows = make_plan(cfg, events)
    print_table(cfg, rows)
    print(f"\n## evaluation protocol: every model, base included, under --agent bash "
          f"(the SmolDataEnvs-sft format the arms are trained in); base also under --agent program "
          f"as a one-turn control. Chat template kwargs {P.CHAT_KWARGS} everywhere.")
    print("\n## the whole session, in order. Nothing below is executed by this command.")
    phase = ""
    for step in steps:
        if step.phase != phase:
            phase = step.phase
        show_step(step)
    print("\n# afterwards: python ops/amd/driver.py status   (ledger says nothing is billing)")


# ── gates and bookkeeping ─────────────────────────────────────────────────────────

def gate(cfg: P.Config, step: P.Step, now: float | None = None) -> None:
    if not step.billed or step.seconds <= 0:
        return
    now = time.time() if now is None else now
    reserve = P.RESERVE_S if step.reserve else 0.0
    v = L.verdict(L.read(Path(cfg.ledger)), now, step.seconds, cfg.price, cfg.budget,
                  cfg.total_cap, reserve)
    print(f"## budget gate for {step.name}: {v.reason}")
    if not v.allowed:
        raise SystemExit(f"\nSTOPPING BEFORE '{step.name}'. {v.reason}\nNothing was started. Lower "
                         "--limit/--samples/--rungs, drop an arm, or raise --budget on purpose.")


def last_go(events: list[dict]) -> bool | None:
    for e in reversed(events):
        if e.get("event") == L.MEASURED and "go" in e:
            return bool(e["go"])
    return None


def run_cmd(cmd: P.Cmd, host: str, capture: bool = False) -> tuple[int, str]:
    argv = [a.replace("<droplet-ip>", host) for a in cmd.argv]
    env = dict(os.environ, **dict(cmd.env))
    if "<droplet-ip>" in " ".join(cmd.argv) and not host:
        raise SystemExit("no droplet IP: create it first or pass --host")
    print("  $ " + cmd.shell(), flush=True)
    if not capture:
        return subprocess.call(argv, env=env), ""
    proc = subprocess.run(argv, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True)
    sys.stdout.write(proc.stdout)
    return proc.returncode, proc.stdout


def run_parallel(cmds: list[P.Cmd], host: str) -> int:
    procs = []
    for cmd in cmds:
        print("  $ " + cmd.shell() + " &", flush=True)
        procs.append(subprocess.Popen(cmd.argv, env=dict(os.environ, **dict(cmd.env))))
    codes = [p.wait() for p in procs]
    return max(codes) if codes else 0


# ── measurements ──────────────────────────────────────────────────────────────────

def parse_measurements(data: dict) -> dict:
    """Fields of the ledger's `measured` event from the smoke's measurements.json."""
    best = data.get("best") or {}
    return {"tokens_per_s": best.get("tokens_per_s"), "batch_size": best.get("per_device_batch_size"),
            "grad_accum": best.get("grad_accum"),
            "checks_ok": data.get("checks_ok"), "resume_ok": (data.get("resume") or {}).get("ok")}


def parse_probe_serve(text: str) -> dict:
    out: dict = {}
    m = re.search(r"READY_AFTER_S=(\d+(?:\.\d+)?)", text)
    if m:
        out["probe_start_s"] = float(m.group(1))
    m = re.search(r"TOOL_CALLS_OK=([01])", text)
    if m:
        out["tool_calls_ok"] = m.group(1) == "1"
    return out


def seconds_per_trial(output: str, wall_seconds: float) -> float | None:
    """Wall seconds per trial at the probe's worker count, from the harness's own progress lines."""
    n = len(TRIAL_LINE.findall(output))
    return wall_seconds / n if n else None


def go_no_go(cfg: P.Config, events: list[dict], now: float) -> tuple[bool, list[str], dict]:
    """The decision the smoke ends with. NO-GO on any failed check, an unmeasured throughput, or a
    projection (money already spent + everything still to run) that passes either cap."""
    meas = P.measured_from_ledger(events)
    rows = P.projection(cfg, tokens_for(cfg), meas)
    spent = L.spend(events, now, cfg.price)
    remaining = P.remaining_after(rows, "sft")
    rem_dollars = P.total_dollars(remaining, cfg.price)
    reasons = []
    for label, val in (("ROCm/stack checklist", meas.checks_ok), ("kill-and-resume", meas.resume_ok),
                       ("LoRA serving + tool calls", meas.tool_calls_ok)):
        if val is not True:
            reasons.append(f"{label}: {'FAILED' if val is False else 'not measured'}")
    if not meas.tokens_per_s:
        reasons.append("training throughput not measured")
    if not meas.sec_per_trial:
        reasons.append("evaluation seconds-per-trial not measured")
    if spent.session + rem_dollars > cfg.budget:
        reasons.append(f"projected session ${spent.session + rem_dollars:.2f} "
                       f"(${spent.session:.2f} spent + ${rem_dollars:.2f} to come) exceeds the "
                       f"${cfg.budget:.2f} budget")
    if spent.total + rem_dollars > cfg.total_cap:
        reasons.append(f"projected total ${spent.total + rem_dollars:.2f} exceeds the "
                       f"${cfg.total_cap:.2f} cap")
    return (not reasons), reasons, {"spent": spent.session, "remaining": rem_dollars}


def cmd_project(cfg: P.Config, events: list[dict]) -> bool:
    ok, reasons, info = go_no_go(cfg, events, time.time())
    rows = P.projection(cfg, tokens_for(cfg), P.measured_from_ledger(events))
    print_table(cfg, rows)
    print(f"\n  spent so far ${info['spent']:.2f}; still to run ${info['remaining']:.2f}")
    if ok:
        print("\n### GO/NO-GO: GO")
    else:
        print("\n### GO/NO-GO: NO-GO (stop; nothing further will run)")
        for r in reasons:
            print(f"  - {r}")
        print("  options: --limit/--samples/--rungs smaller, drop an arm (--arms A,B), "
              "--max-length smaller, or raise --budget on purpose; then `project` again.")
    L.append(Path(cfg.ledger), L.MEASURED, go=ok, reasons=reasons)
    if ok:
        left = P.total_dollars(P.remaining_after(rows, "sft"), cfg.price) / cfg.price * 3600.0
        minutes = int(left * 1.25 / 60.0 + 15)
        print(f"  re-arm the dead-man switch from measurements (stop the old one first):\n"
              f"  $ python ops/amd/deadman.py --deadline-minutes {minutes} --budget {cfg.budget:g} "
              f"--total-cap {cfg.total_cap:g} --tag {cfg.tag}")
    return ok


# ── phases ────────────────────────────────────────────────────────────────────────

def run_steps(cfg: P.Config, steps: list[P.Step], api=None, only: str = "",
              new_session: bool = False) -> None:
    ledger = Path(cfg.ledger)
    Path(cfg.local_logs).mkdir(parents=True, exist_ok=True)
    for step in steps:
        if only and step.phase != only:
            continue
        events = L.read(ledger)
        with_host(cfg, events)
        show_step(step)
        gate(cfg, step)
        L.append(ledger, L.STEP_START, step=step.name, projected_seconds=step.seconds)
        t0 = time.time()
        code = 0
        out = ""
        if step.name == "create":
            code = do_create(cfg, api, new_session)
        elif step.name == "destroy":
            code = 0 if cloud.destroy(api, cfg.tag, ledger) else 1
        elif step.name == "go-no-go":
            code = 0 if cmd_project(cfg, L.read(ledger)) else 3
        elif step.name == "verify-sync":
            code = 0 if verify_sync(cfg) else 1
        elif step.name in ("tunnel", "probe-tunnel"):
            code = ensure_tunnel(cfg)
        elif step.name == "tunnel-down":
            code = subprocess.call(list(step.cmds[0].argv)) and 0
        elif len(step.cmds) > 1:
            code = run_parallel(step.cmds, cfg.host)
        else:
            code, out = run_cmd(step.cmds[0], cfg.host,
                                capture=step.name in ("probe-eval", "probe-serve", "smoke-checks"))
        wall = time.time() - t0
        L.append(ledger, L.STEP_END, step=step.name, code=code, seconds=round(wall, 1),
                 projected_seconds=step.seconds)
        if step.name == "smoke-pull" and code == 0:
            data = json.loads(Path(cfg.local_logs, "measurements.json").read_text())
            L.append(ledger, L.MEASURED, **parse_measurements(data))
        if step.name == "probe-serve":
            L.append(ledger, L.MEASURED, **parse_probe_serve(out))
        if step.name == "probe-eval" and code == 0:
            spt = seconds_per_trial(out, wall)
            if spt:
                L.append(ledger, L.MEASURED, sec_per_trial=round(spt, 2), probe_wall_s=round(wall, 1))
        if code != 0:
            raise SystemExit(f"step '{step.name}' exited {code}. Nothing was cleaned up: "
                             "`driver.py status` shows what is billing; `destroy --yes` stops it.")


def do_create(cfg: P.Config, api, new_session: bool = False) -> int:
    checks = cloud.preflight(api, cfg)
    for name, ok, detail in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}: {detail}")
    if not all(ok for _, ok, _ in checks):
        raise SystemExit("preflight failed; nothing was created")
    info = cloud.create(api, cfg, Path(cfg.ledger), new_session=new_session)
    cfg.host = info["ip"]
    print(f"  created droplet {info['droplet_id']} at {info['ip']} (billing started)")
    return 0


def ensure_tunnel(cfg: P.Config) -> int:
    if subprocess.call(list(P.tunnel_check(cfg).argv), stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL) == 0:
        print("  tunnel already up")
        return 0
    return subprocess.call(list(P.tunnel_up(cfg).argv))


def sha_check(root: Path, sums_name: str = "SHA256SUMS.artifacts") -> list[str]:
    """Names in the droplet's artifact checksum list that are missing or differ locally."""
    from ops.amd.stage import sha256_file
    sums = root / sums_name
    if not sums.exists():
        return [f"{sums_name} (missing)"]
    bad = []
    for line in sums.read_text().splitlines():
        digest, _, name = line.partition("  ")
        path = root / name
        if not name or not path.exists() or sha256_file(path) != digest:
            bad.append(name or line)
    return bad


def hub_repos(cfg: P.Config, namespace: str) -> dict[str, str]:
    base = {"A": "smol-ladder-sft-a", "B": "smol-ladder-sft-b", "AB": "smol-ladder-sft-ab"}
    return {a: f"{namespace}/{base[a]}" for a in P.ARMS if a in cfg.arms}


def verify_sync(cfg: P.Config, hub_files=None, runs_root: Path | None = None,
                namespace: str | None = None) -> bool:
    """After the sync, before the destroy: is everything that matters somewhere that is not the droplet?

    1. every adapter is readable on the Hub (`hub_files(repo)` lists a repo's files; injected so a
       test needs no network),
    2. the droplet's own checksum list matches the copies pulled to logs/amd/,
    3. every run tag has result files in the laptop's results tree (the laptop wrote them directly).
    """
    ok = True
    ns = namespace
    if hub_files is None:
        from huggingface_hub import HfApi
        from ops.amd.stage import resolve_namespace
        ns = ns or resolve_namespace()
        hub_files = HfApi().list_repo_files
    for arm, repo in hub_repos(cfg, ns or "<namespace>").items():
        try:
            files = set(hub_files(repo))
        except Exception as exc:  # noqa: BLE001 - any failure to read means "not verified"
            files, _ = set(), print(f"  hub {repo}: {exc}")
        good = {"adapter_config.json", "adapter_model.safetensors"} <= files
        print(f"  {'PASS' if good else 'FAIL'}  adapter {arm} readable on the Hub ({repo})")
        ok &= good
    bad = sha_check(Path(cfg.local_logs))
    print(f"  {'PASS' if not bad else 'FAIL'}  droplet checksums match local copies"
          + (f": {bad}" if bad else ""))
    ok &= not bad
    root = runs_root or (REPO_ROOT / "data" / "runs")
    for tag, expected in P.expected_trials(cfg).items():
        n = len(list((root / tag / cfg.split).glob("*/*/result.json"))) if (root / tag).exists() else 0
        n += len(list((root / tag / cfg.split).glob("*/*/s*/result.json"))) if (root / tag).exists() else 0
        good = n > 0
        print(f"  {'PASS' if good else 'FAIL'}  run tag {tag}: {n} result files (at most {expected})")
        ok &= good
    return ok


def cmd_status(cfg: P.Config, events: list[dict], api) -> None:
    now = time.time()
    if api is not None:
        try:
            droplets = cloud.tagged(api, cfg.tag)
            if cloud.reconcile(droplets, Path(cfg.ledger), now):
                print("  ledger said billing but no tagged droplet exists: closed it as reclaimed")
                events = L.read(Path(cfg.ledger))
            print(f"  API: {len(droplets)} droplet(s) tagged {cfg.tag}: "
                  f"{[(d.get('name'), d.get('status')) for d in droplets]}")
        except SystemExit as exc:
            print(f"  API check failed: {exc}")
    s = L.summarise(events, now, cfg.price)
    print("## status")
    print(f"  state      {s['state']}")
    print(f"  uptime     {s['uptime_seconds'] / 60.0:.1f} min   ip {s.get('ip') or '-'}")
    print(f"  accrued    session ${s['session_dollars']:.2f} of ${cfg.budget:.2f}; "
          f"total ${s['total_dollars']:.2f} of ${cfg.total_cap:.2f}  (${s['price_per_hour']}/h)")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    names = ["dry-run", "plan", "preflight", "create", "bootstrap", "smoke", "project", "train",
             "serve", "tunnel", "eval", "sync", "destroy", "go", "status", "verify-sync",
             "note-prior-spend"]
    for n in names:
        p = sub.add_parser(n)
        add_common(p)
        p.add_argument("--yes", action="store_true", help="really do the mutating call")
        p.add_argument("--json", action="store_true")
        if n == "create":
            p.add_argument("--new-session", action="store_true",
                           help="reset the session budget (default: only the first create does)")
        if n == "tunnel":
            p.add_argument("action", nargs="?", default="up", choices=["up", "down"])
        if n == "note-prior-spend":
            p.add_argument("dollars", type=float)
    args = ap.parse_args(argv)
    os.chdir(REPO_ROOT)
    if not os.environ.get("AMD_OFFLINE"):   # the tests set it: no .env, so no token, so no network
        load_dotenv(REPO_ROOT / ".env")
    cfg = build_config(args)
    events = L.read(Path(cfg.ledger))
    with_host(cfg, events)

    def client() -> DoApi:
        return DoApi(token_from_env())

    c = args.cmd
    if c == "dry-run":
        dry_run(cfg, events)
        return 0
    if c == "plan":
        _, rows = make_plan(cfg, events)
        table = print_table(cfg, rows)
        if args.json:
            print(json.dumps(table, indent=2))
        return 0
    if c == "status":
        try:
            api = client() if token_from_env() else None
        except SystemExit:
            api = None
        cmd_status(cfg, events, api)
        return 0
    if c == "note-prior-spend":
        L.append(Path(cfg.ledger), L.PRIOR, dollars=args.dollars)
        print(f"recorded ${args.dollars:.2f} of earlier spend toward the total cap")
        return 0
    if c == "preflight":
        res = cloud.preflight(client(), cfg)
        for name, ok, detail in res:
            print(f"  {'PASS' if ok else 'FAIL'}  {name}: {detail}")
        return 0 if all(ok for _, ok, _ in res) else 1
    if c == "project":
        return 0 if cmd_project(cfg, events) else 3
    if c == "verify-sync":
        return 0 if verify_sync(cfg) else 1

    steps, _ = make_plan(cfg, events)
    phases = {"create": ["create"], "bootstrap": ["bootstrap"], "smoke": ["smoke"],
              "train": ["train"], "serve": ["serve"], "eval": ["eval"], "sync": ["sync"],
              "destroy": ["destroy"], "tunnel": ["tunnel"]}

    if c == "create":
        if not args.yes:
            show_step(next(s for s in steps if s.name == "create"))
            print("\n(not sent: add --yes to create the droplet and start billing)")
            return 0
        run_steps(cfg, steps, client(), only="create", new_session=args.new_session)
        return 0
    if c == "destroy":
        if not args.yes:
            show_step(next(s for s in steps if s.name == "destroy"))
            print("\n(not sent: add --yes)")
            return 0
        api = client()
        ok = cloud.destroy(api, cfg.tag, Path(cfg.ledger))
        print(f"  billing stopped (verified by GET): {ok}")
        return 0 if ok else 1
    if c == "tunnel":
        if args.action == "down":
            return subprocess.call(list(P.tunnel_down(cfg).argv))
        return ensure_tunnel(cfg)

    if c in ("train", "go") and last_go(events) is not True:
        raise SystemExit("no GO on record: run `driver.py smoke` (or `project`) first. The "
                         "measured projection must fit the budget before anything is trained.")
    if c == "go":
        api = client()
        try:
            for ph in ("train", "serve", "tunnel", "eval", "sync"):
                run_steps(cfg, steps, api, only=ph)
        finally:
            # The destroy runs whatever happened above: a failed eval must not leave a GPU billing.
            run_steps(cfg, [s for s in steps if s.name in ("tunnel-down", "destroy")], api)
        return 0
    run_steps(cfg, steps, None, only=phases[c][0])
    return 0


if __name__ == "__main__":
    sys.exit(main())
