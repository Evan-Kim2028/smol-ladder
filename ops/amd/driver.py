"""Run the session: a gate on the released adapter, three SFT arms, then one five-model evaluation.

    python ops/amd/driver.py dry-run            every command in order and the costed table; touches nothing
    python ops/amd/driver.py plan [--json]      the costed table only
    python ops/amd/driver.py preflight ...      read-only GETs: is this create going to work
    python ops/amd/driver.py create ... --yes   POST the droplet, wait for the IP, record it
    python ops/amd/driver.py bootstrap          upload the stage dir, run the ONE entry script
    python ops/amd/driver.py smoke              checklist, throughput bench, kill/resume, then the GATE
    python ops/amd/driver.py gate               the gate alone (serve base + released adapter, 60 tasks)
    python ops/amd/driver.py gate-decide        re-read the gate's results and decide (no spend);
                                                --accept-gate continues past a rate that did not clear
    python ops/amd/driver.py project            re-run the go/no-go with new flags (no spend)
    python ops/amd/driver.py train              SFT A, B, A+B (resumable, auto-resume, skips finished arms)
    python ops/amd/driver.py serve              one vLLM per model (merged), every adapter checked
    python ops/amd/driver.py tunnel [up|down]   ssh -L to every served port (loopback)
    python ops/amd/driver.py eval               L1 for all five models, supervised, then STOP: read it,
                                                then buy more with --stage sample2|hints|control|rest
    python ops/amd/driver.py sync               adapters + logs off the droplet, then verify them
    python ops/amd/driver.py destroy --yes      DELETE by tag, then GET until none remains
    python ops/amd/driver.py go                 train -> serve -> tunnel -> eval -> sync -> destroy
    python ops/amd/driver.py status             droplet, uptime, dollars accrued and remaining, the
                                                caps, the $95 hard limit, whether the deadman lives
                                                (restart it after any laptop re-login) and which ssh
                                                agent is in use

Nothing mutates the cloud without `--yes`, and `--yes` is only ever typed by the reviewer. Every
billed step passes a budget gate first (ledger accrual + the step + a reserve for sync/destroy
must stay under --budget and under the total cap, which can never exceed the $95 hard limit) and a
deadman gate (a fresh heartbeat from the independent watcher, ops/amd/deadman.py), and every
lifecycle event is appended to ops/amd/ledger.jsonl. Run long commands detached
(`setsid nohup python ops/amd/driver.py go >> logs/go.log 2>&1 < /dev/null &`): SIGTERM and SIGHUP
are turned into exceptions so `go` still reaches its destroy, but a detached process never gets the
HUP at all. See docs/AMD_RUNBOOK.md.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import re
import signal
import subprocess
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path

OPS = Path(__file__).resolve().parent
REPO_ROOT = OPS.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ops.amd import cloud  # noqa: E402
from ops.amd import ledger as L  # noqa: E402
from ops.amd import plan as P  # noqa: E402
from ops.amd import supervise as SUP  # noqa: E402
from ops.amd import gate as G  # noqa: E402
from ops.amd.doapi import DoApi, child_env, load_dotenv, token_from_env  # noqa: E402

# ── configuration ─────────────────────────────────────────────────────────────────

def local_fingerprint(pub: Path) -> str:
    """MD5 fingerprint of a public key, the form DigitalOcean wants. Empty if unavailable."""
    try:
        out = subprocess.check_output(["ssh-keygen", "-E", "md5", "-lf", str(pub)], text=True)
        return out.split()[1].removeprefix("MD5:")
    except (OSError, subprocess.CalledProcessError, IndexError):
        return ""


def money(text: str) -> float:
    """argparse type for a dollar cap. A total cap above the hard limit is an error, not a clamp:
    typing 120 means the reviewer believes it is allowed, and it is not."""
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a number") from None
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def total_cap_arg(text: str) -> float:
    value = money(text)
    if value > P.HARD_TOTAL_LIMIT:
        raise argparse.ArgumentTypeError(
            f"${value:g} is above the ${P.HARD_TOTAL_LIMIT:g} HARD total limit, which no flag or "
            "environment variable can raise")
    return value


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
    g.add_argument("--budget", type=money, default=P.DEFAULT_BUDGET, help="session cap, dollars")
    g.add_argument("--total-cap", type=total_cap_arg, default=P.TOTAL_CAP,
                   help=f"working total cap, dollars (never above the ${P.HARD_TOTAL_LIMIT:g} hard limit)")
    g.add_argument("--deadline-minutes", type=float, default=0.0)
    g = ap.add_argument_group("work")
    g.add_argument("--arms", default=",".join(P.ARMS), help="arms to TRAIN")
    g.add_argument("--eval-only", default=",".join(f"{n}={r}" for n, r in P.EVAL_ONLY_DEFAULT),
                   help="Hub adapters that are served and evaluated but not trained, NAME=owner/repo "
                        "comma-separated (default: the released upstream adapter as R). R is the "
                        "gate's subject and must stay in the list")
    g.add_argument("--max-length", type=int, default=8192)
    g.add_argument("--limit", type=int, default=250, help="tasks per rung per model")
    g.add_argument("--samples", type=int, default=1,
                   help="samples at L1; a second one is its own purchase (eval --stage sample2)")
    g.add_argument("--late-samples", type=int, default=1, help="samples at L2..L4")
    g.add_argument("--rungs", default=",".join(P.RUNGS))
    g.add_argument("--workers", type=int, default=P.Config.workers,
                   help="parallel trials PER MODEL. 8 is measured safe; 20 hung every engine in "
                        f"session 1. Above {P.MAX_WORKERS_SAFE} needs --i-know")
    g.add_argument("--i-know", action="store_true",
                   help=f"allow --workers above {P.MAX_WORKERS_SAFE} (it stalled session 1's "
                        "evaluation for two hours)")
    g.add_argument("--tag-prefix", default=P.Config.tag_prefix,
                   help="run tags are <prefix>-<model>; reusing a prefix reuses its finished "
                        "trials, which is the point for the gate and a hazard for anything else")
    g.add_argument("--no-base", action="store_true", help="skip the base-model control")
    g.add_argument("--no-program-control", action="store_true")
    g = ap.add_argument_group("gate")
    g.add_argument("--gate-tasks", type=int, default=P.GATE_TASKS)
    g.add_argument("--gate-margin", type=float, default=P.Config.gate_margin,
                   help="the released adapter's pass rate must beat the base's by this much")
    g.add_argument("--gate-max-failures", type=int, default=P.Config.gate_max_failures,
                   help="harness failures tolerated per model")
    g.add_argument("--accept-gate", action="store_true",
                   help="continue although the released adapter did not beat the base by the "
                        "margin (the stack itself must still have passed)")
    g = ap.add_argument_group("supervision (eval and the gate)")
    g.add_argument("--stall-minutes", type=float, default=SUP.DEFAULT_STALL_MIN,
                   help="no new result.json for this long stops the harnesses and restarts the servers")
    g.add_argument("--error-window", type=int, default=SUP.DEFAULT_WINDOW,
                   help="harness-failure share is judged over each model's last N results")
    g.add_argument("--error-share", type=float, default=SUP.DEFAULT_ERROR_SHARE)
    g = ap.add_argument_group("training")
    g.add_argument("--ckpt-steps", type=int, default=P.Config.ckpt_steps,
                   help="checkpoint (and push to the Hub) every N steps")
    g.add_argument("--train-attempts", type=int, default=P.Config.train_attempts,
                   help="automatic resumes of one arm after a crash (a GPU reset costs one interval)")
    g = ap.add_argument_group("where")
    g.add_argument("--host", default="", help="droplet IP (default: from the ledger)")
    g.add_argument("--identity", default="")
    g.add_argument("--commit", default="HEAD")
    g.add_argument("--stage-dir", default="/tmp/smol-ladder-stage")
    g.add_argument("--ledger", default=str(OPS / "ledger.jsonl"))


def parse_eval_only(text: str) -> tuple[tuple[str, str], ...]:
    out = []
    for item in (t.strip() for t in text.split(",") if t.strip()):
        name, sep, repo = item.partition("=")
        if not sep:
            raise SystemExit(f"--eval-only entry {item!r} must be NAME=owner/repo")
        out.append((name.strip().upper(), repo.strip()))
    return tuple(out)


def build_config(args: argparse.Namespace) -> P.Config:
    arms = tuple(a.strip().upper() for a in args.arms.split(",") if a.strip())
    bad = [a for a in arms if a not in P.ARMS]
    if bad:
        raise SystemExit(f"unknown arm(s) {bad}; choose from {list(P.ARMS)}")
    rungs = tuple(r.strip() for r in args.rungs.split(",") if r.strip())
    if not rungs or any(r not in P.RUNGS for r in rungs):
        raise SystemExit(f"--rungs must be a subset of {list(P.RUNGS)}")
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")
    if args.workers > P.MAX_WORKERS_SAFE and not args.i_know:
        raise SystemExit(
            f"--workers {args.workers} is above {P.MAX_WORKERS_SAFE}. Session 1 measured 8 per model "
            "as safe and 20 per model as the setting under which every vLLM engine hung at zero "
            "throughput for two hours (about $5 gone). Pass --i-know to override.")
    fb = args.fallback
    fp = args.ssh_key_fingerprint or local_fingerprint(Path(args.ssh_pubkey))
    try:
        return P.Config(
            size=args.size or (P.SIZE_MI325X if fb else P.SIZE_MI350X),
            region=args.region or (P.REGION_MI325X if fb else P.REGION_MI350X),
            price=args.price or (P.PRICE_MI325X if fb else P.PRICE_MI350X),
            image=args.image, fingerprint=fp, tag=args.tag, budget=args.budget,
            total_cap=args.total_cap, deadline_minutes=args.deadline_minutes, arms=arms,
            eval_only=parse_eval_only(args.eval_only),
            max_length=args.max_length, limit=args.limit, samples=args.samples,
            late_samples=args.late_samples, rungs=rungs, workers=args.workers,
            tag_prefix=args.tag_prefix, include_base=not args.no_base,
            program_control=not args.no_program_control,
            host=args.host, identity=args.identity, commit=args.commit,
            stage_dir=args.stage_dir, ledger=args.ledger, gate_tasks=args.gate_tasks,
            gate_margin=args.gate_margin, gate_max_failures=args.gate_max_failures,
            ckpt_steps=args.ckpt_steps, train_attempts=args.train_attempts,
            stall_minutes=args.stall_minutes, error_window=args.error_window,
            error_share=args.error_share, accept_gate=args.accept_gate)
    except ValueError as exc:
        raise SystemExit(str(exc)) from None


def tokens_for(cfg: P.Config) -> dict[str, P.SetTokens]:
    """Staged token counts if they were made for the data on disk now, else the recount/heuristic.
    tokens.json is the source of truth (stage.py counts A and B with the real tokenizer and
    computes AB as A + B), but one staged from other data is refused, loudly: a plan costed from
    last session's v1 arm B is out by a factor of two."""
    data = REPO_ROOT / "data"
    staged = P.load_tokens(Path(cfg.stage_dir) / "tokens.json", cfg.max_length)
    if staged:
        why = P.stale_tokens(staged, data)
        if not why:
            return staged
        print(f"  note: ignoring the staged token counts ({why}); re-run ops/amd/stage.py",
              file=sys.stderr)
    return P.heuristic_tokens(data, cfg.max_length)


def with_host(cfg: P.Config, events: list[dict]) -> P.Config:
    """The droplet's IP comes from the ledger unless --host was given."""
    if not cfg.host:
        ready = L.latest(events, L.READY)
        live = L.open_interval(events)
        if ready and live and ready.get("droplet_id") == live.get("droplet_id"):
            cfg.host = ready.get("ip", "")
    return cfg


def hardware_of(cfg: P.Config) -> str:
    return f"{cfg.size}|{cfg.region}|{cfg.image}"


def measure_key(cfg: P.Config, events: list[dict]) -> tuple:
    """(droplet id, hardware) that measurements and a GO are valid for: the droplet that is live
    now. With none live the key matches nothing, so a stale GO or throughput from an earlier
    droplet (possibly on other hardware) can never be read as this one's."""
    live = L.open_interval(events)
    return (live.get("droplet_id") if live else "<no live droplet>", hardware_of(cfg))


def stamp(cfg: P.Config, events: list[dict]) -> dict:
    droplet_id, hardware = measure_key(cfg, events)
    return {"droplet_id": droplet_id, "hardware": hardware}


def make_plan(cfg: P.Config, events: list[dict]) -> tuple[list[P.Step], list[P.Row]]:
    meas = P.measured_from_ledger(events, measure_key(cfg, events))
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
          f"session budget ${cfg.budget:.2f}, total cap ${cfg.total_cap:.2f} "
          f"(HARD limit ${P.HARD_TOTAL_LIMIT:.2f}), image {cfg.image}")
    print(f"  {'stage':<38} {'hours':>6} {'dollars':>8}  basis")
    for r in rows:
        print(f"  {r.stage:<38} {r.seconds / 3600.0:>6.2f} {r.dollars(cfg.price):>8.2f}  {r.basis}")
    print(f"  {'TOTAL, every row':<38} {hours:>6.2f} {total:>8.2f}  "
          f"(session headroom ${cfg.budget - total:.2f}, credit headroom ${P.CREDIT - total:.2f})")
    st = P.staged_dollars(rows, cfg.price)
    print(f"  {'-' * 60}")
    print("  what the session costs, by where you stop:")
    print(f"    gate only (then destroy):        ${st['gate_only']:.2f}")
    print(f"    gate + train + L1 eval + sync:   ${st['core']:.2f}   <- the plan")
    for key, label in (("sample2", "+ L1 second sample"), ("hints", "+ L2-L4"),
                       ("control", "+ program control")):
        if st[key]:
            print(f"    {label + ':':<32} ${st[key]:.2f} incremental   (eval --stage {key})")
    print(f"    everything bought:               ${st['all']:.2f}")
    if st["core"] > cfg.budget:
        print(f"  !! THE PLAN IS OVER THE SESSION BUDGET by ${st['core'] - cfg.budget:.2f}")
    elif total > cfg.budget:
        print(f"  note: buying every optional stage would pass the session budget by "
              f"${total - cfg.budget:.2f}")
    if any(not r.measured and "UNMEASURED" in r.basis for r in rows):
        print("  !! rows marked UNMEASURED are placeholders; `project` re-renders this table")
    return {"rows": [{"stage": r.stage, "seconds": round(r.seconds), "dollars":
                      round(r.dollars(cfg.price), 2), "basis": r.basis} for r in rows],
            "total_hours": round(hours, 2), "total_dollars": round(total, 2),
            "price_per_hour": cfg.price, "budget": cfg.budget,
            "within_budget": st["core"] <= cfg.budget,
            "gate_only_dollars": round(st["gate_only"], 2),
            "core_dollars": round(st["core"], 2),
            "incremental_dollars": {"sample2": round(st["sample2"], 2),
                                    "L2-L4": round(st["hints"], 2),
                                    "control": round(st["control"], 2)}}


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
          f"(the SmolDataEnvs-sft format the arms are trained in, --bash-stop model); base also "
          f"under --agent program as a one-turn control. Chat template kwargs {P.CHAT_KWARGS} "
          f"everywhere. Models: {', '.join(P.eval_arms(cfg))} on ports "
          f"{', '.join(str(P.port_for(cfg, a)) for a in P.eval_arms(cfg))}; "
          f"{cfg.workers} workers per model.")
    print("\n## the whole session, in order. Nothing below is executed by this command.")
    phase = ""
    for step in steps:
        if step.phase != phase:
            phase = step.phase
        show_step(step)
    print("\n# afterwards: python ops/amd/driver.py status   (ledger says nothing is billing)")


# ── gates and bookkeeping ─────────────────────────────────────────────────────────

def deadman_command(cfg: P.Config) -> str:
    """The exact command that arms the watcher, detached so closing the terminal cannot kill it."""
    minutes = cfg.deadline_minutes or 600
    ledger = "" if Path(cfg.ledger).resolve() == (OPS / "ledger.jsonl").resolve() \
        else f" --ledger {cfg.ledger}"
    return (f"setsid nohup python ops/amd/deadman.py --deadline-minutes {minutes:g} "
            f"--budget {cfg.budget:g} --total-cap {cfg.total_cap:g} --price {cfg.price:g} "
            f"--tag {cfg.tag}{ledger} >> logs/deadman.log 2>&1 < /dev/null &")


def gate(cfg: P.Config, step: P.Step, now: float | None = None) -> None:
    """Refuse a billed step whose projection would pass either cap, or that has no live deadman.

    Steps with `reserve=False` (sync, tunnel-down, destroy) are the reserve itself: they print
    their verdict but are NEVER refused. A budget guard that can refuse the destroy is a guard that
    keeps the meter running at the exact moment it matters.

    The deadman check is a gate too: the watcher that destroys a runaway droplet is a separate
    process, and one that was never started, or died, looks the same as one with nothing to do. So
    `create` and every other billed step need a heartbeat written within three poll intervals.
    """
    if not step.billed or step.seconds <= 0:
        return
    now = time.time() if now is None else now
    reserve = P.RESERVE_S if step.reserve else 0.0
    v = L.verdict(L.read(Path(cfg.ledger)), now, step.seconds, cfg.price, cfg.budget,
                  cfg.total_cap, reserve)
    print(f"## budget gate for {step.name}: {v.reason}")
    if not v.allowed and step.reserve:
        raise SystemExit(f"\nSTOPPING BEFORE '{step.name}'. {v.reason}\nNothing was started. Lower "
                         "--limit/--samples/--rungs, drop an arm, or raise --budget on purpose.")
    if not v.allowed:
        print(f"## {step.name} runs anyway: it is what stops the spending")
    fresh, why = L.heartbeat_status(Path(cfg.ledger), now, tag=cfg.tag, budget=cfg.budget,
                                    total_cap=cfg.total_cap)
    print(f"## deadman gate for {step.name}: {why}")
    if fresh:
        return
    if not step.reserve:
        print(f"## {step.name} runs anyway: it is what stops the spending")
        return
    raise SystemExit(
        f"\nSTOPPING BEFORE '{step.name}'. No live dead-man switch ({why}).\nNothing was started. "
        "Nothing may bill without the independent watcher that destroys a runaway droplet. Start it "
        f"in another terminal (detached, so closing the terminal cannot kill it):\n\n    "
        f"{deadman_command(cfg)}\n\nthen check it with `driver.py status`. After ANY destroy the "
        "watcher has exited or is stale: re-arm it before the next create.")


def last_go(events: list[dict], key: tuple | None = None) -> bool | None:
    """The latest GO/NO-GO, counting only decisions stamped with `key` (see measure_key)."""
    for e in reversed(events):
        if e.get("event") == L.MEASURED and "go" in e:
            if key is not None and (e.get("droplet_id"), e.get("hardware")) != tuple(key):
                continue
            return bool(e["go"])
    return None


DEFAULT_STEP_TIMEOUT_S = 900.0
TIMEOUT_CODE = 124           # what `timeout(1)` exits with; no real step returns it


def _echo(line: str) -> None:
    sys.stdout.write(line)
    sys.stdout.flush()


def _stream(argv: list[str], env: dict, timeout: float | None) -> tuple[int, str]:
    """Run, echo every line AS IT ARRIVES (a smoke or a probe that prints nothing for ten billed
    minutes looks identical to one that is stuck), and return the full text. A timer kills the
    child at `timeout`: a hung ssh must not hold a billed droplet open forever."""
    proc = subprocess.Popen(argv, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    timed_out = threading.Event()

    def expire() -> None:
        timed_out.set()
        proc.kill()

    timer = threading.Timer(timeout, expire) if timeout else None
    if timer:
        timer.daemon = True
        timer.start()
    chunks: list[str] = []
    try:
        for line in proc.stdout:
            _echo(line)
            chunks.append(line)
        proc.wait()
    finally:
        if timer:
            timer.cancel()
        if proc.poll() is None:          # an exception (a SIGTERM turned into one) mid-stream
            proc.kill()
            proc.wait()
    if timed_out.is_set():
        _echo(f"  !! killed after {timeout:.0f}s (step timeout)\n")
        return TIMEOUT_CODE, "".join(chunks)
    return proc.returncode, "".join(chunks)


def run_cmd(cmd: P.Cmd, host: str, capture: bool = False,
            timeout: float | None = None) -> tuple[int, str]:
    """One command, with its output streamed and a timeout. `capture` only means "also return the
    text": output is never held back."""
    argv = [a.replace("<droplet-ip>", host) for a in cmd.argv]
    env = child_env(dict(cmd.env))
    if "<droplet-ip>" in " ".join(cmd.argv) and not host:
        raise SystemExit("no droplet IP: create it first or pass --host")
    print("  $ " + cmd.shell(), flush=True)
    if capture:
        return _stream(argv, env, timeout)
    try:
        return subprocess.call(argv, env=env, timeout=timeout), ""
    except subprocess.TimeoutExpired:
        print(f"  !! killed after {timeout:.0f}s (step timeout)", flush=True)
        return TIMEOUT_CODE, ""


def run_parallel(cmds: list[P.Cmd], host: str, capture: bool = False,
                 timeout: float | None = None) -> tuple[int, str]:
    """Concurrent commands, one process each. With `capture` their lines are streamed to the
    terminal (interleaved, unprefixed: the trial counter must stay at the start of a line) and
    returned together."""
    procs: list[subprocess.Popen] = []
    chunks: list[str] = []
    readers: list[threading.Thread] = []
    lock = threading.Lock()

    def pump(proc: subprocess.Popen) -> None:
        for line in proc.stdout:
            with lock:
                _echo(line)
                chunks.append(line)

    deadline = time.monotonic() + timeout if timeout else None
    codes: list[int] = []
    try:
        for cmd in cmds:
            print("  $ " + cmd.shell() + " &", flush=True)
            argv = [a.replace("<droplet-ip>", host) for a in cmd.argv]
            kw = dict(stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1) \
                if capture else {}
            proc = subprocess.Popen(argv, env=child_env(dict(cmd.env)), **kw)
            procs.append(proc)
            if capture:
                t = threading.Thread(target=pump, args=(proc,), daemon=True)
                t.start()
                readers.append(t)
        for proc in procs:
            try:
                codes.append(proc.wait(timeout=None if deadline is None
                                       else max(0.0, deadline - time.monotonic())))
            except subprocess.TimeoutExpired:
                print(f"  !! killed after {timeout:.0f}s (step timeout)", flush=True)
                codes.append(TIMEOUT_CODE)
                break
    finally:
        for proc in procs:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
        for t in readers:
            t.join(timeout=5)
    return (max(codes) if codes else 0), "".join(chunks)


# ── measurements ──────────────────────────────────────────────────────────────────

def parse_measurements(data: dict) -> dict:
    """Fields of the ledger's `measured` event from the smoke's measurements.json."""
    best = data.get("best") or {}
    return {"tokens_per_s": best.get("tokens_per_s"), "batch_size": best.get("per_device_batch_size"),
            "grad_accum": best.get("grad_accum"),
            "checks_ok": data.get("checks_ok"), "resume_ok": (data.get("resume") or {}).get("ok")}


ADAPTER_CHECK = re.compile(r"^ADAPTER_CHECK model=(\S+) differs=([01]) tool_calls_ok=([01])", re.M)
MERGE_OK = re.compile(r"^MERGE_OK model=(\S+) modules_applied=(\d+) tensors_changed=(\d+)", re.M)


def parse_serve(text: str) -> dict:
    """What serve.sh printed about each adapter it served: the merge report's verdict, whether a
    parsed `bash` tool call came back, whether its temperature-0 output differs from the base's."""
    checks = {m.group(1): {"differs": m.group(2) == "1", "tool_calls_ok": m.group(3) == "1"}
              for m in ADAPTER_CHECK.finditer(text)}
    merges = {m.group(1): {"modules_applied": int(m.group(2)), "tensors_changed": int(m.group(3))}
              for m in MERGE_OK.finditer(text)}
    return {"adapter_checks": checks, "merge_checks": merges}


def go_no_go(cfg: P.Config, events: list[dict], now: float) -> tuple[bool, list[str], dict]:
    """The decision the smoke ends with. NO-GO on any failed check, an unmeasured throughput, or a
    projection (money already spent + everything still to run) that passes either cap."""
    meas = P.measured_from_ledger(events, measure_key(cfg, events))
    rows = P.projection(cfg, tokens_for(cfg), meas)
    spent = L.spend(events, now, cfg.price)
    remaining = P.remaining_after(rows, "sft")
    rem_dollars = P.total_dollars(remaining, cfg.price)
    reasons = []
    for label, val in (("ROCm/stack checklist", meas.checks_ok), ("kill-and-resume", meas.resume_ok),
                       ("released adapter: tool calls through the served stack", meas.tool_calls_ok),
                       ("released adapter's output differs from the base's on a training prompt",
                        meas.adapter_differs),
                       ("the gate (released adapter vs base through the harness)", meas.gate_go)):
        if val is not True:
            reasons.append(f"{label}: {'FAILED' if val is False else 'not measured'}")
    if not meas.tokens_per_s:
        reasons.append("training throughput not measured")
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
    rows = P.projection(cfg, tokens_for(cfg), P.measured_from_ledger(events, measure_key(cfg, events)))
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
    L.append(Path(cfg.ledger), L.MEASURED, go=ok, reasons=reasons, **stamp(cfg, events))
    if ok:
        left = P.total_dollars(P.remaining_after(rows, "sft"), cfg.price) / cfg.price * 3600.0
        minutes = int(left * 1.25 / 60.0 + 15)
        print(f"  re-arm the dead-man switch from measurements (stop the old one first):\n"
              f"  $ {deadman_command(replace(cfg, deadline_minutes=minutes))}")
    return ok


# ── phases ────────────────────────────────────────────────────────────────────────

CAPTURED = ("gate-serve", "serve", "smoke-checks")   # output parsed for measurements


class SyncUnverified(SystemExit):
    """verify-sync failed. A SystemExit so nothing downstream swallows it, and a type of its own so
    `go` can tell "what matters is not off the droplet" from any other failure."""


class Terminated(SystemExit):
    """SIGTERM or SIGHUP, turned into an exception so the `finally` blocks run."""

    def __init__(self, signum: int):
        super().__init__(128 + signum)
        self.signum = signum


def _raise_terminated(signum, frame) -> None:   # noqa: ARG001 - the signature signal() demands
    raise Terminated(signum)


def install_signal_handlers() -> None:
    """Python runs `finally` on an exception, and by default SIGTERM and SIGHUP are not one: the
    process just dies, with the droplet still billing. Turn both into `Terminated`. Run long
    commands detached (`setsid nohup ... &`, see the runbook) so a closed terminal does not even
    send them; this is the second line of defence, not the first."""
    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, _raise_terminated)


@contextlib.contextmanager
def shielded_signals():
    """Ignore SIGTERM, SIGHUP and SIGINT for the duration. The cleanup in a `finally` must not be
    interrupted by the second Ctrl-C or the second SIGTERM that a nervous operator sends."""
    sigs = (signal.SIGTERM, signal.SIGHUP, signal.SIGINT)
    try:
        old = {s: signal.signal(s, signal.SIG_IGN) for s in sigs}
    except ValueError:                     # not the main thread: nothing to shield
        yield
        return
    try:
        yield
    finally:
        for s, handler in old.items():
            signal.signal(s, handler)


RUNS_ROOT = REPO_ROOT / "data" / "runs"


def progress_log(cfg: P.Config, step: P.Step) -> Path:
    return Path(cfg.local_logs) / f"progress-{step.name}.log"


def record_serve(cfg: P.Config, out: str, gate: bool) -> None:
    """Ledger what serve.sh reported about each adapter, stamped with this droplet. The evaluation
    refuses to start unless the latest record covers every adapter it will evaluate; the gate's
    own record carries the released adapter's tool-call and differs-from-base verdicts."""
    ledger = Path(cfg.ledger)
    found = parse_serve(out)
    extra: dict = {}
    adapter = P.served_name(P.GATE_ARM)
    if gate and adapter in found["adapter_checks"]:
        extra = {"tool_calls_ok": found["adapter_checks"][adapter]["tool_calls_ok"],
                 "adapter_differs": found["adapter_checks"][adapter]["differs"]}
    L.append(ledger, L.MEASURED, serve_gate=gate, **found, **extra, **stamp(cfg, L.read(ledger)))


def eval_blockers(cfg: P.Config, events: list[dict]) -> list[str]:
    """Why the evaluation must not start (empty = it may). Wires the guards into the plan: a
    passed gate on THIS droplet, and a serve step whose merge reports and adapter checks cover
    every adapter that is about to be evaluated."""
    key = measure_key(cfg, events)
    why = []
    if P.measured_from_ledger(events, key).gate_go is not True:
        why.append("no gate GO on record for this droplet (run `driver.py smoke`, or `gate` and "
                   "`gate-decide`): evaluating before the stack is proven is how session 1 wasted "
                   "its money")
    serve = None
    for e in reversed(events):
        if e.get("event") == L.MEASURED and "adapter_checks" in e and not e.get("serve_gate") \
                and (e.get("droplet_id"), e.get("hardware")) == tuple(key):
            serve = e
            break
    if serve is None:
        why.append("no `serve` on record for this droplet (it merges, checks and starts every model)")
    else:
        for arm in P.served_adapters(cfg):
            name = P.served_name(arm)
            if name not in serve.get("merge_checks", {}):
                why.append(f"{name}: no passing merge report on record")
            check = serve.get("adapter_checks", {}).get(name)
            if not check:
                why.append(f"{name}: no adapter check on record")
            elif not check.get("differs"):
                why.append(f"{name}: its temperature-0 output is IDENTICAL to the base's (the "
                           "adapter is not applied)")
            elif not check.get("tool_calls_ok"):
                why.append(f"{name}: no parsed bash tool call came back")
    return why


def restart_servers(cfg: P.Config, serve_step: P.Step, runner=None, tunnel=None) -> bool:
    """Bring the servers back after a stall: the same serve step (it stops everything, waits for
    the GPU memory to be released, and retries a failed engine start once), then the tunnel."""
    runner = runner or run_cmd
    tunnel = tunnel or ensure_tunnel
    code, out = runner(serve_step.cmds[0], cfg.host, capture=True,
                       timeout=P.step_timeout(serve_step))
    record_serve(cfg, out, gate=serve_step.name == "gate-serve")
    return code == 0 and tunnel(cfg) == 0


def run_supervised(cfg: P.Config, step: P.Step, steps: list[P.Step], *, sup_cls=None,
                   runs_root: Path | None = None, launch=None, restart=None, sleep=None,
                   clock=None, probe=None) -> int:
    """An evaluation step under the supervisor (see ops/amd/supervise.py). Everything injectable
    is a parameter so a test drives it without a process, a socket or a sleep."""
    ledger = Path(cfg.ledger)
    serve_name = "gate-serve" if step.name == "gate-eval" else "serve"
    serve_step = next(s for s in steps if s.name == serve_name)
    log_path = progress_log(cfg, step)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(line: str) -> None:
        print(line, flush=True)
        with log_path.open("a") as fh:
            fh.write(line + "\n")

    def start(retry: bool) -> list:
        procs = []
        for cmd in step.cmds:
            c = P.with_retry_failed(cmd) if retry else cmd
            print("  $ " + c.shell() + " &", flush=True)
            procs.append(SUP.Proc(list(c.argv), child_env(dict(c.env))))
        return procs

    extra = {k: v for k, v in (("sleep", sleep), ("clock", clock), ("probe", probe)) if v}
    sup = (sup_cls or SUP.Supervisor)(
        name=step.name, targets=[SUP.Target.from_cmd(c.argv, c.env) for c in step.cmds],
        results_root=runs_root or RUNS_ROOT, launch=launch or start,
        restart_servers=restart or (lambda: restart_servers(cfg, serve_step)), log=log,
        accrued=lambda: L.spend(L.read(ledger), time.time(), cfg.price).session,
        stall_s=cfg.stall_minutes * 60.0, window=cfg.error_window, error_share=cfg.error_share,
        **extra)
    outcome = sup.run()
    if outcome.code == 0 and step.name == "gate-eval":
        L.append(ledger, L.MEASURED, trials_per_min=round(sup.final_rate, 2),
                 **stamp(cfg, L.read(ledger)))
    if outcome.code != 0:
        print(f"\n!! {outcome.message}", flush=True)
        L.append(ledger, L.NOTE, text=outcome.message)
    return outcome.code


def gate_task_ids(cfg: P.Config, runs_root: Path) -> list[str]:
    """The fixed subset the gate ran: recorded when the gate's evaluation finished, so a decision
    re-read after the full L1 run (same run tags, 250 tasks) still judges the same 60."""
    f = Path(cfg.local_logs) / "gate_tasks.json"
    if f.exists():
        return json.loads(f.read_text())
    return G.task_ids_present(runs_root, [P.run_tag(cfg, a) for a in P.gate_arms(cfg)], cfg.split)


def record_gate_tasks(cfg: P.Config, runs_root: Path) -> list[str]:
    ids = G.task_ids_present(runs_root, [P.run_tag(cfg, a) for a in P.gate_arms(cfg)], cfg.split)
    f = Path(cfg.local_logs) / "gate_tasks.json"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(ids))
    return ids


def cmd_gate_decide(cfg: P.Config, runs_root: Path | None = None) -> bool:
    """Read the gate's results and the serve checks, print the report, record the verdict."""
    root = runs_root or RUNS_ROOT
    ledger = Path(cfg.ledger)
    events = L.read(ledger)
    key = measure_key(cfg, events)
    meas = P.measured_from_ledger(events, key)
    ids = gate_task_ids(cfg, root)
    adapter = P.GATE_ARM
    name = P.served_name(adapter)
    merge_ok = None
    for e in reversed(events):
        if e.get("event") == L.MEASURED and e.get("serve_gate") \
                and (e.get("droplet_id"), e.get("hardware")) == tuple(key):
            merge_ok = name in e.get("merge_checks", {})
            break
    base = G.stats_for("base", G.read_results(root, P.run_tag(cfg, "base"), cfg.split), ids)
    other = G.stats_for(adapter, G.read_results(root, P.run_tag(cfg, adapter), cfg.split), ids)
    dec = G.decide(base, other, adapter_name=adapter, differs=meas.adapter_differs,
                   tool_calls_ok=meas.tool_calls_ok, merge_ok=merge_ok, margin=cfg.gate_margin,
                   max_failures=cfg.gate_max_failures, accepted=cfg.accept_gate)
    print("\n".join(dec.lines))
    L.append(ledger, L.MEASURED, gate_go=dec.go, gate_accepted=dec.accepted,
             gate_reasons=dec.hard_failures, **stamp(cfg, L.read(ledger)))
    return dec.go


def run_steps(cfg: P.Config, steps: list[P.Step], api=None, only: str | tuple = "",
              new_session: bool = False, names: tuple = ()) -> None:
    ledger = Path(cfg.ledger)
    Path(cfg.local_logs).mkdir(parents=True, exist_ok=True)
    phases = (only,) if isinstance(only, str) and only else tuple(only or ())
    for step in steps:
        if phases and step.phase not in phases:
            continue
        if names and step.name not in names:
            continue
        events = L.read(ledger)
        with_host(cfg, events)
        show_step(step)
        if step.phase == "eval":
            blockers = eval_blockers(cfg, events)
            if blockers:
                raise SystemExit("\nSTOPPING BEFORE the evaluation. Nothing was started:\n  - "
                                 + "\n  - ".join(blockers))
        gate(cfg, step)
        L.append(ledger, L.STEP_START, step=step.name, projected_seconds=step.seconds)
        t0 = time.time()
        code = 0
        out = ""
        if step.name == "create":
            code = do_create(cfg, api, new_session)
        elif step.name == "destroy":
            code = 0 if cloud.destroy(api, cfg.tag, ledger) else 1
        elif step.name == "wait-ssh":
            code = wait_for_ssh(cfg)
        elif step.name == "upload":
            check_stage_for_upload(cfg)
            code, out = run_cmd(step.cmds[0], cfg.host, timeout=P.step_timeout(step))
        elif step.name == "go-no-go":
            code = 0 if cmd_project(cfg, L.read(ledger)) else 3
        elif step.name == "gate-decide":
            code = 0 if cmd_gate_decide(cfg) else 3
        elif step.name == "verify-sync":
            code = 0 if verify_sync(cfg, ran=ran_steps(L.read(ledger))) else 1
        elif step.name in ("tunnel", "gate-tunnel"):
            code = ensure_tunnel(cfg)
        elif step.name == "tunnel-down":
            code = subprocess.call(list(step.cmds[0].argv), env=child_env()) and 0
        elif step.phase == "eval" or step.name == "gate-eval":
            code = run_supervised(cfg, step, steps)
            if step.name == "gate-eval" and code == 0:
                record_gate_tasks(cfg, RUNS_ROOT)
        elif len(step.cmds) > 1:
            code, out = run_parallel(step.cmds, cfg.host, capture=step.name in CAPTURED,
                                     timeout=P.step_timeout(step))
        else:
            code, out = run_cmd(step.cmds[0], cfg.host, capture=step.name in CAPTURED,
                                timeout=P.step_timeout(step))
        wall = time.time() - t0
        L.append(ledger, L.STEP_END, step=step.name, code=code, seconds=round(wall, 1),
                 projected_seconds=step.seconds)
        if step.name == "bootstrap" and code == 0:
            # The HF token has done its job (it is on the droplet now); it must not sit in /tmp.
            Path(cfg.stage_dir, "remote.env").unlink(missing_ok=True)
        if step.name == "smoke-pull" and code == 0:
            data = json.loads(Path(cfg.local_logs, "measurements.json").read_text())
            L.append(ledger, L.MEASURED, **parse_measurements(data), **stamp(cfg, L.read(ledger)))
        if step.name in ("gate-serve", "serve"):
            record_serve(cfg, out, gate=step.name == "gate-serve")
        if code != 0 and step.name == "verify-sync":
            raise SyncUnverified("verify-sync FAILED: something that matters is not safely off "
                                 "the droplet (the FAIL lines above say what).")
        if code != 0:
            raise SystemExit(f"step '{step.name}' exited {code}. Nothing was cleaned up: "
                             "`driver.py status` shows what is billing; `destroy --yes` stops it.")


BEST_EFFORT_SYNC_TIMEOUT_S = 420.0
HOLD_S = 1800.0      # how long a droplet may be held for the reviewer after a failed verify-sync


def best_effort_sync(cfg: P.Config, steps: list[P.Step], runner=None,
                     timeout: float = BEST_EFFORT_SYNC_TIMEOUT_S) -> bool:
    """Try to get the adapters and logs off the droplet right before it is destroyed. Bounded
    (each command has a timeout) and guarded (nothing it raises may stop the destroy that
    follows). Returns whether both commands exited 0; it verifies nothing, because it exists for
    the case where the normal path did not get to run."""
    runner = runner or run_cmd
    if not cfg.host:
        print("  best-effort sync skipped: no droplet IP is known")
        return False
    ok = True
    for step in steps:
        if step.name not in ("sync-droplet", "sync-pull"):
            continue
        try:
            code, _ = runner(step.cmds[0], cfg.host, timeout=timeout)
        except (Exception, SystemExit) as exc:   # noqa: BLE001 - the destroy must still run
            print(f"  best-effort {step.name} raised {type(exc).__name__}: {exc}")
            ok = False
            continue
        print(f"  best-effort {step.name}: exit {code}")
        ok &= code == 0
    L.append(Path(cfg.ledger), L.NOTE, text=f"best-effort sync before destroy: ok={ok}")
    return ok


def forced_destroy_reason(cfg: P.Config, now: float) -> str:
    """Why the droplet cannot be held for the reviewer to fix a failed sync ("" if it can): holding
    it for HOLD_S plus the destroy would pass a cap, or the deadman's deadline falls inside the
    hold. Money and time win over the sync: the droplet is destroyed and what was lost is reported."""
    ledger = Path(cfg.ledger)
    v = L.verdict(L.read(ledger), now, HOLD_S, cfg.price, cfg.budget, cfg.total_cap, P.DESTROY_S)
    if not v.allowed:
        return v.reason
    beat = L.heartbeat_read(ledger)
    if beat and float(beat.get("deadline") or 0) < now + HOLD_S + P.DESTROY_S:
        return "the dead-man switch's deadline falls within the next half hour"
    return ""


def go(cfg: P.Config, steps: list[P.Step], api, now=time.time) -> int:
    """train -> serve -> tunnel -> eval -> sync -> destroy, and the destroy happens.

    * A step that raises, a refused gate, Ctrl-C, SIGTERM or SIGHUP (see install_signal_handlers):
      the `finally` first makes a bounded best-effort sync, then destroys. A failed evaluation must
      not leave a GPU billing, and must not take the adapters down with it.
    * verify-sync fails in the normal path: STOP BEFORE the destroy and say so. The adapters are
      the one thing that cannot be recomputed cheaply, so the droplet is held for the reviewer
      (the dead-man switch still bounds it) UNLESS the budget or the deadline cannot afford that,
      in which case it is destroyed and what was lost is reported.
    """
    hold = False
    synced = False
    try:
        for ph in ("train", "serve", "tunnel", "eval", "sync"):
            run_steps(cfg, steps, api, only=ph)
        synced = True
    except SyncUnverified as exc:
        forced = forced_destroy_reason(cfg, now())
        if forced:
            print(f"\n!! {exc}\n!! FORCED DESTROY: {forced}\n!! Destroying anyway after one more "
                  "best-effort sync. Anything in the FAIL lines above that is not on the Hub is "
                  "LOST; the private Hub repos are the only copies that survive.")
        else:
            hold = True
            print(f"\n!! {exc}\n!! STOPPING BEFORE THE DESTROY. The droplet is still up and "
                  "BILLING. Fix it and run `driver.py sync`, then `driver.py destroy --yes`. The "
                  "dead-man switch destroys it at its deadline regardless.")
        raise
    finally:
        if not hold:
            with shielded_signals():
                if not synced:
                    best_effort_sync(cfg, steps)
                run_steps(cfg, [s for s in steps if s.name in ("tunnel-down", "destroy")], api)
    return 0


SSH_WAIT_S = 600.0
SSH_PROBE_TIMEOUT_S = 60.0


def ssh_is_ready(cfg: P.Config, code: int, out: str) -> bool:
    """The command really ran: it exited 0 AND printed the exact line only a finished shell makes.
    A first-boot droplet answers sshd and prints "Please wait while we get your droplet ready..."
    while running nothing, so an exit status alone proves nothing."""
    return code == 0 and f"{P.READY_PREFIX}{cfg.user}" in (ln.strip() for ln in out.splitlines())


def wait_for_ssh(cfg: P.Config, runner=None, sleep=time.sleep, clock=time.monotonic,
                 limit: float = SSH_WAIT_S, interval: float = 10.0) -> int:
    """A droplet is `active` in the API before it runs commands (sshd answers first, then the
    image's first-boot setup holds every command for minutes). Poll `echo READY_$(whoami)` until
    its exact output returns, up to `limit` seconds, each try bounded by a timeout."""
    runner = runner or run_cmd
    cmd = P.ssh_ready(cfg)
    start = clock()
    attempt = 0
    while True:
        attempt += 1
        code, out = runner(cmd, cfg.host, capture=True, timeout=SSH_PROBE_TIMEOUT_S)
        if ssh_is_ready(cfg, code, out):
            print(f"  the droplet runs commands (attempt {attempt})")
            return 0
        if clock() - start >= limit:
            print(f"  the droplet did not run a command within {limit:.0f}s ({attempt} attempts)")
            return 1
        print(f"  not ready (attempt {attempt}: exit {code}, no {P.READY_PREFIX}{cfg.user} line); "
              f"retrying in {interval:.0f}s")
        sleep(interval)


def head_commit(repo: Path = REPO_ROOT) -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True,
                          check=True).stdout.strip()


def check_stage_for_upload(cfg: P.Config, head: str | None = None, data_dir: Path | None = None) -> None:
    """Refuse to upload a stage dir that is not the one for THIS commit and these token counts.

    A stage dir from an earlier session still has a remote.env in it and would otherwise be
    uploaded as if current: the droplet would then run older code than the reviewer approved, with
    token counts (hence a projection and a budget) from other data. So `repo.txt` must equal HEAD
    and `tokens.json` must be for this --max-length and these data files; otherwise: re-run stage.py."""
    stage = Path(cfg.stage_dir)
    again = "Re-run `ops/amd/stage.py` (free) and upload again."
    if not (stage / "remote.env").exists():
        raise SystemExit(f"{stage}/remote.env is missing. It is deleted after every successful "
                         f"bootstrap on purpose (it holds the HF token). {again}")
    head = head or head_commit()
    repo_txt = stage / "repo.txt"
    staged = repo_txt.read_text().strip() if repo_txt.exists() else ""
    if staged != head:
        raise SystemExit(f"{stage}/repo.txt is {staged[:10] or 'missing'} but HEAD is {head[:10]}: the "
                         f"stage dir holds other code than the commit you are running. {again}")
    tokens = P.load_tokens(stage / "tokens.json", cfg.max_length)
    if tokens is None:
        raise SystemExit(f"{stage}/tokens.json is missing or was counted at another --max-length "
                         f"(this run: {cfg.max_length}). {again}")
    stale = P.stale_tokens(tokens, data_dir or REPO_ROOT / "data")
    if stale:
        raise SystemExit(f"{stage}/tokens.json is stale: {stale}. {again}")


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
                       stderr=subprocess.DEVNULL, env=child_env()) == 0:
        print("  tunnel already up")
        return 0
    return subprocess.call(list(P.tunnel_up(cfg).argv), env=child_env())


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


EVAL_STAGES = {
    "L1": ("eval-L1",),
    "sample2": ("eval-sample2",),
    "hints": ("eval-L2", "eval-L3", "eval-L4"),
    "control": ("eval-control",),
    "rest": ("eval-sample2", "eval-L2", "eval-L3", "eval-L4", "eval-control"),
    "all": ("eval-L1", "eval-sample2", "eval-L2", "eval-L3", "eval-L4", "eval-control"),
}


def select_eval(steps: list[P.Step], stage: str) -> list[P.Step]:
    """The plan with only the requested evaluation stage's steps in the eval phase. `L1` runs every
    model at L1 and stops: the reviewer reads those numbers, then buys more or not."""
    keep = set(EVAL_STAGES[stage])
    chosen = [s for s in steps if s.phase != "eval" or s.name in keep]
    if not any(s.phase == "eval" for s in chosen):
        raise SystemExit(f"--stage {stage} selects nothing: the plan has no such evaluation step "
                         "(check --rungs / --no-base / --no-program-control)")
    return chosen


def ran_steps(events: list[dict]) -> set[str]:
    """Names of steps that have finished with exit 0 (the evaluation's results live on the laptop,
    so this is read across droplets and sessions)."""
    return {e["step"] for e in events if e.get("event") == L.STEP_END and e.get("code") == 0}


def hub_repos(cfg: P.Config, namespace: str) -> dict[str, str]:
    return {a: f"{namespace}/{P.hub_name(a)}" for a in P.ARMS if a in cfg.arms}


TRIAL_FLOOR = 0.9     # a run tag must hold at least this fraction of the trials it was asked for


def trial_files(root: Path, tag: str, split: str, rung: str) -> int:
    """Result files the harness ran to the end (agent_status "exit 0"). A file that records a
    harness failure is not a trial: a tag of 250 timeouts must not pass for a finished sweep."""
    base = root / tag / split
    if not base.exists():
        return 0
    files = list(base.glob(f"*/{rung}/result.json")) + list(base.glob(f"*/{rung}/s*/result.json"))
    return sum(not SUP.is_failure(f) for f in files)


def required_trials(cfg: P.Config, ran: set[str] | None = None) -> dict[str, tuple[str, int, int]]:
    """run tag -> (rung counted, expected, minimum). Only the FIRST rung of the plan is counted:
    every task gets it, whereas L2-L4 skip tasks that have no verified reference, so for them the
    expected count is an upper bound and a floor on it would fail a healthy run. `ran` limits the
    check to the stages that actually ran (`eval --stage L1` must not be failed for the hint rungs
    it was told not to run)."""
    rung = "L1" if "L1" in cfg.rungs else cfg.rungs[0]
    n = cfg.samples if rung == "L1" else cfg.late_samples
    if rung == "L1" and ran is not None and "eval-sample2" in ran:
        n = max(n, 2)
    out: dict[str, tuple[str, int, int]] = {}
    if ran is None or f"eval-{rung}" in ran:
        for arm in P.eval_arms(cfg):
            out[P.run_tag(cfg, arm)] = (rung, cfg.limit * n, math.ceil(TRIAL_FLOOR * cfg.limit * n))
    if cfg.include_base and cfg.program_control and (ran is None or "eval-control" in ran):
        out[f"{P.run_tag(cfg, 'base')}-program"] = (
            "L1", cfg.limit, math.ceil(TRIAL_FLOOR * cfg.limit))
    return out


def verify_sync(cfg: P.Config, hub_files=None, runs_root: Path | None = None,
                namespace: str | None = None, ran: set[str] | None = None,
                hub_sha256=None) -> bool:
    """After the sync, before the destroy: is everything that matters somewhere that is not the droplet?

    1. every adapter on the Hub is the FINAL one (the `final.done` marker is there) and is byte for
       byte the adapter pulled to the laptop: `hub_sha256(repo, filename)` against the sha256 of
       logs/amd/adapters/<arm>/adapter_model.safetensors (both injected, so a test needs no
       network). Two files existing at the repo root proves nothing: a checkpoint's adapter, or a
       previous session's invalid one, has the same names,
    2. the droplet's own checksum list matches the copies pulled to logs/amd/,
    3. every run tag that was evaluated holds at least 90% of the trials it was asked for, as CLEAN
       results (harness failures do not count), in the laptop's results tree (the laptop wrote
       them directly). A tag with one result file is a sweep that died, not a result. The
       released adapter's tag is checked like the others; only trained arms have a Hub repo.
    """
    ok = True
    ns = namespace
    if hub_files is None:
        from huggingface_hub import HfApi
        from ops.amd.stage import resolve_namespace
        ns = ns or resolve_namespace()
        api = HfApi()
        hub_files = api.list_repo_files
        if hub_sha256 is None:
            hub_sha256 = lambda repo, name: api.get_paths_info(repo, [name])[0].lfs.sha256  # noqa: E731
    if hub_sha256 is None:
        raise SystemExit("verify_sync needs hub_sha256 when hub_files is given")
    from ops.amd.stage import sha256_file
    for arm, repo in hub_repos(cfg, ns or "<namespace>").items():
        local = Path(cfg.local_logs) / "adapters" / arm / "adapter_model.safetensors"
        try:
            files = set(hub_files(repo))
            have = {"adapter_config.json", "adapter_model.safetensors", "final.done"} <= files
            remote = hub_sha256(repo, "adapter_model.safetensors") if have else ""
            same = have and local.exists() and remote == sha256_file(local)
            why = ("" if same else "the final.done marker or an adapter file is missing on the Hub" if not have
                   else f"no local final adapter at {local}" if not local.exists()
                   else "the Hub's adapter is NOT the local final adapter (sha256 differs)")
        except Exception as exc:  # noqa: BLE001 - any failure to read means "not verified"
            same, why = False, str(exc)
        print(f"  {'PASS' if same else 'FAIL'}  adapter {arm} on the Hub is the final adapter ({repo})"
              + ("" if same else f": {why}"))
        ok &= same
    bad = sha_check(Path(cfg.local_logs))
    print(f"  {'PASS' if not bad else 'FAIL'}  droplet checksums match local copies"
          + (f": {bad}" if bad else ""))
    ok &= not bad
    root = runs_root or (REPO_ROOT / "data" / "runs")
    for tag, (rung, expected, minimum) in required_trials(cfg, ran).items():
        n = trial_files(root, tag, cfg.split, rung)
        good = n >= minimum
        print(f"  {'PASS' if good else 'FAIL'}  run tag {tag}: {n} clean {rung} result files "
              f"(expected {expected}, need at least {minimum} = {TRIAL_FLOOR:.0%})")
        ok &= good
    return ok


AGENT_FALLBACK = "/run/user/{uid}/keyring/ssh"


def ensure_ssh_agent(environ=None, uid: int | None = None, exists=os.path.exists) -> tuple[str, str]:
    """Make sure ssh can reach an agent. `SSH_AUTH_SOCK` from the environment if it names a socket
    that exists; otherwise the desktop keyring's /run/user/<uid>/keyring/ssh if that exists (a
    re-login leaves a shell started before it holding a socket that is gone, and every ssh then
    asks for a key it cannot find). Exports the one chosen so every child process inherits it.
    Returns (socket, where it came from); ("", "none") if neither exists."""
    environ = os.environ if environ is None else environ
    uid = os.getuid() if uid is None else uid
    sock = environ.get("SSH_AUTH_SOCK", "")
    if sock and exists(sock):
        return sock, "SSH_AUTH_SOCK from the environment"
    fallback = AGENT_FALLBACK.format(uid=uid)
    if exists(fallback):
        environ["SSH_AUTH_SOCK"] = fallback
        why = "the environment's socket is gone" if sock else "SSH_AUTH_SOCK is not set"
        return fallback, f"fallback: {why}"
    return "", "none: no SSH_AUTH_SOCK and no keyring socket (ssh will rely on --identity)"


def agent_key_state(runner=subprocess.run) -> str:
    """Whether the agent holds a key, from `ssh-add -l` (0 keys listed, 1 none, 2 unreachable)."""
    try:
        proc = runner(["ssh-add", "-l"], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return "could not run ssh-add"
    if proc.returncode == 0:
        return f"{len(proc.stdout.splitlines())} key(s) loaded"
    return "agent reachable but NO key loaded" if proc.returncode == 1 else "agent unreachable"


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
    sock, source = ensure_ssh_agent()
    print_status(cfg, s, now, agent=f"{sock or '-'} ({source}); {agent_key_state()}")


def print_status(cfg: P.Config, s: dict, now: float, agent: str = "") -> None:
    """Everything the reviewer needs to decide whether to continue, from the ledger alone."""
    cap = P.effective_total_cap(cfg.total_cap)
    print("## status")
    print(f"  state      {s['state']}")
    print(f"  droplet    id {s.get('droplet_id') or '-'}   ip {s.get('ip') or '-'}   "
          f"uptime {s['uptime_seconds'] / 60.0:.1f} min   (${s['price_per_hour']}/h)")
    print(f"  accrued    session ${s['session_dollars']:.2f}   total ${s['total_dollars']:.2f}")
    print(f"  remaining  ${cfg.budget - s['session_dollars']:.2f} under the ${cfg.budget:.2f} session "
          f"budget;  ${cap - s['total_dollars']:.2f} under the ${cap:.2f} working total cap")
    print(f"  caps       session ${cfg.budget:.2f}   working total ${cfg.total_cap:.2f}   "
          f"HARD limit ${P.HARD_TOTAL_LIMIT:.2f} (nothing may exceed it; the deadman destroys at "
          f"${min(cfg.total_cap, P.HARD_TOTAL_LIMIT) - P.DESTROY_MARGIN:.2f})")
    fresh, why = L.heartbeat_status(Path(cfg.ledger), now, tag=cfg.tag, budget=cfg.budget,
                                    total_cap=cfg.total_cap)
    print(f"  deadman    {'HEARTBEAT FRESH' if fresh else 'NO LIVE DEADMAN'}: {why}")
    if not fresh:
        print("             a laptop re-login kills the dead-man process: if you logged in again "
              "since starting it, it is gone and the heartbeat above is stale")
        print(f"             start it:  {deadman_command(cfg)}")
    if agent:
        print(f"  ssh agent  {agent}")


def main(argv: list[str] | None = None) -> int:
    install_signal_handlers()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    names = ["dry-run", "plan", "preflight", "create", "bootstrap", "smoke", "gate", "gate-decide",
             "project", "train", "serve", "tunnel", "eval", "sync", "destroy", "go", "status",
             "verify-sync", "note-prior-spend"]
    for n in names:
        p = sub.add_parser(n)
        add_common(p)
        p.add_argument("--yes", action="store_true", help="really do the mutating call")
        p.add_argument("--json", action="store_true")
        if n == "create":
            p.add_argument("--new-session", action="store_true",
                           help="reset the session budget (default: only the first create does)")
        if n in ("eval", "go"):
            p.add_argument("--stage", choices=list(EVAL_STAGES), default="L1",
                           help="evaluation stage: L1 (default) = every model at L1, one sample, "
                                "then stop (look at it, then decide); sample2 = the second L1 "
                                "sample; hints = L2-L4; control = the one-turn control; rest = "
                                "sample2 + hints + control; all = everything")
        if n == "tunnel":
            p.add_argument("action", nargs="?", default="up", choices=["up", "down"])
        if n == "note-prior-spend":
            p.add_argument("dollars", type=float)
    args = ap.parse_args(argv)
    os.chdir(REPO_ROOT)
    ensure_ssh_agent()
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
        return 0 if verify_sync(cfg, ran=ran_steps(events)) else 1
    if c == "gate-decide":
        # Free: reads the gate's result files. The only way past a gate whose rate did not clear
        # the margin is this command with --accept-gate; the project step then re-renders the plan.
        if not cmd_gate_decide(cfg):
            return 3
        return 0 if cmd_project(cfg, L.read(Path(cfg.ledger))) else 3

    steps, _ = make_plan(cfg, events)
    if c in ("eval", "go"):
        steps = select_eval(steps, args.stage)
    phases = {"create": ("create",), "bootstrap": ("bootstrap",), "smoke": ("smoke", "gate"),
              "gate": ("gate",), "train": ("train",), "serve": ("serve",), "eval": ("eval",),
              "sync": ("sync",), "destroy": ("destroy",), "tunnel": ("tunnel",)}

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
            return subprocess.call(list(P.tunnel_down(cfg).argv), env=child_env())
        return ensure_tunnel(cfg)

    if c in ("train", "go") and last_go(events, measure_key(cfg, events)) is not True:
        raise SystemExit("no GO on record for THIS droplet and hardware: run `driver.py smoke` (or "
                         "`project`) first. The measured projection must fit the budget before "
                         "anything is trained, and a GO from an earlier droplet does not carry over.")
    if c == "go":
        return go(cfg, steps, client())
    run_steps(cfg, steps, None, only=phases[c])
    return 0


if __name__ == "__main__":
    sys.exit(main())
