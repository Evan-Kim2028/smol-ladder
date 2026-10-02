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
    python ops/amd/driver.py eval --stage L1    L1 for all four models, then STOP: read it, then buy more
    python ops/amd/driver.py eval --stage rest  L2-L4 and the one-turn control (--stage hints / control / all)
    python ops/amd/driver.py sync               adapters + logs off the droplet, then verify them
    python ops/amd/driver.py destroy --yes      DELETE by tag, then GET until none remains
    python ops/amd/driver.py go                 train -> serve -> tunnel -> eval -> sync -> destroy
    python ops/amd/driver.py status             droplet, uptime, dollars accrued and remaining, the
                                                caps, the $95 hard limit, and whether the deadman lives

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
from ops.amd.doapi import DoApi, child_env, load_dotenv, token_from_env  # noqa: E402

TRIAL_LINE = re.compile(r"^\[\d+/\d+\] .* reward=", re.M)


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
    print(f"  {'stage':<30} {'hours':>6} {'dollars':>8}  basis")
    for r in rows:
        print(f"  {r.stage:<30} {r.seconds / 3600.0:>6.2f} {r.dollars(cfg.price):>8.2f}  {r.basis}")
    print(f"  {'TOTAL':<30} {hours:>6.2f} {total:>8.2f}  "
          f"(session headroom ${cfg.budget - total:.2f}, credit headroom ${P.CREDIT - total:.2f})")
    if total > cfg.budget:
        print(f"  !! OVER THE SESSION BUDGET by ${total - cfg.budget:.2f}")
    staged = P.staged_dollars(rows, cfg.price)
    if staged["hints"] or staged["control"]:
        print(f"  {'-' * 60}")
        print(f"  stop after L1:  ${staged['l1_only']:.2f} in all (everything above except the two "
              "optional rows below)")
        print(f"  + L2-L4:        ${staged['hints']:.2f} incremental   (eval --stage hints)")
        print(f"  + control:      ${staged['control']:.2f} incremental   (eval --stage control)")
    if any(not r.measured and "UNMEASURED" in r.basis for r in rows):
        print("  !! rows marked UNMEASURED are placeholders; the smoke measures them and `project` "
              "re-renders this table before any arm is trained")
    return {"rows": [{"stage": r.stage, "seconds": round(r.seconds), "dollars":
                      round(r.dollars(cfg.price), 2), "basis": r.basis} for r in rows],
            "total_hours": round(hours, 2), "total_dollars": round(total, 2),
            "price_per_hour": cfg.price, "budget": cfg.budget, "within_budget": total <= cfg.budget,
            "l1_only_dollars": round(staged["l1_only"], 2),
            "incremental_dollars": {"L2-L4": round(staged["hints"], 2),
                                    "control": round(staged["control"], 2)}}


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


def parse_probe_serve(text: str) -> dict:
    out: dict = {}
    m = re.search(r"READY_AFTER_S=(\d+(?:\.\d+)?)", text)
    if m:
        out["probe_start_s"] = float(m.group(1))
    m = re.search(r"TOOL_CALLS_OK=([01])", text)
    if m:
        out["tool_calls_ok"] = m.group(1) == "1"
    return out


def seconds_per_trial(output: str, wall_seconds: float, models: int = 1) -> float | None:
    """Wall seconds per trial of ONE model at the probe's worker count and concurrency, from the
    harness's own progress lines. With `models` harnesses running at once the lines of all of them
    are counted, so each model's trial count is the total divided by `models`."""
    n = len(TRIAL_LINE.findall(output))
    return wall_seconds / (n / models) if n else None


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

CAPTURED = ("probe-eval", "probe-serve", "smoke-checks")   # output parsed for measurements


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
        elif step.name == "wait-ssh":
            code = wait_for_ssh(cfg)
        elif step.name == "upload":
            check_stage_for_upload(cfg)
            code, out = run_cmd(step.cmds[0], cfg.host, timeout=P.step_timeout(step))
        elif step.name == "go-no-go":
            code = 0 if cmd_project(cfg, L.read(ledger)) else 3
        elif step.name == "verify-sync":
            code = 0 if verify_sync(cfg, ran=ran_steps(L.read(ledger))) else 1
        elif step.name in ("tunnel", "probe-tunnel"):
            code = ensure_tunnel(cfg)
        elif step.name == "tunnel-down":
            code = subprocess.call(list(step.cmds[0].argv), env=child_env()) and 0
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
        if step.name == "probe-serve":
            L.append(ledger, L.MEASURED, **parse_probe_serve(out), **stamp(cfg, L.read(ledger)))
        if step.name == "probe-eval" and code == 0:
            spt = seconds_per_trial(out, wall, models=len(step.cmds))
            if spt:
                L.append(ledger, L.MEASURED, sec_per_trial=round(spt, 2), probe_wall_s=round(wall, 1),
                         probe_models=len(step.cmds), **stamp(cfg, L.read(ledger)))
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


def wait_for_ssh(cfg: P.Config, runner=None, sleep=time.sleep, clock=time.monotonic,
                 limit: float = SSH_WAIT_S, interval: float = 10.0) -> int:
    """A droplet is `active` in the API before sshd takes logins, so the first scp would fail on a
    perfectly healthy machine. Poll a trivial command until it works, up to `limit` seconds."""
    runner = runner or run_cmd
    cmd = P.ssh(cfg, ["true"])
    start = clock()
    attempt = 0
    while True:
        attempt += 1
        code, _ = runner(cmd, cfg.host, timeout=30.0)
        if code == 0:
            print(f"  sshd answered on attempt {attempt}")
            return 0
        if clock() - start >= limit:
            print(f"  sshd did not answer within {limit:.0f}s ({attempt} attempts)")
            return 1
        print(f"  sshd not ready (attempt {attempt}); retrying in {interval:.0f}s")
        sleep(interval)


def check_stage_for_upload(cfg: P.Config) -> None:
    stage = Path(cfg.stage_dir)
    if not (stage / "remote.env").exists():
        raise SystemExit(f"{stage}/remote.env is missing. It is deleted after every successful "
                         "bootstrap on purpose (it holds the HF token). Re-run `ops/amd/stage.py` "
                         "(free) and upload again.")


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
    "hints": ("eval-L2", "eval-L3", "eval-L4"),
    "control": ("eval-control",),
    "rest": ("eval-L2", "eval-L3", "eval-L4", "eval-control"),
    "all": ("eval-L1", "eval-L2", "eval-L3", "eval-L4", "eval-control"),
}


def select_eval(steps: list[P.Step], stage: str) -> list[P.Step]:
    """The plan with only the requested evaluation stage's steps in the eval phase. `L1` runs the
    four models at L1 and stops: the reviewer reads those numbers, then buys `hints` or not."""
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
    base = {"A": "smol-ladder-sft-a", "B": "smol-ladder-sft-b", "AB": "smol-ladder-sft-ab"}
    return {a: f"{namespace}/{base[a]}" for a in P.ARMS if a in cfg.arms}


TRIAL_FLOOR = 0.9     # a run tag must hold at least this fraction of the trials it was asked for


def trial_files(root: Path, tag: str, split: str, rung: str) -> int:
    base = root / tag / split
    if not base.exists():
        return 0
    return (len(list(base.glob(f"*/{rung}/result.json")))
            + len(list(base.glob(f"*/{rung}/s*/result.json"))))


def required_trials(cfg: P.Config, ran: set[str] | None = None) -> dict[str, tuple[str, int, int]]:
    """run tag -> (rung counted, expected, minimum). Only the FIRST rung of the plan is counted:
    every task gets it, whereas L2-L4 skip tasks that have no verified reference, so for them the
    expected count is an upper bound and a floor on it would fail a healthy run. `ran` limits the
    check to the stages that actually ran (`eval --stage L1` must not be failed for the hint rungs
    it was told not to run)."""
    rung = "L1" if "L1" in cfg.rungs else cfg.rungs[0]
    n = cfg.samples if rung == "L1" else cfg.late_samples
    out: dict[str, tuple[str, int, int]] = {}
    if ran is None or f"eval-{rung}" in ran:
        for arm in P.eval_arms(cfg):
            out[P.run_tag(cfg, arm)] = (rung, cfg.limit * n, math.ceil(TRIAL_FLOOR * cfg.limit * n))
    if cfg.include_base and cfg.program_control and (ran is None or "eval-control" in ran):
        out[f"{P.run_tag(cfg, 'base')}-program"] = (
            "L1", cfg.limit, math.ceil(TRIAL_FLOOR * cfg.limit))
    return out


def verify_sync(cfg: P.Config, hub_files=None, runs_root: Path | None = None,
                namespace: str | None = None, ran: set[str] | None = None) -> bool:
    """After the sync, before the destroy: is everything that matters somewhere that is not the droplet?

    1. every adapter is readable on the Hub (`hub_files(repo)` lists a repo's files; injected so a
       test needs no network),
    2. the droplet's own checksum list matches the copies pulled to logs/amd/,
    3. every run tag that was evaluated holds at least 90% of the trials it was asked for in the
       laptop's results tree (the laptop wrote them directly). A tag with one result file is a
       sweep that died, not a result.
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
    for tag, (rung, expected, minimum) in required_trials(cfg, ran).items():
        n = trial_files(root, tag, cfg.split, rung)
        good = n >= minimum
        print(f"  {'PASS' if good else 'FAIL'}  run tag {tag}: {n} {rung} result files "
              f"(expected {expected}, need at least {minimum} = {TRIAL_FLOOR:.0%})")
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
    print_status(cfg, s, now)


def print_status(cfg: P.Config, s: dict, now: float) -> None:
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
        print(f"             start it:  {deadman_command(cfg)}")


def main(argv: list[str] | None = None) -> int:
    install_signal_handlers()
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
        if n in ("eval", "go"):
            p.add_argument("--stage", choices=list(EVAL_STAGES), default="all",
                           help="evaluation stage: L1 = the four models at L1 only, then stop (look "
                                "at it, then decide); hints = L2-L4; control = the one-turn "
                                "control; rest = hints + control; all = everything (default)")
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
        return 0 if verify_sync(cfg, ran=ran_steps(events)) else 1

    steps, _ = make_plan(cfg, events)
    if c in ("eval", "go"):
        steps = select_eval(steps, args.stage)
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
            return subprocess.call(list(P.tunnel_down(cfg).argv), env=child_env())
        return ensure_tunnel(cfg)

    if c in ("train", "go") and last_go(events, measure_key(cfg, events)) is not True:
        raise SystemExit("no GO on record for THIS droplet and hardware: run `driver.py smoke` (or "
                         "`project`) first. The measured projection must fit the budget before "
                         "anything is trained, and a GO from an earlier droplet does not carry over.")
    if c == "go":
        return go(cfg, steps, client())
    run_steps(cfg, steps, None, only=phases[c][0])
    return 0


if __name__ == "__main__":
    sys.exit(main())
