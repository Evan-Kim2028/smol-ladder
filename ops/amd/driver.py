"""Orchestrate the three SFT arms and their ladder evaluations on an AMD MI300X droplet.

    driver.py --host <ip> --user root --dry-run          print every command, run nothing
    driver.py --host <ip> --user root                    run the whole plan over ssh
    driver.py --mode api --droplet <name|id> --destroy    destroy the droplet (API mode)
    driver.py --mode api --droplet <name|id> --power-off  power it off without destroying

## Two modes, and the difference is a credential

**SSH mode** (`--host`/`--user`, the default) needs nothing but an ssh key: the owner creates the
droplet in the console and attaches the laptop's public key. The agent cannot create or destroy
anything; it runs the plan against a droplet that already exists. This is the mode to start with.

**API mode** (`--mode api`) needs a DigitalOcean personal access token, because AMD Developer Cloud
droplets are DigitalOcean droplets -- DigitalOcean documents `doctl` and `POST /v2/droplets` in its
AMD section, with the MI300X size slugs. The token lives in `AMD_CLOUD_API_TOKEN` in the repo's
git-ignored `.env` and is never passed as a flag, logged or committed.

This module contains no network code: it builds the command list, and `run()` either prints it
(`--dry-run`) or hands it to ssh/doctl. Everything worth testing -- the budget arithmetic, the
step order, the per-step guards -- is pure and lives above `main()`.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

OPS = Path(__file__).resolve().parent
REPO_ROOT = OPS.parent.parent

# Published MI300X rate. See docs/AMD_RUNBOOK.md section 1 for the sources; the runbook's own
# finding is that the $1.99 figure in the plan is unconfirmed and the official rate is higher,
# which changes how much of the credit the plan can buy.
DEFAULT_PRICE_PER_GPU_HOUR = 2.59

ARMS = ("A", "B", "AB")
EVAL_ARMS = ("base", "A", "B", "AB")

# Per-arm SFT estimates, in GPU-hours, for one MI300X at bf16 LoRA, max_length 8192, 1 epoch.
# These are ESTIMATES and the derivation is in the runbook. The smoke run re-measures them: the
# first arm's actual runtime is reported next to its estimate so the rest can be corrected before
# they are spent, which is the whole point of running A first.
SFT_HOURS_ESTIMATE = {"A": 2.5, "B": 1.0, "AB": 3.0}
# Serving and sweeping the ladder, per model, on the same card.
EVAL_HOURS_ESTIMATE = 0.75
SETUP_HOURS = 0.75  # clone, venv, image pull, HF login, data rsync
SMOKE_STEP_HOURS = 0.25  # per smoke step: a 20-step SFT run, then a 5-task sweep
FINAL_SYNC_HOURS = 0.25  # the last rsync + Hub push, which happens while the GPU is still billed


@dataclass
class Step:
    """One thing the driver does, with enough context for --dry-run to be readable."""

    name: str
    argv: list[str]
    note: str = ""
    hours: float = 0.0
    dollars: float = 0.0
    # Steps that must not be skipped when resuming, and steps that are safe to repeat.
    idempotent: bool = True


@dataclass
class Plan:
    steps: list[Step] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def hours(self) -> float:
        return sum(s.hours for s in self.steps)

    @property
    def dollars(self) -> float:
        return sum(s.dollars for s in self.steps)

    def as_json(self) -> dict:
        return {
            "steps": [
                {"name": s.name, "argv": s.argv, "note": s.note,
                 "hours": round(s.hours, 3), "dollars": round(s.dollars, 2)}
                for s in self.steps
            ],
            "totals": {"hours": round(self.hours, 2), "dollars": round(self.dollars, 2)},
            "warnings": self.warnings,
        }


def cost(hours: float, price: float) -> tuple[float, float]:
    return hours, hours * price


def budget_report(price: float, budget: float, arms: list[str],
                  sft_hours: dict[str, float], eval_hours: float,
                  eval_base: bool = True, setup_hours: float = SETUP_HOURS) -> dict:
    """The table in the runbook, as data, so the doc and the tool cannot disagree.

    Every number here is an ESTIMATE. The two inputs it is derived from are the published hourly
    price and the per-arm GPU-hour estimates above; nothing in it is measured, and the smoke run
    is what replaces it with a measurement.
    """
    # Quantise the hours once, then price the quantised hours. Doing it in this order makes the
    # printed table add up to the printed total, which a budget table that does not add up gets
    # ignored over.
    rows = [{"stage": "setup", "hours": round(setup_hours, 2)}]
    for arm in arms:
        rows.append({"stage": f"sft-{arm}", "hours": round(sft_hours[arm], 2)})
    eval_arms = list(arms) + (["base"] if eval_base else [])
    for arm in eval_arms:
        rows.append({"stage": f"eval-{arm}", "hours": round(eval_hours, 2)})
    for row in rows:
        row["dollars"] = round(row["hours"] * price, 2)
    hours = sum(r["hours"] for r in rows)
    dollars = sum(r["dollars"] for r in rows)
    return {
        "rows": rows,
        "total_hours": round(hours, 2),
        "total_dollars": round(dollars, 2),
        "budget_dollars": budget,
        "within_budget": dollars <= budget,
        "headroom_dollars": round(budget - dollars, 2),
        "price_per_gpu_hour": price,
        "note": "ESTIMATES, not measurements. See docs/AMD_RUNBOOK.md section 3 for the derivation.",
    }


def ssh_argv(host: str, user: str, remote: list[str], ident: str | None = None) -> list[str]:
    argv = ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new"]
    if ident:
        argv += ["-i", ident]
    argv.append(f"{user}@{host}")
    argv += ["--"] + remote
    return argv


def plan(cfg) -> Plan:
    """The ordered run plan. Pure: no clock, no filesystem, no network.

    The order is not arbitrary. Setup, then the first-hour ROCm smoke test, then arm A -- the arm
    with the most rows and the published comparison -- so that its measured runtime is available to
    correct the estimates for B and A+B before they are spent. Evaluation of an arm comes right
    after its training, because an adapter that cannot be served should be discovered while the
    droplet is already warm and the run is cheap to redo. Teardown is last and unconditional.
    """
    p = Plan()
    price, budget = cfg.price, cfg.budget
    arms = cfg.arms

    def add(name, remote, note="", hours=0.0, idempotent=True):
        hours_, dollars_ = cost(hours, price)
        p.steps.append(Step(name, ssh_argv(cfg.host, cfg.user, remote, cfg.identity),
                            note, hours_, dollars_, idempotent))

    # 1. Setup. One ssh, one idempotent script: the repo is checked out at a pinned commit, the
    #    venv is built inside the ROCm container, HF is logged in, the data is rsynced in.
    setup = ["bash", f"{cfg.remote_root}/ops/amd/bootstrap.sh",
             "--commit", cfg.commit, "--with-data", "--with-eval"]
    add("bootstrap", setup, "idempotent: env, repo at a pinned commit, venv, HF login, data",
        SETUP_HOURS)

    # 2. The dead-man switch, started before anything that can hang. A window that is a poweroff
    #    is also what keeps the meter honest if the laptop sleeps mid-run.
    add("watchdog", ["bash", f"{cfg.remote_root}/ops/amd/watchdog.sh", "--once"],
        "one tick, to prove the switch is armed before a long run starts")
    p.steps[-1].argv = ssh_argv(cfg.host, cfg.user,
                                ["bash", "-c",
                                 f"nohup bash {cfg.remote_root}/ops/amd/watchdog.sh "
                                 f"--max-minutes {cfg.max_minutes} --idle-minutes {cfg.idle_minutes} "
                                 f">/dev/null 2>&1 & echo $!"],
                                cfg.identity)
    p.steps[-1].note = "start the long-running watchdog loop in the background"

    # 3. The first-hour smoke test. Cheap, and it is the step that finds out whether the ROCm image
    #    can train and serve at all before an arm has burned credit finding out the same thing more
    #    slowly.
    smoke = ["bash", f"{cfg.remote_root}/ops/amd/run_sft.sh", "--arm", "A", "--smoke",
             "--inside-rocm"]
    add("smoke-sft", smoke, "20 steps, proves bf16 LoRA trains on ROCm; measured before committing",
        SMOKE_STEP_HOURS)
    smoke_eval = ["bash", f"{cfg.remote_root}/ops/amd/run_eval.sh", "--arm", "base",
                  "--limit", str(cfg.smoke_limit), "--rungs", "L1"]
    add("smoke-eval", smoke_eval,
        f"{cfg.smoke_limit} tasks, base model, proves vLLM serves and the ladder grades on ROCm",
        SMOKE_STEP_HOURS)

    # 4. The arms. Each: train, then evaluate, then sync its adapter back.
    for arm in arms:
        hours = cfg.sft_hours[arm]
        # Touch the heartbeat before every long step. The watchdog treats a stale heartbeat as a
        # dead orchestrator, so a 3-hour training run must be preceded by a touch rather than left
        # to the one in step 3 -- otherwise a healthy run trips the switch mid-arm.
        add(f"heartbeat-{arm}", ["touch", f"{cfg.remote_root}/.heartbeat"],
            "refresh the watchdog's liveness signal before the long step")
        train = ["bash", f"{cfg.remote_root}/ops/amd/run_sft.sh", "--arm", arm, "--inside-rocm"]
        if cfg.resume:
            train.append("--resume")
        add(f"sft-{arm}", train, f"LoRA bf16, max_length {cfg.max_length}, 1 epoch (ESTIMATE)", hours)
        add(f"sync-{arm}", ["bash", f"{cfg.remote_root}/ops/amd/sync_back.sh", "--push-hub"],
            "push the adapter and log to the Hub before the next arm starts")
        evaluate = ["bash", f"{cfg.remote_root}/ops/amd/run_eval.sh", "--arm", arm]
        if cfg.limit:
            evaluate += ["--limit", str(cfg.limit)]
        add(f"eval-{arm}", evaluate,
            f"serve base+{arm} with vLLM, ladder over {cfg.split}, tag {cfg.tag_prefix}-{arm}",
            cfg.eval_hours)
        add(f"results-{arm}",
            ["bash", f"{cfg.remote_root}/ops/amd/sync_back.sh", "--push-hub",
             "--to-laptop", "--tags", f"{cfg.tag_prefix}-{arm}"],
            "rsync the sweep results back; the measurement, not the adapter, is the result")

    # 6. The base control is evaluated last. It needs no training, so there is no reason to pay for
    #    a server start before the arms that do, and it shares the droplet with whatever ran before.
    if cfg.eval_base:
        evaluate = ["bash", f"{cfg.remote_root}/ops/amd/run_eval.sh", "--arm", "base"]
        if cfg.limit:
            evaluate += ["--limit", str(cfg.limit)]
        add("eval-base", evaluate,
            "arm 0: the base control under --agent program, the floor every arm is read against",
            cfg.eval_hours)
        add("results-base",
            ["bash", f"{cfg.remote_root}/ops/amd/sync_back.sh", "--push-hub",
             "--to-laptop", "--tags", f"{cfg.tag_prefix}-base"],
            "rsync the base model's results back")

    # 7. Teardown, always. A poweroff is not enough: the droplet keeps billing until it is
    #    destroyed, so this is the step that actually stops the meter.
    add("sync-final", ["bash", f"{cfg.remote_root}/ops/amd/sync_back.sh", "--push-hub",
                       "--to-laptop", "--tags", ",".join(f"{cfg.tag_prefix}-{a}" for a in EVAL_ARMS)],
        "final sync: everything off the droplet and on the Hub", FINAL_SYNC_HOURS)
    add("poweroff", ["shutdown", "-h", "now"],
        "stops the work; billing continues until the droplet is DESTROYED")

    if p.dollars > budget:
        p.warnings.append(
            f"PLAN OVER BUDGET: ${p.dollars:.2f} estimated against a ${budget:.2f} cap. "
            "Cut --limit, drop an arm, or raise the cap deliberately.")
    if p.hours * price < budget * 0.5:
        p.warnings.append(
            f"plan uses about half the cap (${p.dollars:.2f} of ${budget:.2f}); "
            "there is room for a second seed or a k=2 sweep.")
    if not cfg.identity:
        p.warnings.append("no --identity given; ssh will use the agent's default key")
    if not cfg.hub_namespace:
        p.warnings.append(
            "AMD_HUB_NAMESPACE is unset: adapters cannot be pushed to the Hub, so the only copy "
            "of a trained adapter is this droplet's disk.")
    return p


@dataclass
class Config:
    host: str
    user: str
    identity: str | None
    commit: str
    arms: list[str]
    price: float
    budget: float
    max_minutes: int
    idle_minutes: int
    max_length: int
    limit: int | None
    split: str
    tag_prefix: str
    remote_root: str
    eval_hours: float
    smoke_hours: float
    smoke_limit: int
    resume: bool
    eval_base: bool
    hub_namespace: str
    sft_hours: dict[str, float] = field(default_factory=lambda: dict(SFT_HOURS_ESTIMATE))


def api_argv(action: str, droplet: str, token_env: str, region: str, size: str,
             image: str, ssh_key_fingerprint: str | None, name: str) -> list[str]:
    """The doctl invocations, built here so --dry-run can print them without doctl installed."""
    base = ["doctl", "--config", "/dev/null", "compute", "droplet"]
    if action == "create":
        argv = base + ["create", name, "--region", region, "--size", size, "--image", image]
        if ssh_key_fingerprint:
            argv += ["--ssh-keys", ssh_key_fingerprint]
        return argv
    if action == "destroy":
        return base + ["delete", droplet, "--force"]
    if action == "power-off":
        return base + ["action", "power-off", droplet, "--wait"]
    if action == "power-on":
        return base + ["action", "power-on", droplet, "--wait"]
    if action == "get":
        return base + ["get", droplet, "--format", "ID,Name,PublicIPv4,Status,Size,Created"]
    if action == "list":
        return base + ["list", "--format", "ID,Name,PublicIPv4,Status,Size,Created"]
    raise ValueError(action)


def run(argv: list[str], dry_run: bool) -> int:
    printable = " ".join(shlex.quote(a) for a in argv)
    if dry_run:
        print(f"  $ {printable}")
        return 0
    print(f"  $ {printable}", flush=True)
    if argv and argv[0] == "ssh":
        # The secret is in the remote .env, never in argv, so nothing has to be scrubbed here.
        return subprocess.call(argv)
    return subprocess.call(argv)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["ssh", "api"], default="ssh")
    ap.add_argument("--host", help="droplet public IP; required in ssh mode")
    ap.add_argument("--user", default="root", help="AMD's own docs use root@<ip>")
    ap.add_argument("--identity", help="path to the laptop's ssh private key")
    ap.add_argument("--droplet", help="droplet name or id, for api mode")
    ap.add_argument("--region", default="atl1", help="ADC GPU droplets are documented as ATL1 only")
    ap.add_argument("--size", default="gpu-mi300x1-192gb",
                    help="1x MI300X slug; 8x is gpu-mi300x8-1536gb")
    ap.add_argument("--image", default="", help="doctl image slug, if creating by API")
    ap.add_argument("--ssh-key-fingerprint", help="ssh key id or fingerprint, for api mode")
    ap.add_argument("--name", default="smol-ladder", help="droplet name, if creating by API")
    ap.add_argument("--commit", default="", help="git commit to check out; defaults to HEAD")
    ap.add_argument("--arms", default=",".join(ARMS), help="which arms to run")
    ap.add_argument("--limit", type=int, help="tasks per ladder sweep; unset means all of them")
    ap.add_argument("--split", default="test")
    ap.add_argument("--tag-prefix", default="amd1")
    ap.add_argument("--remote-root", default="/opt/smol-ladder")
    ap.add_argument("--price", type=float, default=float(
        os.environ.get("AMD_PRICE_PER_GPU_HOUR", DEFAULT_PRICE_PER_GPU_HOUR)))
    ap.add_argument("--budget", type=float, default=float(os.environ.get("AMD_BUDGET_USD", 80)))
    ap.add_argument("--max-minutes", type=int, default=int(
        os.environ.get("AMD_WALLCLOCK_LIMIT_MIN", 0)) or None)
    ap.add_argument("--idle-minutes", type=int, default=int(os.environ.get("AMD_IDLE_LIMIT_MIN", 45)))
    ap.add_argument("--max-length", type=int, default=8192)
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    ap.add_argument("--no-eval-base", dest="eval_base", action="store_false", default=True)
    ap.add_argument("--smoke-hours", type=float, default=0.5)
    ap.add_argument("--smoke-limit", type=int, default=5)
    ap.add_argument("--hub-namespace", default=os.environ.get("AMD_HUB_NAMESPACE", ""))
    # The one-shot lifecycle verbs, which are the only steps that are not a plan.
    ap.add_argument("--create", action="store_true", help="api mode: create the droplet")
    ap.add_argument("--destroy", action="store_true", help="api mode: DESTROY the droplet (stops billing)")
    ap.add_argument("--power-off", action="store_true", help="api mode: power off (does NOT stop billing)")
    ap.add_argument("--status", action="store_true", help="api mode: print the droplet")
    ap.add_argument("--dry-run", action="store_true", help="print every command; run nothing")
    ap.add_argument("--json", action="store_true", help="emit the plan as JSON")
    args = ap.parse_args()

    load_dotenv(REPO_ROOT / ".env")
    token = os.environ.get("AMD_CLOUD_API_TOKEN", "")

    # -- API mode: the lifecycle verbs, one at a time.
    if args.mode == "api":
        if not args.dry_run and not token:
            raise SystemExit(
                "AMD_CLOUD_API_TOKEN is not set. Put it in the repo's git-ignored .env as\n"
                "  AMD_CLOUD_API_TOKEN=<digitalocean personal access token>\n"
                "with `read:write` scope, or use --mode ssh which needs no token at all.")
        if not args.dry_run and not shutil_which("doctl"):
            raise SystemExit("doctl is not installed; see "
                             "https://docs.digitalocean.com/reference/doctl/how-to/install/")
        verb = None
        if args.create:
            verb = "create"
        elif args.destroy:
            verb = "destroy"
        elif args.power_off:
            verb = "power-off"
        elif args.status:
            verb = "get"
        if verb:
            argv = api_argv(verb, args.droplet or args.name, "AMD_CLOUD_API_TOKEN", args.region,
                            args.size, args.image, args.ssh_key_fingerprint, args.name)
            if verb in ("create", "destroy", "power-off", "power-on"):
                print(f"## api: {verb} {args.droplet or args.name}")
                if verb == "power-off":
                    print("NOTE: a powered-off GPU droplet keeps billing. Destroy it to stop.")
                if verb == "destroy":
                    print("NOTE: destroying is irreversible; sync_back.sh --push-hub first.")
                sys.exit(run(argv, args.dry_run))
            sys.exit(run(argv, args.dry_run))
        if not args.dry_run:
            print(run(["doctl", "compute", "droplet", "list",
                       "--format", "ID,Name,PublicIPv4,Status,Size,Created"], False))
            sys.exit(0)
        run(["doctl", "compute", "droplet", "list",
             "--format", "ID,Name,PublicIPv4,Status,Size,Created"], True)
        return

    # -- SSH mode: the plan.
    if not args.host and not args.dry_run:
        raise SystemExit("--host is required in ssh mode (the droplet's public IP).")
    host = args.host or "0.0.0.0"
    price = args.price
    max_minutes = args.max_minutes or int(args.budget / price * 60)
    commit = args.commit or os.environ.get("AMD_COMMIT") or default_commit()
    cfg = Config(
        host=host, user=args.user, identity=args.identity, commit=commit,
        arms=[a.strip() for a in args.arms.split(",") if a.strip()],
        price=price, budget=args.budget, max_minutes=max_minutes,
        idle_minutes=args.idle_minutes, max_length=args.max_length, limit=args.limit,
        split=args.split, tag_prefix=args.tag_prefix, remote_root=args.remote_root,
        eval_hours=EVAL_HOURS_ESTIMATE, smoke_hours=args.smoke_hours,
        smoke_limit=args.smoke_limit, resume=args.resume, eval_base=args.eval_base,
        hub_namespace=args.hub_namespace)
    bad = [a for a in cfg.arms if a not in ARMS]
    if bad:
        raise SystemExit(f"unknown arm(s) {bad}; want from {list(ARMS)}")

    report = budget_report(price, args.budget, cfg.arms, cfg.sft_hours, EVAL_HOURS_ESTIMATE,
                           eval_base=cfg.eval_base, setup_hours=SETUP_HOURS + 2 * SMOKE_STEP_HOURS)
    p = plan(cfg)  # the smoke steps are already steps in the plan, so the totals already include them

    if args.json:
        print(json.dumps({"budget": report, "plan": p.as_json()}, indent=2))
        return

    print(f"## budget (ESTIMATES at ${price}/GPU-h, cap ${args.budget:.2f})")
    for row in report["rows"]:
        print(f"  {row['stage']:<12} {row['hours']:>6.2f} h  ${row['dollars']:>7.2f}")
    print(f"  {'TOTAL':<12} {report['total_hours']:>6.2f} h  ${report['total_dollars']:>7.2f}"
          f"   (cap ${args.budget:.2f}, headroom ${report['headroom_dollars']:.2f})")
    print()
    for w in p.warnings:
        print(f"WARNING: {w}")
    if p.warnings:
        print()
    print(f"## plan: {len(p.steps)} steps, {p.hours:.2f} h, ${p.dollars:.2f} estimated")
    for step in p.steps:
        print(f"- {step.name:<16} {step.hours:>5.2f} h  ${step.dollars:>6.2f}  {step.note}")
    print()

    if args.dry_run:
        print("## dry run: the commands, in order, with nothing executed")
        for step in p.steps:
            print(f"\n# {step.name}: {step.note}")
            run(step.argv, True)
        print(f"\n## end of plan ({len(p.steps)} steps, ${p.dollars:.2f} estimated)")
        return

    for step in p.steps:
        print(f"\n## {step.name}")
        code = run(step.argv, False)
        if code != 0 and not step.idempotent:
            print(f"step {step.name} failed ({code}); stopping.")
            sys.exit(code)
        if code != 0:
            print(f"WARNING: step {step.name} failed ({code}); continuing.")


def default_commit() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
                                       text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "HEAD"


def shutil_which(name: str) -> str | None:
    from shutil import which
    return which(name)


def load_dotenv(path: Path) -> None:
    """Read .env without a dependency and without clobbering the real environment.

    `set -a; source .env` is the shell way but this is python and the file may hold values with
    spaces in them; a tiny parser that only accepts KEY=VALUE is safer and has no side effects.
    """
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("'\"")
        os.environ.setdefault(key, value)


if __name__ == "__main__":
    main()
