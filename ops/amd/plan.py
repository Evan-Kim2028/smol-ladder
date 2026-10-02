"""The session as data: hardware, the ordered steps with every command, and the costed table.

Pure. No clock, no network, no filesystem except reading token counts that `stage.py` wrote. Both
`driver.py` (which executes the steps) and the tests (which assert on them) consume this module,
and the runbook's numbers are produced by it, so the plan, the table and the docs cannot drift.

## Why this hardware (re-verified by read-only GET on 2026-10-01)

    gpu-mi300x1-192gb        $2.59/h   listed, offered in NO region     <- the old plan; cannot be created
    gpu-mi350x1-288gb-spot   $2.46/h   ric1                             <- first choice
    gpu-mi355x1-288gb-spot   $2.97/h   mem1
    gpu-mi325x1-256gb        $3.80/h   nyc2, tor1 (on-demand)           <- fallback

MI350X spot is both the cheapest and the biggest (288 GB). Spot droplets are reclaimed, normally
with at least two hours' email notice, and are destroyed on reclaim; that risk is why training
checkpoints to a private Hub repo every few minutes and why evaluation runs L1 for all models
first. On-demand MI325X is the fallback if a run must not be interruptible.

## Why the vLLM image is the default

Both application images carry ROCm 7.2 and a ROCm build of PyTorch 2.10 (vLLM cannot run without
one). They differ in the other half. The two halves are not equally cheap to add:

  * training stack on top of the vLLM image: peft, trl, accelerate, datasets, transformers are
    pure-Python wheels (tens of MB) that install in about a minute into a venv layered over the
    image's site-packages, so the image's torch is untouched;
  * vLLM on top of the PyTorch image: a ROCm vLLM build is tied to an exact torch / triton / aiter
    set. Installing it means either a multi-GB torch replacement or a source build, each many
    billed minutes, and each a way to end up with a torch that no longer matches the ROCm driver.

So the default is `amddeveloperclou-vllm0171` (vLLM 0.17.1, above the 0.16.2 floor for Qwen3.5).
The fallback is `amddeveloperclou-pytorch2100rocm7` + the vLLM ROCm container.

## The evaluation protocol

SmolDataEnvs-sft, which arms A, B and A+B are trained on, is the single-`bash`-tool agent format
with submission to /workdir/answer.txt. In this repo that protocol is `run_ladder --agent bash`
(smol_ladder.upstream.BASH_TOOL). Every adapter is evaluated under it, and so is the base-model
control, so that adapter minus base is the effect of the SFT and not a difference of protocols. A
second, cheap control runs the base under `--agent program` (one turn, no tools), the protocol
upstream publishes a number for (docs/LOCAL_MODELS.md).
"""

from __future__ import annotations

import json
import math
import os
import shlex
from dataclasses import dataclass, field
from pathlib import Path

# ── hardware ─────────────────────────────────────────────────────────────────────
SIZE_MI350X = "gpu-mi350x1-288gb-spot"
REGION_MI350X = "ric1"
PRICE_MI350X = 2.46
SIZE_MI325X = "gpu-mi325x1-256gb"
REGION_MI325X = "tor1"
PRICE_MI325X = 3.80
LISTED_BUT_NOT_OFFERED = "gpu-mi300x1-192gb"  # $2.59/h; no region offers it

IMAGE_DEFAULT = "amddeveloperclou-vllm0171"
IMAGE_FALLBACK = "amddeveloperclou-pytorch2100rocm7"
TAG = "smol-ladder"
NAME = "smol-ladder"

DEFAULT_BUDGET = 35.0       # this session
TOTAL_CAP = 90.0            # the WORKING total cap: the $100 credit minus a margin
# The owner's rule: total spend (every earlier dollar included) has a HARD cutoff that nothing may
# override. It is a constant, not a setting: no flag, no env var and no Config field can raise it.
# Every cap in this package goes through `effective_total_cap`, which clamps to it.
HARD_TOTAL_LIMIT = 95.0
# Held back from the hard limit by the dead-man switch for the destroy itself: a poll interval
# plus the verify loop is about two minutes, which is about $0.10 at $2.46/h.
DESTROY_MARGIN = 0.50
CREDIT = 100.0
CREDIT_EXPIRES = "2026-10-18"



def effective_total_cap(cap: float) -> float:
    """The cap that is actually enforced: never above the hard limit, whatever was asked for."""
    return min(float(cap), HARD_TOTAL_LIMIT)


ARMS = ("A", "B", "AB")           # trained in this session
# One suffix on every Hub repo this session writes. Session 1's repos (smol-ladder-sft-{a,b,ab})
# hold invalid adapters and must never be reused. common.sh's amd_hub_name has the same names.
DEFAULT_SESSION = "s2"
HUB_BASE_NAMES = {"A": "smol-ladder-sft-a", "B": "smol-ladder-sft-b", "AB": "smol-ladder-sft-ab",
                  "artifacts": "smol-ladder-runs"}


def session() -> str:
    return os.environ.get("AMD_SESSION") or DEFAULT_SESSION


def hub_name(key: str, sess: str | None = None) -> str:
    """The Hub repo name (without namespace) of an arm ("A", "B", "AB") or of the logs dataset."""
    return f"{HUB_BASE_NAMES[key]}-{sess or session()}"
BASE_MODEL = "Qwen/Qwen3.5-2B"
VLLM_MIN = "0.16.2"
SERVE_PORT = 8000
LORA_RANK = 16
# Adapters that are evaluated but not trained: arm name -> Hub repo. `R` is the released upstream
# adapter (same base, r=16). It is also the gate's subject (see GATE_ARM): a model somebody else
# already trained, so a bad serving stack shows up on it before any of OUR GPU hours are spent.
EVAL_ONLY_DEFAULT = (("R", "AdithyaSK/smoldataenvs-sft-2b-v0"),)
GATE_ARM = "R"
GATE_TASKS = 60                   # the gate's fixed subset: the first 60 tasks of the split
MAX_WORKERS_SAFE = 10             # session 1: 8 per model was fine, 20 hung every engine

# ── time estimates, in seconds ───────────────────────────────────────────────────────────
# Every figure below is either MEASURED in session 1 (a spot MI350X in ric1, vLLM 0.17.1 image) and
# says so, or an ESTIMATE and says so. A measurement taken on this droplet replaces the estimate:
# `driver.py project` re-renders the table.
BOOT_S = 240.0               # create -> ssh answers
BOOTSTRAP_S = 720.0          # MEASURED in session 1: about 12 min billed, first-boot wait included
SMOKE_CHECK_S = 120.0
BENCH_BATCHES = (4,)         # session 1 trained at batch 4; `smoke.sh --bench-batches 2 4 8` compares more
BENCH_CONFIGS = len(BENCH_BATCHES)
BENCH_LOAD_S = 90.0          # model load per configuration, billed on top of the timed window
RESUME_CHECK_S = 300.0
SFT_OVERHEAD_S = 150.0       # per arm: load, evals, final save and push
SYNC_S = 300.0
DESTROY_S = 60.0
RESERVE_S = SYNC_S + DESTROY_S   # held back so the last two steps are always affordable
SAFETY = 1.15                # multiplier on measured training time
GATE_SERVE_S = 420.0         # ESTIMATE: download + merge the released adapter, start base + R, check
GATE_STOP_S = 15.0
SERVE_START_S = 420.0        # ESTIMATE: merge A, B, AB (R is already merged) and start five engines

# MEASURED in session 1 on that hardware. The training number is real trained tokens per second at
# per-device batch 4 (effective batch 8) with the reference fallback for the two fast kernels the
# image lacks; the evaluation number is trials per minute summed over all five servers at 8 workers
# each (a 16-turn bash trial took about 47 s). 1,250 trials at 22 per minute is about 57 minutes.
MEASURED_TOKENS_PER_S = 5446.0
MEASURED_TRIALS_PER_MIN = 22.0
MEASURED_SEC_PER_TRIAL = 47.0
PROGRAM_COST_FACTOR = 0.3      # a one-turn trial is far cheaper than a bash trial
ASSUMED_TOKENS_PER_S = MEASURED_TOKENS_PER_S
ASSUMED_TRIALS_PER_MIN = MEASURED_TRIALS_PER_MIN

# Effective batch is held at upstream's 8 sequences/step so arm A stays comparable to the
# published recipe; the batch sizes trade per-device width for accumulation.
EFFECTIVE_BATCH = 8
BENCH_SECONDS = 120

RUNGS = ("L1", "L2", "L3", "L4")


@dataclass
class Config:
    # hardware / account
    size: str = SIZE_MI350X
    region: str = REGION_MI350X
    price: float = PRICE_MI350X
    image: str = IMAGE_DEFAULT
    fingerprint: str = ""
    tag: str = TAG
    name: str = NAME
    # money
    budget: float = DEFAULT_BUDGET
    total_cap: float = TOTAL_CAP
    deadline_minutes: float = 0.0       # 0 = derive from the projection
    # work
    arms: tuple[str, ...] = ARMS
    max_length: int = 8192
    split: str = "test"
    limit: int = 250
    eval_only: tuple[tuple[str, str], ...] = EVAL_ONLY_DEFAULT   # (arm, Hub repo): evaluated, not trained
    samples: int = 1                    # L1; a second sample is its own stage (eval --stage sample2)
    late_samples: int = 1               # L2..L4
    rungs: tuple[str, ...] = RUNGS
    workers: int = MAX_WORKERS_SAFE - 2   # per model; 8 was measured safe, 20 hung every engine
    max_turns: int = 16
    # amd1 holds session 1's results, which were produced by models that were not what they were
    # named; a tag that is reused is read back as finished, so this session needs a fresh one.
    tag_prefix: str = "amd2"
    include_base: bool = True
    program_control: bool = True
    max_model_len: int = 16384
    gate_tasks: int = GATE_TASKS
    gate_margin: float = 0.05           # the released adapter must beat the base by this much
    gate_max_failures: int = 3          # harness failures tolerated per model in the gate
    ckpt_steps: int = 50                # a checkpoint (pushed to the Hub) every this many steps
    train_attempts: int = 3             # automatic resumes of one arm after a crash
    accept_gate: bool = False           # continue past a gate whose rate did not clear the margin
    stall_minutes: float = 5.0          # supervision: no new result for this long is a stall
    error_window: int = 10              # ... or this share of a model's last N results failing
    error_share: float = 0.5
    # where things are
    host: str = ""
    user: str = "root"
    identity: str = ""
    commit: str = "HEAD"
    remote_root: str = "/opt/smol-ladder"
    remote_stage: str = "/var/tmp/smol-ladder-stage"
    remote_log: str = "/var/log/smol-ladder"
    stage_dir: str = "/tmp/smol-ladder-stage"
    local_logs: str = "logs/amd"
    ledger: str = "ops/amd/ledger.jsonl"
    tunnel_socket: str = "/tmp/smol-ladder-tunnel.sock"
    env_file: str = ".env"

    def __post_init__(self) -> None:
        if self.total_cap > HARD_TOTAL_LIMIT:
            raise ValueError(f"total cap ${self.total_cap:g} is above the ${HARD_TOTAL_LIMIT:g} "
                             "hard limit, which nothing may raise")
        names = [n for n, _ in self.eval_only]
        taken = {"BASE", *ARMS}
        for n, repo in self.eval_only:
            if not n.isalnum() or n != n.upper() or n in taken or names.count(n) > 1:
                raise ValueError(f"eval-only arm name {n!r} must be unique, upper-case alphanumeric "
                                 f"and not one of {sorted(taken)}")
            if "/" not in repo:
                raise ValueError(f"eval-only arm {n}: {repo!r} is not a Hub repo id (owner/name)")
        if GATE_ARM not in names:
            raise ValueError(f"the gate evaluates the released adapter {GATE_ARM}: keep "
                             f"{GATE_ARM}=<repo> in the eval-only list")

    @property
    def spot(self) -> bool:
        return self.size.endswith("-spot")


@dataclass(frozen=True)
class Cmd:
    argv: tuple[str, ...]
    env: tuple[tuple[str, str], ...] = ()

    def shell(self) -> str:
        prefix = " ".join(f"{k}={shlex.quote(v)}" for k, v in self.env)
        body = " ".join(shlex.quote(a) for a in self.argv)
        return f"{prefix} {body}".strip()


@dataclass
class Step:
    phase: str                     # stage create bootstrap smoke train serve tunnel eval sync destroy
    name: str
    where: str                     # laptop | droplet | api
    cmds: list[Cmd] = field(default_factory=list)   # more than one = run concurrently
    seconds: float = 0.0           # projected BILLED seconds
    note: str = ""
    api: str = ""                  # human description of an API call (where == "api")
    reserve: bool = True           # hold back the sync/destroy reserve when gating
    billed: bool = True
    timeout: float = 0.0           # wall seconds before the driver kills the step; 0 = derive


# ── arm / model naming: defined once, read by the serve and the eval commands ───────────────

def served_name(arm: str) -> str:
    return "amd-base-2b" if arm == "base" else f"amd-{arm.lower()}-2b"


def run_tag(cfg: Config, arm: str) -> str:
    return f"{cfg.tag_prefix}-{arm.lower()}"


def eval_only_arms(cfg: Config) -> list[str]:
    return [name for name, _ in cfg.eval_only]


def eval_arms(cfg: Config) -> list[str]:
    """Every model that is evaluated: the base, the arms trained here, the Hub adapters."""
    return (["base"] if cfg.include_base else []) + [a for a in ARMS if a in cfg.arms] \
        + eval_only_arms(cfg)


def served_adapters(cfg: Config) -> list[str]:
    """Adapters the full evaluation serves (merged, one server each), in port order."""
    return [a for a in ARMS if a in cfg.arms] + eval_only_arms(cfg)


def port_order(cfg: Config) -> list[str]:
    """Port = 8000 + position here. Fixed, whichever arms are in this run, so `A` is always :8001
    and the tunnel, the harness and the scripts (common.sh's amd_port) cannot disagree."""
    return ["base", *ARMS, *eval_only_arms(cfg)]


def port_for(cfg: Config, arm: str) -> int:
    return SERVE_PORT + port_order(cfg).index(arm)


def ports(cfg: Config) -> list[int]:
    """Every port that anything is served on: the tunnel forwards all of them."""
    return sorted({port_for(cfg, a) for a in ["base", *served_adapters(cfg)]})


def gate_arms(cfg: Config) -> list[str]:
    return ["base", GATE_ARM]


def gate_limit(cfg: Config) -> int:
    return min(cfg.gate_tasks, cfg.limit)


# ── ssh / scp ─────────────────────────────────────────────────────────────────────

CONTAINER = "smol"          # common.sh's AMD_CONTAINER; a test keeps the two in step


def ssh_opts(cfg: Config) -> list[str]:
    # Host keys are deliberately NOT remembered. The droplet is ephemeral and reached by the IP the
    # API just returned over TLS; providers reuse addresses, so a second session's droplet at an
    # old IP has a new key and `accept-new` would refuse it ("REMOTE HOST IDENTIFICATION HAS
    # CHANGED") at the worst moment, while a remembered key would only ever protect a machine that
    # no longer exists. Authentication is by OUR key, which the droplet cannot forge; the secrets
    # it receives (the HF token) are scoped to this session. UserKnownHostsFile=/dev/null also
    # keeps the laptop's real known_hosts out of it.
    opts = ["-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null", "-o", "LogLevel=ERROR",
            "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=30",
            "-o", "ServerAliveCountMax=6"]
    if cfg.identity:
        opts += ["-i", cfg.identity]
    return opts


def host_of(cfg: Config) -> str:
    return f"{cfg.user}@{cfg.host or '<droplet-ip>'}"


def ssh(cfg: Config, remote: list[str]) -> Cmd:
    return Cmd(tuple(["ssh", *ssh_opts(cfg), host_of(cfg), "--", *remote]))


def remote_script(cfg: Config, script: str, *args: str) -> Cmd:
    # Every droplet step runs inside the long-lived container (entrypoint.sh starts it); the
    # paths are bind-mounted identically. Secrets are read from remote_root/.env, never passed.
    return ssh(cfg, ["docker", "exec", CONTAINER, "bash", f"{cfg.remote_root}/ops/amd/{script}", *args])


def scp_up(cfg: Config) -> Cmd:
    # No trailing "/." and no pre-made directory: scp -r creates the destination when it does not
    # exist, and fails when it is asked to copy "dir/." into a directory that is not there yet.
    return Cmd(tuple(["scp", "-r", *ssh_opts(cfg), cfg.stage_dir,
                      f"{host_of(cfg)}:{cfg.remote_stage}"]))


def scp_down(cfg: Config, remote: str, local: str) -> Cmd:
    return Cmd(tuple(["scp", "-r", *ssh_opts(cfg), f"{host_of(cfg)}:{remote}", local]))


def tunnel_up(cfg: Config) -> Cmd:
    """Control-socket tunnel: `-f` returns once the forward is up, `-S` names the socket that
    `tunnel_down` and `tunnel_check` talk to, and ExitOnForwardFailure makes a busy local port a
    failure instead of a silent tunnel to nowhere. Bound to loopback on both ends."""
    fwd: list[str] = []
    for p in ports(cfg):
        fwd += ["-L", f"127.0.0.1:{p}:127.0.0.1:{p}"]
    return Cmd(tuple(["ssh", *ssh_opts(cfg), "-f", "-N", "-M", "-S", cfg.tunnel_socket,
                      "-o", "ExitOnForwardFailure=yes", *fwd, host_of(cfg)]))


def tunnel_down(cfg: Config) -> Cmd:
    return Cmd(tuple(["ssh", "-S", cfg.tunnel_socket, "-O", "exit", host_of(cfg)]))


def tunnel_check(cfg: Config) -> Cmd:
    return Cmd(tuple(["ssh", "-S", cfg.tunnel_socket, "-O", "check", host_of(cfg)]))


# ── the laptop-side evaluation commands ────────────────────────────────────────────

CHAT_KWARGS = '{"enable_thinking": false}'


def eval_cmd(cfg: Config, arm: str, rungs: str, samples: int, *, agent: str = "bash",
             tag: str | None = None, model: str | None = None, limit: int | None = None,
             max_turns: int | None = None, retry_failed: bool = False) -> Cmd:
    """One sweep of one model at one rung set, run on the laptop against the tunnel.

    `--agent bash` is the protocol the SFT data is in, with the harness's default `--bash-stop
    model` (only the model ends an episode, as in the SFT rows): the flag is NOT passed, so the
    default cannot be overridden here by accident. `--no-climb` runs every requested rung on
    every task, so a rung's pass rate has the same denominator for all models.
    `SMOL_LADDER_BASE_URL` points at the tunnel (loopback, so no API key is needed); the
    non-thinking kwargs match both the SFT rows and the server's default. `--workers` is per
    model and `cfg.workers` defaults to the 8 that session 1 measured safe. Each harness process
    has its own scratch directory (the harness makes it unique per invocation); the override
    variable is kept so a hand run can still point it somewhere.
    """
    turns = max_turns if max_turns is not None else (1 if agent == "program" else cfg.max_turns)
    argv = ["uv", "run", "python", "-m", "smol_ladder.run_ladder",
            "--split", cfg.split, "--model", model or served_name(arm),
            "--run-tag", tag or run_tag(cfg, arm), "--agent", agent,
            "--rungs", rungs, "--samples", str(samples), "--no-climb",
            "--limit", str(limit if limit is not None else cfg.limit),
            "--workers", str(cfg.workers), "--max-turns", str(turns)]
    if retry_failed:
        argv.append("--retry-failed")
    return Cmd(tuple(argv),
               (("SMOL_LADDER_BASE_URL", f"http://127.0.0.1:{port_for(cfg, arm)}/v1"),
                ("SMOL_LADDER_CHAT_TEMPLATE_KWARGS", CHAT_KWARGS),
                ("SMOL_LADDER_SCRATCH", scratch_dir(cfg, arm))))


def scratch_dir(cfg: Config, arm: str) -> str:
    """One scratch root per model's harness process: concurrent evaluations once shared one and
    each one's cleanup deleted the other's working directory mid-trial."""
    return f"/var/tmp/smol-ladder/scratch-{cfg.tag_prefix}-{arm.lower()}"


def with_retry_failed(cmd: Cmd) -> Cmd:
    return cmd if "--retry-failed" in cmd.argv else Cmd((*cmd.argv, "--retry-failed"), cmd.env)


def eval_phases(cfg: Config) -> list[tuple[str, list[Cmd]]]:
    """Evaluation in the order a spot reclaim hurts least, one purchase at a time.

    L1 for every model first (that single comparison is the result; one sample), then, only if
    bought, a second L1 sample, then L2, L3, L4 across all models, then the one-turn `program`
    control of the base. Within a phase the models run concurrently, one harness process each,
    each against its own server. The gate's 60 trials are already in the L1 run tags, so L1 does
    not pay for them again (the harness reuses a clean result).
    """
    phases: list[tuple[str, list[Cmd]]] = []
    for rung in cfg.rungs:
        phases.append((rung, [eval_cmd(cfg, a, rung, cfg.samples if rung == "L1" else cfg.late_samples)
                              for a in eval_arms(cfg)]))
        if rung == "L1" and cfg.samples == 1:
            phases.append(("sample2", [eval_cmd(cfg, a, "L1", 2) for a in eval_arms(cfg)]))
    if cfg.include_base and cfg.program_control:
        phases.append(("control", [eval_cmd(
            cfg, "base", "L1", 1, agent="program", tag=f"{run_tag(cfg, 'base')}-program")]))
    return phases


def gate_cmds(cfg: Config) -> list[Cmd]:
    """The gate: the same sweep as L1 (same run tags, same flags), on the first 60 tasks, for the
    base and the released adapter at once. Same tags, so the full L1 run finds these 60 trials
    done and does not pay for them again."""
    return [eval_cmd(cfg, a, "L1", 1, limit=gate_limit(cfg)) for a in gate_arms(cfg)]


def expected_trials(cfg: Config) -> dict[str, int]:
    """Upper bound of result files per run tag (L2-L4 skip tasks that have no reference)."""
    per = 0
    for rung in cfg.rungs:
        per += cfg.limit * (cfg.samples if rung == "L1" else cfg.late_samples)
    out = {run_tag(cfg, a): per for a in eval_arms(cfg)}
    if cfg.include_base and cfg.program_control:
        out[f"{run_tag(cfg, 'base')}-program"] = cfg.limit
    return out


# ── token sets, measurements, projection ───────────────────────────────────────────

@dataclass(frozen=True)
class SetTokens:
    rows: int
    trained_tokens: int     # after truncation at max_length
    method: str


def load_tokens(path: Path, max_length: int) -> dict[str, SetTokens] | None:
    """Token counts written by stage.py. None unless they were counted at this max_length."""
    path = Path(path)
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    if int(data.get("max_length", -1)) != int(max_length):
        return None
    return {k: SetTokens(int(v["rows"]), int(v["trained_tokens"]), v.get("method", "?"))
            for k, v in data["sets"].items()}


# Trained tokens at max_length 8192, recounted for this session's data with the Qwen3.5 chat
# template and tokenizer (tools and tool results rendered, rows cut at the window), the way
# stage.py counts them. B is ja3_sft_v2 (1,122 rows); the session-1 figure of 6.2M was v1. They are
# the fallback when no tokens.json was staged; a staged tokens.json is the source of truth and
# `stage.py` computes AB as A + B from these same two counts.
RECOUNTED_TOKENS = {8192: {"A": (4439, 9_085_233), "B": (1122, 3_123_047)}}


def _line_count(path: Path) -> int:
    return sum(1 for line in path.open() if line.strip())


def heuristic_tokens(data_dir: Path, max_length: int) -> dict[str, SetTokens]:
    """No staged token counts: the recount above if the data on disk has the rows it was made
    from, else bytes / 3.0. Measured on the real sets the true ratio is ~3.3 bytes per token, so the
    estimate over-counts by roughly 10%: the safe direction for a budget."""
    out: dict[str, SetTokens] = {}
    sets = {"A": [data_dir / "train" / "sft_upstream" / "train.jsonl"],
            "B": [data_dir / "train" / "ja3_sft_v2.jsonl"]}
    known = RECOUNTED_TOKENS.get(max_length, {})
    for key, files in sets.items():
        files = [f for f in files if f.exists()]
        if not files:
            continue
        rows = sum(_line_count(f) for f in files)
        if key in known and known[key][0] == rows:
            out[key] = SetTokens(rows, known[key][1],
                                 f"recounted with the Qwen3.5 tokenizer, {rows:,} rows")
            continue
        out[key] = SetTokens(rows, int(sum(f.stat().st_size for f in files) / 3.0),
                             "bytes/3.0 heuristic (pessimistic)")
    if "A" in out and "B" in out:
        out["AB"] = SetTokens(out["A"].rows + out["B"].rows,
                              out["A"].trained_tokens + out["B"].trained_tokens,
                              "sum of A and B")
    return out


def stale_tokens(tokens: dict[str, SetTokens], data_dir: Path) -> str:
    """Why a staged tokens.json must not be trusted ("" if it can be): its row counts are not the
    rows of the files on disk now. The first session staged counts for ja3_sft (v1, 2,029 rows)
    and a plan costed from them is wrong by a factor of two for arm B."""
    files = {"A": data_dir / "train" / "sft_upstream" / "train.jsonl",
             "B": data_dir / "train" / "ja3_sft_v2.jsonl"}
    for key, path in files.items():
        if key in tokens and path.exists() and tokens[key].rows != _line_count(path):
            return (f"tokens.json counts {tokens[key].rows} rows for arm {key} but {path} has "
                    f"{_line_count(path)}: it was staged from other data")
    if {"A", "B", "AB"} <= set(tokens) and tokens["AB"].trained_tokens != (
            tokens["A"].trained_tokens + tokens["B"].trained_tokens):
        return "tokens.json's AB is not A + B"
    return ""


@dataclass
class Measured:
    """What this droplet measured. Zero/None means "not measured yet": the table says so."""
    tokens_per_s: float = 0.0
    batch_size: int = 0
    grad_accum: int = 0
    trials_per_min: float = 0.0      # the gate's trials per minute over its 2 concurrent models
    tool_calls_ok: bool | None = None     # the released adapter produced a parsed `bash` tool call
    adapter_differs: bool | None = None   # its temperature-0 output differs from the base's
    gate_go: bool | None = None           # the gate's decision (GO, or explicitly accepted)
    checks_ok: bool | None = None
    resume_ok: bool | None = None


_MEASURED_FIELDS = ("tokens_per_s", "batch_size", "grad_accum", "trials_per_min", "tool_calls_ok",
                    "adapter_differs", "gate_go", "checks_ok", "resume_ok")


def measured_from_ledger(events: list[dict], key: tuple | None = None) -> Measured:
    """Fold the ledger's `measured` events into one Measured. With `key` = (droplet_id, hardware)
    only events stamped with exactly that key count: a number measured on another droplet, or on
    other hardware, must never carry over. `key=None` is for pure tests and reads everything."""
    m = Measured()
    for e in events:
        if e.get("event") != "measured":
            continue
        if key is not None and (e.get("droplet_id"), e.get("hardware")) != tuple(key):
            continue
        for key_ in _MEASURED_FIELDS:
            if e.get(key_) is not None:
                setattr(m, key_, e[key_])
    return m


def sft_seconds(tokens: int, tokens_per_s: float) -> float:
    if tokens_per_s <= 0:
        raise ValueError("tokens_per_s must be measured or assumed explicitly, not zero")
    return tokens / tokens_per_s * SAFETY + SFT_OVERHEAD_S


def hint_rungs(cfg: Config) -> tuple[str, ...]:
    return tuple(r for r in cfg.rungs if r != "L1")


def trials_per_min_for(meas: Measured) -> tuple[float, str]:
    """The rate the evaluation rows are costed at, and what it rests on. The session-1 figure is
    for five models at once; the gate measures two, which says nothing good about five, but a gate
    that ran SLOWER than session 1 did is a reason to cost the evaluation lower."""
    if meas.trials_per_min and meas.trials_per_min < MEASURED_TRIALS_PER_MIN:
        return meas.trials_per_min, (f"{meas.trials_per_min:.1f} trials/min, measured by this "
                                     "droplet's gate (below session 1's)")
    return MEASURED_TRIALS_PER_MIN, (
        f"{MEASURED_TRIALS_PER_MIN:g} trials/min over all servers, measured in session 1 on this "
        f"hardware (5 models x 8 workers, about {MEASURED_SEC_PER_TRIAL:g} s per 16-turn trial)")


def eval_breakdown(cfg: Config, rate: float) -> dict[str, float]:
    """Wall seconds the servers must stay up, per purchase: the gate's trials, L1 for all models
    (minus the gate's 60 x 2, already done), the optional second sample, the hint rungs and the
    one-turn control. A rate is trials per minute summed over every model."""
    n = len(eval_arms(cfg))
    per_trial = 60.0 / rate
    reuse = len(gate_arms(cfg)) * gate_limit(cfg) * 1     # the gate ran sample 0 of L1
    out = {"gate": len(gate_arms(cfg)) * gate_limit(cfg) * per_trial,
           "L1": 0.0, "sample2": 0.0, "hints": 0.0, "control": 0.0}
    if "L1" in cfg.rungs:
        out["L1"] = max(n * cfg.limit * cfg.samples - reuse, 0) * per_trial
        if cfg.samples == 1:
            out["sample2"] = n * cfg.limit * per_trial
    for rung in hint_rungs(cfg):
        out["hints"] += n * cfg.limit * cfg.late_samples * per_trial
    if cfg.include_base and cfg.program_control:
        out["control"] = cfg.limit * PROGRAM_COST_FACTOR * per_trial
    return out


ROW_GATE_SERVE = "gate: serve base + released adapter"
ROW_GATE_EVAL = "gate: 60 tasks x 2 models"
ROW_SERVE = "serve: start vLLM"
ROW_L1 = "eval L1"
ROW_SAMPLE2 = "eval L1 second sample (incremental)"
ROW_HINTS = "eval L2-L4 (incremental)"
ROW_CONTROL = "eval program control"
OPTIONAL_ROWS = {"sample2": ROW_SAMPLE2, "hints": ROW_HINTS, "control": ROW_CONTROL}


@dataclass
class Row:
    stage: str
    seconds: float
    basis: str
    measured: bool

    def dollars(self, price: float) -> float:
        return self.seconds / 3600.0 * price


def projection(cfg: Config, tokens: dict[str, SetTokens], meas: Measured) -> list[Row]:
    """Every billed stage, in order, with what each number rests on."""
    tps = meas.tokens_per_s or ASSUMED_TOKENS_PER_S
    tps_basis = ("measured on this droplet" if meas.tokens_per_s else
                 f"{MEASURED_TOKENS_PER_S:,.0f} tok/s measured in session 1 (MI350X, batch 4)")
    rate, rate_basis = trials_per_min_for(meas)
    ev = eval_breakdown(cfg, rate)
    bench = BENCH_CONFIGS * (BENCH_SECONDS + BENCH_LOAD_S)
    rows = [
        Row("create + boot", BOOT_S, "estimate", False),
        Row("bootstrap", BOOTSTRAP_S, "session 1: about 12 min billed, first-boot wait included",
            False),
        Row("smoke: checklist", SMOKE_CHECK_S, "estimate", False),
        Row("smoke: throughput bench", bench,
            f"{BENCH_CONFIGS} batch size x ({BENCH_SECONDS}s timed + {BENCH_LOAD_S:.0f}s load)",
            False),
        Row("smoke: kill and resume", RESUME_CHECK_S, "estimate", False),
        Row(ROW_GATE_SERVE, GATE_SERVE_S + GATE_STOP_S,
            "estimate: download + merge the released adapter, start 2 engines, adapter check", False),
        Row(ROW_GATE_EVAL, ev["gate"], f"{len(gate_arms(cfg))} models x {gate_limit(cfg)} trials; "
            f"{rate_basis}; the L1 run reuses these trials", bool(meas.trials_per_min)),
    ]
    for arm in ARMS:
        if arm not in cfg.arms:
            continue
        t = tokens.get(arm)
        if t is None:
            rows.append(Row(f"sft {arm}", 0.0, "NO TOKEN COUNT: data missing", False))
            continue
        rows.append(Row(f"sft {arm}", sft_seconds(t.trained_tokens, tps),
                        f"{t.trained_tokens / 1e6:.2f}M tokens / {tps:,.0f} tok/s x {SAFETY} "
                        f"+ {SFT_OVERHEAD_S:.0f}s; {tps_basis}", bool(meas.tokens_per_s)))
    n_models = len(eval_arms(cfg))
    rows.append(Row(ROW_SERVE, SERVE_START_S,
                    f"estimate: merge {len(ARMS)} trained adapters (R is already merged), start "
                    f"{n_models} engines", False))
    if "L1" in cfg.rungs:
        trials = n_models * cfg.limit * cfg.samples
        reuse = len(gate_arms(cfg)) * gate_limit(cfg)
        rows.append(Row(ROW_L1, ev["L1"], f"{n_models} models x {cfg.limit} tasks x {cfg.samples} "
                        f"sample = {trials:,} trials - {reuse} done in the gate = {trials - reuse:,} "
                        f"at {rate:g}/min; {rate_basis}", bool(meas.trials_per_min)))
        if ev["sample2"]:
            rows.append(Row(ROW_SAMPLE2, ev["sample2"], f"{n_models} models x {cfg.limit} tasks, "
                            f"sample 2 of L1; {rate_basis}; optional", bool(meas.trials_per_min)))
    if hint_rungs(cfg):
        rows.append(Row(ROW_HINTS, ev["hints"], f"{n_models} models, {','.join(hint_rungs(cfg))}, "
                        f"{cfg.limit} tasks x {cfg.late_samples} sample; {rate_basis}; "
                        "optional: look at L1 first", bool(meas.trials_per_min)))
    if ev["control"]:
        rows.append(Row(ROW_CONTROL, ev["control"], f"base under --agent program, {cfg.limit} "
                        f"tasks; {PROGRAM_COST_FACTOR}x a bash trial; optional",
                        bool(meas.trials_per_min)))
    rows.append(Row("sync back + verify", SYNC_S, "reserve", False))
    rows.append(Row("destroy + verify", DESTROY_S, "reserve", False))
    return rows


def total_dollars(rows: list[Row], price: float) -> float:
    return sum(r.dollars(price) for r in rows)


def staged_dollars(rows: list[Row], price: float) -> dict[str, float]:
    """What the session costs at each decision point, and what each optional purchase adds.

    gate_only   everything up to and including the gate, then the destroy (no training, no sync:
                nothing exists yet that is worth pulling)
    core        gate + train + L1 evaluation + sync + destroy: everything except the optional rows
    sample2, hints, control   incremental, bought after reading L1
    all         core plus every optional row"""
    add = {k: sum(r.dollars(price) for r in rows if r.stage == name)
           for k, name in OPTIONAL_ROWS.items()}
    total = total_dollars(rows, price)
    gate_end = next(i for i, r in enumerate(rows) if r.stage == ROW_GATE_EVAL)
    destroy = sum(r.dollars(price) for r in rows if r.stage == "destroy + verify")
    return {"gate_only": total_dollars(rows[:gate_end + 1], price) + destroy,
            "core": total - sum(add.values()), **add, "all": total}


def remaining_after(rows: list[Row], first_stage_prefix: str) -> list[Row]:
    """Rows from the one whose name starts with `first_stage_prefix` to the end."""
    for i, r in enumerate(rows):
        if r.stage.startswith(first_stage_prefix):
            return rows[i:]
    return rows


def default_deadline_minutes(cfg: Config, rows: list[Row]) -> float:
    """The dead-man switch's wall clock: 1.25x the projected duration plus 15 minutes, never more
    than the session budget buys. Re-derived from measurements after the smoke (`project` prints
    the command), because the first value rests on placeholders."""
    projected_s = sum(r.seconds for r in rows)
    by_budget = cfg.budget / cfg.price * 60.0
    return float(math.ceil(min(by_budget, projected_s * 1.25 / 60.0 + 15.0)))


# ── the ordered plan ──────────────────────────────────────────────────────────────

TIMEOUT_FLOOR_S = 900.0
TIMEOUT_FACTOR = 3.0           # a step may take this many times its projection before it is killed


def step_timeout(step: Step) -> float:
    """Generous (3x the projection, at least 15 minutes) and finite: a hung ssh or a stuck harness
    must not keep a billed droplet open until the deadman's deadline."""
    return step.timeout or max(TIMEOUT_FLOOR_S, TIMEOUT_FACTOR * step.seconds)


READY_PREFIX = "READY_"


def ssh_ready(cfg: Config) -> Cmd:
    """The command that proves a login really executes. A droplet's first boot holds ssh commands
    behind "Please wait while we get your droplet ready..." for minutes while sshd already
    answers, so `true` exits 0 on a machine that is not running anything yet. This one echoes a
    string that only a finished shell can produce, and the driver requires the exact line."""
    return ssh(cfg, [f"echo {READY_PREFIX}$(whoami)"])


def serve_cmd(cfg: Config, *, gate: bool = False) -> Cmd:
    """Serve every model merged, one vLLM per model on its fixed port (LoRA mode does not work for
    this model on vLLM 0.17.1). `gate` serves only the base and the released adapter. Every
    adapter is checked after it is up (tool calls, output differs from the base's) and the script
    exits non-zero if one is not."""
    args = ["--wait", "--verify"]
    arms = [] if gate else [a for a in ARMS if a in cfg.arms]
    args += ["--arms", ",".join(arms)]
    for name, repo in cfg.eval_only:
        if gate and name != GATE_ARM:
            continue
        args += ["--hub", f"{name}={repo}:{port_for(cfg, name)}"]
    return remote_script(cfg, "serve.sh", *args)


def build_plan(cfg: Config, tokens: dict[str, SetTokens], meas: Measured) -> list[Step]:
    """Every step, in the order the reviewer runs them. Each step is one command (or one group of
    concurrent commands) and carries the billed seconds the budget gate prices."""
    rows = {r.stage: r for r in projection(cfg, tokens, meas)}

    def secs(stage: str) -> float:
        return rows[stage].seconds if stage in rows else 0.0

    flags = ["--size", cfg.size, "--region", cfg.region, "--image", cfg.image,
             "--ssh-key-fingerprint", cfg.fingerprint or "<ssh-key-fingerprint>",
             "--budget", f"{cfg.budget:g}"]
    steps: list[Step] = []

    steps.append(Step("stage", "stage-inputs", "laptop", [Cmd((
        "python", "ops/amd/stage.py", "--out", cfg.stage_dir, "--commit", cfg.commit,
        "--max-length", str(cfg.max_length)))], 0.0,
        "code tarball of the pinned commit, both SFT sets, token counts, sha256 sums, the one "
        "entry script and the secrets file; all built before the droplet exists", billed=False))
    steps.append(Step("stage", "preflight", "laptop", [Cmd((
        "python", "ops/amd/driver.py", "preflight", *flags))], 0.0,
        "read-only GETs: size offered in the region, image slug exists, ssh key fingerprint is "
        f"registered, no droplet already tagged {cfg.tag}", billed=False))
    steps.append(Step("stage", "deadman", "laptop", [Cmd((
        "python", "ops/amd/deadman.py", "--deadline-minutes",
        f"{cfg.deadline_minutes or default_deadline_minutes(cfg, list(rows.values())):g}",
        "--budget", f"{cfg.budget:g}", "--total-cap", f"{cfg.total_cap:g}",
        "--price", f"{cfg.price:g}", "--tag", cfg.tag))], 0.0,
        "run DETACHED (setsid nohup ... >> logs/deadman.log 2>&1 < /dev/null &) BEFORE create: "
        "destroys the tagged droplet at the deadline, at the cap (never above the $95 hard limit) "
        "or if it is found powered off, and writes a heartbeat the driver requires before it "
        "creates or bills anything; independent of the droplet. RESTART it after any laptop "
        "re-login: a logout kills it and `status` then shows a stale heartbeat", billed=False))

    body = create_body(cfg)
    steps.append(Step("create", "create", "api", [Cmd((
        "python", "ops/amd/driver.py", "create", *flags, "--yes"))],
        secs("create + boot"),
        "POST /v2/droplets, then poll GET until active and record the IP in the ledger",
        api=f"POST https://api.digitalocean.com/v2/droplets  {json.dumps(body)}"))

    stage_parts = cfg.remote_stage.strip("/").split("/")
    if not cfg.remote_stage.startswith("/") or len(stage_parts) < 2 or ".." in stage_parts:
        raise ValueError(f"remote_stage {cfg.remote_stage!r} is not a safe path to rm -rf")
    steps.append(Step("bootstrap", "wait-ssh", "laptop", [ssh_ready(cfg)], 0.0,
                      f"poll `echo {READY_PREFIX}$(whoami)` until the exact line comes back (a "
                      "droplet's first boot blocks commands behind 'Please wait while we get your "
                      "droplet ready...' for minutes while sshd already answers), retrying for up "
                      "to 10 minutes", billed=False))
    steps.append(Step("bootstrap", "clean-stage", "droplet", [ssh(cfg, [
        "rm", "-rf", "--", cfg.remote_stage])], 0.0,
        "remove any earlier copy of the stage dir: scp -r onto an existing directory nests the "
        "upload one level down, and the entry script would verify a stale copy", billed=False))
    steps.append(Step("bootstrap", "upload", "laptop", [scp_up(cfg)], secs("bootstrap") * 0.1,
                      "stage dir to the droplet (tens of MB)"))
    steps.append(Step("bootstrap", "bootstrap", "droplet", [ssh(cfg, [
        "bash", f"{cfg.remote_stage}/entrypoint.sh"])], secs("bootstrap") * 0.9,
        "ONE non-interactive command: verify sums, unpack, stop jupyter, start the `smol` container, "
        "venv over the image's torch, training deps, HF login, model download, arm the watchdog"))

    steps.append(Step("smoke", "smoke-checks", "droplet", [remote_script(
        cfg, "smoke.sh", "--max-length", str(cfg.max_length),
        "--bench-seconds", str(BENCH_SECONDS),
        "--bench-batches", *(str(b) for b in BENCH_BATCHES))],
        secs("smoke: checklist") + secs("smoke: throughput bench") + secs("smoke: kill and resume"),
        f"scripted ROCm PASS/FAIL checklist, SFT throughput at batch {', '.join(map(str, BENCH_BATCHES))} "
        "on the real base and data, one real kill-and-resume"))
    steps.append(Step("smoke", "smoke-pull", "laptop", [scp_down(
        cfg, f"{cfg.remote_log}/measurements.json", f"{cfg.local_logs}/measurements.json")],
        0.0, "bring the measurements home; the projection is computed here", billed=False))

    # The gate: the released adapter, served the way every model will be, run through the harness
    # that will evaluate everything. It exists to find a broken serving stack, tool-call parser,
    # merge or harness on somebody else's already-trained model, for about what a minute of
    # training costs, BEFORE any of our own training. Its 60 trials per model are in the run tags
    # the L1 evaluation uses, so they are paid for once.
    steps.append(Step("gate", "gate-serve", "droplet", [serve_cmd(cfg, gate=True)],
                      secs(ROW_GATE_SERVE) - GATE_STOP_S,
                      f"download {GATE_ARM}, merge it into its own copy of the base and CHECK the "
                      "merge report, start base + it (one vLLM each), tool-call probe and a "
                      "temperature-0 comparison on a fixed training prompt"))
    steps.append(Step("gate", "gate-tunnel", "laptop", [tunnel_up(cfg)], 0.0,
                      "ssh -L to every served port, loopback on both ends", billed=False))
    steps.append(Step("gate", "gate-eval", "laptop", gate_cmds(cfg), secs(ROW_GATE_EVAL),
                      f"the fixed first-{gate_limit(cfg)}-task subset at L1, base and {GATE_ARM} "
                      "concurrently, one sample each, supervised (stall, error-rate and health "
                      "checks); run tags are the L1 tags, so these trials are not bought twice"))
    steps.append(Step("gate", "stop-gate-server", "droplet", [remote_script(
        cfg, "serve.sh", "--stop")], GATE_STOP_S,
        "STOP the gate servers before anything else, including the decision: they hold most of the "
        "GPU, a live server counts as work for the idle watchdog, and a decision that needs a human "
        "must not be made on a billing droplet's time", reserve=False))
    steps.append(Step("gate", "gate-decide", "laptop", [Cmd((
        "python", "ops/amd/driver.py", "gate-decide"))], 0.0,
        f"GO only if the harness failures are within tolerance, {GATE_ARM}'s temperature-0 output "
        f"differs from the base's, and {GATE_ARM} beats the base by the margin; otherwise print "
        "both rates, a paired comparison and the stop-reason histograms and require "
        "--accept-gate. Re-runnable at no cost", billed=False))
    steps.append(Step("gate", "go-no-go", "laptop", [Cmd((
        "python", "ops/amd/driver.py", "project", "--budget", f"{cfg.budget:g}"))], 0.0,
        "measured projection for A, B, A+B and the evaluation; STOPS unless the gate passed and "
        "the plan fits the budget", billed=False))

    for arm in ARMS:
        if arm in cfg.arms:
            steps.append(Step("train", f"sft-{arm}", "droplet", [remote_script(
                cfg, "run_sft.sh", "--arm", arm, "--max-length", str(cfg.max_length),
                "--save-steps", str(cfg.ckpt_steps), "--max-attempts", str(cfg.train_attempts))],
                secs(f"sft {arm}"),
                f"LoRA r={LORA_RANK} bf16, effective batch {EFFECTIVE_BATCH}, one epoch; batch "
                f"size from the smoke; checkpoint every {cfg.ckpt_steps} steps, latest pushed to "
                f"the private Hub repo; up to {cfg.train_attempts} automatic resumes after a crash "
                "(a GPU reset costs one checkpoint interval); skipped if already finished"))

    steps.append(Step("serve", "serve", "droplet", [serve_cmd(cfg)], secs(ROW_SERVE),
                      f"merge each trained adapter (the released one is already merged and "
                      f"checked), one vLLM per model on ports {ports(cfg)[0]}-{ports(cfg)[-1]}, then "
                      "check every adapter; the base server starts with a GPU-memory wait and one "
                      "retry"))
    steps.append(Step("tunnel", "tunnel", "laptop", [tunnel_up(cfg)], 0.0,
                      "re-opened if the gate tunnel was closed; forwards every served port",
                      billed=False))
    phases = eval_phases(cfg)
    hints = max(len(hint_rungs(cfg)), 1)
    phase_seconds = {"L1": secs(ROW_L1), "sample2": secs(ROW_SAMPLE2), "control": secs(ROW_CONTROL)}
    for rung in hint_rungs(cfg):
        phase_seconds[rung] = secs(ROW_HINTS) / hints
    for phase, cmds in phases:
        note = f"{len(cmds)} sweep{'s' if len(cmds) != 1 else ''} concurrently, each on its own " \
               "server, supervised"
        if phase == "L1":
            note += "; `eval --stage L1` stops after this one so everything else can be bought or not"
        steps.append(Step("eval", f"eval-{phase}", "laptop", cmds, phase_seconds[phase], note))
    steps.append(Step("sync", "sync-droplet", "droplet", [remote_script(
        cfg, "sync_back.sh", "--push-hub", "--arms", ",".join(cfg.arms))], secs("sync back + verify") * 0.6,
        "adapters, checkpoints and logs to the private Hub repos", reserve=False))
    steps.append(Step("sync", "sync-pull", "laptop", [scp_down(
        cfg, f"{cfg.remote_log}/*", f"{cfg.local_logs}/")], secs("sync back + verify") * 0.4,
        "logs and the checksum list to logs/amd/ (each entry by name: scp refuses a bare '.')",
        reserve=False))
    steps.append(Step("sync", "verify-sync", "laptop", [Cmd((
        "python", "ops/amd/driver.py", "verify-sync"))], 0.0,
        "adapters readable from the Hub, local run trees complete and clean, remote checksums match",
        reserve=False, billed=False))
    steps.append(Step("destroy", "tunnel-down", "laptop", [tunnel_down(cfg)], 0.0,
                      "close the tunnel", reserve=False, billed=False))
    steps.append(Step("destroy", "destroy", "api", [Cmd((
        "python", "ops/amd/driver.py", "destroy", "--tag", cfg.tag, "--yes"))],
        secs("destroy + verify"),
        "DELETE every droplet tagged, then GET until none remains; the ledger closes the interval "
        "only when the GET confirms",
        api=f"DELETE https://api.digitalocean.com/v2/droplets?tag_name={cfg.tag}  then  "
            f"GET /v2/droplets?tag_name={cfg.tag}  (must be empty)", reserve=False))
    return steps


def create_body(cfg: Config) -> dict:
    """The droplet-create request. The key is attached BY FINGERPRINT and the tag is set at
    creation: a droplet that exists without the tag is one the dead-man switch cannot find."""
    return {"name": cfg.name, "region": cfg.region, "size": cfg.size, "image": cfg.image,
            "ssh_keys": [cfg.fingerprint or "<ssh-key-fingerprint>"], "tags": [cfg.tag],
            "backups": False, "ipv6": False, "monitoring": False}
