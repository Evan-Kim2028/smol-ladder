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
checkpoints to a private Hub repo every few minutes and why evaluation runs L1 for all four models
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


ARMS = ("A", "B", "AB")
BASE_MODEL = "Qwen/Qwen3.5-2B"
VLLM_MIN = "0.16.2"
SERVE_PORT = 8000
LORA_RANK = 16

# ── time estimates, in seconds, used until the smoke replaces them with measurements ─────────
BOOT_S = 240.0               # create -> ssh answers
BOOTSTRAP_S = 720.0          # upload, unpack, venv, pip, model download
SMOKE_CHECK_S = 120.0
BENCH_CONFIGS = 3            # batch sizes timed; see BENCH_BATCHES
BENCH_LOAD_S = 90.0          # model load per configuration, billed on top of the timed window
RESUME_CHECK_S = 300.0
PROBE_START_S = 300.0        # vLLM up with base + the smoke's adapter
SFT_OVERHEAD_S = 150.0       # per arm: load, evals, final save and push
SERVE_START_S = 300.0
SYNC_S = 300.0
DESTROY_S = 60.0
RESERVE_S = SYNC_S + DESTROY_S   # held back so the last two steps are always affordable
SAFETY = 1.15                # multiplier on measured training time
CONTENTION = 1.5             # four concurrent sweeps share one server and the laptop's cores

# Placeholders. Every one is labelled UNMEASURED in the table and replaced by the smoke.
ASSUMED_TOKENS_PER_S = 6000.0
ASSUMED_SEC_PER_TRIAL = 10.0   # wall seconds per trial with the probe's worker count
PROGRAM_COST_FACTOR = 0.3      # a one-turn trial is far cheaper than a bash trial

# Effective batch is held at upstream's 8 sequences/step so arm A stays comparable to the
# published recipe; the batch sizes trade per-device width for accumulation.
EFFECTIVE_BATCH = 8
BENCH_BATCHES = (2, 4, 8)      # never 1: 288 GB is not a reason to train at batch 1
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
    samples: int = 2                    # L1
    late_samples: int = 1               # L2..L4
    rungs: tuple[str, ...] = RUNGS
    workers: int = 12
    max_turns: int = 16
    tag_prefix: str = "amd1"
    include_base: bool = True
    program_control: bool = True
    serve_mode: str = "lora"            # "lora" (one server) | "merged" (one server per model)
    max_model_len: int = 16384
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


# ── arm / model naming: defined once, read by the serve and the eval commands ───────────────

def served_name(arm: str) -> str:
    return "amd-base-2b" if arm == "base" else f"amd-{arm.lower()}-2b"


def run_tag(cfg: Config, arm: str) -> str:
    return f"{cfg.tag_prefix}-{arm.lower()}"


PROBE_NAME = "amd-probe-2b"


def eval_arms(cfg: Config) -> list[str]:
    return (["base"] if cfg.include_base else []) + [a for a in ARMS if a in cfg.arms]


def port_for(cfg: Config, arm: str) -> int:
    """One server carries every model in `lora` mode; `merged` mode is one server per model."""
    if cfg.serve_mode == "lora":
        return SERVE_PORT
    return SERVE_PORT + (["base", "A", "B", "AB"].index(arm))


def ports(cfg: Config) -> list[int]:
    return sorted({port_for(cfg, a) for a in eval_arms(cfg)} | {SERVE_PORT})


# ── ssh / scp ─────────────────────────────────────────────────────────────────────

def ssh_opts(cfg: Config) -> list[str]:
    opts = ["-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new",
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
    return ssh(cfg, ["bash", f"{cfg.remote_root}/ops/amd/{script}", *args])


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
             max_turns: int | None = None) -> Cmd:
    """One sweep of one model at one rung set, run on the laptop against the tunnel.

    `--agent bash` is the protocol the SFT data is in. `--no-climb` runs every requested rung on
    every task, so a rung's pass rate has the same denominator for all four models.
    `SMOL_LADDER_BASE_URL` points at the tunnel (loopback, so no API key is needed); the
    non-thinking kwargs match both the SFT rows and the server's default.
    """
    turns = max_turns if max_turns is not None else (1 if agent == "program" else cfg.max_turns)
    return Cmd(
        ("uv", "run", "python", "-m", "smol_ladder.run_ladder",
         "--split", cfg.split, "--model", model or served_name(arm),
         "--run-tag", tag or run_tag(cfg, arm), "--agent", agent,
         "--rungs", rungs, "--samples", str(samples), "--no-climb",
         "--limit", str(limit if limit is not None else cfg.limit),
         "--workers", str(cfg.workers), "--max-turns", str(turns)),
        (("SMOL_LADDER_BASE_URL", f"http://127.0.0.1:{port_for(cfg, arm)}/v1"),
         ("SMOL_LADDER_CHAT_TEMPLATE_KWARGS", CHAT_KWARGS)))


def eval_phases(cfg: Config) -> list[tuple[str, list[Cmd]]]:
    """Evaluation in the order a spot reclaim hurts least.

    L1 for all four models first (that single comparison is the result), then L2, L3, L4 across
    all four, then the one-turn `program` control of the base. Within a phase the models run
    concurrently, one harness process each, against the same server.
    """
    phases: list[tuple[str, list[Cmd]]] = []
    for rung in cfg.rungs:
        n = cfg.samples if rung == "L1" else cfg.late_samples
        phases.append((rung, [eval_cmd(cfg, a, rung, n) for a in eval_arms(cfg)]))
    if cfg.include_base and cfg.program_control:
        phases.append(("control", [eval_cmd(
            cfg, "base", "L1", 1, agent="program", tag=f"{run_tag(cfg, 'base')}-program")]))
    return phases


def probe_cmd(cfg: Config) -> Cmd:
    """The smoke's 20-task measurement: base model, L1, same protocol and workers as the real run."""
    return eval_cmd(cfg, "base", "L1", 1, tag=f"{cfg.tag_prefix}-probe", limit=20)


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


def heuristic_tokens(data_dir: Path, max_length: int) -> dict[str, SetTokens]:
    """No tokenizer, no stage dir: bytes / 3.0. Measured on the real sets, the true ratio is
    ~3.3 bytes per token for A and ~3.4 for B after truncation at 8192, so this over-counts by
    roughly 10-15%: the safe direction for a budget."""
    out: dict[str, SetTokens] = {}
    sets = {"A": [data_dir / "train" / "sft_upstream" / "train.jsonl"],
            "B": [data_dir / "train" / "ja3_sft.jsonl"]}
    for key, files in sets.items():
        files = [f for f in files if f.exists()]
        if not files:
            continue
        rows = sum(1 for f in files for line in f.open() if line.strip())
        out[key] = SetTokens(rows, int(sum(f.stat().st_size for f in files) / 3.0),
                             "bytes/3.0 heuristic (pessimistic)")
    if "A" in out and "B" in out:
        out["AB"] = SetTokens(out["A"].rows + out["B"].rows,
                              out["A"].trained_tokens + out["B"].trained_tokens,
                              "sum of A and B")
    return out


@dataclass
class Measured:
    """What the smoke measured. Zero/None means "not measured yet": the table says so."""
    tokens_per_s: float = 0.0
    batch_size: int = 0
    grad_accum: int = 0
    sec_per_trial: float = 0.0       # wall seconds per bash trial at cfg.workers, from the probe
    probe_start_s: float = 0.0       # vLLM start time with base + adapter
    tool_calls_ok: bool | None = None
    checks_ok: bool | None = None
    resume_ok: bool | None = None


def measured_from_ledger(events: list[dict]) -> Measured:
    m = Measured()
    for e in events:
        if e.get("event") != "measured":
            continue
        for key in ("tokens_per_s", "batch_size", "grad_accum", "sec_per_trial",
                    "probe_start_s", "tool_calls_ok", "checks_ok", "resume_ok"):
            if e.get(key) is not None:
                setattr(m, key, e[key])
    return m


def sft_seconds(tokens: int, tokens_per_s: float) -> float:
    if tokens_per_s <= 0:
        raise ValueError("tokens_per_s must be measured or assumed explicitly, not zero")
    return tokens / tokens_per_s * SAFETY + SFT_OVERHEAD_S


def eval_seconds(cfg: Config, sec_per_trial: float) -> float:
    """Wall seconds the one server must stay up for the whole evaluation.

    Phases run in series, the models inside a phase run concurrently, so a phase takes
    (trials per model) x (measured wall seconds per trial) x CONTENTION. `sec_per_trial` was
    measured with one model driving the server and the laptop's cores to itself, which is what
    CONTENTION pays for.
    """
    total = 0.0
    for rung in cfg.rungs:
        n = cfg.samples if rung == "L1" else cfg.late_samples
        total += cfg.limit * n * sec_per_trial * CONTENTION
    if cfg.include_base and cfg.program_control:
        total += cfg.limit * sec_per_trial * PROGRAM_COST_FACTOR * CONTENTION
    return total


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
    tps_basis = ("measured" if meas.tokens_per_s
                 else f"UNMEASURED: assumes {ASSUMED_TOKENS_PER_S:,.0f} tok/s")
    spt = meas.sec_per_trial or ASSUMED_SEC_PER_TRIAL
    spt_basis = ("measured" if meas.sec_per_trial
                 else f"UNMEASURED: assumes {ASSUMED_SEC_PER_TRIAL:.0f} s per trial")
    bench = BENCH_CONFIGS * (BENCH_SECONDS + BENCH_LOAD_S)
    rows = [
        Row("create + boot", BOOT_S, "estimate", False),
        Row("bootstrap", BOOTSTRAP_S, "estimate: upload, unpack, venv, pip, model download", False),
        Row("smoke: checklist", SMOKE_CHECK_S, "estimate", False),
        Row("smoke: throughput bench", bench,
            f"{BENCH_CONFIGS} batch sizes x ({BENCH_SECONDS}s timed + {BENCH_LOAD_S:.0f}s load)",
            False),
        Row("smoke: kill and resume", RESUME_CHECK_S, "estimate", False),
        Row("smoke: serve + 20-task probe",
            (meas.probe_start_s or PROBE_START_S) + 20 * spt,
            "estimate: vLLM start + 20 trials", bool(meas.probe_start_s and meas.sec_per_trial)),
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
    ev = eval_seconds(cfg, spt)
    rows.append(Row("serve + evaluate (4 models)", SERVE_START_S + ev,
                    f"{len(eval_arms(cfg))} models, {len(cfg.rungs)} rungs, {cfg.limit} tasks; "
                    f"{spt_basis}; x{CONTENTION} contention", bool(meas.sec_per_trial)))
    rows.append(Row("sync back + verify", SYNC_S, "reserve", False))
    rows.append(Row("destroy + verify", DESTROY_S, "reserve", False))
    return rows


def total_dollars(rows: list[Row], price: float) -> float:
    return sum(r.dollars(price) for r in rows)


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
        "creates or bills anything; independent of the droplet", billed=False))

    body = create_body(cfg)
    steps.append(Step("create", "create", "api", [Cmd((
        "python", "ops/amd/driver.py", "create", *flags, "--yes"))],
        secs("create + boot"),
        "POST /v2/droplets, then poll GET until active and record the IP in the ledger",
        api=f"POST https://api.digitalocean.com/v2/droplets  {json.dumps(body)}"))

    steps.append(Step("bootstrap", "upload", "laptop", [scp_up(cfg)], secs("bootstrap") * 0.1,
                      "stage dir to the droplet (tens of MB)"))
    steps.append(Step("bootstrap", "bootstrap", "droplet", [ssh(cfg, [
        "bash", f"{cfg.remote_stage}/entrypoint.sh"])], secs("bootstrap") * 0.9,
        "ONE non-interactive command: verify sums, unpack, venv over the image's torch, "
        "training deps, HF login, model download, arm the watchdog"))

    steps.append(Step("smoke", "smoke-checks", "droplet", [remote_script(
        cfg, "smoke.sh", "--max-length", str(cfg.max_length),
        "--bench-seconds", str(BENCH_SECONDS))],
        secs("smoke: checklist") + secs("smoke: throughput bench") + secs("smoke: kill and resume"),
        "scripted ROCm PASS/FAIL checklist, SFT throughput at 3 batch sizes on the real base "
        "and data, one real kill-and-resume"))
    steps.append(Step("smoke", "smoke-pull", "laptop", [scp_down(
        cfg, f"{cfg.remote_log}/measurements.json", f"{cfg.local_logs}/measurements.json")],
        0.0, "bring the measurements home; the projection is computed here", billed=False))
    steps.append(Step("smoke", "probe-serve", "droplet", [remote_script(
        cfg, "serve.sh", "--probe", "--wait",
        *(["--merged"] if cfg.serve_mode == "merged" else []))],
        secs("smoke: serve + 20-task probe") * 0.5,
        "serve the base plus the smoke's adapter with the final flags: proves LoRA loads and "
        "the tool-call parser returns tool calls BEFORE any arm is trained"))
    steps.append(Step("smoke", "probe-tunnel", "laptop", [tunnel_up(cfg)], 0.0,
                      "ssh -L to the droplet, loopback on both ends", billed=False))
    steps.append(Step("smoke", "probe-eval", "laptop", [probe_cmd(cfg)],
                      secs("smoke: serve + 20-task probe") * 0.5,
                      "20 tasks of L1 under --agent bash: measures seconds per trial"))
    steps.append(Step("smoke", "go-no-go", "laptop", [Cmd((
        "python", "ops/amd/driver.py", "project", "--budget", f"{cfg.budget:g}"))], 0.0,
        "measured projection for A, B, A+B and the evaluation; STOPS unless it fits the budget",
        billed=False))

    for arm in ARMS:
        if arm in cfg.arms:
            steps.append(Step("train", f"sft-{arm}", "droplet", [remote_script(
                cfg, "run_sft.sh", "--arm", arm, "--max-length", str(cfg.max_length))],
                secs(f"sft {arm}"),
                f"LoRA r={LORA_RANK} bf16, effective batch {EFFECTIVE_BATCH}, one epoch; batch "
                "size from the smoke; checkpoint every 100 steps, latest pushed to the private "
                "Hub repo; resumes from disk or Hub; skipped if already finished"))

    steps.append(Step("serve", "serve", "droplet", [remote_script(
        cfg, "serve.sh", "--all", "--arms", ",".join(cfg.arms), "--wait",
        *(["--merged"] if cfg.serve_mode == "merged" else []))],
        SERVE_START_S,
        "ONE vLLM server: base + every adapter as named LoRA modules" if cfg.serve_mode == "lora"
        else "merge each adapter, one vLLM server per model (fallback mode)"))
    steps.append(Step("tunnel", "tunnel", "laptop", [tunnel_up(cfg)], 0.0,
                      "re-opened if the probe tunnel was closed", billed=False))
    ev_total = secs("serve + evaluate (4 models)") - SERVE_START_S
    phases = eval_phases(cfg)
    weight = {rung: (cfg.samples if rung == "L1" else cfg.late_samples) for rung in cfg.rungs}
    weight["control"] = PROGRAM_COST_FACTOR
    wsum = sum(weight[p] for p, _ in phases) or 1.0
    for phase, cmds in phases:
        steps.append(Step("eval", f"eval-{phase}", "laptop", cmds, ev_total * weight[phase] / wsum,
                          f"{len(cmds)} sweep{'s' if len(cmds) != 1 else ''} concurrently against the one server"))
    steps.append(Step("sync", "sync-droplet", "droplet", [remote_script(
        cfg, "sync_back.sh", "--push-hub", "--arms", ",".join(cfg.arms))], secs("sync back + verify") * 0.6,
        "adapters, checkpoints and logs to the private Hub repos", reserve=False))
    steps.append(Step("sync", "sync-pull", "laptop", [scp_down(
        cfg, f"{cfg.remote_log}/.", f"{cfg.local_logs}/")], secs("sync back + verify") * 0.4,
        "logs and the checksum list to logs/amd/", reserve=False))
    steps.append(Step("sync", "verify-sync", "laptop", [Cmd((
        "python", "ops/amd/driver.py", "verify-sync"))], 0.0,
        "adapters readable from the Hub, local run trees complete, remote checksums match",
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
