# AMD Developer Cloud runbook: three SFT arms and a four-model ladder, frugally

Code in [`ops/amd/`](../ops/amd/), tests in [`tests/test_ops_amd.py`](../tests/test_ops_amd.py). Nothing
here has run on AMD hardware. Every number is either read off the account by a GET on 2026-10-01,
quoted from a source, or labelled an estimate; the costed table is computed by the driver and a test
keeps this page in step with it.

The task: LoRA-SFT `Qwen/Qwen3.5-2B` three ways on one GPU droplet, **A** (upstream's
`FineEnvs/SmolDataEnvs-sft`), **B** (our exported ja3 traces), **A+B**, then evaluate the base and the
three adapters on the SmolDataEnvs `test` split, then **destroy the droplet and stop** for an owner
checkpoint. No GRPO in this session.

**What changed from the first version of this plan.** It was costed on `gpu-mi300x1-192gb` at $2.59/h
in ATL1. On this account that size is listed but **offered in no region**, so it could never have
been created. The plan is rebuilt for the hardware that exists, for frugality, and for a split
where the droplet only trains and serves while the laptop (which has the sandbox, the task tables
and 32 cores) runs the ladder through an ssh tunnel.

---

## 1. Hardware and price (verified by read-only GET, 2026-10-01)

| size slug | $/h | where offered | role |
|---|---|---|---|
| `gpu-mi350x1-288gb-spot` | **$2.46** | **ric1** | **first choice**: cheapest and largest (288 GB), spot |
| `gpu-mi355x1-288gb-spot` | $2.97 | mem1 | alternative, spot |
| `gpu-mi325x1-256gb` | **$3.80** | **tor1**, nyc2 | **fallback hardware**: on-demand, so it cannot be reclaimed |
| `gpu-mi300x1-192gb` | $2.59 | **no region** (listed, not offered) | the old plan; cannot be created |

- Billing is **per second**, and a **powered-off GPU droplet still bills**: only destroying it stops
  the meter ([DO pricing](https://docs.digitalocean.com/products/amd/details/pricing/)). That is why the
  switch that matters runs on the laptop (section 6).
- **Spot** droplets are reclaimed, normally with at least two hours' email notice, and destroyed on
  reclaim. The plan is built so a reclaim costs minutes of training and nothing of the evaluation.
- Credit: **$100**; the portal says it **expires 2026-10-18**. Caps: **$35** for this session
  (`--budget`), **$90** working total for the whole account (`--total-cap`, the credit minus a
  margin), and a **$95 HARD limit** on total spend that nothing overrides: it is a constant in
  `plan.py`, `--total-cap` above 95 is rejected by the driver and the dead-man switch, no environment
  variable is read for it, the budget gate refuses any step whose projection would pass it, and the
  dead-man switch destroys at the lower of the working cap and $95, minus $0.50 so that the destroy
  itself finishes before the limit. All three are enforced before every billed step. Money already
  spent elsewhere: `driver.py note-prior-spend USD`.
- Images offered in ric1 and tor1 (all regions, min disk 720 GB): `amddeveloperclou-vllm0171` (vLLM
  0.17.1), `amddeveloperclou-pytorch2100rocm7` (PyTorch 2.10.0, ROCm 7.2),
  `amddevelopercloud-rocm72software`, base `gpu-amd-base`.
- Account state when checked: 0 droplets, **0 registered SSH keys** (the reviewer registers one; an
  API create attaches it by fingerprint), droplet limit 10, team "My AMD Team".

### The image: why `amddeveloperclou-vllm0171`

Both application images carry ROCm 7.2 and a ROCm build of PyTorch 2.10 (vLLM cannot run without
one). The question is which *other* half is cheaper to add while billed:

- **Training stack on the vLLM image**: peft, trl, accelerate, datasets, transformers are pure-Python
  wheels, tens of MB, about a minute to install into a venv layered over the image's site-packages.
  The image's torch is never touched (a `torch==` constraint makes pip fail rather than replace it).
- **vLLM on the PyTorch image**: a ROCm vLLM build is tied to an exact torch / triton / aiter set.
  Adding it means a multi-GB torch replacement or a source build: many billed minutes, and a way to
  end with a torch that no longer matches the driver.

So the default is the vLLM image. **Fallback**: `--image amddeveloperclou-pytorch2100rocm7` plus the
vLLM ROCm container (section 8). An adapter trained on one image loads on the other.

## 2. The plan, costed

Costed total at the planning placeholders: **$19.96** (8.11 GPU-hours) of the $35 session budget.
It is computed by `python ops/amd/driver.py plan`, and every row marked UNMEASURED is replaced by the
smoke's measurements before any arm trains (`driver.py project` re-renders it). Measurements and the
GO belong to the droplet and hardware they were taken on: after a destroy, or on a new droplet, the
table is UNMEASURED again.

**Stop after L1: $11.50. L2-L4 add $7.69. The control adds $0.77.** The plan prints the same three
lines under the table. L1 for the four models is the result; the hint rungs are a second purchase
the reviewer makes after reading L1 (`--stage L1`, then `--stage rest`, section 4).

| stage | hours | $ | basis |
|---|---|---|---|
| create + boot | 0.07 | 0.16 | estimate |
| bootstrap | 0.20 | 0.49 | estimate: upload, unpack, venv, pip, model download |
| smoke: checklist | 0.03 | 0.08 | estimate |
| smoke: throughput bench | 0.17 | 0.43 | 3 batch sizes x (120 s timed + 90 s load) |
| smoke: kill and resume | 0.08 | 0.20 | estimate |
| smoke: serve + 20-task probe | 0.14 | 0.34 | estimate; measured by the probe |
| sft A | 0.50 | 1.24 | 8.70M tokens / 6,000 tok/s x 1.15 + 150 s; UNMEASURED |
| sft B | 0.37 | 0.92 | 6.20M tokens, same basis |
| sft AB | 0.83 | 2.05 | 14.90M tokens, same basis |
| serve: start vLLM | 0.08 | 0.20 | estimate: vLLM start with base + 3 adapters |
| eval L1 (4 models) | 2.08 | 5.12 | 250 tasks x 2 samples, 10 s per trial placeholder, x1.5 contention (guess) |
| eval L2-L4 (incremental) | 3.12 | 7.69 | 3 hint rungs x 250 tasks x 1 sample; **optional: look at L1 first** |
| eval program control | 0.31 | 0.77 | base under `--agent program`, 0.3x a bash trial; optional |
| sync back + verify | 0.08 | 0.20 | reserve |
| destroy + verify | 0.02 | 0.04 | reserve |
| **TOTAL** | **8.11** | **19.96** | |

What the table rests on:

- Token counts, measured with the Qwen3.5 tokenizer, rows cut at the 8192 window as training cuts
  them: A 8,695,863 trained tokens (4,439 rows), B 6,203,984 trained tokens (2,029 rows). `stage.py`
  recounts them exactly when the tokenizer is cached, else falls back to a pessimistic bytes/3.0.
- Training time = tokens / **measured real tokens per second** x 1.15 safety + 150 s per arm.
- **Evaluation is 68% of the bill**, and it is also the easiest to shrink: `--rungs L1` costs about a
  third of the full ladder; `--limit 100`, `--samples 1`, `--late-samples`, `--no-base`,
  `--no-program-control` are all flags. The evaluation order is L1 for all four models first
  (2 samples), then L2, L3, L4, then the base's one-turn control, so a reclaim or a cap hit still
  leaves the most valuable numbers, and `--stage L1` makes the stop after L1 a command, not a hope.
- The smoke measures real seconds per trial on 20 tasks, with the base and the probe adapter driven
  **concurrently** so contention is measured with two models instead of assumed for four (about a
  minute of extra wall time; the factor for 4-way then drops from the 1.5 guess to 1.25), and the
  projection stops the session (**NO-GO**) if the measured plan, plus what is already spent, would
  pass the budget.

### Evaluation protocol (stated, because it decides what the numbers mean)

SmolDataEnvs-sft, which A, B and A+B are trained on, is the single-`bash`-tool agent format that
submits to `/workdir/answer.txt`. In this repo that protocol is **`run_ladder --agent bash`**
(`smol_ladder.upstream.BASH_TOOL`, 16 turns, loop ends on submission). Every adapter is evaluated
under it. The **base-model control is evaluated under `--agent bash` too**, so that adapter minus
base is the effect of the SFT and not a difference of protocols. A second, nearly free control runs
the base under `--agent program` (one turn, no tools), the protocol upstream publishes a number for
(`docs/LOCAL_MODELS.md`). Everywhere the chat template is non-thinking
(`{"enable_thinking": false}`), as the SFT rows were rendered.

All four models share **one** vLLM server (base plus three LoRA modules); each sweep is a separate
harness process with its own run tag: `amd1-base`, `amd1-a`, `amd1-b`, `amd1-ab`, `amd1-base-program`.
`--no-climb` gives every rung the same denominator for all models.

## 3. Before the droplet exists (all free)

1. Commit everything. The droplet runs one pinned commit's `git archive`; there is no clone and no
   GitHub dependency. `stage.py` refuses a commit that lacks `train/sft_lora.py` (merge `main`
   first) and refuses uncommitted changes to the code it pins.
2. Register your SSH public key on the AMD Developer Cloud team (console: Settings, Security, SSH
   keys). The driver attaches it by fingerprint:
   `FP=$(ssh-keygen -E md5 -lf ~/.ssh/id_ed25519.pub | awk '{print $2}' | sed 's/^MD5://')`.
3. `.env` (git-ignored) holds `HF_TOKEN` (write access) and optionally `AMD_HUB_NAMESPACE` (default:
   the token's own user). The DigitalOcean token (`AMD_CLOUD_API_TOKEN`, else
   `DIGITALOCEAN_ACCESS_TOKEN`) is used by the driver and the dead-man switch in-process only; it
   never appears in an argv, a log or the ledger. `.env` wins over a variable left in the shell by an
   older session. ssh, scp and the evaluation harness are started with both DigitalOcean tokens and
   the Hugging Face tokens removed from their environment: none of them needs one.
4. `stage.py` builds the directory the droplet receives: `code.tar.gz`, both SFT tarballs,
   `tokens.json`, `repo.txt`, `SHA256SUMS`, the single `entrypoint.sh`, and `remote.env` (mode 600,
   never in the sums). The driver **deletes `remote.env` from the laptop stage dir after a
   successful bootstrap** (it holds the HF token); to bootstrap again, for example after a spot
   reclaim, run `stage.py` again first (it is free). The upload refuses to run without it.

## 4. The exact reviewer command sequence

Run from the worktree, with the same plan flags on every command (or none). Nothing mutates the
cloud without `--yes`, and **nothing bills without a live dead-man switch**: `create` and every other
billed step are refused unless the watcher's heartbeat is less than 90 seconds old.

```sh
cd /home/evan/Documents/smol-ladder-wt-amd
FP=$(ssh-keygen -E md5 -lf ~/.ssh/id_ed25519.pub | awk '{print $2}' | sed 's/^MD5://')
mkdir -p logs

# 0. free: stage everything, read the plan, check the account (GETs only)
uv run --with tokenizers python ops/amd/stage.py --out /tmp/smol-ladder-stage --commit HEAD --max-length 8192
python ops/amd/driver.py dry-run --ssh-key-fingerprint "$FP"
python ops/amd/driver.py preflight --ssh-key-fingerprint "$FP"

# 1. BEFORE create: the dead-man switch, DETACHED (setsid nohup), so closing this terminal cannot kill it
setsid nohup python ops/amd/deadman.py --deadline-minutes 624 --budget 35 --total-cap 90 \
    --price 2.46 --tag smol-ladder >> logs/deadman.log 2>&1 < /dev/null &
sleep 5; python ops/amd/driver.py status      # the last line must say HEARTBEAT FRESH

# 2. billing starts here
python ops/amd/driver.py create --ssh-key-fingerprint "$FP" --yes
python ops/amd/driver.py bootstrap     # waits for sshd, clears the remote stage dir, uploads, installs
python ops/amd/driver.py smoke         # checklist, bench, kill/resume, probe (2 models), STOPS the probe server, GO / NO-GO

# 3. only if the smoke printed GO (a GO is valid for this droplet only). Optional: tighten the
#    deadman with the minutes the smoke printed (start the new one, then `kill` the old pid from
#    logs/deadman.log). Then the long steps, DETACHED, polling their logs:
setsid nohup python ops/amd/driver.py train >> logs/train.log 2>&1 < /dev/null &    # SFT A, B, A+B; stops any vLLM, asserts a free GPU
until ! pgrep -f 'ops/amd/driver[.]py train' >/dev/null; do sleep 30; done; tail -n 5 logs/train.log
python ops/amd/driver.py serve         # ONE vLLM: base + 3 adapters
python ops/amd/driver.py tunnel        # ssh -L 8000, loopback on both ends

# 4. evaluation in two purchases. L1 first: the four models at L1, then STOP.
setsid nohup python ops/amd/driver.py eval --stage L1 >> logs/eval-L1.log 2>&1 < /dev/null &
until ! pgrep -f 'ops/amd/driver[.]py eval' >/dev/null; do sleep 30; done; tail -n 20 logs/eval-L1.log
python ops/amd/driver.py status        # read the L1 numbers and the money, THEN decide:
#    either buy the hint rungs and the control ($7.69 + $0.77 at the placeholders)...
setsid nohup python ops/amd/driver.py eval --stage rest >> logs/eval-rest.log 2>&1 < /dev/null &
until ! pgrep -f 'ops/amd/driver[.]py eval' >/dev/null; do sleep 30; done; tail -n 20 logs/eval-rest.log
#    ...or skip straight to the teardown.

# 5. teardown
python ops/amd/driver.py sync          # adapters + logs off the droplet, then verified three ways
python ops/amd/driver.py destroy --yes # DELETE by tag, then verify by tag, by id and by account audit
python ops/amd/driver.py status        # the ledger must say nothing is billing
# 6. RE-ARM before any further create: after a destroy the deadman has exited (or will); start a
#    new one (step 1) before the next `create`. The driver refuses to create without it.
```

`eval --stage` takes `L1`, `hints` (L2-L4), `control`, `rest` (hints and control) or `all`
(default). `verify-sync` only expects the stages that actually ran.

After a GO, the whole remainder can run unattended with
`setsid nohup python ops/amd/driver.py go >> logs/go.log 2>&1 < /dev/null &` (`go` takes
`--stage` too). Prefer the manual sequence above: `go` destroys after its evaluation, so a
`go --stage L1` cannot be followed by the hint rungs without a new droplet. What `go` guarantees:

- The destroy runs in a `finally`, and SIGTERM and SIGHUP are turned into an exception so the
  `finally` runs for them too (it does not otherwise: a bare `kill` skips it). A second signal during
  the cleanup is ignored. Run long commands **detached**, as above; a detached process never gets the
  SIGHUP of a closed terminal at all, so the handler is the second line of defence, not the first.
- On any failure, refused gate, Ctrl-C or signal it first makes a bounded best-effort sync (each
  command has a timeout and its failure is ignored), then destroys.
- If `verify-sync` fails in the normal path it **stops before the destroy** and says the droplet is
  still billing: fix it, `driver.py sync`, then `destroy --yes`. The dead-man switch still destroys
  at its deadline. The exception: if holding the droplet for another half hour would pass a cap or
  run into the deadman's deadline, it destroys anyway after one more best-effort sync and prints
  that what the FAIL lines name is lost (the private Hub repos are the surviving copies).

If the smoke prints **NO-GO**, nothing further runs. `driver.py project` re-evaluates the decision
with new flags and no spend (smaller `--limit`, fewer `--rungs`, `--arms A,B`); then continue. The
cheapest NO-GO exit is `driver.py destroy --yes` (the smoke costs about $1.3).

## 5. What each step does

| step | where | what |
|---|---|---|
| `stage.py` | laptop | tarballs, checksums, token counts, entry script, secrets file |
| `preflight` | laptop | GETs: size offered in region, price unchanged, image slug, key registered, nothing already tagged |
| `create` | API | refused without a fresh deadman heartbeat, with an open ledger interval, or with anything already tagged (never two droplets); records the interval **before** POSTing (billing starts at creation), POSTs size/region/image/key-fingerprint/tag `smol-ladder`, polls until active, records the IP. A timeout, 429 or 5xx is **not** "nothing created": the tag listing is polled for about 90 s, a droplet that shows up is adopted, and otherwise the interval stays open and a new create is refused until it resolves |
| `bootstrap` | both | `wait-ssh` (polls until sshd answers, up to 10 min), `clean-stage` (removes the remote stage dir so a re-upload cannot nest), `upload`, then `entrypoint.sh`, one non-interactive command: verify sums, unpack, find the image's python, a venv layered over it (the `.pth` points at the directory torch is installed in) with only the pure-Python training deps (never the repo's `train` extra: it pins CUDA torch and bitsandbytes), HF login, private Hub repos created and **asserted private even if they pre-existed** (datasets too), base model downloaded, watchdog armed (apt runs under a lock timeout with bounded retries). `remote.env` is deleted from the laptop afterwards |
| `smoke` | both | `smoke.sh`: 12-item ROCm/stack checklist (PASS/FAIL each, a critical FAIL stops before the benchmark); `bench.py`: the real trainer at per-device batch 2, 4, 8 (effective batch held at 8, never batch 1) for 120 s each on a mixed A/B sample; `resume_check.sh`: SIGKILL after the first checkpoint, then the real restart logic; `serve.sh --probe`: base plus the smoke's adapter with the final flags and a tool-call probe; 20 tasks per model from the laptop, base and adapter at once, to measure seconds per trial under 2-way contention; **`stop-probe-server`**, because that server holds about 85% of the GPU; then the costed GO/NO-GO, valid for this droplet and hardware only |
| `train` | droplet | `run_sft.sh` per arm: stop any vLLM, skip if finished (on disk, or **on the Hub: a `final.done` marker next to the adapter, which a fresh droplet honours**), assert at least 90% of GPU memory is free (refuses otherwise), move aside half-written checkpoints, restore `last-checkpoint/` from the Hub on a fresh droplet, then the trainer with `--resume`; final adapter verified and pushed, marker last |
| `serve` | droplet | `serve.sh --all`: one `vllm` process, `--enable-lora --max-loras 3 --max-lora-rank 16 --lora-modules amd-a-2b=... amd-b-2b=... amd-ab-2b=...`, `--enable-auto-tool-choice --tool-call-parser qwen3_coder`, `--default-chat-template-kwargs '{"enable_thinking": false}'`, bound to 127.0.0.1 |
| `eval` | laptop | `uv run python -m smol_ladder.run_ladder` per model, concurrently, with `SMOL_LADDER_BASE_URL=http://127.0.0.1:8000/v1`; `--stage L1` runs only the four L1 sweeps |
| `sync` | both | adapters and logs to the private Hub repos (works when no arm has finished), pulled to `logs/amd/`, then `verify-sync`: adapters readable on the Hub, SHA-256 of the pulled adapters equals the droplet's, every evaluated run tag holds **at least 90%** of its first rung's trials on the laptop (a tag with one result is a dead sweep) |
| `destroy` | API | DELETE by tag, then verified three ways before the ledger closes: the tag listing is empty, every droplet id the ledger ever recorded answers GET with 404 (one that lost its tag is deleted by id), and the whole account is listed. Anything else found, above all an **untagged GPU droplet**, is reported loudly and written to the ledger, and is never deleted: it is not ours |

Every ssh step has a timeout (three times its projection, at least 15 minutes) and every step
streams its output as it arrives, so a hung step is killed instead of billing and a quiet one is
visible. ssh does not remember host keys (`StrictHostKeyChecking=no`, `UserKnownHostsFile=/dev/null`):
the droplet is ephemeral, reached by the IP the API just returned, and providers reuse addresses,
so `accept-new` would refuse a second session's droplet at an old IP.

Checkpoint cadence: `train/sft_lora.py` saves every 100 steps (it has no flag for this) and
`hub_strategy="every_save"` pushes the newest to the arm's private repo. At an effective batch of 8 and
the measured throughput that is a few minutes, so a reclaim costs minutes. The kill-and-resume in the
smoke exercises exactly this path once.

Knobs `train/sft_lora.py` does not expose, and that would cut cost if it did: `--packing`,
`--save-steps`, turning gradient checkpointing off (288 GB can afford it). They are outside this
change's scope, so the benchmark covers batch size only.

## 6. Safety rails

**Ledger** (`ops/amd/ledger.jsonl`, git-ignored, append-only). Every lifecycle event with a
timestamp and the hourly rate it bills at. Spend is *recomputed* from the events, so an interval with
no closing event (a spot reclaim, an unresolved create) still counts. `status` shows the droplet,
uptime, accrued and remaining dollars, the caps, the $95 hard limit and the deadman's state, and
reconciles against the API: a ledger that thinks a droplet is billing when none is tagged is closed as
reclaimed (an unresolved create gets ten minutes before that).

**Budget gate.** Before every billed step: already accrued + the step's projected seconds at the
current rate + a reserve for sync and destroy must stay at or under `--budget` (session) and the
total cap, which is never above **$95**. A step that would pass either is refused ("STOPPING BEFORE
..."). Sync and destroy are exempt from the reserve they exist to spend, and are never refused.

**Deadman gate.** `create` and every other billed step are also refused unless the dead-man switch's
heartbeat file (`ops/amd/ledger.jsonl.heartbeat`) is fresh: written within three poll intervals
(90 s), the last successful droplet listing within the same window (a watcher whose every call
fails is alive but blind), the same tag, armed with caps no looser than this session's, deadline not
passed. The refusal prints the exact detached command that starts it. A missing token, a crash or a
closed terminal therefore cannot go unnoticed: they stop the next `create`.

**Dead-man switch** (`deadman.py`, on the laptop, independent of the droplet; start it detached).
Polls every 30 s; finds the droplet by **tag**; destroys by tag when any of: the wall-clock deadline
passes; the session cost reaches `--budget`; the total reaches the lower of `--total-cap` and $95,
**minus $0.50 for the destroy itself**; a tagged droplet is found **powered off** (still billing).
Cost is the larger of the ledger's and the API's own (droplet age times its size's hourly price), so
a create the ledger never recorded is still priced. Every action is appended to the ledger and
printed. It survives any API failure (timeouts, resets, non-JSON answers) with capped backoff, and it
**exits only after a destroy that was verified** (tag listing empty, every recorded id 404), or past
the deadline with a successful empty listing; an unverified destroy is retried every poll for as long
as it takes. After a verified destroy it exits on purpose, which makes the heartbeat go stale:
**re-arm it before the next create**. `--once --dry-run` shows the decision and destroys nothing.

**On-droplet watchdog** (`watchdog.sh`). Trips on wall clock, no work process for 30 min, or no ssh
session for 20 min. It pushes everything to the Hub, then powers off. A droplet cannot destroy itself;
the power-off is the signal the laptop's switch acts on. A live vLLM counts as work, which is one more
reason the probe server is stopped before training. Do not leave the droplet idle for more than about
25 minutes between steps.

## 7. Failure playbook

| situation | what to do |
|---|---|
| **Spot reclaim email** (two hours' notice, normally) | If the current arm will finish inside the notice, let it; else `driver.py sync`. After the reclaim: `driver.py status` closes the ledger interval. Then re-arm the deadman, `stage.py` again (the bootstrap deleted `remote.env`), `create --yes`, `bootstrap`, **`smoke` again** (a GO and the measurements belong to the droplet they were taken on; a new droplet has none), then `train` (an arm whose final adapter is on the Hub is skipped, the unfinished one resumes from the Hub's `last-checkpoint/`), and carry on; `eval` resumes because the harness reuses finished trials. It is the same session, so the spend so far still counts. |
| **ROCm failure** in the checklist (no `/dev/kfd`, torch cannot see the GPU, bf16 matmul wrong) | The smoke stops before the benchmark. `destroy --yes` (about $0.7 spent). Retry once on the **fallback hardware**: `--fallback` (MI325X, tor1, $3.80/h, same image slug, on-demand) on every command. If that fails the same way, the image is the problem: `--image amddeveloperclou-pytorch2100rocm7`. |
| **vLLM too old**, or not importable on the host python (`vllm >= 0.16.2` fails; "Model architectures ['Qwen3_5ForConditionalGeneration'] are not supported") | The entrypoint or the smoke stops. Options in order: the other image; on the PyTorch image, the vLLM ROCm container with `/opt/smol-ladder` and `~/.cache/huggingface` bind-mounted and `--device /dev/kfd --device /dev/dri`. |
| **Tool calls empty** (the probe prints `leaked` or `prose`, `TOOL_CALLS_OK=0`) | A parser mismatch, not a bad model. Set `AMD_TOOL_PARSER` (for example `hermes`) in the droplet's `/opt/smol-ladder/.env`, rerun `serve.sh --probe --wait`, and re-run `project`. Never train past a NO-GO on this: every sweep would score about zero. |
| **LoRA will not load** in vLLM 0.17 for Qwen3.5 (the probe server dies on the adapter) | Fallback mode, one flag on `serve`, `tunnel` and `eval`: `--serve-mode merged`. Each adapter is merged into its own 4.6 GB model and served on its own port (8000 base, 8001 A, 8002 B, 8003 AB, 20% GPU each). Costs about a minute per adapter. | Every merge writes `merge_report.json` (modules applied, max relative delta, changed tensors, adapter sha256) and exits non-zero if nothing was applied or the merged weights equal the base's; `serve.sh --merged` refuses a directory without a passing report (`merge_adapter.py --check DIR`).
| **Adapter output identical to the base's** (`ADAPTER_DIFFERS_FROM_BASE=0` in the probe, a NO-GO) | The adapter is not applied: a no-op merge, or a LoRA module that did not attach. Never train or evaluate past it: every arm would score as the base. The probe prints both outputs on a fixed training prompt and each one's token agreement with that row's training target. |
| **NO-GO on budget** | Cheapest levers first: `--rungs L1`, `--limit`, `--late-samples 1`, `--no-program-control`; then `--arms A,B`; then `--max-length 4096`; or raise `--budget` on purpose. Then `driver.py project`. |
| **Training OOM** | The benchmark already skips configurations that run out of memory and picks the best that fits; if every one OOMs the smoke stops. |
| **ssh refused / key rejected** | The fingerprint was not registered on the team, or the image does not inject keys. `destroy --yes`, fix, re-run `preflight`. |
| **The dead-man switch destroyed it** | Intended. Adapters are on the Hub; `create --yes`, `bootstrap`, `train`, and continue. |
| **`create` returns 422 / no capacity** | The API refused: nothing was created (the ledger closes the pending interval). Try `--fallback`. |
| **`create` says "outcome UNKNOWN"** (timeout, 429, 5xx) | The droplet may still land. The ledger interval stays open and counts; a second create is refused. Wait a few minutes and run `driver.py status` (it adopts or closes the pending create), check the console, or `destroy --yes`. Never create by hand meanwhile. |
| **`create` is refused: "No live dead-man switch"** | Start the deadman (section 4, step 1) and run `driver.py status` until it says HEARTBEAT FRESH. |
| **`go` stopped before the destroy: verify-sync failed** | The droplet is up and billing. Read the FAIL lines; `driver.py sync`; `destroy --yes`. The deadman destroys it at its deadline regardless. |
| **destroy "NOT verified"** | A recorded droplet id still answers, or the tag listing still shows one. Re-run `destroy --yes`; check the console. The deadman keeps retrying by itself. |
| **"UNTAGGED GPU DROPLET ... NOT touched"** | The destroy found a GPU droplet on the account that is not one the ledger recorded. It bills whether or not it is tagged and the driver will not delete what is not its own: look at the console. |
| **A step was killed by its timeout** (`exit 124`) | The droplet is still up. `status`; fix the cause (usually ssh); re-run the step; it is resumable. |

## 8. vLLM image layout (an unknown that decides the first minutes)

Measured on the live droplet: the host python has no torch or vLLM; they live in docker images
(`rocm:latest`, `vllm/vllm-openai-rocm:v0.17.1`, same torch 2.9.1 / vllm 0.17.1 / transformers 4.57.6)
and a `rocm` jupyter container holds the GPU. So `entrypoint.sh` (host) stops `rocm`, starts ONE
long-lived container `smol` from the vLLM image (`--network host`, `/dev/kfd` + `/dev/dri`,
`--ipc host`, `--restart no`, `sleep infinity`) with `/opt/smol-ladder`, the stage dir,
`/var/log/smol-ladder` and an HF cache (`/var/cache/smol-hf`) mounted at the same paths, then runs
`container_setup.sh` in it. Every plan step is `ssh ... docker exec smol bash .../<script>`; only the
watchdog (poweroff, host `pgrep`) stays on the host and calls `sync_back.sh` through `docker exec`.
Setup installs jq/procps with apt and a venv (`--system-site-packages`) holding transformers 5,
trl and peft, all under a constraints file pinning the image's torch, torchvision, torchaudio,
triton and vllm. The system transformers 4.57.6 cannot read Qwen3.5 and vLLM requires `<5`, hence
the venv. Re-running `entrypoint.sh` is idempotent (a running `smol` on the right image is kept).

Measured there: vLLM 0.17.1 with `--enable-lora` crashes at cuda-graph warmup for Qwen3.5
(`IndexError` in `set_lora`), and with `--enforce-eager` fails to load a peft all-linear adapter
(size mismatch on the fused linear-attention projections). So the default is `--serve-mode merged`
(merge, then serve each model; probe tool calls verified). The 3-step LoRA smoke trained fine.

## 9. Teardown checklist

1. `driver.py sync` finished and `verify-sync` printed PASS for every line.
2. `driver.py destroy --yes` printed "billing stopped (verified by GET): True".
3. `driver.py status` says nothing is billing; the DigitalOcean console shows no droplet, volume or
   snapshot (a snapshot or volume bills after the droplet is gone; this plan creates none).
4. The dead-man switch exited by itself after the verified destroy (`pgrep -f 'ops/amd/deadman[.]py'`
   should find nothing; if it did not, stop it). **Re-arm it before any further create.** Revoke the
   DigitalOcean token if it was only for this session.
5. Compare the ledger's total with the console's credit balance and record both next to the table in
   section 2: that comparison is the input to every future estimate.

## 10. Remaining unknowns

Ranked by how likely each is to fail on the real instance; none can be verified from the laptop.

1. **Where vLLM lives on the image** (host python vs a container). Fails at `bootstrap`, about $0.7.
2. **LoRA serving for Qwen3.5 on vLLM 0.17.1** (a multimodal architecture with gated-delta-net
   projections). The probe exercises it before any arm trains; `--serve-mode merged` is the default.
3. **The training stack installing cleanly** over the image's torch: transformers >= 5.17, trl >= 1.13
   and peft >= 0.21 resolving against a `torch==2.10.0` constraint, and the `.pth` layering exposing the
   image's torch to the venv.
4. **Qwen3.5 training speed and memory on ROCm**: the gated-delta-net layers fall back to unfused
   kernels, so tokens per second may be far below the placeholder; the benchmark measures it and the
   projection decides. `docs/TRAINING.md` measured only the laptop.
5. **The tool-call parser**: `qwen3_coder` is from the vLLM Qwen3.5 recipe for a 397B MoE, not
   validated for a 2B dense model or for an SFT adapter's output. The probe checks it.
6. **The checkpoint file names the restart logic treats as "complete"** (`optimizer.pt`,
   `scheduler.pt`, `rng_state*.pth`) under transformers 5. If they differ, the smoke's kill-and-resume
   reports FAIL (it would otherwise restart from zero silently) and the projection is NO-GO.
7. **Hub push of checkpoints** (`hub_strategy="every_save"`, private repos, `last-checkpoint/`) working
   with these library versions and a token with write scope. The entrypoint checks write access by
   creating the repos; the first checkpoint push is only exercised in the real arm.
8. **Spot capacity at the moment of creation**, and the two-hour notice actually being given.
9. **Evaluation seconds per trial and contention**: placeholder 10 s per trial with a 1.5x factor for
   four concurrent sweeps on one laptop. The 20-task probe replaces the first; the factor is a guess.
10. **ssh as root with the registered key** (host keys are not remembered, see section 5), and the
    laptop's connection surviving hours of tunnel (`ServerAliveInterval`).
11. **The $100 credit's real expiry and balance**: the portal's 2026-10-18 date is the reviewer's
    reading; the ledger is an estimate of spend, not the billing system's number.

## Sources

- [AMD Developer Cloud](https://www.amd.com/en/developer/resources/cloud-access/amd-developer-cloud.html)
- [DigitalOcean AMD pricing and billing](https://docs.digitalocean.com/products/amd/details/pricing/)
- [How to create AMD GPU Droplets](https://docs.digitalocean.com/products/amd/how-to/create/)
- [AMD credits](https://docs.digitalocean.com/products/amd/details/credits/) and
  [limits](https://docs.digitalocean.com/products/amd/details/limits/)
- [DigitalOcean API: droplets, delete by tag, SSH keys](https://docs.digitalocean.com/reference/api/)
- `docs/LOCAL_MODELS.md` (the upstream protocols, the vLLM flags, the 0.16.2 floor), `docs/TRAINING.md`
