# AMD Developer Cloud runbook, session 2: a gate, three SFT arms and a five-model ladder, frugally

Code in [`ops/amd/`](../ops/amd/), tests in [`tests/test_ops_amd.py`](../tests/test_ops_amd.py). Every
number is either read off the account by a GET on 2026-10-01, quoted from a source, **measured in
session 1** (a spot MI350X in ric1, the vLLM 0.17.1 image; stated as such wherever it is used), or
labelled an estimate. The costed table is computed by the driver and a test keeps this page in step
with it.

The task: LoRA-SFT `Qwen/Qwen3.5-2B` three ways on one GPU droplet, **A** (upstream's
`FineEnvs/SmolDataEnvs-sft`), **B** (our exported ja3 traces, `ja3_sft_v2`, 1,122 rows), **A+B**, then
evaluate the base, the three adapters and the **released upstream adapter `R`**
(`AdithyaSK/smoldataenvs-sft-2b-v0`, same base, r=16, evaluated but not trained) on the SmolDataEnvs
`test` split, then **destroy the droplet and stop** for an owner checkpoint. No GRPO in this session.

**What session 1 was, and why this one is different.** Session 1 spent $14.87 and produced nothing
usable: training dropped tool calls, the merge copied the base's weights over the merged ones, the
evaluation prompt contradicted the training format, concurrent evaluations shared a scratch
directory, and an evaluation hung for two hours without anyone noticing. All of those are fixed in
the repo. This session also makes the operational failures impossible or cheap: a **gate** proves the
whole stack on a model somebody else already trained before a cent goes on training (section 2a),
the evaluation is **supervised** so a stall costs minutes (section 6), and section 11 lists what
session 1 taught, each item with the guard that now exists.

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

**Costed plan: gate only $1.65. Gate + train + L1 eval $8.08. Everything bought $17.53.** The first is
what a NO-GO at the gate costs, the second is the plan, and the third buys every optional stage below.
It is computed by `python ops/amd/driver.py plan`, which prints the same lines under the table. The
rows below are session 1's measurements on that hardware until this droplet replaces them: after the
gate `driver.py project` re-renders the table, and after a destroy or on a new droplet it is the
session-1 table again.

**Gate only: $1.65. Gate + train + L1 eval: $8.08. The second L1 sample adds $2.33. L2-L4 add $6.99.
The control adds $0.14.** Each optional stage is a separate purchase made after reading L1:
`eval --stage sample2`, `hints`, `control` (or `rest` for all three).

| stage | hours | $ | basis |
|---|---|---|---|
| create + boot | 0.07 | 0.16 | estimate |
| bootstrap | 0.20 | 0.49 | session 1: about 12 min billed, first-boot wait included |
| smoke: checklist | 0.03 | 0.08 | estimate |
| smoke: throughput bench | 0.06 | 0.14 | 1 batch size x (120 s timed + 90 s load); session 1 trained at batch 4 |
| smoke: kill and resume | 0.08 | 0.20 | estimate |
| gate: serve base + released adapter | 0.12 | 0.30 | estimate: download and merge R, start 2 engines, adapter check |
| gate: 60 tasks x 2 models | 0.09 | 0.22 | 120 trials at 22/min; the L1 run reuses them |
| sft A | 0.57 | 1.41 | 9.09M tokens / 5,446 tok/s x 1.15 + 150 s |
| sft B | 0.22 | 0.55 | 3.12M tokens, same basis |
| sft AB | 0.76 | 1.86 | 12.21M tokens, same basis |
| serve: start vLLM | 0.12 | 0.29 | estimate: merge 3 trained adapters (R is already merged), start 5 engines |
| eval L1 | 0.86 | 2.11 | 5 models x 250 tasks x 1 sample = 1,250 trials - 120 done in the gate = 1,130 at 22/min |
| eval L1 second sample (incremental) | 0.95 | 2.33 | 1,250 trials at 22/min; **optional** |
| eval L2-L4 (incremental) | 2.84 | 6.99 | 3 hint rungs x 5 models x 250 tasks x 1 sample; **optional: look at L1 first** |
| eval program control | 0.06 | 0.14 | base under `--agent program`, 0.3x a bash trial; optional |
| sync back + verify | 0.08 | 0.20 | reserve |
| destroy + verify | 0.02 | 0.04 | reserve |
| **TOTAL, every row** | **7.13** | **17.53** | |

What the table rests on, **as measured in session 1 on that hardware** (not yet on this droplet):

- **Training**: 5,446 real tokens per second at per-device batch 4 (effective batch 8). The image
  lacks the two fast kernels the gated-delta-net layers want (`causal_conv1d`,
  `flash-linear-attention`), so training runs the slow reference code; section 10, item 4, says
  what installing them would take and why it is not done.
- **Trained tokens**: A 9,085,233 trained tokens (4,439 rows), B 3,123,047 trained tokens (1,122
  rows of `ja3_sft_v2`), AB 12,208,280 (A + B, computed that way by `stage.py`), counted with the
  Qwen3.5 chat template and tokenizer, rows cut at the 8192 window as training cuts them. The
  session-1 B figure (6.2M) was v1. A `tokens.json` staged from other data (the stage directory
  of session 1 holds v1) is **refused** by the driver with a note, and the recount above is used.
- **Evaluation**: about 47 s per 16-turn bash trial, and **about 22 trials per minute summed over
  five servers at 8 workers each**. One L1 block of five models is therefore 5 x 250 = 1,250
  trials / 22 per minute = **57 minutes**; the gate has already done 2 x 60 of them, so the L1 row
  is 1,130 trials = 51 minutes. (An earlier plan said 11-12 minutes. That was wrong.)
- **Bootstrap**: about 12 minutes of billed time including the first-boot wait.
- **Evaluation is the largest optional cost**, and every lever is a flag: `--limit`, `--samples`,
  `--late-samples`, `--rungs`, `--no-base`, `--no-program-control`. The order is the order a reclaim
  hurts least: L1 for every model (one sample), then, each bought on its own, a second L1 sample,
  L2, L3, L4, and the base's one-turn control.
- The gate **measures** the trials per minute with two models; the evaluation is costed at the lower
  of that and session 1's 22 (two models say nothing good about five). The go/no-go stops the session
  if the plan, plus what is already spent, would pass the budget.

### 2a. The gate

Before any training, the smoke (after its checklist, one-batch-size benchmark and
kill-and-resume) runs the gate on the **released adapter**, whose recipe and output somebody else has
already validated, so a fault in our stack shows up on a model we did not train:

1. `serve.sh` downloads `R` from the Hub, **merges** it into its own copy of the base, checks the
   merge report (`merge_adapter.py --check`: modules applied, tensors changed, the loader's weight
   files are not the base's), and starts **two** servers, the base (:8000) and `R` (:8004).
2. It checks `R` through the served stack: a parsed `bash` tool call comes back (the parser matches),
   and its **temperature-0 output on a fixed training prompt differs from the base's** (the adapter
   is applied; identical output is what the first merge bug looked like).
3. From the laptop, through the tunnel, the harness runs `--agent bash` at **L1 on the first 60 tasks
   of the SmolDataEnvs test split, one sample, both models concurrently**, supervised (section 6).
   The run tags are the **L1 tags** (`amd2-base`, `amd2-r`), so the full L1 run finds those 60 trials
   done and does not pay for them again.
4. The gate servers are **stopped first**, then `gate-decide` reads the results:

   | verdict | when |
   |---|---|
   | **GO** | (a) harness failures within tolerance (default 3 of 60 per model; a trial that was never written counts as a failure), **and** (b) the merge report passed, `R` produced a tool call, and its output differs from the base's, **and** (c) `R`'s pass rate is at least the base's plus the margin (default 0.05) |
   | **NO-GO** | (a) or (b) failed. `--accept-gate` cannot override it: the stack is broken |
   | **STOP** | only (c) failed. It prints both rates, a **paired comparison** (both pass / only base / only `R` / neither, and an exact sign test) and **stop-reason histograms** for both models, and requires `driver.py gate-decide --accept-gate` (free; it re-reads the results and then re-runs the projection) |

   The histograms show the share of `max_turns`, `context_exhausted`, `answer_submitted` (and
   `model_stopped`, which is how an episode ends under the default `--bash-stop model`) and the share
   that **ended with an answer**: a healthy SFT model should mostly end that way, and one that mostly
   runs out of turns is telling you about the prompt or the stack, not about its skill.

The droplet is idle while you decide, and the on-droplet watchdog powers it off after 30 idle minutes
(the dead-man then destroys a powered-off droplet): **decide within about 25 minutes** of the
`STOP`, or `destroy --yes` and keep the gate's trials, which live on the laptop and are reused by the
next droplet.

A GO is recorded in the ledger for **this droplet and hardware only**, and **`go-no-go` (and so
`train`) and the evaluation both refuse without it**; the evaluation additionally refuses unless the
latest `serve` on this droplet merged, checked and verified every adapter it is about to evaluate
(section 11, item 2).

### Evaluation protocol (stated, because it decides what the numbers mean)

SmolDataEnvs-sft, which A, B and A+B are trained on, is the single-`bash`-tool agent format that
submits to `/workdir/answer.txt`. In this repo that protocol is **`run_ladder --agent bash`**
(`smol_ladder.upstream.BASH_TOOL`, 16 turns) with the harness's default **`--bash-stop model`**: only
the model ends an episode, as in the SFT rows. The flag is never passed by the driver, so the default
cannot be overridden by accident. Every adapter is evaluated under it. The **base-model control is
evaluated under `--agent bash` too**, so that adapter minus base is the effect of the SFT and not a
difference of protocols. A second, nearly free control runs the base under `--agent program` (one
turn, no tools), the protocol upstream publishes a number for (`docs/LOCAL_MODELS.md`). Everywhere the
chat template is non-thinking (`{"enable_thinking": false}`), as the SFT rows were rendered.

Each model is its own **merged** checkpoint on its own vLLM server and port (LoRA serving does not
work for this model on vLLM 0.17.1): base 8000, A 8001, B 8002, AB 8003, R 8004, fixed by model, and
the tunnel forwards every served port. Each is a separate harness process with its own run tag
(`amd2-base`, `amd2-a`, `amd2-b`, `amd2-ab`, `amd2-r`, `amd2-base-program`) and its **own scratch
directory** (the harness makes one per invocation; `SMOL_LADDER_SCRATCH` is still set per model).
`--no-climb` gives every rung the same denominator for all models. The prefix is `amd2`, not session
1's `amd1`: **a run tag that is reused is read back as finished**, so session 1's results (made by
models that were not what they were named) must never be found by this session's tags. Any other
Hub adapter can be added with `--eval-only NAME=owner/repo,...`: it gets the next port, a run tag, a
tunnel forward, a row in the table and a check, but is never trained or pushed.

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
   reclaim, run `stage.py` again first (it is free). The upload refuses to run without it. Run it again before
   every session in any case: a stage directory left by session 1 holds `tokens.json` for the superseded v1
   arm B, which the driver now refuses (and says so) in favour of the recount.

## 4. The exact reviewer command sequence

Run from the worktree, with the same plan flags on every command (or none). Nothing mutates the
cloud without `--yes`, and **nothing bills without a live dead-man switch**: `create` and every other
billed step are refused unless the watcher's heartbeat is less than 90 seconds old. **After any laptop
re-login the dead-man process is gone** (a logout kills it): `driver.py status` then shows a stale
heartbeat and `NO LIVE DEADMAN`, and the next billed step is refused until step 1 is run again.
`status` also prints which ssh agent socket is in use and whether a key is loaded; ssh needs
`SSH_AUTH_SOCK`, and the driver exports the one from the environment, or falls back to
`/run/user/<uid>/keyring/ssh` when the variable is unset or names a socket that is gone.

```sh
cd /home/evan/Documents/smol-ladder
FP=$(ssh-keygen -E md5 -lf ~/.ssh/id_ed25519.pub | awk '{print $2}' | sed 's/^MD5://')
mkdir -p logs

# 0. free: stage everything (re-stage: the tokens must be the v2 counts), read the plan, check the account (GETs only)
uv run --with tokenizers python ops/amd/stage.py --out /tmp/smol-ladder-stage --commit HEAD --max-length 8192
python ops/amd/driver.py dry-run --ssh-key-fingerprint "$FP"
python ops/amd/driver.py preflight --ssh-key-fingerprint "$FP"

# 1. BEFORE create: the dead-man switch, DETACHED (setsid nohup), so closing this terminal cannot kill it.
#    Start it again after ANY laptop re-login.
setsid nohup python ops/amd/deadman.py --deadline-minutes 550 --budget 35 --total-cap 90 \
    --price 2.46 --tag smol-ladder >> logs/deadman.log 2>&1 < /dev/null &
sleep 5; python ops/amd/driver.py status      # must say HEARTBEAT FRESH, and name the ssh agent and a loaded key

# 2. billing starts here
python ops/amd/driver.py create --ssh-key-fingerprint "$FP" --yes
python ops/amd/driver.py bootstrap     # waits until `echo READY_$(whoami)` really runs, clears the stage dir, uploads, installs
python ops/amd/driver.py smoke         # checklist, bench, kill/resume, then THE GATE; ends in GATE: GO|NO-GO|STOP and GO/NO-GO
#    GATE: STOP (only the rate did not clear the margin): read the paired comparison and the histograms, then, free:
#      python ops/amd/driver.py gate-decide --accept-gate
#    GATE: NO-GO, or GO/NO-GO: NO-GO, means stop and destroy (step 5; about $1.65 is spent by now)

# 3. only if the smoke printed GO (valid for this droplet only). Optional: tighten the deadman with the
#    minutes it printed (start the new one, then `kill` the old pid from logs/deadman.log). Then the long
#    steps, DETACHED, polling their logs:
setsid nohup python ops/amd/driver.py train >> logs/train.log 2>&1 < /dev/null &    # SFT A, B, A+B; auto-resumes; stops any vLLM, asserts a free GPU
until ! pgrep -f 'ops/amd/driver[.]py train' >/dev/null; do sleep 30; done; tail -n 5 logs/train.log
python ops/amd/driver.py serve         # merge A, B, AB, one vLLM per model, every adapter checked; R is already merged
python ops/amd/driver.py tunnel        # ssh -L to every served port (8000-8004), loopback on both ends

# 4. evaluation in separate purchases. L1 first: five models, one sample, SUPERVISED, then STOP.
setsid nohup python ops/amd/driver.py eval --stage L1 >> logs/eval-L1.log 2>&1 < /dev/null &
tail -f logs/amd/progress-eval-L1.log  # one line a minute: trials per model, trials/min, ETA, accrued dollars
until ! pgrep -f 'ops/amd/driver[.]py eval' >/dev/null; do sleep 30; done; tail -n 20 logs/eval-L1.log
python ops/amd/driver.py status        # read the L1 numbers and the money, THEN decide:
#    either buy more (a second L1 sample $2.33, L2-L4 $6.99, the control $0.14 at the table's rates)...
setsid nohup python ops/amd/driver.py eval --stage rest >> logs/eval-rest.log 2>&1 < /dev/null &
until ! pgrep -f 'ops/amd/driver[.]py eval' >/dev/null; do sleep 30; done; tail -n 20 logs/eval-rest.log
#    ...or buy one at a time (--stage sample2, hints, control), or skip straight to the teardown.

# 5. teardown
python ops/amd/driver.py sync          # adapters + logs off the droplet, then verified three ways
python ops/amd/driver.py destroy --yes # DELETE by tag, then verify by tag, by id and by account audit
python ops/amd/driver.py status        # the ledger must say nothing is billing
# 6. RE-ARM before any further create: after a destroy the deadman has exited (or will); start a
#    new one (step 1) before the next `create`. The driver refuses to create without it.
```

`eval --stage` takes `L1` (the default), `sample2` (the second L1 sample, on the same run tags, so
sample 0 is reused), `hints` (L2-L4), `control`, `rest` (sample2, hints and control) or `all`.
`verify-sync` only expects the stages that actually ran. **Concurrency** is `--workers` per model,
default **8** (measured safe); the driver **refuses values above 10 unless `--i-know` is passed**
(20 per model hung every engine in session 1).

After a GO, the whole remainder can run unattended with
`setsid nohup python ops/amd/driver.py go >> logs/go.log 2>&1 < /dev/null &` (`go` takes
`--stage` too, default L1). Prefer the manual sequence above: `go` destroys after its evaluation, so a
`go --stage L1` cannot be followed by the other stages without a new droplet. What `go` guarantees:

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

If the smoke prints **NO-GO** (a failed check, a failed gate, or a projection past the budget),
nothing further runs. `driver.py project` re-evaluates the decision with new flags and no spend
(smaller `--limit`, fewer `--rungs`, `--arms A,B`); then continue. The cheapest exit is
`driver.py destroy --yes`: the whole smoke and gate cost about $1.65.

## 5. What each step does

| step | where | what |
|---|---|---|
| `stage.py` | laptop | tarballs, checksums, token counts (AB = A + B, from `ja3_sft_v2`), entry script, secrets file |
| `preflight` | laptop | GETs: size offered in region, price unchanged, image slug, key registered, nothing already tagged |
| `create` | API | refused without a fresh deadman heartbeat, with an open ledger interval, or with anything already tagged (never two droplets); records the interval **before** POSTing (billing starts at creation), POSTs size/region/image/key-fingerprint/tag `smol-ladder`, polls until active, records the IP. A timeout, 429 or 5xx is **not** "nothing created": the tag listing is polled for about 90 s, a droplet that shows up is adopted, and otherwise the interval stays open and a new create is refused until it resolves |
| `bootstrap` | both | `wait-ssh` runs **`echo READY_$(whoami)`** and requires that exact line, retrying for up to 10 minutes (each try bounded by a 60 s timeout): the image's first boot holds commands behind "Please wait while we get your droplet ready..." while sshd already answers, so an exit status of `true` proved nothing. Then `clean-stage` (removes the remote stage dir so a re-upload cannot nest), `upload`, then `entrypoint.sh`, one non-interactive command: verify sums, unpack, find the image's python, a venv layered over it with only the pure-Python training deps (never the repo's `train` extra: it pins CUDA torch and bitsandbytes), HF login, private Hub repos created and **asserted private even if they pre-existed**, base model downloaded, watchdog armed (apt runs under a lock timeout with bounded retries). `remote.env` is deleted from the laptop afterwards |
| `smoke` | both | `smoke.sh`: ROCm/stack checklist (PASS/FAIL each, a critical FAIL stops before the benchmark); `bench.py`: the real trainer at per-device batch 4 for 120 s on a mixed A/B sample; `resume_check.sh`: SIGKILL after the first checkpoint, then the real restart logic; then **the gate** (section 2a): `gate-serve`, `gate-tunnel`, `gate-eval` (supervised), **`stop-gate-server`**, `gate-decide`, and the costed GO/NO-GO, valid for this droplet and hardware only. `driver.py gate` runs the gate steps alone |
| `train` | droplet | `run_sft.sh` per arm: stop any vLLM, skip if finished (on disk, or **on the Hub: a `final.done` marker next to the adapter, which a fresh droplet honours**), assert at least 90% of GPU memory is free, move aside half-written checkpoints, restore `last-checkpoint/` from the Hub on a fresh droplet, then the trainer with `--resume` and a **checkpoint every `--ckpt-steps` (default 50) steps**. If the trainer dies after running at least two minutes (a GPU "device wedged" reset killed one in session 1), the loop **resumes up to `--train-attempts` (default 3) times**, each after the card has recovered, so a reset costs one checkpoint interval; a run that dies sooner is a bug, not a reset, and is not retried. Final adapter verified and pushed, marker last |
| `serve` | droplet | `serve.sh --wait --verify --arms A,B,AB --hub R=...:8004`: stop everything, **wait for the GPU to give its memory back**, merge every trained adapter and check each merge report **before any server starts**, then start one `vllm` per model (0.17 of the GPU each, `--enable-auto-tool-choice --tool-call-parser qwen3_coder`, `--default-chat-template-kwargs '{"enable_thinking": false}'`, bound to 127.0.0.1). A server that **dies during startup is restarted once** after waiting for GPU memory again (the base server failed once with "Engine core initialization failed" right after the others were killed). Then every adapter gets the tool-call probe and the temperature-0 comparison; the script fails if any adapter's output is identical to the base's |
| `eval` | laptop | `uv run python -m smol_ladder.run_ladder` per model, concurrently, **supervised** (section 6), each against its own port; the guards refuse it without a gate GO and a verified serve |
| `sync` | both | adapters and logs to the private Hub repos (works when no arm has finished), pulled to `logs/amd/` (the remote entries by name, `.../smol-ladder/*`: new scp rejects a bare `.` with "unexpected filename"), then `verify-sync`: adapters readable on the Hub, SHA-256 of the pulled adapters equals the droplet's, every evaluated run tag holds **at least 90%** of its first rung's trials **as clean results** (a harness failure is not a trial; a tag with one result is a dead sweep) |
| `destroy` | API | DELETE by tag, then verified three ways before the ledger closes: the tag listing is empty, every droplet id the ledger ever recorded answers GET with 404 (one that lost its tag is deleted by id), and the whole account is listed. Anything else found, above all an **untagged GPU droplet**, is reported loudly and written to the ledger, and is never deleted: it is not ours |

Every ssh step has a timeout (three times its projection, at least 15 minutes) and every step
streams its output as it arrives, so a hung step is killed instead of billing and a quiet one is
visible. ssh does not remember host keys (`StrictHostKeyChecking=no`, `UserKnownHostsFile=/dev/null`):
the droplet is ephemeral, reached by the IP the API just returned, and providers reuse addresses,
so `accept-new` would refuse a second session's droplet at an old IP.

Checkpoint cadence: `train/sft_lora.py` has no flag for it (it saves every 100 steps), so
`ops/amd/sft_run.py` takes `--save-steps` and forces it into the trainer's `SFTConfig`;
`hub_strategy="every_save"` pushes each checkpoint to the arm's private repo. A reclaim or a reset
costs one interval. The kill-and-resume in the smoke exercises exactly this path once.

Knobs `train/sft_lora.py` does not expose, and that would cut cost if it did: `--packing`, turning
gradient checkpointing off (288 GB can afford it). They are outside this change's scope.

## 6. Safety rails

**Ledger** (`ops/amd/ledger.jsonl`, git-ignored, append-only). Every lifecycle event with a
timestamp and the hourly rate it bills at. Spend is *recomputed* from the events, so an interval with
no closing event (a spot reclaim, an unresolved create) still counts. `status` shows the droplet,
uptime, accrued and remaining dollars, the caps, the $95 hard limit, the deadman's state and the ssh
agent, and
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

**Supervision of the evaluation (the stall policy).** Every harness step (`gate-eval` and each `eval-*`)
runs under `ops/amd/supervise.py`, which watches three independent signals, because from inside the
harness a hung engine looks like a long run of slow, ordinary-looking failures:

| signal | default | meaning |
|---|---|---|
| stall | no new `result.json` from any live sweep for **5 minutes** (`--stall-minutes`) | the engines are not producing |
| errors | at least **50%** (`--error-share`) of a model's last **10** (`--error-window`) results written since the last (re)start are harness failures (`agent_status != "exit 0"`) | trials are failing as fast as they arrive |
| health | a one-token chat completion to every live model's port, once a minute with a 30 s timeout, **fails twice in a row** | the engine accepts connections and never answers |

Any of them (and a harness process exiting non-zero) triggers the same bounded recovery: the harness
processes are stopped **cleanly, by the process group of the child the driver started** (never by
pattern-matching process names, and never the group the driver itself runs in), the **servers are
restarted once** (the same `serve` step, with its GPU-memory wait and start retry) and the tunnel
checked, and the sweeps resume with `--retry-failed` (a clean result is never re-run). If it happens
**again**, the stage is stopped and the driver exits non-zero (75) with the reason, the droplet
**still billing** (`status`, then `destroy --yes` or fix and re-run: finished trials are reused).
The waste from a stall is therefore bounded by two detection windows and one server start, about
15-20 billed minutes, not the two hours session 1 lost. Every minute one line goes to
`logs/amd/progress-<step>.log` and the terminal: trials done per model (and failed), trials per
minute, ETA, accrued dollars. A sweep that finished is also not trusted blindly: if its last results
are mostly failures it is retried once before it is called done, and `verify-sync` counts only clean
results.

**ssh agent and the dead-man after a re-login.** A laptop re-login kills the dead-man process and can
leave a shell holding an `SSH_AUTH_SOCK` that no longer exists. `status` shows a stale heartbeat
(`NO LIVE DEADMAN`, with the start command) and prints the agent socket in use and whether a key is
loaded; the driver exports `SSH_AUTH_SOCK` from the environment, or falls back to
`/run/user/<uid>/keyring/ssh` when the variable is unset or its socket is gone. **Restart the dead-man
(section 4, step 1) after any re-login, before any billed command.**

**On-droplet watchdog** (`watchdog.sh`). Trips on wall clock, no work process for 30 min, or no ssh
session for 20 min. It pushes everything to the Hub, then powers off. A droplet cannot destroy itself;
the power-off is the signal the laptop's switch acts on. A live vLLM counts as work, which is one more
reason the gate's servers are stopped before training. Do not leave the droplet idle for more than about
25 minutes between steps.

## 7. Failure playbook

| situation | what to do |
|---|---|
| **Spot reclaim email** (two hours' notice, normally) | If the current arm will finish inside the notice, let it; else `driver.py sync`. After the reclaim: `driver.py status` closes the ledger interval. Then re-arm the deadman, `stage.py` again (the bootstrap deleted `remote.env`), `create --yes`, `bootstrap`, **`smoke` again** (a GO and the measurements belong to the droplet they were taken on; a new droplet has none), then `train` (an arm whose final adapter is on the Hub is skipped, the unfinished one resumes from the Hub's `last-checkpoint/`), and carry on; `eval` resumes because the harness reuses finished trials. It is the same session, so the spend so far still counts. |
| **ROCm failure** in the checklist (no `/dev/kfd`, torch cannot see the GPU, bf16 matmul wrong) | The smoke stops before the benchmark. `destroy --yes` (about $0.7 spent). Retry once on the **fallback hardware**: `--fallback` (MI325X, tor1, $3.80/h, same image slug, on-demand) on every command. If that fails the same way, the image is the problem: `--image amddeveloperclou-pytorch2100rocm7`. |
| **vLLM too old**, or not importable on the host python (`vllm >= 0.16.2` fails; "Model architectures ['Qwen3_5ForConditionalGeneration'] are not supported") | The entrypoint or the smoke stops. Options in order: the other image; on the PyTorch image, the vLLM ROCm container with `/opt/smol-ladder` and `~/.cache/huggingface` bind-mounted and `--device /dev/kfd --device /dev/dri`. |
| **Tool calls empty** (`serve.sh --verify` prints `ADAPTER_CHECK ... tool_calls_ok=0`, or `leaked`/`prose`) | A parser mismatch, not a bad model. Set `AMD_TOOL_PARSER` (for example `hermes`) in the droplet's `/opt/smol-ladder/.env`, run `serve` (or `gate`) again, then `gate-decide`. Never train past a NO-GO on this: every sweep would score about zero. |
| **LoRA serving** | Not used: LoRA mode does not work for this model on vLLM 0.17.1 (it crashes at cuda-graph warmup, and with `--enforce-eager` cannot load a peft all-linear adapter). Every adapter is merged and served as an ordinary model, one process and port each. Every merge writes `merge_report.json` (modules applied, max relative delta, changed tensors, adapter sha256) and exits non-zero if nothing was applied or the merged weights equal the base's; `serve.sh` refuses a directory without a passing report (`merge_adapter.py --check DIR`). |
| **Adapter output identical to the base's** (`ADAPTER_CHECK ... differs=0`, `GATE: NO-GO`) | The adapter is not applied: a no-op merge, or a wrong directory served. `serve.sh` exits non-zero, the gate is a NO-GO that `--accept-gate` cannot override, and the evaluation refuses to start. Never train or evaluate past it: every arm would score as the base. The probe prints both outputs on a fixed training prompt and each one's token agreement with that row's training target. |
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
| **The evaluation stalled** (the progress log stops moving, or `driver.py eval` exits 75 with "STOPPED after 1 recovery attempt(s)") | The supervisor already stopped the sweeps, restarted the servers once and resumed. If it happened again the droplet is still billing: read `/var/log/smol-ladder/vllm_*.log` on the droplet and the tunnel (`driver.py tunnel up`), fix, and re-run `eval --stage L1`: finished trials are reused. Do not raise `--workers` above 8 to "go faster": 20 per model is what hung session 1's engines. |
| **A server did not start** ("Engine core initialization failed", a cuda-graph capture assertion), usually right after the others were killed | `serve.sh` waits for the GPU's memory to be released before it starts anything and restarts a server that dies during startup once, after waiting again. If it still fails it exits with the log path; a manual `serve.sh` a minute later has always worked. |
| **A GPU "device wedged" reset killed training** | `run_sft.sh` resumes up to `--train-attempts` times from the newest checkpoint (one `--ckpt-steps` interval lost), after the card has recovered. If it ran out of attempts, `driver.py train` again resumes from disk or the Hub. |
| **`wait-ssh` is slow on a new droplet** | Expected: the first boot holds commands behind "Please wait while we get your droplet ready..." for a few minutes. It is not "ready" until `READY_root` comes back; nothing else runs before that. |
| **ssh says it has no key / "Permission denied" after a re-login** | The shell's `SSH_AUTH_SOCK` is dead. `driver.py status` shows the socket in use and whether a key is loaded (the driver falls back to `/run/user/<uid>/keyring/ssh`); `ssh-add -l` should list your key. Then restart the dead-man. |
| **The gate printed `GATE: STOP`** | Only the pass-rate margin failed. Read the paired comparison and the stop-reason histograms: if the released adapter is roughly level with the base and mostly runs out of turns, suspect the prompt or the stack rather than the adapter, and do not train. If you judge the protocol sound, `driver.py gate-decide --accept-gate` continues. |

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
(size mismatch on the fused linear-attention projections). So the only serving mode is merged (merge,
then serve each model on its own process and port; tool calls and adapter effect verified). The
3-step LoRA smoke trained fine. Session 1 fitted five 2B servers on the card (0.15-0.2 of it each;
the script uses 0.17) and measured about 180-200 tokens per second per server with about 7
concurrent requests each, because the five engines share the GPU.

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
2. **Merged serving of the released adapter**: it is downloaded from the Hub in the gate, so a repo
   layout that differs from `adapter_config.json` + `adapter_model.safetensors` fails there, at about
   $0.30 of server time, before any training. The gate is exactly the place for that to fail.
3. **The training stack installing cleanly** over the image's torch: transformers >= 5.17, trl >= 1.13
   and peft >= 0.21 resolving against a `torch==2.10.0` constraint, and the `.pth` layering exposing the
   image's torch to the venv.
4. **Training speed: the two fast kernels.** Session 1 measured 5,446 tok/s because the image's
   container lacks `causal_conv1d` and `flash-linear-attention`, so the gated-delta-net layers run the
   slow reference code. **Not installed, deliberately.** `causal-conv1d` is a CUDA/HIP extension that
   compiles from source at install (many billed minutes, a compiler toolchain the container may not
   have, and no ROCm build was verified for this card). `flash-linear-attention` is pure Triton and
   would install in seconds, but its `triton`/`torch` requirements sit next to the image's pinned
   ones (the training venv installs under a constraints file, so pip would fail rather than replace
   them), and nothing here proves its kernels compile and give finite losses on gfx950. It could cut
   training time substantially (training is $3.82 of the $8.08 plan), but a broken kernel on a spot
   droplet costs more than it saves, and it cannot be verified offline. If you want to try it, do it
   in the smoke, not the real run: in the container,
   `/opt/smol-ladder/.venv/bin/python -m pip install -c /var/log/smol-ladder/constraints.txt flash-linear-attention`,
   re-run `bash /opt/smol-ladder/ops/amd/smoke.sh`, and keep it only if the benchmark's tok/s goes up
   and the kill-and-resume still passes; otherwise `pip uninstall flash-linear-attention`. The plan
   and the code do not depend on it.
5. **The tool-call parser**: `qwen3_coder` is from the vLLM Qwen3.5 recipe for a 397B MoE, not
   validated for a 2B dense model or for an SFT adapter's output. The gate checks it on the released adapter.
6. **The checkpoint file names the restart logic treats as "complete"** (`optimizer.pt`,
   `scheduler.pt`, `rng_state*.pth`) under transformers 5. If they differ, the smoke's kill-and-resume
   reports FAIL (it would otherwise restart from zero silently) and the projection is NO-GO.
7. **Hub push of checkpoints** (`hub_strategy="every_save"`, private repos, `last-checkpoint/`) working
   with these library versions and a token with write scope. The entrypoint checks write access by
   creating the repos; the first checkpoint push is only exercised in the real arm.
8. **Spot capacity at the moment of creation**, and the two-hour notice actually being given.
9. **Evaluation throughput at five models**: 22 trials/min is session 1's measurement at 5 x 8
   workers. The gate measures two models only; the table uses the lower of the two.
10. **ssh as root with the registered key** (host keys are not remembered, see section 5), and the
    laptop's connection surviving hours of tunnel (`ServerAliveInterval`).
11. **The $100 credit's real expiry and balance**: the portal's 2026-10-18 date is the reviewer's
    reading; the ledger is an estimate of spend, not the billing system's number.

## 11. Lessons from session 1

Each item is what went wrong, what it cost, and the guard that now exists. A test pins every guard.

1. **A stalled evaluation went unnoticed for two hours (about $5).** Raising concurrency from 8 to 20
   parallel trials per model hung every vLLM engine at zero throughput; every trial then timed out
   slowly and was recorded as a harness error, so results kept "arriving". Now: concurrency defaults
   to the measured-safe 8 and the driver refuses more than 10 without `--i-know`; the evaluation is
   supervised by stall, error-share and health-probe signals with one bounded recovery (section 6).
2. **Models were evaluated that were byte-identical to the base** (the merge copied the base's
   weights over the merged ones). Now: `merge_adapter.py` verifies what the loader reads and writes a
   report; `serve.sh` refuses a directory without a passing report, merges and checks every adapter
   before any server starts, and checks each adapter's tool calls and temperature-0 output against the
   base's afterwards; `serve` records the verdicts in the ledger, and **the evaluation refuses to
   start** unless the gate was a GO on this droplet and the latest serve covered every adapter it will
   evaluate. The gate repeats the check on the released adapter before any training.
3. **The released upstream adapter was merged and served by hand as a fifth model.** Now it is arm
   `R`, a first-class eval-only arm (`--eval-only NAME=owner/repo` adds more): downloaded, merged,
   checked, served on its own port, tunnelled, evaluated under its own run tag, in the cost table and
   in `verify-sync`, never trained or pushed, and it is the gate's subject.
4. **`driver.py serve` after a hang: the base server failed to start once** right after the others
   were killed ("Engine core initialization failed", a cuda-graph capture assertion), and **a GPU
   "device wedged" reset killed training once**. Now: `serve.sh` waits for the GPU memory to be
   released before starting anything and restarts a server that dies during startup once; training
   resumes automatically up to `--train-attempts` times with a checkpoint every `--ckpt-steps` (50)
   steps.
5. **The first boot blocks ssh commands** ("Please wait while we get your droplet ready...") for a
   few minutes while `wait-ssh` already reported ready. Now it runs `echo READY_$(whoami)` and requires
   the exact line, with retries and a per-try timeout.
6. **`sync-pull` failed** with scp "error: unexpected filename: ." (new OpenSSH's scp refuses a bare
   `.` entry). Now it names the remote entries (`.../smol-ladder/*`).
7. **A laptop re-login killed the dead-man process**, and the default ssh agent socket was gone
   (ssh needs `SSH_AUTH_SOCK=/run/user/1000/keyring/ssh` in this environment). Now `status` shows a
   stale heartbeat with the start command, and the driver exports `SSH_AUTH_SOCK` (environment, else
   the keyring fallback) and shows which agent and whether a key is loaded. **Restart the dead-man
   after any re-login.**
8. **Serving measurements.** LoRA mode does not work on vLLM 0.17.1 for this model (merged only);
   one vLLM process per model on ports 8000+; five 2B servers fit (0.15-0.2 of the card each); about
   180-200 tok/s per server at about 7 concurrent requests each, because the engines share the GPU;
   a 16-turn bash trial took about 47 s; training ran at 5,446 tok/s because the container lacks
   `causal_conv1d` and `flash-linear-attention` (section 10, item 4). All of it is in the costed table.
9. **Results of an invalid session must not be reused.** A run tag that exists is read back as
   finished. This session's tags are `amd2-*`; session 1's `amd1-*` trees are left alone and never
   read. `verify-sync` counts clean results only.

## Sources

- [AMD Developer Cloud](https://www.amd.com/en/developer/resources/cloud-access/amd-developer-cloud.html)
- [DigitalOcean AMD pricing and billing](https://docs.digitalocean.com/products/amd/details/pricing/)
- [How to create AMD GPU Droplets](https://docs.digitalocean.com/products/amd/how-to/create/)
- [AMD credits](https://docs.digitalocean.com/products/amd/details/credits/) and
  [limits](https://docs.digitalocean.com/products/amd/details/limits/)
- [DigitalOcean API: droplets, delete by tag, SSH keys](https://docs.digitalocean.com/reference/api/)
- `docs/LOCAL_MODELS.md` (the upstream protocols, the vLLM flags, the 0.16.2 floor), `docs/TRAINING.md`
