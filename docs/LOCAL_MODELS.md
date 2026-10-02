# Running the released 2B models locally

Research (2026-10-01) on what `Qwen3.5-2B`, `AdithyaSK/smoldataenvs-sft-2b-v0` and
`AdithyaSK/smoldataenvs-grpo-2b-v0` were actually trained and evaluated with, what we had to
change to run them here, and whether this machine can serve them.

**The headline:** the two released models do not share a protocol. GRPO is a one-turn,
no-tools, write-a-program agent; SFT is a multi-turn `bash` agent that submits to a file. Our
solver is a third protocol. Running any of them through the wrong one produces a number about the
protocol, not the model — and the ladder's whole claim is a comparison *between* models, so it
needs all of them on a protocol or explicitly none.

---

## 1. The upstream protocol, exactly

Sources: [`scripts/eval_pass1.py`](https://github.com/adithya-s-k/FineEnvs/blob/main/04-smoldataenvs/scripts/eval_pass1.py),
[`scripts/rollout.py`](https://github.com/adithya-s-k/FineEnvs/blob/main/04-smoldataenvs/scripts/rollout.py),
[`scripts/train_grpo.py`](https://github.com/adithya-s-k/FineEnvs/blob/main/04-smoldataenvs/scripts/train_grpo.py),
[`FineEnvs/SmolDataEnvs-sft`](https://huggingface.co/datasets/FineEnvs/SmolDataEnvs-sft).
Copied into `smol_ladder/upstream.py` and asserted verbatim in `tests/test_local_models.py`.

### 1a. GRPO / `eval_pass1.py` — the protocol the published ~0.40 came from

| | |
|---|---|
| turns | **1.** `rollout.py` is the entire environment: one program, one sandbox run, one reward |
| tools | **none.** The request carries no `tools` key at all |
| system | `"You are a data analyst. You answer questions about CSV files by writing a short Python program and reading what it prints."` |
| user | `{question}` / blank line / `The files are in /home/user/input and your program runs in that directory:` / the `- filename` list / `Write one Python program in a ```python block, then stop.` / three bullets (answer is the last thing printed, bare; keep it under 40 lines; pandas/numpy/scipy/sklearn/statsmodels installed) |
| files | `row["files"]`. **When empty**, upstream does not print an empty list — it substitutes a paragraph telling the model to `os.listdir('/home/user/input')` |
| chat template | the model's own, with **`enable_thinking=False`** |
| decoding | greedy, `do_sample=False`, **`MAX_NEW_TOKENS=1024`** |
| answer | last fenced block wins (models think in one block and answer in the next); unfenced output is treated as code; a completion that fails `compile()` is scored 0 without spending a sandbox |
| guard | the last printed line that looks like a shell command scores 0 (`2.14 > answer.txt` is a redirect that never ran; but `>50K` is a real gold answer, so the regex needs something *before* the `>`) |
| grading | SmolDataEnvs' own `grader.py`, exact → numeric w/ per-task tolerances → list → math-verify. No LLM in the path |

`train_grpo.py` confirms the same shape: `reward_weights=[1.0, 0.0]`, `num_generations=8`,
`max_completion_length=1024`, `chat_template_kwargs={"enable_thinking": False}`,
`repetition_penalty=1.05`, `mask_truncated_completions=True`. **Both released models' card and
eval go through this path.** Upstream's own README is explicit that "everything runs non-thinking"
and that SFT, RL and eval must render the same template.

### 1b. SmolDataEnvs-sft — the bash agent

The SFT dataset is 4,677 verified trajectories, each with `messages` + `tools`. The protocol is
uniform across all of them:

| | |
|---|---|
| tool | exactly one, named **`bash`**, `{"command": str}` → combined stdout+stderr, *non-stateful between calls* |
| system | `"You are an autonomous data-analysis agent operating in a sandboxed Linux container. Your only tool is `bash`. The dataset files are in /home/user/input/. … To submit your final answer you MUST call the `bash` tool to write it to /workdir/answer.txt … Do NOT end your turn without submitting."` |
| user | files list, installed-packages list, `Question:`, the question, an optional per-task `Answer as: …` format line, then the generic "single clean value" sentence and the write-to-`answer.txt` instruction |
| answer | the contents of `/workdir/answer.txt` |
| turns | `n_turns` runs 3–12 in the published rows; the shape is always *inspect → compute → `echo -n "…" > /workdir/answer.txt` → one closing sentence* |
| chat template | same non-thinking template |

### 1c. Ours, and every point of difference

| | ours (`--agent tools`) | upstream GRPO (`--agent program`) | upstream SFT (`--agent bash`) |
|---|---|---|---|
| tools offered | `run_shell`, `write_solution` | none | `bash` |
| turns | up to 40 | 1 | up to 16, stops at submission |
| where the tables live | `./input`, read-only symlink | `/home/user/input`, and the program's **cwd** | `/home/user/input` |
| the deliverable | a persistent `./solution.py` | one program in a fence | a value in `/workdir/answer.txt` |
| how it is graded | `solution.py` re-run **offline, sealed, no network**, last line | same | the file's contents |
| model id | OpenRouter `stealth/space-bunny-alpha` | local/Hub | local/Hub |
| thinking | not sent | `enable_thinking=False` | `enable_thinking=False` |

Four differences matter, and they are the ones that will move a number:

1. **The loop.** A 2B model trained on "one program, then stop" has never produced a multi-turn
   tool transcript. Run it in our 40-turn loop and you measure its ability to imitate an agent
   shape it was not taught, not its ability to answer.
2. **The deliverable.** Ours is a *file that must still run when re-run offline, with no network*.
   That is strictly harder than "the last line it printed", and a model trained on the simpler
   contract will be scored down for a difference in the contract.
3. **The answer channel.** Upstream SFT's `answer.txt` is a value. Our solver has no answer
   channel at all — the shell tool's last output is explicitly not the prediction.
4. **The paths.** Upstream programs are written against `/home/user/input` with the tables as
   cwd. Ours is `./input` one level up.

### What a faithful evaluation actually requires

For each released model, under its own protocol, with its own prompt text byte-for-byte:

- **GRPO model → `--agent program`.** One turn, no tools, 1024 tokens, `enable_thinking=False`,
  greedy, the program extracted from the last fence and re-run sealed. That is exactly
  `eval_pass1.py`. This is the number to compare against the published ~0.28→0.40.
- **SFT model → `--agent bash`.** The `bash` schema verbatim, `/workdir/answer.txt` rewritten to
  the trial directory, the loop ending on submission, upstream's command guard applied.
- **Our solver on either of them → `--agent tools`.** Expect a large drop. That drop is itself a
  finding and belongs in the post (it is a format-transfer measurement, which is the PLAN's
  stated risk "A vs B confounded by format"), but it must be *labelled* as such.
- **Cross-arm comparison on the ladder.** Arms 0 / R-SFT / R-GRPO must each run in the protocol
  they were trained in, or the ladder compares protocols. If you want all three on one protocol,
  the honest one is `program`, since that is the only one upstream ever reports a number for —
  but then the SFT arm is being measured on a protocol it was not trained for, and that asymmetry
  has to be stated in the table.
- **The base model (arm 0) is a control.** Qwen3.5-2B is a *base* model: it has no instruction
  tuning for either protocol. Its `program` score is the floor the whole comparison rests on.
- **Non-thinking is not optional.** Omitting `enable_thinking=False` changes the prompt template
  (`<think>\n\n</think>\n\n` vs `<think>\n`), and a model trained under one rendered under the
  other is being measured on a template mismatch. The base model will not stop generating at all
  without it.

### Two traps in the repos themselves

- **`smoldataenvs-sft-2b-v0` is a LoRA adapter, not a model.** 93 MB, `adapter_config.json`,
  r=16 α=32 on `Qwen/Qwen3.5-2B`, `target_modules=all-linear`, `exclude_modules=visual.*`. You
  must serve the base model *with the adapter loaded*; pointing vLLM at the repo id alone will
  serve an adapter directory and fail.
- **`smoldataenvs-grpo-2b-v0` is stored in fp32**: 2,213,241,664 F32 parameters, an 8.85 GB
  `model.safetensors`. That does not fit in a 6 GB card in any precision this card runs well;
  it must be cast to bf16/fp16 first (4.4 GB) or quantised.

### 1d. Context budget and tool-output truncation (`--agent bash`)

Served models have a finite window (16,384 tokens on the AMD vLLM), and a model that `cat`s a CSV
used to overflow it, get a 400, burn five retries and crash with no transcript. Now:

- Tool output is cut to its first 8,000 characters plus `\n... [truncated]` (`TOOL_OUTPUT_MAX_CHARS`).
  Upstream's loop is not published; this is what its SFT rows show (max tool result 8,016 chars, 126
  of them exactly head-8,000 + that marker; p99 4,687, p95 1,541). Each request's `max_tokens` is
  1,024 (eval_pass1's cap) clamped to the room left. The window comes from `GET /v1/models`
  `max_model_len`, or `SMOL_LADDER_MAX_MODEL_LEN`.
- Context exhausted (no room left, or a context-length 400) ends the episode as a normal exit-0
  trial with `stop_reason: "context_exhausted"`, graded on whatever `answer.txt` holds: a model
  failure. Other 4xx (not 429) are never retried; 429/5xx/timeouts/empty choices still are, and a
  failure after retries is still a harness failure (`exit 1`).
- `transcript.json` is written from a `finally` for every trial. `result.json` gains `stop_reason`
  (`answer_submitted`, `model_stopped`, `max_turns`, `context_exhausted`, `single_turn`, `error`),
  `turns_used`, `last_prompt_tokens`, `truncated_outputs`, `context_length`.

---

## 2. What we changed in the code

`smol_ladder/or_agent.py` — the endpoint is configuration, not a constant:

| env var | meaning |
|---|---|
| `SMOL_LADDER_BASE_URL` | the server. A loopback URL needs **no** API key (vLLM and llama.cpp both ignore a bearer they did not ask for, and making a local run invent a token to satisfy the remote path is how a local eval ends up not runnable at all) |
| `SMOL_LADDER_API_KEY_ENV` | which variable holds the key. Default `OPENROUTER_API_KEY` |
| `SMOL_LADDER_CHAT_TEMPLATE_KWARGS` | JSON passed through as `chat_template_kwargs`. **Defaults to `{"enable_thinking": false}`** — that is how every model in this study was trained and how upstream scores it. Set to `""` to send none, for a server whose template has no such flag |

The jail's environment is allowlisted rather than inherited, so only the endpoint variables and
the named key travel into a trial; the base URL is passed even when empty because a local server
cannot do without it.

`smol_ladder/upstream.py` — the two upstream protocols verbatim, plus `localise_paths` (rewrites
`/home/user/input` → `input`, `/workdir` → `.`) and `looks_like_a_command`.

`run_ladder.py --agent {tools,program,bash}`. All three run in the same bubblewrap jail and are
graded by the same offline sealed pass and the same `smol_ladder.grade.grade`, so the comparison
differs only where the protocol differs. Two portability fixes the tests pin:

- **Both path idioms resolve.** Upstream programs use bare filenames because their cwd *is* the
  table directory; our prompts say `input/a.csv`. The offline pass keeps cwd at the trial
  directory and links each table beside `solution.py`, so both work.
- **The links are relative.** The offline pass bind-mounts the trial directory at `/tmp/work`, and
  bwrap does not resolve a symlink pointing outside the mount — an absolute link made every
  bare-filename program raise `FileNotFoundError` and score 0.0.

`tests/test_local_models.py` — 21 tests against a **stub HTTP server** on a loopback port (no
model, no weights): the prompts are byte-checked against upstream, the endpoint resolution is
checked, and both upstream modes are driven end-to-end through `once()` including the
command guard, submission-stop, and empty-output cases.

---

## 3. This machine

Measured 2026-10-01.

```
GPU     NVIDIA GeForce RTX 4050 Laptop GPU, 6141 MiB, 5.3 GB free, compute capability 8.9 (Ada)
RAM     94 GB total, ~61 GB available
CPU     32 threads
Disk    908 GB, 256 GB free
Swap    4 GB disk + 16 GB zram
HF cache 34 GB already present
```

Installed: **`vllm` no. `llama.cpp` no. `ollama` no. `torch` no. `transformers` no.**
Only `huggingface_hub` is importable (the repo's own dependency). The CUDA *driver* is present
and current (580.173.02, CUDA 13.0), so nothing blocks a GPU install — there is simply no
inference stack on the box, and we were asked not to install one.

### Does a 2B model fit in 6 GB with tool calling?

Yes for the *weights*, barely, and only with the settings below.

| | weights | leaves for KV + activations |
|---|---|---|
| base Qwen3.5-2B bf16 | 4.55 GB (Hub ships bf16) | ~1.0 GB at `--gpu-memory-utilization 0.92` |
| GRPO model as shipped | 8.85 GB fp32 — **does not fit** | — |
| GRPO cast to bf16 | 4.43 GB | ~1.2 GB |
| SFT model | 93 MB adapter on the 4.55 GB base | as base |

The arithmetic is uncomfortable but workable: with `--max-model-len 4096` (our prompts are
question + file list + a handful of rungs; upstream's SFT trajectories run 3–12 turns and the
dataset's own SFT config used `max_length=8192`, so 4096 is tight for a 12-turn bash trace and
**8192 is the safer setting if it fits**) the KV cache is a few hundred MB. Note the card is a
laptop part: expect it to be power-limited well below its 50 W ceiling in most chassis,
which moves the decode rate more than anything else below.

### Expected throughput, and how long the sweep takes

Decode for a 2B bf16 model is memory-bandwidth-bound. This card is 6 GB GDDR6 on a 96-bit bus at
16 Gbps effective → **192 GB/s** theoretical. Weights at 4.5 GB gives an arithmetic ceiling of
~43 tok/s; realistically **~20–30 tok/s at batch 1** on a laptop-capped 4050, and roughly
**150–250 tok/s aggregate** once vLLM is continuous-batching our 20 concurrent trials (the
ceiling is arithmetic bandwidth again, so the batch buys concurrency, not per-stream speed).

Per-task generation budget:

| protocol | generated tokens per trial (upstream's own numbers) |
|---|---|
| program | 250–900 for a program that finishes; the cap is 1024 |
| bash | 3–12 turns × ~100–250 tokens ≈ 400–1500, with the prompt growing each turn |
| tools | up to 40 turns; a 2B model under an unfamiliar protocol will often run to the cap |

Sweep cost, **250 tasks × up to 5 rungs**, climbing so ≈ 550 trials average (1 rung on a pass, up
to 4 on a total fail):

| | generation tokens | @ 200 tok/s aggregate | plus sandbox (550 trials over 20 workers) | total |
|---|---|---|---|---|
| `--agent program`, k=1 | ~220k | ~18 min | ~5 min | **~25 min** |
| `--agent bash`, k=1 | ~500k | ~42 min | ~20 min | **~1–1.5 h** |
| `--agent tools`, k=1 | ~1–3M | ~1.5–4 h | ~25 min | **~2–5 h** |
| any of the above, k=4 | ×4 | | | program ~1.5 h, bash ~5 h, tools ~8–20 h |

These are estimates from the bandwidth arithmetic and upstream's own token counts, not
measurements — nothing is installed here. Treat them as ±2× and check the first 10 tasks against
the `agent_seconds` field each `result.json` records.

The ladder's own optimisation still applies and matters more at this speed: climbing stops at the
first pass, so run L1 across all three models first and only then spend rungs.

**Recommendation: run the full 250×5 sweep on Kaggle T4s, not here.** The laptop is a fine place
to develop the harness and sanity-check 5–20 tasks; a 6 GB laptop GPU is not where a
three-arm × five-rung × k-sample sweep should be discovered.

---

## 4. Commands (for the owner to approve — nothing here was run)

Everything below installs packages and downloads weights. **Not run on this machine.**

### 4a. Laptop: vLLM (best fidelity, tightest memory)

```sh
# ~9 GB download: vLLM + its CUDA 12.9 torch
uv venv --python 3.12 .venv-vllm && source .venv-vllm/bin/activate
uv pip install vllm==0.30.0

export SMOL_LADDER_BASE_URL=http://127.0.0.1:8000/v1   # no API key needed
```

**Arm 0 — the base control** (bf16, 4.55 GB):

```sh
vllm serve Qwen/Qwen3.5-2B \
  --served-model-name qwen3.5-2b \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.92 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --default-chat-template-kwargs '{"enable_thinking": false}'

uv run python -m smol_ladder.run_ladder --split test --rungs L1 --agent program \
  --model qwen3.5-2b --max-turns 1 --workers 16
```

**Arm R-GRPO** — the fp32 checkpoint must be cast, or it will not fit in 6 GB:

```sh
# Either cast it offline and serve the local directory:
uv run --with 'torch>=2.4' --with 'transformers>=5.17' --with peft python -c "
from transformers import AutoModelForCausalLM
import torch
m = AutoModelForCausalLM.from_pretrained('AdithyaSK/smoldataenvs-grpo-2b-v0',
                                         dtype=torch.bfloat16)
m.save_pretrained('models/grpo-2b-bf16')"
# ...or let vLLM cast it at load time, which is one flag and no extra disk:
vllm serve AdithyaSK/smoldataenvs-grpo-2b-v0 \
  --served-model-name grpo-2b \
  --dtype bfloat16 --max-model-len 8192 \
  --gpu-memory-utilization 0.92 \
  --default-chat-template-kwargs '{"enable_thinking": false}'

uv run python -m smol_ladder.run_ladder --split test --rungs L1 --agent program \
  --model grpo-2b --max-turns 1 --workers 16
```

**Arm R-SFT** — the adapter, on its base:

```sh
vllm serve Qwen/Qwen3.5-2B \
  --served-model-name sft-2b \
  --enable-lora --lora-modules sft-2b=AdithyaSK/smoldataenvs-sft-2b-v0 \
  --max-lora-rank 16 \
  --max-model-len 8192 --gpu-memory-utilization 0.92 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --default-chat-template-kwargs '{"enable_thinking": false}'

uv run python -m smol_ladder.run_ladder --split test --rungs L1 --agent bash \
  --model sft-2b --max-turns 16 --workers 16
```

Pass `--model` the **`--served-model-name`**, not the repo id: the adapter runs are served under
the base model's name and the LoRA is selected per-request.

### 4b. Laptop: llama.cpp (quantised, comfortable, less faithful)

```sh
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
cmake -B build -DGGML_CUDA=ON && cmake --build build -j --target llama-server
pip install -r requirements/requirements-convert_hf_to_gguf.txt
python convert_hf_to_gguf.py AdithyaSK/smoldataenvs-grpo-2b-v0 --outfile grpo-q8_0.gguf
./build/bin/llama-quantize grpo-q8_0.gguf grpo-Q4_K_M.gguf Q4_K_M

./build/bin/llama-server -m grpo-Q4_K_M.gguf --host 127.0.0.1 --port 8080 \
  --jinja --ctx-size 8192 -ngl 99
export SMOL_LADDER_BASE_URL=http://127.0.0.1:8080/v1
export SMOL_LADDER_CHAT_TEMPLATE_KWARGS=      # empty: llama-server's --jinja path
```

Q4_K_M of a 2B is ~1.4 GB, so this is the option that leaves room for a long context. The cost
is fidelity: quantisation perturbs the model, and llama.cpp's tool-call parsing is not
`qwen3_coder`. **Use it for development and small sanity runs; use vLLM bf16 for anything
reported.**

### 4c. Kaggle T4 (recommended for the real sweep)

Two T4s, 16 GB each — the base model in bf16 fits three times over, so `--max-model-len` can go
to 32768 and two servers can run concurrently while the harness drives both.

```sh
# In a Kaggle notebook, T4 accelerator, Internet ON (phone-verified), add HF_TOKEN as a Secret.
!pip install -q vllm==0.30.0

!vllm serve AdithyaSK/smoldataenvs-grpo-2b-v0 \
  --served-model-name grpo-2b --dtype bfloat16 \
  --max-model-len 16384 --gpu-memory-utilization 0.90 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --default-chat-template-kwargs '{"enable_thinking": false}'
```

Then point the harness at it. Two practical notes:

- On Kaggle the model and the harness are in **different processes**, so `SMOL_LADDER_BASE_URL`
  must be reachable from the harness — run the server with `--host 0.0.0.0` and use the
  notebook's own `localhost:8000` from a second cell, or run the server and the sweep in the
  same cell sequentially.
- The Kaggle disk is wiped between sessions: push any merged/converted model to the Hub, and
  write results to `data/runs/` (a symlink to the shared volume) rather than `/kaggle/working`.
- vLLM does not support P100; the notebook must select a **T4**.

---

## 5. What is uncertain

- **The throughput and wall-time figures are bandwidth arithmetic, not measurements.** Nothing
  could be installed, so no tok/s was observed on this card. ±2×.
- **`--tool-call-parser qwen3_coder` is from the vLLM Qwen3.5 recipe**, and that recipe documents
  a 397B MoE, not a 2B dense. The parser name should hold (it is keyed on the template, and the
  template ships with the model) but the combination is untested here. If auto tool calls come
  back empty, that flag is the first thing to check.
- **vLLM ≥ 0.16.2 is required.** `qwen3_5` support was cut before v0.16.1 was tagged; v0.30.0
  (2026-09-22) has it, including LoRA and GatedDeltaNet kernels for Qwen3.5. An older pin fails
  with `Model architectures ['Qwen3_5ForConditionalGeneration'] are not supported for now`.
- **Whether the SFT adapter's tool behaviour survives vLLM's parser** is unknown. It was trained
  with `chat_template_kwargs={"enable_thinking": False}` on the same template, so the wire format
  should match, but the model's raw output has not been inspected.
- **Whether GRPO started from SFT is still unclear** in upstream's own materials, so "GRPO"
  may mean SFT-then-RL or RL-from-base. It does not change how we serve it; it does change what
  the arm means in the writeup.
- **The `n_turns` cap of 16 for the bash mode** is ours: upstream's rows stop at 12, and there is
  no published turn cap. A model that has not submitted by 16 is recorded as a failure, which is
  a judgement call we made, not a measurement upstream made.
- **Quantised llama.cpp results are not comparable to bf16 vLLM results.** Treat any 4b number as
  a development signal only.