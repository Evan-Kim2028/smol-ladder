# Training: exporting data, SFT, and GRPO

Everything needed to run the first post-training pass of a ~2B model, on the laptop or on Kaggle.
Research and measurements 2026-10-01. Code in [`train/`](../train/), tests in
[`tests/test_train.py`](../tests/test_train.py).

```
uv run --extra train python -m train.export_sft --source ... --out ...
uv run --extra train python -m train.sft_lora  --data ... --out ...
uv run --extra train python -m train.grpo     --adapter ... --rung-schedule ladder
```

---

## 1. The headline finding: our sweeps were throwing away the training data

**Every sweep run before 2026-10-01 saved no transcripts.** Every trial wrote `turns.json` --
which contains `json.dumps(len(log))`, a single integer -- copied `solution.py` (and `answer.txt`)
out of the per-trial scratch, and deleted the scratch. The assistant/tool conversation was never
written down. A verified trial was therefore a program and an answer, and `data/runs/**` had
**zero** trainable transcripts in 2,178 verified trials.

**`gen_solutions` does save them** (`data/solutions/<split>/<task>/transcript.jsonl`), but it has
only ever been run on the held-out splits:

| source | verified tasks | full transcript | trainable |
|---|---|---|---|
| `data/solutions/test` | 211 | 211 | **no — held out** |
| `data/solutions/eval` | 116 | 116 | **no — held out** |
| `data/solutions/jupyter-agent` | 479 | **0** | — |
| `data/runs/test` | 217 | **0** | — |
| `data/runs/synthetic` | 211 | **0** | — |
| `data/runs/jupyter-agent.bak-…` | 290 | **0** | — |

**327 full multi-turn transcripts exist on this machine and every one is from `test` or `eval`.**
There is no trainable multi-turn trace here yet. `gen_solutions.py` says so itself: *"Never train on
them: they come from the held-out splits."*

**The consequence, stated where it is easy to miss: the existing verified trials yield only
single-turn examples.** They are not unusable -- `collect_traces` still emits a row for each one,
through the fallback that writes the verified program, submits the known answer and stops. That
teaches the contract and nothing about exploration, so an SFT run on it learns *how to answer* and
not *how to find out*. A training set of fallback rows is a smaller, honest dataset and must never
be reported as a trace dataset; the export says how many came from which path.

**Fixed, on both paths.** `run_ladder.once()` and `gen_refs.attempt()` now save the conversation
to `<task>/<rung>/transcript.json` (and to the reference slot `promote()` copies up), default-on,
`--no-transcript` opts out on either. The file is the **whole conversation**, not just the replies:
`or_agent.solve_loop` and `bash_loop` return the message list they sent, so it opens with the
system and user turns and carries every assistant message, tool call and tool result in order. The
earlier version of this fix saved only the assistant turns, which is enough to count turns and not
enough to train on — a chat template has to re-attach a prompt and a system message, and it cannot
know which prompt the turns answered.

Tested end to end in `tests/test_local_models.py`: `once()` runs against a stub completions
endpoint, so the file is written by `or_agent`'s own loop and read back by `once()`. They assert
the roles and their order, that each call carries its arguments, that each result carries its real
output and the `tool_call_id` of the call that produced it, that the user turn is the rung text
verbatim, and that the bash protocol's submission is in the transcript with its value.

**This cannot be applied retroactively.** Arm B's data has to come from a new sweep, not from the
tree already on disk — that is what the `ja3` transcript sweep on the v3 ladder-grade pool is for.

**That sweep has now been run and exported: §5.** The `ja3` sweep (run tag `ja3`, split
`jupyter-agent-v3`, rung L1, 3,880 tasks, model `stealth/space-bunny-alpha`) produced 2,109
verified trials and `train/export_ja3.py` turns them into **2,029 real SFT trajectories** in
upstream's format. The numbers above describe the tree as it stood *before* it; they are kept
because they are the reason the exporter exists, and because `--source traces` still reads that old
tree and still finds nothing in it.

---

## 2. The format decision

**One target format: upstream's `FineEnvs/SmolDataEnvs-sft`.** `messages` + `tools`, one tool named
`bash`, the answer submitted by `echo -n "<value>" > /workdir/answer.txt`, the non-thinking chat
template.

Two reasons:

1. **It is what makes the arms comparable.** Upstream's released model was trained in it, and the
   published ~0.28 → ~0.40 was measured in it. A different target format means "did our training
   beat theirs" is unanswerable, because the arms differ in format as well as in data — the exact
   confound `PLAN.md` names as the thing that decides whether A vs B means anything.
2. **It is what TRL consumes natively.** Upstream: *"there is no preprocessing step here and none is
   hiding in a helper: load, train, push."* The format that is easiest to train is the format that
   is comparable.

**The cost, stated plainly.** Our traces are a *different solver* (`run_shell` +
`write_solution`, `./input` paths, a persistent `solution.py`). Re-expressing them as bash
trajectories is a **translation**, not a passthrough. `train/traces.py` documents exactly what it
preserves, and the fallback path exists because it is the only one that can run today.

**One deliberate split.** `train/rungs.py` emits the **program** protocol (one fenced program, no
tools) rather than a bash trajectory: a rung is prompt-level information for a one-shot program,
and a bash trajectory would need invented exploration turns to carry it. Mixing rung rows and trace
rows in one export would produce a model that speaks a third protocol. So: rung data trains
`program`, trace data trains `bash`, and `sft_lora.py --protocol` says which.

---

## 3. The firewall

`FineEnvs/SmolDataEnvs-sft` has **no** `test`/`eval` rows by `task_id` (all 4,677 ids are in
SmolDataEnvs' `train`). The audit's leak is real but subtler, and it needs the `question` key:

| leak | rows | caught by |
|---|---|---|
| a held-out **task id** | 0 | `task_id` |
| the same **question text**, different task id | **4** | `question` |
| a different question over a held-out **table** | 110 | `bucket_prefix` (off by default) |

The 4: *"How many samples belong to each species in the dataset?"*, *"Which publisher has the
highest total global sales in the dataset?"*, and two correlation questions — all in the SFT set,
all in `test`. One leaked question is 0.4 points of pass@1 on 250 tasks.

**Default: `task_id,question`. Keeps 4,673 rows, drops 4. This is the owner's decision, taken
2026-10-01 and unchanged for now** — it replicates arm A, and the 4 dropped rows are the only
leak the default actually catches. §7 item 1 is the version of this that is still open.

**Why `bucket_prefix` is off by default.** It is the right idea at the wrong threshold:
**115 of the 170 held-out tables also appear in SmolDataEnvs' own `train` split.** Upstream
therefore treats a shared table as training material, not as held out. A table-level firewall
refuses **3,019 of 4,677 rows** — including every row upstream itself would have trained on — and an
arm-A replication at 1,658 rows is no longer arm A. (55 tables appear *only* in held-out splits, and
no SFT row sits on any of them.) For the **ladder** the coarse check is defensible: knowing a
table's shape is the signal being measured. So it stays available as `--heldout-columns
bucket_prefix,task_id,question` and is off unless asked for.

**The contamination is reported, not hidden.** Whatever the setting, the export prints the rows it
dropped **by column**, so a training run can always state how many held-out rows it refused and
which check refused them: `dropped {"question": 4}` is a fact about the run, not a footnote.

---

## 4. Data counts, right now

| source | available | format | notes |
|---|---|---|---|
| `smoldataenvs-sft` | **4,673** | bash | 4,439 train / 234 val after a 5% deterministic split |
| **`ja3_sft`** (§5) | **2,029** | bash | **real traces.** Arm B's data. 80 refused, 0 by the firewall |
| **`ja3_fallback`** (§5) | **682** | bash | single-turn contract rows from `solutions/jupyter-agent`. **Not traces** |
| `traces` | **1,258** (0 real / 1,258 fallback) | bash | 223 refused: 217 held-out tasks + 6 held-out questions. **Superseded for arm B by `ja3_sft`** |
| `rungs L1` | **590** | program | jupyter-agent only |
| `rungs L2` | **591** | program | |
| `rungs L3` | **591** | program | |
| `rungs L4` | **592** | program | |
| `rungs`, synthetic | **0** | program | **all 1,830 synthetic tasks sit on held-out tables** |
| `rungs`, train | **0** | program | `solutions/train` is empty; L2+ needs a reference |

Three zeros worth explaining:

- **synthetic: 0, by the firewall working.** Every one of the 1,830 synthetic tasks is built on a
  table that also appears in a held-out split (verified, not suspected). The coarse check removes
  the whole split. This is the firewall being correct, and it is why the count is reported rather
  than returning an empty set quietly.
- **train: 0, because no references exist.** `L2`–`L4` are gated on a verified reference solution
  and `data/solutions/train/` is empty. `gen_solutions --split train` fills it.
- **`traces`: 0 real, and that is now history.** `--source traces` reads the pre-2026-10-01 tree
  and still finds no conversation in it. `ja3_sft` is what the `ja3` sweep produced, and §5 is
  the dataset to use for arm B.

---

## 5. Arm B's dataset: `ja3_sft`

**Exported 2026-10-01 from the `ja3` transcript sweep** (run tag `ja3`, split
`jupyter-agent-v3`, rung L1, model `stealth/space-bunny-alpha`). This is the first training data in
the project that is *our own solver's* real conversations rather than another model's or a
contract stub, which is the whole difference between arm B and arm A.

```sh
uv run python -m train.export_ja3 --out data/train     # writes the two files and two manifests
uv run python -m train.export_ja3 --dry-run            # counts only, writes nothing
```

| | |
|---|---|
| **path** | `data/train/ja3_sft.jsonl` (20.9 MB, 2,029 rows) |
| manifest | `data/train/ja3_sft.manifest.json` |
| fallback | `data/train/ja3_fallback.jsonl` (682 rows) + `ja3_fallback.manifest.json` |
| format | upstream's `SmolDataEnvs-sft`: `messages` + `tools`, one tool named `bash`, and nothing else |
| source | 2,109 verified trials (reward 1.0, clean sealed verify) → 2,029 rows exported, **80 refused** |
| row shape | identical to upstream's: `{"messages": [...], "tools": [...]}` |

### The 80 refusals, by reason

| reason | rows | what it means |
|---|---|---|
| `transcript does not end with the graded answer` | 35 | truncated, or the model's last words are not the value the sealed pass graded |
| `gold answer in the prompt (answer-substring)` | 28 | `ladder.leaks` found the gold answer in the task's own question |
| `gold answer in the prompt (answer-numeric:…)` | 16 | the same, via the grader: a number in the question grades as the answer |
| `heldout:question` | 1 | a SmolDataEnvs test/eval **question** text |
| `heldout:kaggle_table` | **0** | no v3 task sits on a held-out table — which is the expected answer |

**The firewall dropped 0 by table and 1 by question.** One jupyter-agent question happens to match
a SmolDataEnvs held-out question verbatim, which the bare-name table rule cannot see; the question
key catches it. Everything else survived because v3 was built with `jtasks_v2.sde_overlap` on
`dataset_key` — the *bare* Kaggle name, so a mirror under another owner cannot slip through — and
the export re-derives the keys from the live split load rather than trusting the pool's own
`sde_overlap` tag. A tag is a claim; the split is the fact.

**983 rows (48%) have a gold answer under four characters**, below the floor `ladder.leaks` uses
for a *hint*, where a short answer matches by chance. That floor is not widened here, because the
text being checked is the task's own question — which every row of a dataset must contain — and
holding questions to a stricter rule than hints would make arm B's population depend on the
answer's length. They are counted in the manifest instead of silently dropped.

### The scrub, and what it caught

Every message, command and tool result is scrubbed before it is written, and the export **refuses
a row** rather than writing one that still contains `/home/`, `/var/tmp/`, the local username, a
hostname, or anything matching a key/token pattern. Rules: local absolute paths
(`/home/<user>`, `/Users/<user>`, `/var/tmp/...`, worktrees) and the Kaggle cache layout become
`./input/<file>` or `./input`; the username and hostname are removed.

Two things it found on the live tree, both of which a path-only rule would have shipped:

- **`ls -la ./input`** is the first command of essentially every transcript, and its output opens
  with three columns of permissions, **the local account name**, and a date. Replacing paths
  cannot fix that column; a bare-name rule is what removes it.
- **`/var/tmp/synprobe2/...`** — another session's scratch directory, present in 7 of the first
  320 verified trials. Not this project's path, and not something the pool knows about. The rule
  is now *any* `/var/tmp` path.

Verified on the written files: 0 occurrences of `/home/`, `/Users/`, `/var/tmp/` or
`kaggle/datasets`. The remaining hits for the username are inside words (`relevant`, `relevance`)
and inside a table's data (`Evans, Fred` in a baseball CSV), both of which are correct to keep.

### It is a translation, and the cost is stated

The sweep ran `run_shell` + `write_solution` against `./input` with a persistent `solution.py`;
upstream's protocol is one `bash` tool against `/home/user/input` submitting to
`/workdir/answer.txt`. Same contract, different spelling, so the trajectory is re-expressed:

- both tools become `bash`; the model's own command text is kept verbatim; call ids are rewritten
  so results stay attached to the calls that produced them; `cd . &&` is dropped (it is the
  sweep's scratch, not something the model said);
- `./input` is **kept**, not rewritten to `/home/user/input`: it is the spelling the model's own
  commands already use, so rewriting it would misattribute a convention to the translator;
- the **submission turn is appended**: the trajectory is cut at the model's final answer — which is
  checked to actually state the graded prediction, compared through the dataset's own grader so
  "141" and "141.0" are one answer — and `echo -n "<prediction>" > /workdir/answer.txt` is added
  after it. What the model learns is "compute it, state it, submit it"; what it does *not* learn
  is our re-formatting of the value. That is the whole of the invention, and it is one turn.

### Token statistics

Measured with the base model's own chat template — `Qwen/Qwen3.5-0.8B`, cached locally,
`enable_thinking=False` — because that is the text a `--max-length` has to be set against. The
manifest names the method; a chars/4 fallback names itself in the same field.

| | trace rows (`ja3_sft`) | contract rows (`ja3_fallback`) |
|---|---|---|
| turns, median | **7** | 4 (always 4) |
| turns, p90 | **11** | 4 |
| turns, min / max | 4 / 41 | 4 / 4 |
| tokens, median | **2,364** | 1,079 |
| tokens, p90 | **6,956** | 1,370 |
| tokens, min / max | 1,249 / 54,648 | 894 / 20,709 |
| **share over 8,192 tokens** | **7.93%** | 0.15% |

**op_family breakdown** (2,029 rows): count 823 · agg 400 · string 239 · filter 195 · stat_test
144 · other 67 · ml_fit 66 · argmax 48 · groupby 35 · lookup 12.

### How it compares to arm A

| | arm A (`smoldataenvs-sft`) | arm B (`ja3_sft`) |
|---|---|---|
| rows | 4,673 | **2,029 (43%)** |
| protocol | bash | bash |
| trajectories | another model's, on SmolDataEnvs' tasks | **our own solver's**, on jupyter-agent tasks |
| median turns / tokens | — | 7 / 2,364 |
| held-out leakage caught | 4 (question) | 1 (question), 0 (table) |

Arm B is **43% the size of arm A and different in kind**, which is the point of the experiment: the
same format, the same grader, the same held-out splits, and trajectories from a different solver on
a non-overlapping corpus. Two caveats that belong next to the number: 2,029 rows is one epoch of
about 250 optimizer steps at batch 1 × accum 8, which is small enough that seed variance matters
(PLAN asks for ≥2 seeds), and the 7.9% over 8,192 tokens means a `--max-length 8192` run silently
truncates ~160 rows — `--max-length 4096` would truncate ~25% and is the wrong default here.

### The fallback file, and why it is separate

`data/solutions/jupyter-agent` has 855 verified references and **zero** conversations, so all that
can come out of it is the contract trajectory: write the verified program, submit the graded
answer, stop. 173 more are not in the v3 pool at all. Those 682 rows are written to their own
file with a `note` in their manifest saying exactly what they are, because a dataset that mixed
them in with real traces without saying so would report something it is not.

---

## 6. Running it

### Export

```sh
# Arm A: upstream's trajectories, through the firewall
uv run --extra train python -m train.export_sft \
    --source smoldataenvs-sft --out data/train/sft_upstream
# -> kept 4673, dropped {"question": 4}, train 4439 / val 234

# The coarse, ladder-appropriate variant (drops 3,019 — read §3 first)
uv run --extra train python -m train.export_sft \
    --source smoldataenvs-sft --heldout-columns bucket_prefix,task_id,question \
    --out data/train/sft_upstream_strict

# Our traces, and a rung curriculum
uv run --extra train python -m train.export_sft --source traces --out data/train/sft_traces
uv run --extra train python -m train.export_sft --source rungs --rung L2 --out data/train/rungs_L2
uv run python -m train.rungs --counts      # availability, per rung, per source

# Arm B: the ja3 transcript sweep, real traces + the contract rows (see §5)
uv run python -m train.export_ja3 --out data/train
# -> ja3_sft.jsonl 2029 rows, ja3_fallback.jsonl 682, firewall drops 0 table / 1 question
```

### SFT — laptop (this box)

```sh
# Arm A
uv run --extra train python -m train.sft_lora \
    --data data/train/sft_upstream \
    --model Qwen/Qwen3.5-2B \
    --out runs/sft_a \
    --max-length 4096 --load-in-4bit        # QLoRA: 4-bit weights, ~1.2 GB

# Arm B. --max-length 8192, not 4096: 7.9% of the trace rows are longer than 8,192 tokens and the
# median is 2,364, so 4096 would truncate about a quarter of the set (§5).
uv run --extra train python -m train.sft_lora \
    --data data/train/ja3_sft.jsonl \
    --model Qwen/Qwen3.5-2B \
    --out runs/sft_b \
    --max-length 8192 --load-in-4bit
```

`--load-in-4bit` is **required** on 6 GB: the base model is 4.55 GB bf16 and leaves nothing for
activations at any useful context.

### SFT — Kaggle T4

```sh
kaggle kernels push -p train/kaggle        # after setting `id` in kernel-metadata.json
```

Secrets from Kaggle Secrets (`HF_TOKEN`, optional `HUB_MODEL_ID`). Internet on, T4 (never P100).
**No kernel was pushed.** See [`train/kaggle/sft_kaggle.ipynb`](../train/kaggle/sft_kaggle.ipynb).

### GRPO

```sh
uv run --extra train python -m train.grpo \
    --adapter runs/sft_a --rung-schedule ladder --rung-start L3 --steps 200
```

---

## 7. Measured, on this laptop

**Stack** (the `train` extra): torch 2.14.1+cu130, transformers 5.18.0, trl 1.14.1, peft 0.21.2,
accelerate 1.15.0, datasets 5.0.1, bitsandbytes 0.50.2. GPU: RTX 4050 Laptop, 6,141 MiB total,
**5,269 MiB free** at start, compute capability 8.9.

### The smoke run

```sh
uv run --extra train python -m train.sft_lora \
    --data data/train/sft_upstream --model Qwen/Qwen3.5-0.8B \
    --out /var/tmp/smol-ladder/smoke_sft \
    --max-steps 30 --max-length 2048 --load-in-4bit --logging-steps 2 --grad-accum 4
```

**Qwen/Qwen3.5-0.8B, not the 2B.** A 0.8B QLoRA is the only Qwen3.5 that fits a *smoke* run in ~5 GB.
It is the right proxy: identical `Qwen3_5ForConditionalGeneration` architecture and a
byte-identical `enable_thinking` chat template, so the code path exercised here is the one the 2B
will take. The 2B (4.55 GB bf16) leaves ~1 GB for activations at any useful context and belongs on
Kaggle.

| | |
|---|---|
| steps | 30 optimizer steps, 4,439 train / 234 val rows |
| **loss** | **2.472 → 1.166** (logged every 2 steps) |
| mean token accuracy | 0.561 → **0.729** |
| eval loss | 1.224 (step 25) → **1.207** (step 30) |
| grad norm | 22.5 → 1.94, no spikes, no NaN |
| train runtime | 1,307 s (~22 min); **0.023 steps/s** |
| **VRAM peak** | **2.9 GB** of 6.1 GB (`nvidia-smi`, incl. 4-bit weights + LoRA + 2048-token activations) |
| adapter reload for generation | 0.87 GB peak, one fence emitted a working `print(len(df))` |

Loss falls monotonically over 30 steps and eval loss tracks it, so the pipeline learns rather than
merely runs. 30 steps is 2.7% of one epoch; this proves the path, not a converged model.

**The non-thinking template is verified in the render**, not assumed: `apply_chat_template(...,
enable_thinking=False)` emits the empty `<think>\n\n</think>\n\n` block, and the trained adapter
loads on top of the 4-bit base and generates. Two transformers fallbacks were logged and are correct
on this machine (`causal_conv1d` and `flash-linear-attention` are not installed — neither is on
Kaggle either; neither is needed for correctness).

### Time estimates

Measured: **30 steps in 22 min ≈ 44 s/step** at 2048 tokens, 4-bit, grad-accum 4, on a
power-limited laptop 4050 (the log shows the eval passes inflating the per-step average; the pure
train steps were 26–35 s).

| run | laptop (6 GB) | Kaggle T4 (16 GB) |
|---|---|---|
| **smoke** (0.8B, 30 steps, 2048 tok) | **22 min measured** | ~8 min |
| **arm A, 0.8B** (4,439 rows, 1 epoch, 2048 tok) | **~11 h** | ~3 h |
| **arm A, 2B** (4,439 rows, 1 epoch, **8192** tok) | **does not fit** — bf16 weights alone are 4.55 GB, QLoRA activations at 8k do not fit in 5 GB | **~7–9 h**, one session |

**The 2B number is an estimate, not a measurement** — the only thing measured here is 0.8B at 2048
tokens, and the 2B run is 2.5x the parameters at 4x the context, with unfused linear-attention
kernels on both. A T4 is roughly 3–4x a power-limited 4050 on this workload (more FP32 throughput,
no power cap), and bf16 LoRA instead of 4-bit removes the quantization overhead. Treat ±2x, and
check the first 10 steps against the logged rate. One epoch at batch 1 x accum 8 on 4,439 rows is
~555 optimizer steps.

**The 12 h Kaggle session cap is the binding constraint, not throughput.** 4,439 rows at 8192
tokens is close to one session on one T4. `--resume` plus `hub_strategy="every_save"` is the
safety net; on a wiped disk the checkpoint has to come back *from the Hub*, so `HUB_MODEL_ID` must
be set on the first run or an interrupted run loses everything.

---

## 8. Decisions the owner must make

1. **Firewall strictness** (§3) — **decided 2026-10-01: keep the default**, `task_id,question`, which
   keeps **4,673** rows and replicates arm A. The table-level `bucket_prefix` check stays an **opt-in
   flag** and is not turned on for now. Adding it keeps 1,658 rows and is the right call *if* the
   ladder's L1-vs-L2 comparison becomes the primary result — knowing a table's shape is exactly what
   L2 hands over. Whichever is used, the export reports the drops by column, so the contamination is
   stated in the run record rather than left in a report nobody reads. Changing this changes the
   dataset and must never be done silently.

   **For arm B this decision is already made by construction and it is not the owner's to make
   twice.** `ja3_sft` is exported from the v3 pool, which excludes every SmolDataEnvs test/eval
   *table* on the bare dataset name; the export re-asserts that and reports 0 drops (§5). There is
   no arm-B variant of the `bucket_prefix` flag, because there is no arm-B row to apply it to.
2. **Base model.** Everything defaults to `Qwen/Qwen3.5-2B`, the exact `base_model_name_or_path` in
   both released models' configs. The smoke run used `Qwen/Qwen3.5-0.8B` (same architecture, same
   `enable_thinking` template) because 2B does not fit a LoRA smoke run on 6 GB. Real runs use the
   2B.
3. **Which SFT arm is "ours".** — **resolved 2026-10-01.** Arm A is upstream's 4.7K. Arm B is
   **`data/train/ja3_sft.jsonl`, 2,029 rows of our own solver's real trajectories** (§5), exported
   by `train/export_ja3.py` from the `ja3` transcript sweep. `--source traces` is superseded for
   arm B and must not be quoted as a trace dataset: it still reads the pre-2026-10-01 tree and finds
   no conversation in it. What is still the owner's call: whether to train arm B on `ja3_sft`
   alone (2,029 traces), on `ja3_fallback` alone (682 contract rows), or on the two concatenated
   (2,711, with the contract rows teaching only the submission contract and the traces teaching the
   exploration). §5 argues for the traces alone, and the fallback file exists so the alternative can
   be run without re-deriving anything.
4. **Where the arms run** — **leaning, not settled: the AMD Developer Cloud credit** (MI300X 192 GB,
   $1.99/h, $100 ≈ 50 h, expiring 30 days after applying) carries arms A–D, with **Kaggle as the
   fallback** and the place the first SFT runs. The credit is to be **applied only once SFT and GRPO
   can run back to back**: it is a 30-day clock, and spending it on an SFT with nowhere to put its RL
   arm wastes the window. So the order is Kaggle SFT → GRPO script proven end to end → apply the
   credit → arms A–D. On AMD: smoke-test vLLM + TRL in the first hour, and do **not** use QLoRA
   (bitsandbytes on ROCm is less mature, and memory is not the bottleneck on a 192 GB card).
5. **Rung curriculum format** (§2). Rungs emit the `program` protocol; traces emit `bash`. Whether
   to add a bash-protocol rung exporter, or keep the two protocols separate, is a design call.
6. **GRPO shaping weight.** Defaults to **0**, matching what upstream now ships. Their history:
   `+0.1` for "no traceback" was collected by empty programs; requiring *printed output* fixed that;
   weighting it then bought verbosity. Turning it on means owning that risk, and the run record
   stores the mean bonus collected so the run cannot be silently compared to one without it.
7. **Rung schedule parameters.** Default `start=L3, floor=L1, window=8, demote>0.75, promote<0.25`.
   The gap between the thresholds is deliberate — a task near 50% would otherwise flicker rung every
   step, and a rung that flickers is not a rung.
8. **Seeds.** PLAN asks for ≥2 seeds for A and B. Nothing here sets one beyond `--seed 42`; a
   second seed is a re-run with a different value and a different `--out`.