# Plan: post-train a 2B data-analysis agent, and use the ladder to measure it

> The priority is **data first, then a training run**. Evaluation of the *released* models
> (the old "post #1") is no longer the gate; the ladder is now an instrument for training and
> measurement rather than a deliverable of its own. Checkboxes work as a task list.

## Where we are

**What exists.** A complete eval harness: the four-rung ladder with a Blackwell-tested ordering
(`tests/test_blackwell.py`), the offline sealed grading pass, the SmolDataEnvs grader unmodified, a
solver that talks to any OpenAI-compatible endpoint, and the two upstream 2B protocols (`program`,
`bash`) alongside ours (`tools`) so every arm can be run under the protocol it was trained in
(`docs/LOCAL_MODELS.md`). Reference generation, plain-language hint generation and validation, the
tagged jupyter-agent pool, per-rung summarisation with a bootstrap CI, offline regrading, and
`--run-tag`/`RUN.json` so a sweep records the code, command line, model, protocol, rungs, samples,
climb setting and reference denominator at launch. 256 tests pass.

**Measured, on `test` (250 tasks, `stealth/space-bunny-alpha`):** L1 187/250 = 74.8%; schema control
23/63 = 36.5% on L1 failures; L2 167/181 = 92.3%; L3 8/14 = 57.1%; L4 6/6 = 100%. Reference funnel:
181/250 (72%) have a verified reference. First-passing rung: L1 187, L2 17, L3 3, L4 3, and 40 tasks
with no reference. 20 of the 23 information-rung rescues are also control rescues, so on this split a
hint rung and a no-information prompt are close to interchangeable. `test` L1 by family: `agg` 68.0%,
`stat_test` 78.1%, label-typed answers 66.3% vs numeric 77.2%, hard tier 61.6%.

**Withdrawn — the 92% schema-control claim is dead.** The numbers audit recomputed every figure in
the old headline from raw per-trial results and withdrew the `test` table in full: no artefact of the
92% run survives, its denominator (179) is impossible against its own L1 count (184), and it predates
a verifier-mount fix. The surviving control is 23/63 = 36.5%, which does **not** support "the failures
are overwhelmingly an exploration problem". Withdrawn with it: `L1 184/250`, `L2 9/14`, `L3 2/5`,
`L4 3/3`, and the "of 19 climbed tasks" summary.

**Found wrong and fixed on this branch.** The schema control leaked the gold answer on 30/250 tasks
(it printed sample rows; it is now built from dtypes and column profiles that emit no value), and it
was *vacuous* on 243/250 — `schema_dump` passed `usecols=range(40)` and a bare `except` swallowed the
error, so the median dump was 37 characters and 191/250 named no column; median is now 1404 chars and
250/250 name a column. `Method: join` was emitted for `os.path.join` on 56/181 references.
`summarize.py` double-counted, summing 266 for 250 tasks. The `ANSWER:` prefix cost 8 of 490 stored
predictions; the fix went into the prompt, because the grader is not the defect and was left matching
upstream. L2/L3 are now written in plain language from the verified reference instead of extracted
from its AST (98/181 references are hand-written csv/sqlite, so the AST gave empty filters on 71% and
a method line reading "get, items, values"): 167/181 hints validate, 14 fall back to the AST,
non-empty content 62.4% → 88.4% (columns), 28.2% → 56.9% (filters), 66.9% → 92.3% (method).

**Still broken or open.** The `synthetic` split's gold answers are wrong: `iter_tables` reads 50,000
rows to compute the answer but the agent reads the untruncated file, so 58 of 275 tables over 50k
rows are ungradeable by construction — the 87 "never" there is a grading artefact, not a result.
jupyter-agent has 274 attempted references and 23 passing, so the ladder cannot yet run on the source
that needs it most. 8/181 tasks pass at L1 and fail at L2, a real monotonicity violation. 18.9% of
tasks flip across four identical L1 runs, so "first passing rung" is not identifiable at k=1. No
instruction stack is installed on this machine.

## Objective

Get to a first SFT and then an RL training run of the 2B model as fast as the data allows, and use
the information ladder three ways:

1. **Verify tasks** before they enter training — a reference solution that reproduces the gold answer
   is what makes a task's gold trustworthy, and it is what L2/L3 are written from.
2. **Produce hint rungs L2/L3 as a curriculum for RL.** At the ~28% pass rate of the released GRPO
   eval, many GRPO groups score all-zero and give no gradient. Train the hard tasks *with* a hint rung
   and withdraw it as per-task pass rate rises (the adaptive curriculum the ladder post suggests).
3. **Measure skill versus information before and after training** — does training lower the rung
   needed, or close the L1 vs L1+schema gap, or both?

Every outcome is reportable. "Hints do not help RL" is a result, not a failure.

## Data assets

| Asset | Count | Status | For | Blocked by |
|---|---|---|---|---|
| SmolDataEnvs `train` | 5,000 tasks | on disk, graded by `grader.py` | RL tasks + reward | nothing for RL; SFT needs traces we do not have |
| SmolDataEnvs-sft | 4,677 verified trajectories | **usable for SFT today** | SFT traces | nothing — but it is `bash`-protocol; conversion needed if arms must share one tool format |
| SmolDataEnvs `test` / `eval` | 250 / 144 | held out, never trained on | in-distribution eval | nothing; 69/250 `test` tasks have no reference, so L2–L4 are measurable on 181 only |
| jupyter-agent pool (v2) | 9,187 tasks, **5,124 ladder-grade** | built and tagged; references being generated | RL tasks + reward; eval (L1 41.9%, twice the headroom) | verified references (23 passing so far); 519 tasks across 148 uncached datasets need ~69 GB |
| Plain-language hints (L2/L3) | 167/181 `test` refs validate | written, validated, cached | RL curriculum; measurement | 14 fall back to the AST, and 9 of those are label-answers that *are* column names — unfixable; needs generating on jupyter-agent refs |
| Synthetic tasks | 275 | **gold broken, repair outstanding** | RL tasks (unlimited supply) | the `nrows=50,000` truncation is still in `synthetic.py`: regenerate all gold and re-verify every spec against the shipped table before any trial |

The `jtasks_v2` ladder-grade subset deliberately keeps the method-ambiguous families (`ml_fit`,
`stat_test`, `groupby`, `lookup`, `join` — 1,055 tasks) and does not tune towards `count`/`agg`,
because those are the families the ladder has nothing to disambiguate. 8,668 of 9,187 pool tasks
(94.3%) reuse datasets already in the Kaggle cache.

**Contamination, measured.** 74% of `test` tasks share a source table with a `train` task (185/250 by
`bucket_prefix`), and 4 `test` questions appear verbatim in the released SmolDataEnvs-sft blob. The
split is by question, not by table, so an arm trained on SmolDataEnvs-sft has seen the tables L4 is
written about. The clean version partitions the 471 Kaggle datasets, not the 5,394 tasks.

## Critical path to the first training run

**Ordered; steps marked [non-blocking] are not on the path to the first training run.**

| # | step | depends on |
|---|---|---|
| 1 | **SFT (LoRA) on SmolDataEnvs-sft** — the first training run | a serving/training stack, nothing else. The 4,677 trajectories are `bash`-protocol (`messages` + `tools`, answer in `/workdir/answer.txt`, 3–12 turns; upstream's config used `max_length=8192`) |
| 2 | Converter: the two trace formats → one tool format + chat template | (1), which defines the format. Without it, arm-vs-arm differences are format, not data |
| 3 | SFT on our own verified traces | (2) for the format, plus our own trace collection — **and our trace collection is [non-blocking] for step 1** |
| 4 | GRPO (LoRA) on SmolDataEnvs `train` | (1) as its starting point, and a non-degenerate reward. `num_generations=8` at the ~28% pass rate is the problem this project exists to solve |
| 5 | Hint-curriculum GRPO | verified references + validated L2/L3 hints for the *training* tasks. Hints exist for `test` refs today, not for `train` |
| 6 | Ladder measurement of every arm, before and after | each arm existing. Cheap relative to training, so it can run on the baseline as soon as (1) lands |

**Reference generation** is needed for 5 and 6 but **[non-blocking]** for the first training run. Run it
early anyway: it is the long pole for the ladder and it is what validates a task's gold.

- [ ] 1. SFT (LoRA) on SmolDataEnvs-sft, 4,677 traces — first training run
- [ ] 2. Converter: upstream `bash` traces and our traces → one tool format + chat template
- [ ] 3. Reference solutions for the training split (`gen_refs` / `gen_solutions`), verified offline
- [ ] 4. Plain-language L2/L3 hints for the training split (`gen_hints`), validated and cached
- [ ] 5. Synthetic gold repair: drop the 50k truncation, regenerate, re-verify every spec
- [ ] 6. Our own verified SFT traces in the converted format
- [ ] 7. GRPO (LoRA) on SmolDataEnvs `train` from the step-1 model
- [ ] 8. Hint-curriculum GRPO: hints on at low pass rate, withdrawn as per-task pass rate rises
- [ ] 9. jupyter-agent references at scale (the 5,124 ladder-grade subset)
- [ ] 10. Ladder measurement of every arm, before and after, same tasks and protocol
- [ ] 11. Open decisions below, then the write-up

## Training arms

| Arm | Model | Training | Notes |
|---|---|---|---|
| 0 | Qwen3.5-2B | none | the control; base model, no instruction tuning for either protocol |
| R-SFT | `smoldataenvs-sft-2b-v0` | theirs | a **93 MB LoRA adapter** (r=16, α=32) on the base, not a model |
| R-GRPO | `smoldataenvs-grpo-2b-v0` | theirs | shipped in **fp32, 8.85 GB**; must be cast to bf16 (4.43 GB) |
| A | Qwen3.5-2B | SFT (LoRA) on SmolDataEnvs-sft, ~4.7K traces | the baseline we can run today |
| B | Qwen3.5-2B | SFT on our verified traces | depends on our trace collection |
| C | best of A/B | + GRPO (LoRA) on SmolDataEnvs `train` | |
| D | C | + hint-curriculum GRPO (L2/L3 withdrawn as pass rate rises) | **promoted from stretch goal**; the ~28% pass rate makes all-zero groups the binding constraint |

Controls that still decide whether the numbers mean anything:

- **Same chat template and tool format across A and B**, or A vs B measures format. Render
  non-thinking everywhere (`enable_thinking=False`): upstream trains and scores that way, and
  omitting it changes the rendered template and is itself a measurement.
- **Each arm runs under the protocol it was trained in.** A 2B model trained on "one program, then
  stop" has never produced a multi-turn tool transcript; run it in a 40-turn loop and you measure
  format transfer, not the model. Cross-arm ladder comparison needs all arms on one protocol, or the
  asymmetry stated in the table.
- **Held-out discipline.** No `test` or `eval` task, and no reference solution, ever enters training
  data. References come only from held-out splits, and the ladder's rungs nest within a task, so any
  split keeps all of a task's rungs on one side.
- **Seeds and CIs.** ≥2 seeds for A and B if hours allow; 95% CIs throughout. On ~250–450 tasks,
  differences under ~5 points are noise. Report the run-to-run spread next to every headline: four
  identical L1 runs spanned 71.5–74.8%.
- **Reward integrity.** SmolDataEnvs' "+0.1 if the code runs" shaping reward got hacked by empty
  programs. Inspect samples every N steps and require printed output.

## Evaluation protocol

- **The ladder** (`docs/LADDER.md`): L1 < L2 < L3 < L4, cumulative, each rung Blackwell-garbling the
  one below. Metric is the lowest passing rung **and** the per-rung pass-rate curve, from **k samples
  per rung, no climb** — every rung on every task, so each pass rate shares a denominator and the
  monotonicity test is well defined. `--no-climb` and `--samples` both exist; climbing biases the
  lowest-passing-rung histogram toward whichever rung a task happened to be reached at.
- **Run tags**: every sweep gets `--run-tag TAG` → `data/runs/TAG/<split>/`, plus a `RUN.json`
  written at launch. Cached results are never silently inherited across ladder versions.
- **In-distribution first**: SmolDataEnvs `test` (250) and `eval` (144), with the reference funnel
  reported alongside every rung table — L2–L4 exist only where a verified reference was built
  (181/250), so every rung number is conditional on that.
- **The schema control** (L1+schema) is not a rung: it adds no information, so it isolates
  processing/skill gain. Run it on L1 failures.
- **OOD later**: DABstep, after the in-distribution protocol is stable. No public gold code there,
  so only L1 and the control.
- Reference solutions are generated by the same model family under evaluation, so "training lowered
  the rung" cannot be separated from recognising its own teacher's code without a shuffled-reference
  control.

## Compute

| Resource | Amount | Use for |
|---|---|---|
| Laptop RTX 4050 Laptop GPU | 6 GB VRAM (5.3 GB free), 94 GB RAM, **no inference stack installed** | pipeline dev, grader tests, harness sanity checks on 5–20 tasks. Base bf16 weights are 4.55 GB, leaving ~1.0 GB for KV at `--gpu-memory-utilization 0.92`; `--max-model-len 4096`, or 8192 if it fits |
| Kaggle | 2× T4 16 GB, 30 h/week | all real training, and every multi-arm × five-rung sweep |
| AMD Developer Cloud (MI300X 192 GB, $1.99/h) | $100 ≈ 50 h, **expires 30 days after applying** | GRPO arms only. Whether the credit has been applied is **unknown** |

Corrections from `docs/LOCAL_MODELS.md`: the SFT release is a **LoRA adapter** (93 MB, r=16 on
`all-linear` — it must be served with the base model and the adapter loaded), and the GRPO release is
**fp32 at 8.85 GB**, which does not fit this card in any precision it runs well — cast to bf16
(4.43 GB) first. The released models only evaluate meaningfully under each one's own protocol, and
that doc's throughput figures are bandwidth arithmetic rather than measurements (±2×).

Gotchas: Kaggle needs a **T4** (vLLM does not support P100), fp16 only (watch for NaN loss spikes with
Qwen — LoRA and a lower LR), no FlashAttention 2, a 12 h session cap, and the disk is wiped between
sessions, so checkpoint to the Hub and write results to `data/runs/`. Internet must be enabled (phone
verification). On AMD, smoke-test vLLM + TRL in the first hour, do not use QLoRA (bitsandbytes on ROCm
is less mature and memory is not the bottleneck), and do not apply the credit until the GRPO script
already runs on laptop/Kaggle. Use WSL2 if on Windows.

Secrets, from `.env` (git-ignored, never committed): `HF_TOKEN`, `KAGGLE_USERNAME`/`KAGGLE_KEY` (or
`~/.kaggle/kaggle.json`), `OPENROUTER_API_KEY`, and optionally `WANDB_API_KEY`. On Kaggle add them as
notebook **Secrets**, never inline.

## Open decisions for the owner

1. Which protocol is the project protocol — upstream's one-turn `program`, the `bash` agent, or our
   `tools` loop? This decides the converter, the held-out discipline and every cross-arm table.
2. Does the first run go on the laptop or straight to Kaggle? The laptop has no inference stack and
   6 GB; a 2B LoRA SFT is small enough that Kaggle is likely the cheaper path to a first run.
3. Is the AMD credit still valid, and do we spend it on arm C or hold it?
4. Do we hold the 74%-table-overlap contamination and report it, or partition the 471 Kaggle datasets
   for a clean held-out set (which shrinks `test`)?
5. Synthetic: repair the gold and keep it as unlimited RL task supply, or drop the split?
6. Do we do our own trace collection at all, or is arm B's question ("is our data better than
   SmolDataEnvs-sft?") answered by a smaller, higher-precision set built only from verified traces?

## Risks

| Risk | Mitigation |
|---|---|
| All-zero GRPO groups at the ~28% pass rate give no gradient | hint-curriculum arm D: train with L2/L3, withdraw as pass rate rises |
| Reward hacking on the shaping reward | require printed output; inspect samples every N steps |
| Effects within noise | CIs, ≥2 seeds, whole splits, rerun spread reported beside every headline |
| A vs B confounded by format or chat template | one converter, one template, non-thinking everywhere |
| Protocol mismatch between arms | each arm run under its training protocol; asymmetry stated |
| Contamination through shared tables (74% of `test`) | no `test`/`eval` tables or references in training; dataset-level partition if the owner prefers |
| Ladder hints leak the answer | already enforced (text, differential, run-and-grade oracle) and a hint that fails is regenerated or falls back |
| Ladder hints are useless for RL | that is the measurement; arm D is designed to be able to return "no" |
| Hint cost degrades the rung (L3 below L2) | already observed; report monotonicity violations as results, with k samples and no climb |
| Training on broken gold | synthetic's 50k truncation is the known case; the rule is that a task's gold must be re-derivable from the table the agent sees |
| fp16 instability on T4 | LoRA, lower LR, watch the first few hundred steps |
| Ladder eval cost | no-climb grid only where needed; run L1 across all arms first, spend rungs second |

## Background (facts that still hold)

### SmolDataEnvs ([dataset](https://huggingface.co/datasets/FineEnvs/SmolDataEnvs), [code](https://github.com/adithya-s-k/FineEnvs/tree/main/04-smoldataenvs))
- 5,394 verified tasks: train 5,000 / test 250 / eval 144, from 471 Kaggle datasets. Held-out splits
  are deliberately harder (~38–40% hard vs 14% in train). A row is question, gold answer,
  `reward_mode` + `atol`/`rtol`, a pointer into HF bucket `AdithyaSK/jupyter-agent-kaggle-all` (use
  `bucket_prefix`), and the full agent `instruction`.
- `grader.py`: exact → numeric with tolerance → list → math-verify. **No LLM in the reward path.**
- "Verified" means the grader accepted an answer, not that this is the unique correct computation; on
  a loose `rtol` a materially different method also grades 1.0. Pass rates were never published.
- `-sft`: 4,677 verified trajectories, TRL-ready `messages` + `tools`.
- Released: [`smoldataenvs-sft-2b-v0`](https://huggingface.co/AdithyaSK/smoldataenvs-sft-2b-v0)
  (LoRA adapter), [`smoldataenvs-grpo-2b-v0`](https://huggingface.co/AdithyaSK/smoldataenvs-grpo-2b-v0)
  (full weights, fp32). Their published result, read off `curves.gif`: pass@1 ~0.28 → 0.40, pass@4 →
  0.60 after 1,119 GRPO steps. Whether GRPO started from SFT is **unknown**.
- Their lessons: the "+0.1 if the program runs" shaping reward was hacked by empty programs (fixed by
  requiring the program to print something); run everything non-thinking with one shared template;
  reuse one sandbox per process rather than one per rollout.

### jupyter-agent ([dataset](https://huggingface.co/datasets/jupyter-agent/jupyter-agent-dataset))
51,389 rows across 103 shards, duplicated as `thinking`/`non_thinking`, ~72 GB mostly
`original_notebook` — load only the columns you need. `executor_type = llm` rows have simulated
outputs, so their answers are fiction (66% of rows); some `kaggle_dataset_name` values are wrong;
answers are free text, so a gradable-answer filter is required before SmolDataEnvs' grader applies.

### Information ladder ([Evan Kim, "Difficulty is an information gap"](https://evan-kim2028.github.io/evan_writings/writings/difficulty-is-an-information-gap/))
- Thesis: difficulty is a relation between a task, a model and an *amount of information*.
- Method: nested prompts, each containing everything below; measure the lowest rung where the model
  first passes. Findings (242 Go tasks): extra information replaces search (24–32% fewer read calls);
  the two agents disagreed on what was hard (56% agreement, 0.29 rank correlation); 59% of graded
  tasks failed at L1 and became solvable with more information.
- Caveat: small study, SWE tasks. We borrow the method, not the findings. Blackwell's theorem
  compares information *sets* and says nothing about cost, which is why a higher rung can genuinely
  lower the pass rate for a real agent with a turn budget.

### Benchmarks
| Benchmark | Role | Notes |
|---|---|---|
| SmolDataEnvs `test` (250) | in-distribution | 74.8% L1 for a strong agent; the ladder's funnel is 181/250 |
| [DABstep](https://huggingface.co/spaces/adyen/DABstep) (450) | OOD, later | answers hidden; no public gold code, so L1 and the control only |
| [AgenticDataBench](https://arxiv.org/html/2607.01647) (344) | stretch OOD | heavy (~490 MB, 6.4 files/task); a 2B model will likely score near 0 |
| [DataAgentBench](https://github.com/ucbepic/DataAgentBench) | skip | top 94.7%, essentially beaten |

## Links
- Ladder definition and the Blackwell argument: [`LADDER.md`](LADDER.md) · serving the released
  models: [`LOCAL_MODELS.md`](LOCAL_MODELS.md) · read-only audits and the superseded-findings list:
  [`audits/`](audits/README.md)
- SmolDataEnvs: https://huggingface.co/datasets/FineEnvs/SmolDataEnvs · code: https://github.com/adithya-s-k/FineEnvs/tree/main/04-smoldataenvs
- SmolDataEnvs-sft: https://huggingface.co/datasets/FineEnvs/SmolDataEnvs-sft
- jupyter-agent: https://huggingface.co/datasets/jupyter-agent/jupyter-agent-dataset · blog: https://huggingface.co/blog/jupyter-agent-2
- DABstep: https://huggingface.co/spaces/adyen/DABstep · AgenticDataBench: https://arxiv.org/html/2607.01647
- Information ladder post: https://evan-kim2028.github.io/evan_writings/writings/difficulty-is-an-information-gap/ · AMD credit notes: https://lilting.ch/en/articles/amd-developer-cloud-credit-journey