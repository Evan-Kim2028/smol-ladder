# Smol Data Transfer: does verified data-agent training generalise?

> Umbrella plan. Paste it into a GitHub issue as-is, or keep it as `PLAN.md` in the repo.
> The checkboxes work as a task list in both.

## TL;DR

Small (2B) models can be trained with RL to answer data-analysis questions by writing and running code.
[SmolDataEnvs](https://huggingface.co/datasets/FineEnvs/SmolDataEnvs) (released 2026-09-24) reports a
2B model going from ~0.28 to ~0.40 pass@1 after GRPO. **That number is only measured on held-out tasks
from the same source**, so nobody has checked whether it transfers. This project checks, then asks
whether small clean data or big noisy data is the better starting point.

- **Post #1 (fast, ~1–2 weeks):** evaluate the *released* models on an in-distribution and an
  out-of-distribution benchmark. No training needed.
- **Post #2 (~3–4 more weeks):** train our own arms: clean-5K vs noisy-51K SFT, plus GRPO, and do
  error analysis.

Goal: learning + a portfolio piece + a blog post people cite. Being first to evaluate SmolDataEnvs
independently is the hook, so move fast on post #1.

---

## Research questions (commit to these before running anything)

- **RQ1: Transfer.** Do SmolDataEnvs-trained models improve on an out-of-distribution data-analysis
  benchmark (DABstep), or only on in-distribution tasks (SmolDataEnvs `test`)?
- **RQ2: Quality vs quantity.** For SFT, which is better: ~5K verified traces (SmolDataEnvs-sft) or
  ~51K noisy traces (jupyter-agent, real-execution subset)? Measured in- and out-of-distribution.
- **RQ3: RL on top.** Does GRPO after the better SFT add anything, and does *that* gain transfer?

- **RQ4 — Skill or information? (via an information ladder, see below)** When a model fails, is it
  missing *skill* or missing *information*? Does training (SFT / GRPO) lower the amount of information a
  model needs to pass?

Primary metric: pass@1 (greedy, one attempt per task) with 95% CIs.
Secondary: pass@4, and per-difficulty-tier breakdown.
Every outcome is reportable. "It doesn't transfer" is a result, not a failure.

---

## Background (facts gathered so far)

### SmolDataEnvs ([dataset](https://huggingface.co/datasets/FineEnvs/SmolDataEnvs), [code](https://github.com/adithya-s-k/FineEnvs/tree/main/04-smoldataenvs))
- 5,394 verified tasks: train 5,000 / test 250 / eval 144. Built from 471 Kaggle datasets.
- Held-out splits are deliberately harder (~38–40% hard vs 14% in train).
- Row = question, gold answer, `reward_mode` + `atol`/`rtol`, pointer to data in HF bucket
  `AdithyaSK/jupyter-agent-kaggle-all` (use `bucket_prefix`), full agent `instruction`.
- `grader.py`: exact → numeric w/ tolerance (+ percent/fraction bridge) → list → math-verify.
  No LLM in the reward path. `--json` mode also reports `tool_efficiency` (1 − calls/15).
- Tasks were "verified": strong agent models had to reproduce the gold answer in a sandbox.
  Pass rate / attempt counts are **not published**.
- `-sft`: 4,677 verified trajectories (TRL-ready `messages` + `tools`).
- Released models (Qwen3.5-2B base, **no eval numbers in the model cards**):
  - [`AdithyaSK/smoldataenvs-sft-2b-v0`](https://huggingface.co/AdithyaSK/smoldataenvs-sft-2b-v0)
  - [`AdithyaSK/smoldataenvs-grpo-2b-v0`](https://huggingface.co/AdithyaSK/smoldataenvs-grpo-2b-v0)
- Their published result (read off `curves.gif`, `eval` split, 144 tasks): pass@1 ~0.28 → 0.41 / 0.40
  (shuffled / curriculum), pass@4 → 0.60 after 1,119 GRPO steps. Unclear whether GRPO started from SFT.
- Their eval script: `scripts/eval_pass1.py` (greedy, `SPLIT=test` for the benchmark split).
- Lessons from their README:
  - The "+0.1 if the program runs" shaping reward got hacked: empty programs collected the bonus.
    Fixed by requiring the program to *print* something.
  - Run everything non-thinking (same chat template for SFT, RL and eval).
  - Reuse one sandbox per process; don't start one per rollout.

### jupyter-agent ([dataset](https://huggingface.co/datasets/jupyter-agent/jupyter-agent-dataset))
- 51,389 examples, duplicated as `thinking` / `non_thinking` splits (same examples, different formatting).
- ~72 GB download, mostly the `original_notebook` column (0.2–5.6 MB per row). Load only the columns you need.
- Columns: `id`, `messages`, `tools`, `question`, `answer`, `edu_score`, `files_used`, `packages_used`,
  `kaggle_dataset_name`, `executor_type`, `original_notebook`.
- Traces use tools `add_and_execute_jupyter_code_cell` + `final_answer` (a different format from SmolDataEnvs).
- Known noise:
  - `executor_type = llm` means the outputs were simulated by an LLM, not executed.
  - Some `kaggle_dataset_name` values are wrong. Seen: an SVM-on-`vehicle.csv` question labelled `ukveteran/retail-food`.
  - Answers are free text (e.g. "61,257 USD (70,187 for failed minus 8,930 for successful)").
- HF's own result: Qwen3-4B fine-tuned on it gained up to 20% on DABstep-easy.

### Information ladder ([Evan Kim, "Difficulty is an information gap"](https://evan-kim2028.github.io/evan_writings/writings/difficulty-is-an-information-gap/))
- Thesis: difficulty is a relation between a task, a model and an *amount of information*, not a fixed property of the task.
- Method: a ladder of prompts, each rung containing everything below it (bug report → full description →
  test names → signatures → hidden tests). Measure the **lowest rung where each model first passes**.
- Findings (242 certified Go coding tasks, 2 agents, ~$1.4k of tokens):
  - Extra information replaces search: 24–32% fewer read/search calls when it turned a fail into a pass.
  - The two models disagree on what's hard: 56% agreement, 0.29 rank correlation.
  - 59% of graded tasks fail at L1 and become solvable with more information.
- Suggested uses: adaptive curricula (withhold information as the model improves), multiple prompt variants per task.
- Caveat: small study, SWE tasks, not data analysis. We borrow the *method*, not the findings.

### Benchmarks
| Benchmark | Role | Notes |
|---|---|---|
| SmolDataEnvs `test` (250) | in-distribution | No published score yet |
| [DABstep](https://huggingface.co/spaces/adyen/DABstep) (450) | **out-of-distribution** | Top of leaderboard saturated by benchmark-specific agents, still discriminative for small models. Answers hidden: submit to the leaderboard; a small dev split has public answers (verify details) |
| [AgenticDataBench](https://arxiv.org/html/2607.01647) (344) | stretch OOD | Best 48.8%. Heavy tasks (~490 MB, 6.4 files avg), so a 2B model will likely score near 0 |
| [DataAgentBench](https://github.com/ucbepic/DataAgentBench) | skip | Top 94.7%, essentially beaten |

---

## Experiment design

| Arm | Model | Training | Used in |
|---|---|---|---|
| 0 | Qwen3.5-2B | none | post #1, #2 |
| R-SFT | `smoldataenvs-sft-2b-v0` | theirs | post #1 |
| R-GRPO | `smoldataenvs-grpo-2b-v0` | theirs | post #1 |
| A | Qwen3.5-2B | SFT on SmolDataEnvs-sft (~4.7K) | post #2 |
| B | Qwen3.5-2B | SFT on jupyter-agent real-execution subset (all of it) | post #2 |
| C | best of A/B | + GRPO (LoRA) on SmolDataEnvs train | post #2 |

Controls that decide whether results mean anything:
- **Same format** for A and B. Convert both trace sets into one tool/prompt format and the same chat
  template, or A vs B measures format rather than data.
- Same hyperparameters, sequence length and template across arms. Log token counts per arm.
- ≥2 seeds for A and B if budget allows. Report 95% CIs (on ~250–450 tasks, <~5 pt differences are noise).
- **Leakage check:** drop jupyter-agent rows whose notebook/dataset overlaps SmolDataEnvs `test`/`eval`
  tasks (match on `source_row_id` notebook id and `kaggle_dataset`). Also check DABstep overlap with training data.

### Information ladder for data-analysis tasks (RQ4)

| Rung | Model gets | Source of the extra info |
|---|---|---|
| L1 | question + file names (the normal prompt) | as-is |
| L2 | + schema / `df.head()` of the relevant files | generated automatically from the data |
| L3 | + which files/columns/filters matter | extracted from verified solution code |
| L4 | + method hint (e.g. "Pearson correlation after dropping nulls") | extracted from verified solution code |

- Metric per task and model: **lowest passing rung** (or "never"). Compare distributions across arms:
  "GRPO lowered the rung needed on X% of tasks" is more informative than a single pass@1.
- Full ladder only on SmolDataEnvs `test`: the SmolDataEnvs-sft traces contain verified code to derive L3/L4 from.
  DABstep has no public gold code, so only L1–L2 there, plus DABstep's own docs (e.g. the manual) as an optional rung.
- Hints must not leak the answer. Check with a script that the final value doesn't appear in any hint.
- Cost: each rung is a full eval pass (~3–4× eval time). To save budget, only run L2+ on tasks failed at the rung below.

---

## Compute plan

| Resource | Amount | Use for |
|---|---|---|
| Laptop RTX 4050 (6 GB) | free | pipeline dev, grader tests, tiny GRPO smoke test (sub-1B, QLoRA via Unsloth), few-task evals |
| Kaggle (2× T4 16 GB) | 30 h/week free | all evals, SFT arms A and B |
| AMD Developer Cloud (MI300X 192 GB, $1.99/h) | $100 ≈ 50 h, **expires 30 days after applying** | GRPO arm C only |

Gotchas:
- **Kaggle:**
  - Use T4 (vLLM doesn't support P100).
  - fp16 only: watch for NaNs/loss spikes with Qwen.
  - No FlashAttention 2.
  - 12 h session cap and the disk is wiped: push checkpoints to the Hub regularly.
  - Enable internet (needs phone verification).
  - Split evals across both T4s.
- **AMD:**
  - ROCm: smoke-test vLLM + TRL in the first hour.
  - Don't use QLoRA (bitsandbytes on ROCm is less mature and memory isn't the bottleneck).
  - Only apply the credit once the GRPO script already works on laptop/Kaggle.
- **Sandbox:** the upstream scripts run code in HF Jobs sandboxes, which bill your HF account. Replace
  them with a local sandbox: a subprocess with a timeout + a working dir with the task's files, or
  Docker where available.
- **Laptop:** use WSL2 if on Windows.

### Environment variables / secrets
| Var | Needed for |
|---|---|
| `HF_TOKEN` | downloading bucket data, pushing models/datasets, DABstep submission |
| `KAGGLE_USERNAME`, `KAGGLE_KEY` (or `~/.kaggle/kaggle.json`) | `kagglehub` downloads for jupyter-agent source data |
| `WANDB_API_KEY` or trackio (optional) | experiment tracking |

On Kaggle, add these as notebook **Secrets** (Add-ons → Secrets), never inline.

---

## Milestones

### M1: Eval harness (laptop → Kaggle)
- [ ] Repo scaffold (`uv` or `pip` + `requirements.txt`), config via env vars
- [ ] Download a SmolDataEnvs task's files from the HF bucket (see the dataset card snippet)
- [ ] Local sandbox runner: run model code with a timeout in a working dir containing the task's files, capture stdout/answer file
- [ ] Wire in upstream `grader.py`; unit test: gold answer → 1.0, wrong → 0.0
- [ ] Agent loop: vLLM generation → tool call → sandbox → … → answer (cap turns/tool calls)
- [ ] Run on 5 tasks on the laptop, then the full `eval` split on Kaggle
- [ ] Sanity check: reproduce their chart numbers roughly with `smoldataenvs-grpo-2b-v0` on `eval` (~0.40)
- [ ] DABstep adapter: load tasks + context files, same agent loop, write the submission file; validate on the dev split

### M2: Post #1, "Does it transfer?"
- [ ] Evaluate arms 0, R-SFT, R-GRPO on SmolDataEnvs `test` and DABstep (submit to leaderboard)
- [ ] CIs, per-tier breakdown, 20–30 failures read by hand
- [ ] Write the post: question, setup, table + one chart, limitations, cost
- [ ] Release the harness on GitHub, results on the Hub
- [ ] Post a friendly note in the SmolDataEnvs Community tab with the results

### M3: Data prep
- [ ] Load jupyter-agent `non_thinking` **without** `original_notebook`; filter `executor_type == "e2b"`
- [ ] Deduplicate; leakage filter vs SmolDataEnvs `test`/`eval`
- [ ] Converter: both trace formats → one common tool format + chat template
- [ ] Data card: row counts before/after each filter, token counts per arm
- [ ] (Optional) release the cleaned real-execution subset as a dataset

### M4: SFT arms (Kaggle)
- [ ] Arm A: SFT on SmolDataEnvs-sft
- [ ] Arm B: SFT on the cleaned jupyter-agent subset (multi-session, resume from Hub checkpoints)
- [ ] Second seed for A and B if hours allow
- [ ] Evaluate both with the M1 harness

### M5: GRPO (laptop smoke test → AMD)
- [ ] Reward: correctness from `grader.py` + small shaping ("program printed something"); watch for hacking
- [ ] Laptop/Kaggle: 20–50 step smoke run on a tiny model; confirm non-zero, varied rewards
- [ ] AMD: ROCm smoke test, then GRPO (LoRA) from the better SFT arm; checkpoint to Hub
- [ ] Evaluate arm C on both benchmarks

### M6: Error analysis
- [ ] Build ladder prompts L2–L4 for SmolDataEnvs `test` (L2 automatic; L3/L4 from verified sft code); leak check
- [ ] Run the ladder for every arm, only on tasks failed at the rung below; record the lowest passing rung
- [ ] DABstep: L1 vs L2 (schema) for every arm; the skill-vs-information split for the transfer question
- [ ] Tool-call analysis: read/inspect calls per trace by arm and rung (does training or information replace exploration?)
- [ ] Sample ~50 failures per arm per benchmark
- [ ] Categories, e.g.: wrong column/file, bad join/filter, wrong aggregation, answer formatting, crashed code, gave up/ran out of turns, misread question
- [ ] Table: failure categories × arm × benchmark

### M7: Post #2 + release
- [ ] Write-up: RQ1–RQ4, results table, transfer chart, ladder chart (lowest passing rung per arm), error analysis, limitations
- [ ] Cost accounting (free hours used + $ spent)
- [ ] Release models, configs, cleaned dataset, harness

### Stretch: hint-scaffolded GRPO (possible post #3)
Problem it targets: at ~28% pass rate many GRPO groups score all zeros and give no gradient, and SmolDataEnvs'
"+0.1 if the code runs" shaping reward got gamed. Idea: train hard tasks with a higher ladder rung and
withdraw hints as per-task pass rate rises (the adaptive curriculum the ladder post suggests).
- [ ] Literature check first: hint- or guidance-based RL for LLMs already exists; find what's new here
- [ ] Only start after post #2 ships

---

## Risks

| Risk | Mitigation |
|---|---|
| Effects within noise | CIs, 2 seeds, whole splits (no subsets) |
| A vs B confounded by format | common converter + template (M3) |
| DABstep submission limits/format | check leaderboard rules early (M1) |
| fp16 instability on T4 | LoRA, lower LR, watch the first few hundred steps |
| Reward hacking in GRPO | require printed output, inspect samples every N steps |
| AMD credit clock | don't apply the credit until M5 smoke test passes |
| Scope creep | post #1 ships before M3 starts; scaffolded GRPO only after post #2 |
| Ladder hints leak the answer | automatic leak check; hand-inspect a sample |
| Ladder eval cost | only climb on failed tasks; full ladder on SmolDataEnvs `test` only |

## Links
- SmolDataEnvs: https://huggingface.co/datasets/FineEnvs/SmolDataEnvs · code: https://github.com/adithya-s-k/FineEnvs/tree/main/04-smoldataenvs
- SmolDataEnvs-sft: https://huggingface.co/datasets/FineEnvs/SmolDataEnvs-sft
- jupyter-agent: https://huggingface.co/datasets/jupyter-agent/jupyter-agent-dataset · blog: https://huggingface.co/blog/jupyter-agent-2
- DABstep: https://huggingface.co/spaces/adyen/DABstep · paper: https://arxiv.org/abs/2506.23719
- AgenticDataBench: https://arxiv.org/html/2607.01647
- Information ladder post: https://evan-kim2028.github.io/evan_writings/writings/difficulty-is-an-information-gap/
- AMD Developer Cloud notes: https://lilting.ch/en/articles/amd-developer-cloud-credit-journey
