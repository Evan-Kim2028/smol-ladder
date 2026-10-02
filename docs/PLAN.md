# Plan: post-train a 2B data-analysis agent, and use the ladder to measure it

> The priority is **data first, then a training run**. Evaluation of the *released* models
> (the old "post #1") is no longer the gate; the ladder is now an instrument for training and
> measurement rather than a deliverable of its own. Checkboxes work as a task list.

## Where we are

**What exists.** A complete eval harness: the four-rung ladder with a Blackwell-tested ordering
(`tests/test_blackwell.py`), the offline sealed grading pass, the SmolDataEnvs grader unmodified, a
solver that talks to any OpenAI-compatible endpoint, and the two upstream 2B protocols (`program`,
`bash`) alongside ours (`tools`) so every arm can be run under the protocol it was trained in
(`docs/LOCAL_MODELS.md`). Reference generation with a retry budget (`gen_refs --attempts N`,
keeping every attempt in its own directory and treating only reward >= 1.0 as final), plain-language
hint generation and validation, the tagged jupyter-agent pools v1/v2/**v3** (`jtasks_v3`, 7,518
tasks, with the corrected `test`/`eval` bare-name overlap firewall), per-rung summarisation with a
bootstrap CI, offline regrading, and `--run-tag`/`RUN.json` so a sweep records the code, command
line, model, protocol, rungs, samples, climb setting and reference denominator at launch.
453 tests pass (as of 2026-10-01), 4 of them marked `slow` because they read a whole split from
disk; `pytest -m "not slow"` is the ~2-minute quick loop (449 tests) and the full suite ~15.

**Measured, on `test`, run `v2` (250 tasks x 5 conditions x 2 samples, `--no-climb`,
`stealth/space-bunny-alpha`) — as of 2026-10-01, re-measured from `data/runs/v2` by me:** every
rung on the same 213 referenced tasks reads L1 87.6% [83.6, 91.3], control 88.0%, L2 93.4%,
L3 95.3%, L4 98.6%; on all 250 L1 is 77.9% [73.1, 82.6] (452/500 trials scored, 48 harness
failures) and the control 79.4%. **The ladder is not monotone**: L1→L2 11 of 212 paired tasks
fell and 27 rose (p=0.014), L2→L3 10 fell / 16 rose (p=0.33), L3→L4 3 fell / 13 rose (p=0.021).
The paired control effect is +1.2% [−2.1, +4.6], p=0.36, i.e. the schema dump buys nothing over L1.
**154 of the 213 referenced tasks already pass L1 in both samples**, so the full-set curve is near
ceiling; on the 56 tasks that failed L1 in at least one sample the curve is 22.3% → 97.2% (control
37.5%), which is the only place a hint effect is visible. Full tables and the caveats:
`docs/LADDER.md`, "Run v2". **The referenced set is biased toward tasks this model can do**, since
a reference exists only where this model family already produced a verified solution.

**Tooling fixed on this branch, 2026-10-01.** `summarize` refused v2 without `--allow-mixed`
because its mixed-prompt check compared prompt hashes across *tasks* rather than within one
(task, rung) cell — and every rung prompt embeds its own question, so 250 clean tasks read as 250
ladders. It now checks per (task, rung) on three axes (prompt hash, a ladder fingerprint over the
six files that write the rungs and grade them, and the hint prompt version), and a launch's
fingerprint is checked against the results on disk. v2 summarises clean with no override.
`smol_ladder.reverify` re-runs the sealed offline grading pass for trials whose first pass failed
(90 of them, all harness failures under load, not model failures) and records the outcome in a new
`reverify.json`; it never writes `result.json`. Agent timeouts stay harness failures.

**jupyter-agent L1 on the v1 pool (2,000 tasks) — claimed 838/1,990 = 42.1%, and as of
2026-10-01 I cannot re-derive it from disk.** The per-trial results of that sweep do not survive:
`data/runs/jupyter-agent.bak-pre-rerun-20261001/` holds **692 results over 692 distinct tasks**,
not 1,990, and `data/runs/summary_jupyter-agent.json` reports `tasks: 0` with an all-zero L1 block.
The figure is consistent with `logs/ja_L1_full_20261001.log`, which prints every tenth task and ends
on "done: 2000 tasks, 840 verified references" — but that log line counts *reference* generations,
not L1 passes, and a log is not a per-trial artefact. **Quote 42.1% as historical, not as a
measurement**; the number to quote for a clean pool is the v3 sweep, and its per-trial results do
survive. What is verifiable today: `data/solutions/jupyter-agent` holds **2,005 reference attempts,
855 of them `keep: true`**.

**383 of the 2,000 v1 tasks are excluded from `jtasks_v3`** (verified on disk 2026-10-01: 1,617 v1
ids survive into v3, 383 do not). The corrected overlap firewall fires on SmolDataEnvs `test`/`eval`
tables by bare dataset name and those 383 sit on one. So the 42.1% was measured over a population
whose overlap with the held-out splits was, in v1, only partly removed — a further reason to quote
the v3 sweep instead.

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
On disk as of 2026-10-01: 209 `test` hints (193 validated, 16 failed) and 60 `eval` hints; the
`test` figure above is the 181 that also have a ladder reference, which is the set L2/L3 are
measured on.

**Every sweep before 2026-10-01 saved no transcripts, and that is what made arm B unbuildable.**
`run_ladder.once()` copied `solution.py` out of the per-trial scratch and deleted the scratch, so
the assistant/tool conversation existed only in memory; `turns.json` held `json.dumps(len(log))` —
a turn *count*. So every verified trial already on disk is a program and an answer, and the
**existing verified trials yield only single-turn examples** through `train/traces.py`'s fallback:
write the program, submit the known answer, stop. That teaches the contract and nothing about
exploration, and it is emphatically not a substitute for a real trace. Measured on this machine
before the fix: `data/solutions/test` 211 and `data/solutions/eval` 116 transcripts, **all of them
held out**, and **0** under `data/solutions/jupyter-agent`, `data/runs/test`,
`data/runs/synthetic` and `data/runs/jupyter-agent.bak-*` — 2,178 verified trials with no
trainable conversation among them.

`run_ladder` and `gen_refs` now save the whole conversation by default (`--no-transcript` opts
out), on both paths, including the transcript a promoted reference carries — and **this cannot be
applied retroactively**, so arm B's data has to come from a new sweep rather than from the tree
already on disk. That is what the `ja3` transcript sweep is for. See `docs/TRAINING.md` §1.

**Repaired on this branch.** The `synthetic` split's gold answers were wrong and are now fixed.
`iter_tables` read 50,000 rows to compute the answer while the agent read the untruncated file, so
any table over that size was ungradeable by construction (58 of the 275 old tables); separately,
`.6g` printing against a 1e-6 tolerance rejected the task's own correct answer on tasks small
enough that truncation could not explain them. Both are fixed at the root and enforced by a
shipped-file gate that
re-executes each task's reference against the file the agent is shipped. The corpus is **6,956
tasks over 42 tables**, all gate-passing, of which **only 275 have any trial at all** — so the
split has no measured curve, and regrading the stored 757 trials moves 287 of them (L3 1/65 → 61/65,
L4 0/64 → 62/64). The old synthetic numbers were a measurement of a broken gold and are
superseded; see `docs/LADDER.md` for the before/after table and the 87 owed rung-trials.

**The training pipeline exists and is exercised, but not at the target scale.** `train/` carries
the exporter, the LoRA SFT trainer, the format converter and the GRPO loop, with
`docs/TRAINING.md`. As of 2026-10-01 the **only measured training run is an SFT smoke test on
Qwen3.5-0.8B** (30 steps, 2048 tokens, ~22 min) — chosen because 2B does not fit a LoRA smoke run
in 6 GB of VRAM. **The GRPO loop has never been run for real**, and no 2B arm (A, B, C, D) has been
trained. Every timing for 2B in `docs/TRAINING.md` is an extrapolation from the 0.8B measurement.

**Still open.** jupyter-agent references are accumulating but still far short of the pool: 2,005
attempts on disk, **855 verified**, as of 2026-10-01, against 7,518 tasks in v3 (4,217
ladder-grade). The ladder is **not monotone** on run v2: 11 of 212 paired tasks fall from L1 to L2
(p=0.014) and 3 of 209 from L3 to L4 (p=0.021) — a real budget effect, unattributed. 12% of L1
tasks flip verdict between two samples, so "first passing rung" is not identifiable at k=2 either.
No inference stack is installed on this machine.

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
| SmolDataEnvs-sft | 4,677 verified trajectories | **usable for SFT today**; the export keeps **4,673** and drops 4 | SFT traces | nothing — but it is `bash`-protocol; conversion needed if arms must share one tool format |
| SmolDataEnvs `test` / `eval` | 250 / 144 | held out, never trained on | in-distribution eval | nothing; **as of 2026-10-01, verified from disk: `test` 213/250 and `eval` 118/144 have a verified reference**, so L2–L4 are measurable on those only. In run v2 the 213 referenced tasks were all measured at every rung (`--no-climb`); 154 of them already pass L1 in both samples |
| jupyter-agent pool (v3) | 7,518 tasks, **4,217 ladder-grade** (both re-measured from `data/jtasks_v3.jsonl` and `smol_ladder.pool` on 2026-10-01) | built and tagged; references being generated | RL tasks + reward; eval — the `ja3` transcript sweep over the 3,880 ladder-grade tasks whose tables are cached has yielded **2,029 exported SFT traces** (arm B) and continues | verified references (855 in `data/solutions/jupyter-agent` as of 2026-10-01, all on the v1 pool); 337 ladder-grade tasks are skipped for uncached datasets |
| Plain-language hints (L2/L3) | 167/181 `test` refs validate | written, validated, cached; **measured on v2: 20 tasks fall back to the AST per rung** | RL curriculum; measurement | the AST arm is 20 tasks, far too few to read a difference off (L3 85.0% AST vs 96.3% hint — not a finding at that size); needs generating on jupyter-agent refs |
| Synthetic tasks | **6,956** | **gold repaired and gated; unmeasured** | RL tasks (unlimited supply) | the shipped-file gate now refuses any task whose reference does not grade 1.0 against the file the agent is shipped, and the id cache is committed so a regeneration cannot renumber the corpus. But only **275** of the 6,956 have ever been run, so there is no curve: run a full `--no-climb` grid over the corpus before quoting any synthetic number |

The `jtasks_v3` ladder-grade subset (4,217 tasks) deliberately keeps the method-ambiguous families
(`ml_fit`, `stat_test`, `groupby`, `lookup`, `join`) and does not tune towards `count`/`agg`, because
those are the families the ladder has nothing to disambiguate. 7,018 of 7,518 v3 tasks (93.4%)
reuse datasets already in the Kaggle cache, as of 2026-10-01; restricted to the ladder-grade subset
it is **3,880 of 4,217**, and the remaining 337 are skipped rather than downloaded (their datasets
span owners the endpoint was refusing). Run `--split jupyter-agent-v3` to select that pool, which
never writes `data/jtasks.jsonl`; v2's corresponding figures are 5,124 ladder-grade and 8,668/9,187
cached.

**Contamination in the jupyter-agent pool, corrected.** v1 and v2 were not clean and the docs
previously claimed overlap with SmolDataEnvs was "impossible". It was neither: the firewall banned
only `test`'s 122 slugs (not `eval`'s 81) and matched full `owner/name` slugs, which misses a
different owner's mirror of the same table. **1,669 of v2's 9,187 tasks (18%) — 383 of v1's 2,000 —
sit on a table SmolDataEnvs `test` or `eval` has already scored.** `data/jtasks_v3.jsonl` (7,518
tasks, 4,217 ladder-grade) fires on `test` and `eval` only and matches on the bare dataset name, so
a mirror is caught and a task sharing a table with SmolDataEnvs `train` is kept and tagged
`sde_overlap: "train"`. Every v3 task was already in v2 with identical content apart from the two
new fields, so no `task_id` moved and no existing result is invalidated. See `docs/LADDER.md`,
"The overlap firewall was wrong".

**Contamination, measured.** 74% of `test` tasks share a source table with a `train` task (185/250 by
`bucket_prefix`), and 4 `test` questions appear verbatim in the released SmolDataEnvs-sft blob. The
split is by question, not by table, so an arm trained on SmolDataEnvs-sft has seen the tables L4 is
written about. The clean version partitions the 471 Kaggle datasets, not the 5,394 tasks.

**The SFT export firewall, and the decision behind it (owner's call, made 2026-10-01).** The export
drops 4 rows and keeps **4,673** of SmolDataEnvs-sft's 4,677, at the `task_id,question` default. The
4 are held-out *questions* appearing verbatim in `test` under a different task id, which a `task_id`
check cannot see; one leaked question is 0.4 points of pass@1 on 250 tasks. The table-level
(`bucket_prefix`) firewall stays an **opt-in flag**, off by default and not changed here: 115 of the
170 held-out tables also appear in SmolDataEnvs' own `train` split, so a table-level check refuses
3,019 of 4,677 rows including every row upstream itself would have trained on, and arm A at 1,658
rows is no longer arm A. The coarse check is the defensible one for the ladder, where knowing a
table's shape is the signal being measured, and it must not be flipped silently. **The contamination
is reported either way** — every export prints the drops by column, so a run can always state how
many rows it refused and why. Full reasoning and the measured rows: `docs/TRAINING.md` §3 and §7.

## Critical path to the first training run

**Ordered; steps marked [non-blocking] are not on the path to the first training run.**

| # | step | depends on |
|---|---|---|
| 1 | **SFT (LoRA) on SmolDataEnvs-sft** = arm A — the first training run | a serving/training stack on AMD, nothing else. The 4,673 exported rows are `bash`-protocol (`messages` + `tools`, answer in `/workdir/answer.txt`, 3–12 turns; upstream's config used `max_length=8192`) |
| 2 | Converter: the two trace formats → one tool format + chat template | (1), which defines the format. Without it, arm-vs-arm differences are format, not data |
| 3 | SFT on our own traces = arm B, then A+B | (2) for the format, plus our trace collection — **the collection is done** (2,029 exported ja3 traces) but the runs are **[non-blocking] for step 1** |
| 4 | GRPO (LoRA) on SmolDataEnvs `train` = arm C | the best SFT arm as its starting point, and a non-degenerate reward. `num_generations=8` at the ~28% pass rate is the problem this project exists to solve |
| 5 | Hint-curriculum GRPO = arm D | verified references + validated L2/L3 hints for the *training* tasks. Hints exist for `test` refs today, not for `train` |
| 6 | Ladder measurement of every arm, before and after | each arm existing. Cheap relative to training, so it can run on the baseline as soon as (1) lands |

**Reference generation** is needed for 5 and 6 but **[non-blocking]** for the first training run. Run it
early anyway: it is the long pole for the ladder and it is what validates a task's gold.

- [ ] 1. SFT (LoRA) on SmolDataEnvs-sft, **arm A**, 4,673 exported traces (4,677 minus 4 leaked
      questions) — first training run
- [ ] 2. Converter: upstream `bash` traces and our traces → one tool format + chat template
- [ ] 3. Reference solutions for the training split (`gen_refs` / `gen_solutions`), verified offline
- [ ] 4. Plain-language L2/L3 hints for the training split (`gen_hints`), validated and cached
- [x] 5. Synthetic gold repair: the 50k truncation and the print/tolerance bug are fixed, the
      corpus is regenerated under the committed id cache, and every spec passes the shipped-file gate
- [ ] 5b. Run the synthetic split: only 275 of 6,956 tasks have ever been run, so `--no-climb`
      across the corpus before any synthetic number is quoted
- [x] 5c. Re-measure `test` cleanly at k=2 with `--no-climb` (run `v2`), re-verify the trials
      whose grading pass failed under load, and report the curve on one task set, the paired
      control effect, rerun consistency, the hint-source split and the ceiling — `docs/LADDER.md`,
      "Run v2"
- [x] 6. Our own verified SFT traces in the converted format — **the exporter is written and tested,
      and the data now exists**: `data/train/ja3_sft.jsonl` holds **2,029 traces from 2,109 verified
      trials** (`data/train/ja3_sft.manifest.json`; 0 rows dropped by the widest-key firewall). This
      item was open only because the old tree could not yield conversations; the `ja3` sweep is what
      closed it. The training run itself is item 6b below
- [ ] 6b. Train **arm B** on those 2,029 traces, then **arm A+B** on 4,673 + 2,029, in that order
- [ ] 7. GRPO (LoRA) on SmolDataEnvs `train` from the best SFT arm — arm C
- [ ] 8. Hint-curriculum GRPO: hints on at low pass rate, withdrawn as per-task pass rate rises — arm D
- [ ] 9. jupyter-agent references at scale (the 4,217 ladder-grade tasks in `jtasks_v3`).
      **In progress**: the `ja3` transcript sweep over the 3,880 ladder-grade tasks whose tables
      are cached produced the 2,029 traces above; reference generation at pool scale continues
- [ ] 10. Ladder measurement of every arm, before and after, same tasks and protocol
- [ ] 11. Open decisions below, then the write-up

**Gates before the first instance is created.** Three things must be true before a MI300X droplet
exists, because each one is cheaper to check here than at $2.59/h:

1. [x] **Arm B's export exists.** Done: 2,029 traces, 2,109 verified trials, 0 firewall drops, scrub
   enforced — `data/train/ja3_sft.manifest.json`. Arm B is not a plan any more.
2. [ ] **The AMD runbook and the `ops/amd` scripts are reviewed and dry-run — in progress.** Note
   that `ops/` does not exist in the tree yet, so this gate is on the runbook's own creation as much
   as on the scripts. A dry run must cover create, a real training step, and **destroy**.
3. [ ] **The GRPO loop has had a real local smoke run on the 0.8B model — in progress.** Until a GRPO
   step has actually executed end to end, arm C's budget line is a guess and its results would be
   uninterpretable.

## Immediate next steps

1. Finish the export and the hints: close out the ja3 sweep and `gen_hints` for the training split.
2. Review the AMD runbook and the `ops/amd` scripts, and dry-run them (gate 2).
3. ROCm smoke test on a short-lived MI300X instance — vLLM + TRL + one training step — then
   **DESTROY it**, pass or fail.
4. Train SFT **A**, then **B**, then **A+B** on the instance.
5. Ladder-evaluate the base model and each of the three adapters on `test`, same tasks, same protocol.
6. GRPO smoke on the 0.8B locally (gate 3), then **C** and **D** on the remaining credit.

## Training arms

The SFT arms are **A, then B, then A+B** — the owner's decision of 2026-10-01 — and the two GRPO arms
follow from whichever SFT arm wins. A is run first because it is the one that can be run today and it
replicates a known target; B is the interesting comparison; A+B is the practical best.

| Arm | Model | Training | Notes |
|---|---|---|---|
| 0 | Qwen3.5-2B | none | the control; base model, no instruction tuning for either protocol |
| R-SFT | `smoldataenvs-sft-2b-v0` | theirs | a **93 MB LoRA adapter** (r=16, α=32) on the base, not a model |
| R-GRPO | `smoldataenvs-grpo-2b-v0` | theirs | shipped in **fp32, 8.85 GB**; must be cast to bf16 (4.43 GB) |
| A | Qwen3.5-2B | SFT (LoRA) on SmolDataEnvs-sft, **4,673** exported rows | the replication of upstream, and the first run. The export drops the 4 questions that appear verbatim in `test`; the table-level firewall stays an opt-in flag and the contamination is reported either way |
| B | Qwen3.5-2B | SFT on **our** exported ja3 trajectories, **2,029** traces | **had no data at all until this branch**: every sweep before 2026-10-01 saved no transcript, so arm B exists because `run_ladder` and `gen_refs` keep the conversation by default. The export is done; the counts and stats are in "Arm B's data" below |
| A+B | Qwen3.5-2B | SFT on A's 4,673 rows plus B's 2,029 | the pooled arm; A+B − A is the read on whether our traces add anything on top of theirs |
| C | best of A / B / A+B | + GRPO (LoRA) on SmolDataEnvs `train` | plain GRPO, no curriculum |
| D | C | + hint-curriculum GRPO (L2/L3 withdrawn as pass rate rises) | **promoted from stretch goal**; the ~28% pass rate makes all-zero groups the binding constraint |

**Arm B's data, as of 2026-10-01** (`data/train/ja3_sft.manifest.json`): **2,029 rows**, each
`messages` + `tools` (the SmolDataEnvs-sft format, one tool named `bash`) and nothing else, from
**2,109 verified trials**; 983 of those trials had the gold answer below the leak floor. The firewall
ran at its **widest key, `kaggle_table`** (170 tables; also `bucket_prefix` 170, `question` 393,
`task_id` 394) and **dropped 0 rows** — the pool is ja3, not SmolDataEnvs. Tokens under the
Qwen3.5-0.8B chat template with `enable_thinking=False`: min 1,249, **median 2,364**, p90 6,956, max
54,648, **7.93% over 8,192**; turns min 4, **median 7**, p90 11, max 41. The path/identity scrub fails
the export if a local absolute path, the local username, a hostname or an API key survives. Op family:
count 823, agg 400, string 239, filter 195, stat_test 144, other 67, ml_fit 66, argmax 48, groupby 35,
lookup 12.

A second export, `data/train/ja3_fallback.jsonl`, holds **682 single-turn contract rows** built from
the 855 v1 verified references in `data/solutions/jupyter-agent` (173 dropped as not in the v3 pool).
Its own manifest says they are **not traces** — no conversation exists for those trials, so there is no
exploration in them. They can pad an arm but they cannot stand in for B.

**How to read A vs B, stated plainly.** B is **smaller** (2,029 vs 4,673), it comes from **a different
solver**, and it shares most of its tables with the SmolDataEnvs `train` split, so the two arms differ
in size, in teacher and in table coverage at once — A beating B is not by itself evidence that their
data is worse. The clean version is a **size-matched subsample of A**, an optional later run rather
than one of the three committed arms. A dataset comparison report (`docs/DATASET_COMPARISON.md`) is
**in progress** and is not linked here until it exists on disk.

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
  (213 have one, and 181 have a rung above L1 actually run), so every rung number is conditional
  on that.
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
| AMD Developer Cloud (MI300X 192 GB, **$2.59/h**) | **$100.00** of promotional credit, **expires 2026-10-18**, so ≈**38 GPU-hours** at the verified rate | arms A–D. The credit is live and unused, so it is not "applied once SFT and GRPO run back to back" any more — it is spent **AMD first**, while it lasts |
| Kaggle | 2× T4 16 GB, 30 h/week | the **fallback** if the ROCm smoke test fails, and the home of everything that must happen after 2026-10-18. Credentials are **not yet on this machine**, so it is not a same-day path |

Corrections from `docs/LOCAL_MODELS.md`: the SFT release is a **LoRA adapter** (93 MB, r=16 on
`all-linear` — it must be served with the base model and the adapter loaded), and the GRPO release is
**fp32 at 8.85 GB**, which does not fit this card in any precision it runs well — cast to bf16
(4.43 GB) first. The released models only evaluate meaningfully under each one's own protocol, and
that doc's throughput figures are bandwidth arithmetic rather than measurements (±2×).

**The verified compute facts, 2026-10-01.** The AMD Developer Cloud credit is **administered through
the DigitalOcean API** — instances are created and destroyed against that API — and the token lives
in the git-ignored `.env` as `AMD_CLOUD_API_TOKEN` (aliased `DIGITALOCEAN_ACCESS_TOKEN`). It is never
committed. Two facts here are *not* from the API, and are labelled as such rather than dressed up as
measurements:

- **$100.00 available, expiring 2026-10-18** is read off the **owner's portal**, because the API does
  not expose credits at all. If this number has moved, only the portal knows.
- **MI300X 192 GB is $2.59/h**, per the [Droplet pricing
  docs](https://docs.digitalocean.com/products/droplets/details/pricing/) *and* this account's own
  GPU size listing via the API. The earlier **$1.99/h** in this document was wrong; **$100 at $2.59/h
  is ≈38 GPU-hours, not ≈50.** The working deadline is therefore **2026-10-17**, one day before
  expiry, with a **hard spend cap of $90** — the $10 headroom is what stops a runaway loop from
  turning a credit run into a real invoice.

Three billing rules decide how the run must be operated, all from the provider's own terms:

- **Invoices land on the first of the month for the previous month's usage**
  ([invoices](https://docs.digitalocean.com/platform/billing/invoices/)). October's GPU time is
  therefore invoiced on **2026-11-01**, *after* the credit expires on 2026-10-18. There is no early
  warning that the cap was blown.
- **The credit applies to the whole billing cycle it expires in.** The
  [promotional-credit terms](https://www.digitalocean.com/legal/promotional-credit-discount-terms) say
  redeemed credits "will be applied to offset eligible fees and charges incurred during the entire
  billing cycle in which it expires", and charges **above** the credit are billed to the payment
  method. So spending inside the cycle is what the credit pays for, and the cap is the only thing
  standing between the run and a real card charge.
- **A powered-off GPU droplet still bills, so teardown means DESTROY.** Not powering it down, not
  leaving it idle overnight, not "just pausing for the weekend" — the machine keeps charging until
  the API call destroys it. GPU droplets bill **per second with a 60-second minimum**
  ([pricing](https://docs.digitalocean.com/products/droplets/details/pricing/)), so a
  smoke-test-and-destroy costs about a minute, not an hour.

Spend is tracked from **instance uptime**, not from a timer someone remembers: the harness records
the create and destroy timestamps, and $2.59 × those hours is the number that matters.

**Budget.** These are *estimates*, to be replaced by measured numbers after the first hour on the
instance — an hour of real 2B SFT throughput moves every row below.

| Item | Estimate | Note |
|---|---|---|
| ROCm smoke test (vLLM + TRL + a training step) | 1 h | per-second billing with a 60 s minimum, so a throwaway instance is cheap |
| SFT A, then B, then A+B | 3–6 h total | all three share the one 192 GB card |
| Ladder evaluation: base + three adapters on `test` | 4–8 h | `--no-climb`, k samples |
| GRPO C (plain) | 8–10 h | |
| GRPO D (hint curriculum) | 8–10 h | |
| Margin | 3–5 h | |
| **Total** | **≈27–38 h** | the top of that range is the whole credit; the bottom leaves real slack |

**If the budget runs short, the cut order is GRPO before SFT/eval.** Drop D first, then C, and keep
A/B/A+B plus their ladder evaluation — an SFT arm that was measured on the ladder is a result; an
unmeasured GRPO run is nothing. Nothing runs past $90.

**Kaggle's role.** 2× T4 16 GB, 30 h/week, credentials not yet on this machine. It is the fallback if
ROCm fails the smoke test — if the MI300X cannot do a training step in the first hour, do not burn
the credit discovering why — and after 2026-10-18 it carries the second seeds, the size-matched
subsample of A, the eval-split ladder evaluation, and any overflow.

Gotchas: Kaggle needs a **T4** (vLLM does not support P100), fp16 only (watch for NaN loss spikes with
Qwen — LoRA and a lower LR), no FlashAttention 2, a 12 h session cap, and the disk is wiped between
sessions, so checkpoint to the Hub and write results to `data/runs/`. Internet must be enabled (phone
verification). On AMD, smoke-test vLLM + TRL in the first hour and **destroy the instance whether
the test passes or fails** — a failed smoke test on a live droplet is $2.59 per hour of pure lesson.
Do not use QLoRA (bitsandbytes on ROCm is less mature and memory is not the bottleneck). Use WSL2 if
on Windows.

Secrets, from `.env` (git-ignored, never committed): `HF_TOKEN`, `KAGGLE_USERNAME`/`KAGGLE_KEY` (or
`~/.kaggle/kaggle.json`), `OPENROUTER_API_KEY`, `AMD_CLOUD_API_TOKEN` /
`DIGITALOCEAN_ACCESS_TOKEN` for the AMD credit's DigitalOcean management API, and optionally
`WANDB_API_KEY`. On Kaggle add them as notebook **Secrets**, never inline.

## Open decisions for the owner

1. Which protocol is the project protocol — upstream's one-turn `program`, the `bash` agent, or our
   `tools` loop? This decides the converter, the held-out discipline and every cross-arm table.
2. **Resolved 2026-10-01: the SFT arms are A, then B, then A+B**, and GRPO arms C (plain) and
   (hint curriculum) start from whichever SFT arm wins. A is first because it runs today and
   replicates a known target; A+B is last because pooling is only interesting once each half is
   measured.
3. **Resolved for now: the firewall stays at `task_id,question` (4,673 rows kept, 4 dropped) and the
   table-level check stays an opt-in flag.** Revisit if the ladder's L1-vs-L2 comparison becomes the
   primary result rather than arm A's replication; it must not be changed silently either way.
4. **Resolved 2026-10-01: AMD first, Kaggle as fallback and for everything after the credit.**
   The credit is already applied and expires **2026-10-18**, so there is no longer a question of
   *when* to spend it — the earlier "apply it only once SFT and GRPO run back to back" ordering was
   written for a 30-day clock that does not exist. What replaces it is the **$90 hard cap**, the
   **2026-10-17** working deadline, and **DESTROY** as the only teardown. The first GPU hour goes to
   a ROCm smoke test; if it fails, the credit goes back to waiting and Kaggle takes over.
5. Do we hold the 74%-table-overlap contamination and report it, or partition the 471 Kaggle datasets
   for a clean held-out set (which shrinks `test`)?
6. Synthetic: the gold is repaired and gated, so the split is kept as unlimited RL task supply —
   but it has never been run beyond 275 of 6,956 tasks. Run it, or drop it?
7. **Resolved 2026-10-01: we do our own trace collection, and it is arm B** — 2,029 exported ja3
   trajectories exist, so the question is answered by running B rather than by debating it. The
   "smaller, higher-precision set built only from verified traces" is already what B is; what
   remains open is whether A vs B needs the **size-matched subsample of A** to be read fairly, which
   is an optional later run, and the dataset comparison is being written up separately.

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
| Training on broken gold | both known cases (synthetic's 50k truncation and its `.6g`/1e-6 tolerance) are fixed and gated; the rule stands that a task's gold must be re-derivable from the table the agent sees |
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