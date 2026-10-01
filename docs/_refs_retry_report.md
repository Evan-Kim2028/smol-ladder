# Reference retry report (test / eval)

Sweep of 2026-10-01 on branch `feat/refretry`, run from the `smol-ladder-wt-refretry` worktree
with `--attempts 3 --workers 8`, model `stealth/space-bunny-alpha`. Logs:
`logs/refs_test_retry_20261001.log`, `logs/refs_eval_retry_20261001.log`.

## Coverage

| split | before | retried | new references | after | of |
|-------|--------|---------|----------------|-------|-----|
| test  | 181/250 (72.4%) | 69 | 32 | **213/250 (85.2%)** | 250 |
| eval  | 95/144 (66.0%)  | 49 | 23 | **118/144 (81.9%)** | 144 |

287 fresh attempts were written (168 test, 119 eval), none overwriting an earlier one. The
previous cached failures were genuinely unreachable: `reference()` gated on
`agent_status == "exit 0"`, so the 56 test and 32 eval tasks whose solution ran cleanly and
printed the wrong answer could never be retried. Roughly half of each blocked set turned out to
be solvable on a second or third try.

## Why the rest have no reference

63 tasks across the two splits still have none after 3 attempts. The buckets below are read off
the attempt directories, not inferred from the log, and each implies a different decision.

| bucket | test | eval | what it means |
|--------|-----:|-----:|---------------|
| inconsistent answers | 18 | 13 | model disagreed with itself |
| timeout | 9 | 3 | agent never produced an answer |
| consistent same wrong answer | 6 | 5 | see below |
| some attempts produced nothing | 4 | 4 | too little evidence to call |
| no output at all | 0 | 1 | the agent finished without answering |

### Consistent same wrong answer (11) — the interesting list

These are the only tasks where every attempt that answered gave the same wrong thing, so they
are the only ones that say anything about the *gold*. I checked the underlying data for the
three cleanest ones; **the model is not right and the gold is not reproducible either**.

**`0000_684_684849_qa_4` (test) — the data does not contain the column.**
Q: total number of hospitals owned by "Government - Hospital District or Authority".
Gold `566`; all 4 attempts said `561`.
`HospInfo.csv` here is the 2017 CMS Hospital General Information file. It has exactly three
`Hospital Type` values — `Acute Care Hospitals`, `Critical Access Hospitals`, `Childrens` — and
no row anywhere matching "Government" in any column. The model correctly identified that the
label belongs to a *Hospital Ownership* column, found that column missing, and then fell back to
walking the filesystem, where it counted 561 unrelated CSVs. The gold came from a different
vintage of the file that had the column. **Unanswerable from the shipped data — drop.**

**`0000_667_667350_qa_3` (test) — "case" is undefined.**
Q: which year 2009–2012 had the highest total number of water pollution cases across all
districts and pollutants. Gold `2010`; all 4 attempts said `2009`.
The file has one row per village-pollutant reading (550,242 rows) with a `Year` column holding
four snapshot dates, not a case id. Counting rows gives 2009 (179,999 vs 144,582); the gold's
2010 must count distinct reported conditions, which the file does not identify. Both answers are
defensible from the shipped data. **Ambiguous — drop.**

**`0000_764_764519_qa_3` (test) and `0001_325_1325260_qa_1` (test) — format, not substance.**
Gold `Male`, all attempts said `M`; gold `1 engine`, all attempts said `1`
(`reward_mode=exact_short`). The model identified the right row every time and lost on surface
form. These are answerable by a prompt that asks for the label verbatim from the column, so
**fix the prompt before dropping.**

**`0000_455_455476_qa_5`, `0001_043_1043420_qa_4` (test), `0001_078_1078204_qa_2` (eval) —
unstated thresholds and splits.** Gold `575` vs `580` (a "total stat threshold" no rule defines);
gold `3` vs `1` benign cases misclassified by a Random Forest with no seed or split given; gold
`37.79` vs `37.17` mean game duration. **Not reproducible as specified — drop.**

**`0001_875_1875604_qa_4` (eval) — mismatched question and gold.** Q asks *which cereal* has the
highest potassium; gold is the number `330`, every attempt answered with the cereal name. The
gold answers a different question. **Drop.**

**`0002_335_2335686_qa_2` (eval) — column naming.** Gold `accel. y`; all attempts said
`acceleration_y`, i.e. the same feature under its real header. Prompt/answer-normalisation fix,
not a bad question.

**`0016_567_16567793_qa_4` (eval) — genuinely close.** Gold `SVM`, all 4 attempts said
`Logistic Regression` (one attempt timed out). A real disagreement about model accuracy, likely
seed-dependent. **Drop or re-seed.**

### Inconsistent answers (31)

Almost none of these are nondeterministic in the strict sense; they split into three causes.

1. **Unstated hyperparameters (the majority).** The question names a model but never a seed,
   split, or K, so the metric is a free parameter: KNN K answered `13`, `9`, `6` against gold
   `5`; precision answered `0.7526`, `0.9277` against `0.6940`; per-iteration accuracy varied
   `0.7264`–`0.8146` against `0.8208`. Also `0001_922_1922748_qa_4` (gold
   `Decision Tree Classifier`, answers `Decision Tree`, `DecisionTree`) and `0013_884_13884693_qa_4`
   (gold `KNN`, answers `Decision Tree`, `QuadraticDiscriminantAnalysis`) — these look like label
   mismatches in the gold rather than modelling noise.
2. **Unit and scale errors that a stricter answer spec would catch.** Gold `0.8846` vs answers
   `0.88464`, `0.9988`; gold `157` vs `157.11`; gold `88` vs `0.8846`; gold `9` vs `10`, `12`, `7`.
   The model is within tolerance in several cases and the grader's `atol` is what rejects it.
3. **Answer-form failures** — `USA` vs `United States`, `São Paulo` vs `SP`, `no` vs `yes`/`False`,
   `Romance` vs `Action`, `4th` vs `4`. The reasoning is often right and the surface form is
   rejected by `exact_short`.

### Timeouts (12) and near-silent agents (1)

All 12 timeouts are the same family: questions that require training or scanning several models
and reporting the best one's metric (XGBoost sweep, LightGBM with hyperparameter optimisation,
KNN/SVM/RF comparison, "which model achieved the highest F1"). Four ran past the 1200 s cap on
every attempt. These are not wrong, just too slow for the harness; they would need a longer cap
or a cheaper question, not a better prompt.

The one remaining `no output` task is eval `0012_337_12337856_qa_2`, which exited cleanly on every
attempt without printing anything — a prompt failure rather than a broken harness. No attempt in
either split ended in a genuine harness error (the single `exit 8` the earlier sweep had recorded
for test `0001_632_1632381_qa_1` was retried and resolved).

## Recommendation

Drop from the ladder, in this order:

1. **Unanswerable / wrong gold (8).** `0000_684_684849_qa_4`, `0000_667_667350_qa_3` (test);
   `0001_875_1875604_qa_4`, `0001_078_1078204_qa_2` (eval); the two unstated-threshold test tasks
   `0000_455_455476_qa_5`, `0001_043_1043420_qa_4`; plus the three label-mismatch suspects
   `0001_922_1922748_qa_4`, `0013_884_13884693_qa_4`, `0016_567_16567793_qa_4`.
2. **Timeouts (12)** unless the agent cap is raised; they are otherwise legitimate.
3. **Keep but fix the answer spec (7).** The `M`/`Male`, `1`/`1 engine`, `accel. y`,
   `USA`/`United States`, `4th`/`4`, scale-vs-percent cases. These are a prompt and
   normalisation fix, and dropping them throws away tasks the model can actually do.
4. **Keep as-is (rest).** Genuinely nondeterministic model-comparison tasks are fine for measuring
   rungs, since a reference is not required for L1 or the L1+schema control, and their
   reward-0 baseline is honest.

The two big wins worth doing first are the answer-format prompt fix (recovers ~7 tasks) and a
decision on the unstated-hyperparameter family (~15 tasks), which together would put both splits
in the low 90s.