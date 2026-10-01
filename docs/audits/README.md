# Audits

Read-only audits of this repository, kept as written. Each one was produced by analysis over
the results tree in `data/` and the logs, without running a model or modifying any file here;
the report texts are not edited after the fact, so a finding that later work contradicts is
corrected in the list below rather than in place. Scripts the audits ran lived in `/tmp` and
are not in the repo, but the numeric appendix of each report is self-contained and reruns.

## 2026-10-01 — the ladder ([`2026-10-01-ladder.md`](2026-10-01-ladder.md))

Asks whether the information ladder is well defined and whether the tasks are too easy for it
to measure anything, and finds both worries are correct. It traces the headline table in
`LADDER.md` back to a run that no longer exists on disk, measures hint quality per rung
(L2 empty on 65% of test references, L3 adding ~16 characters on median), shows the schema
control leaks the gold answer on 12% of tasks and blows up to a 38.5k-token prompt on some,
and establishes from four independent L1 runs that 18.9% of tasks flip between runs, so
"first passing rung" is not identifiable at one sample per rung. Its failure analysis
attributes 34% of the 63 test L1 failures to non-reasoning causes.

## 2026-10-01 — the numbers ([`2026-10-01-numbers.md`](2026-10-01-numbers.md))

Recomputes every number in the `LADDER.md` headline block from raw per-trial `result.json`,
and withdraws the `test` table in full: no artifact of the 92% control run survives, and the
surviving control shows 23/63 = 36.5% instead. It dates every result file against the logs
that wrote it to show the on-disk tree is one consistent ladder version, traces the
`summarize.py` double-count that made 250 tasks summarise to 266, and identifies a separate
`nrows=50_000` truncation in `synthetic.py` that makes the synthetic gold answers ungradeable
for any table over 50k rows.

## Superseded findings

Findings above that later work on this branch has overturned. The reports keep the original
text; this is the current position.

- **"Grader strictness explains 21 of 63 L1 failures"** (ladder audit, B3). Overstated. A
  regrade of every stored prediction on the split found 15 of 490 graded predictions carrying
  an `ANSWER:` label, and 8 of those flip to a pass when the one prefix is removed. The audit's
  own re-grading had 7 of 21 flipping, but by reformulation rather than by prefix alone, so the
  category conflated our prompt's ambiguity with genuine label-vocabulary mismatches that the
  benchmark itself cannot arbitrate (`NA` vs `North America`, `SP` vs `São Paulo`).
- **"`ANSWER: <gold>` fails to grade on 110 of 116 exact-mode tasks"** (both audits). True as
  stated, and it overstates the practical effect by construction: the measurement evaluates a
  string the model never actually produced. The 15-of-490 figure above is the one that counts
  real stored predictions.
- **"The grader is too strict and should normalise the prefix"** (ladder audit, C2 option 4).
  The grader is not the defect and was not changed. Neither prompt asks for the prefix,
  SmolDataEnvs asks for no prefix either, and its grader's `_normalize` lowercases and collapses
  whitespace and nothing more — so `grade()` already matches the benchmark's definition, and
  loosening it would quietly measure a grader of our own. The fix went into the prompt, which
  now states that the last line is graded on its own and that `"Answer: 42"` grades 0 while
  `"42"` grades 1. `smol_ladder.regrade` measures the label's cost offline and writes nothing.
- **The `summarize.py` double-count** (numbers audit, section 3). Fixed. `first_passing_rung`
  is now a partition over the ladder rungs only, one bucket per task, asserted to sum to the
  task count; the control is reported separately and cannot be one of its buckets, because the
  bucket names are fixed by `BUCKETS`. "Tried and failed" and "never tried" are now distinct
  buckets, and harness failures are counted and excluded from pass rates rather than scored as
  model failures.
- **The bogus `Method: join` on 31% of test references** (ladder audit, A3). Fixed. `method_hint`
  reads the parse rather than a regex: a method name is an operation when its receiver holds
  data and string formatting when the receiver is a literal or a path, so `os.path.join` and
  `", ".join(...)` are no longer read as dataframe merges. Measured over all 181 test
  references, bogus `join` goes 60 → 0, empty columns 118 → 58, empty filters 177 → 129.

Findings the later work has *not* addressed are still open and are not listed here: the schema
control's 12% answer leak and its uncapped prompt size, the `nrows=50_000` synthetic gold
truncation, the 40 turns/timeout share of failures, and the 8-of-181 L1-pass/L2-fail
monotonicity violation.
