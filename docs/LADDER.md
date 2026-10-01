# The information ladder for data-analysis tasks

## What we found first (SmolDataEnvs `test`, 250 tasks, space-bunny-alpha)

**The table this section used to open with is withdrawn, and the headline with it.** It reported
L1 184/250 = 73.6% and an L1+schema control of 165/179 = 92.2% of L1 failures, and concluded that
"the failures on this benchmark are overwhelmingly an exploration problem, not a missing-information
problem."

Three reasons, in the numbers audit's words ([`audits/2026-10-01-numbers.md`](audits/2026-10-01-numbers.md)):

- **No artefact of that run survives.** The results it summarised were deleted in the regrade; the
  tree is not on disk, so no figure in it can be recomputed from per-trial results.
- **Its denominator is impossible against its own numerator.** The section gives L1 as 184/250,
  which leaves 66 L1 failures, and then reports a control denominator of 179. 184 + 179 = 363 of 250
  tasks. The source of 179 is unknown and we do not guess it.
- **The figure that survives does not support the conclusion.** On the same split the control
  rescues 23 of 63 L1 failures = 36.5%, a minority. "Overwhelmingly an exploration problem" rested
  entirely on the 92% number and goes with it.

Withdrawn with it: `L1 184/250`, `L2 9/14`, `L3 2/5`, `L4 3/3`, and the "of 19 climbed tasks"
summary. The audit's superseded-findings table lists each one against its current replacement.

### What the surviving legacy tree shows

Recomputed on 2026-10-01 by running the summarise tool over the legacy `data/runs/test` tree
(`python -m smol_ladder.summarize --split test`). Quoting only what it prints:

| | |
|---|---|
| L1 | 76.2% [70.9%, 81.6%], 244/250 trials scored, 6 harness failures |
| L1+schema (control, not a rung) | 23/63 = 36.5% of L1 failures, 7 harness failures |
| control with a reference | 20/31 = 64.5% |
| control without a reference | 3/32 = 9.4% |

First passing rung, a partition over the ladder rungs with the control excluded, one bucket per
task, summing to 250:

| L1 | L2 | L3 | L4 | never | not climbable (no reference) | not scored |
|---|---|---|---|---|---|---|
| 186 | 17 | 3 | 3 | 7 | 28 | 6 |

Reference funnel for this run, as the tool reports it: 250 tasks in the split, 213 with a reference
in the source, and 181 that are actually measurable at L2–L4 — so every L2–L4 figure below is
conditional on that 181. The 28 "not climbable" tasks never had a rung above L1 run on them (the
climb stopped before the last 32 references existed), so they were never tested
against an information rung; they are not a case of the model needing more information, and they
are never booked as "never".

Two caveats that bound all of the above, both of which the tool reports itself:

- **k = 1.** Every rung has a single stored trial per task, so these are one sample per rung, not
  pass probabilities over a distribution. Four identical L1 runs on this split spanned 71.5–74.8%,
  and 18.9% of tasks flip between runs, so the first-passing-rung buckets at k=1 are not
  identifiable findings. The one number the tool flags as robust is that 0 of 244 tasks pass on a
  fraction within 0.25 of the majority line at L1, so no bucket there is a coin flip.
- **Climbing.** Rungs were run until the first pass, so the per-rung denominators are not
  comparable to each other as independent pass rates: L4's tasks are by construction the ones the
  control and L2 both failed.

### A clean re-measurement is in progress

The section above is the best reading of the legacy tree, not the study's result. Run tag `v2` is
measuring this again from scratch: every rung on every task (`--no-climb`, no climbing), 2 samples
per rung, the plain-language hints, and an answer-free schema control. It will replace this section
when it lands. Until then nothing here should be quoted as a finding about the ladder.

## What a rung is

A task has a gold answer y* and tables E, which sit in the sandbox at every rung. Each rung Lk is
a signal s_k, so at rung k the agent sees (s_k, E). The ladder is **Blackwell-ordered** because
prompt Lk contains prompt L(k-1) verbatim: (s_{k-1}, E) is a function of (s_k, E), a garbling.
By Blackwell's theorem, a Bayes-optimal agent's **expected** payoff is then weakly higher at
higher rungs, for every decision problem.

### Two requirements the ordering actually depends on

The post states the ladder as *"each level contains everything in the one below"* and *"holds
the task fixed and changes only what the solver sees."* Both are load-bearing, and we violated
both before `tests/test_blackwell.py` existed to check them.

1. **Garbling, not adjacency.** L1 must be recoverable from every L(k) by dropping the added
   block. Our L4 did not contain L3 — it replaced L3's block instead of appending — so L4 was a
   sibling of L3 and "the lowest rung that passes" stopped meaning "the least information that
   sufficed". Now a test asserts `L(k).startswith(L(k-1))` for every pair, and that L(k) adds a
   *suffix*.

2. **The task includes the environment.** L1 listed the dataset row's `files`, which is what the
   original notebook happened to use, not what is in `./input`. On 56% of synthetic and 28% of
   SmolDataEnvs tasks L1 announced one file while `ls ./input` showed several. That is a prompt
   asserting something false about the solver's own environment, and every rung inherits it. It
   is also not neutral: naming a *subset* of the files is a hint about which table the question
   is about, which is exactly the information the schema control exists to isolate. L1 now reads
   the directory and lists what is actually there.

   This also fixes a labeling lie. L2's block is "Files read / Columns used / Filters applied",
   but L1 already gave the file names, so L2 added only columns and filters. The rungs were not
   a partition of increasing information. L2 is now honestly the first rung that says anything
   about *the computation*.

A third requirement, from the same post and inherited from its Go ladder: the rungs of one task
nest by design, so a train/eval split must keep all of a task's rungs on one side. We never
train on these, but any future use of the ladder as a curriculum inherits the constraint.

Three things the theorem does not give us, and how the analysis handles each:

1. **It is about expectations, not single tasks.** Even a Bayes-optimal agent can do worse on a
   particular task at a higher rung. So monotonicity is a claim about **aggregate pass rates**.
   A single task that passes at Lk and fails at L(k+1) is not a violation by itself. It becomes
   evidence of one only when it happens systematically across tasks or samples.
2. **It assumes a Bayes-optimal agent.** An LLM is not one. Drops in aggregate pass rate at a
   higher rung are the real **monotonicity violations**. We report them as results.
3. **It says nothing about costly information.** Blackwell compares what the agent knows, not
   what it costs to find out. The next section is built on this gap.

Metric: the lowest rung at which the model first passes (or "never"), computed from k samples
per rung, never from a single greedy sample. The headline numbers are aggregate pass-rate
curves by rung.

## Every rung adds information

A rung counts only if it tells the agent something it could not compute from E. A schema dump
(columns, dtypes, sample rows) fails this test: it is a function of E, so it would be
Blackwell-equivalent to the rung below and would add no information. It is therefore **not a
rung**. It runs as a separate control condition (below).

The ladder is strictly ordered: L1 < L2 < L3 < L4.

| Rung | Adds | Source |
|---|---|---|
| L1 | question + file names (the normal prompt) | dataset row |
| L2 | L1 + which files and columns the computation uses, and which filters it applies | verified reference solution |
| L3 | L2 + the method, e.g. "count rows per value of X, take the mode" | verified reference solution |
| L4 | L3 + the full reference `solution.py` with its final print removed | verified reference solution |

### L2 and L3 are written in plain language, and validated because of it

L2 and L3 were extracted statically from the reference's AST. That is deterministic but it is
mostly empty: 98/181 test references are hand-written csv or sqlite code rather than pandas, so
the filters came back empty on 71% of them and L3 read "get, items, values" — a bag of method
names, not a method. So the hints are now written by a model, from the verified reference, in
plain language, and cached to `data/hints/<split>/<task_id>.json` with the model, the prompt
version and a hash of the reference, which makes a rerun resumable and deterministic once
generated.

That trades a guarantee away, so it is bought back with validation. Every hint is checked before
it is used, and a hint that fails is regenerated up to three times before the task falls back to
the AST extraction:

1. **No hallucination.** A named file must exist in `./input` and a named column in that table's
   header, including the `table_info` of a sqlite file (13 test references are hand-written SQL).
   A fabricated name is dropped and counted; the ladder would otherwise be asserting something
   false about the solver's own environment, which is the failure mode requirement 2 above.
2. **No leak**, by the same three checks as everything else in section "Construction rules": no
   gold answer by text match, no numeric literal the dataset's grader accepts, and the hit must
   be one L1 does not already have.
3. **L3 must add something.** A method that only restates L2's columns and filters is not a rung.

On the 181 verified test references: 167 (92.3%) got a usable hint and 14 fell back to the AST.
The share of tasks with non-empty content went from 62.4% to 88.4% for columns, from 28.2% to
56.9% for filters, and from 66.9% to 92.3% for the method. Every failed hint failed on a leak
rejection. The cost is context: 506 characters added at L3 against 114 for the AST, which is
exactly the budget the monotonicity discussion below is about, and it applies to these rungs
more than it did to the old ones.

Each rung carries information about which computation the question intends, and none of it is
recoverable from the tables alone. How to read the first passing rung:
- **L1**: the model solves the task as posed.
- **L2/L3**: an information failure. The question underdetermines the intended computation
  for this model, or the model cannot turn the question into it.
- **L4**: an execution or format failure. The model had the program and still failed.
- **Never passes, even at L4**: a harness bug or a broken task. Inspect by hand.

## Exploration control (not a rung)

**L1+schema**: the L1 prompt plus a schema dump of the relevant files, generated by script and
identical for every model. The dump is a function of the tables, so for a Bayes-optimal agent
it is Blackwell-equivalent to L1 and worth nothing. A real agent is not Bayes-optimal, and we
cannot separate *why* it gains from the dump: it may be that reading E is expensive (tool
calls, a turn cap, limited context, mistakes on the way), or that the dump states the
answer-relevant column names in a directly attendable form, or that it removes distractor
files. Only the first of those is an exploration cost; the other two are informational in
effect, even though the information is a function of E. So the honest claim is: **a gain here
is a processing gain, not evidence that the task was underdetermined.** We run it on the
tasks that fail at L1.

Measured on the legacy tree, that gain is 23/63 = 36.5% of L1 failures, split by reference status
as 20/31 with a reference and 3/32 without. That is a minority of failures, not an overwhelming
share, and it is not the strongest result in this study: it rests on 63 attempts at k=1 on a tree
built by a ladder version that has since changed, and the per-rung denominators there were produced
by climbing. It is the number a clean re-measurement has to beat or explain, not a finding. The
checks that the control is not simply leaking remain worth stating, because they are what makes
the number interpretable at all: the dump is generated by script from the tables alone, never from
the reference solution or the gold answer; a column whose name equals the answer is replaced before
the dump is built; and it is identical across models by construction.

RQ4 then reads: does training lower the first passing rung (information), close the L1 vs
L1+schema gap (skill), or both?

## Monotonicity is not free, and we treat violations as results

Blackwell's theorem compares information sets. It says nothing about a **budget**: a real
agent has 40 turns and a context window, and a rung adds text that must be read before the
work can start. So higher rungs can genuinely lower the pass rate, and L3 scoring below L2 is
that effect (small n here; the mechanism is real and expected).

We therefore report monotonicity violations rather than explaining them away: for every
adjacent pair of rungs, the fraction of tasks that pass at k and fail at k+1, with an exact
binomial test. A drop that survives at 2x the turn cap is about the *content* of the hint
(its style, its length, its leaks). A drop that disappears at 2x the cap is about the budget.
Either way it is a finding, and the current numbers do not separate the two.

## Task sources

The ladder needs more tasks than SmolDataEnvs can supply: 250 in `test`, 213 with a reference
and only 181 with a rung above L1 actually run, and — as above — a third of the L1 failures left
after the control. The second source is
**jupyter-agent** (51,389 rows). It does *not* share no dataset with SmolDataEnvs: that was the
old claim and it was wrong in both directions, see "The overlap firewall was wrong" below. The pool
is much noisier, and the work is in the filter:

- `executor_type == "e2b"` only; the `llm` rows have simulated outputs, so their answers are
  fiction. That is 66% of the rows.
- The answer must reduce to a number, a short label, or yes/no. jupyter-agent answers are
  sentences — "61,257 USD (70,187 for failed minus 8,930 for successful)" — and SmolDataEnvs'
  grader needs a value plus a `reward_mode`. Over 1,704 e2b rows: 945 numeric, 324 label,
  43 bool, 388 rejected.
- Rejected even though they look gradable: "Not explicitly stated in the notebook outputs"
  (the dataset recording that no answer exists, which matches a label pattern); answers with a
  unit or a parenthetical derivation; answers that restate the question with a count.

Numeric tolerances come from each answer's own printed precision, floored at 1e-4 relative and
capped at 0.05. The floor matters: an answer stored as `8.89663104713` with a 1e-12 tolerance
marks a model that printed `8.896631` wrong, which measures the grader, not the agent. The cap
matters too, or an integer `453` gets a whole-unit tolerance and admits `453.4`.

### The v2 pool

The first extract stopped at 2,000 tasks after 8 of 103 shards, and shipped rows carrying only a
question and an answer. `data/jtasks_v2.jsonl` keeps the same filters and the same `ja_<slug>` id
rule, so **every one of the 2,000 v1 task_ids is still present**, and adds the tags the audit said
the pool was missing. Nothing is deleted; the subset is a selection over tags.

| Stage | count | |
|---|---|---|
| rows in all 103 shards | 51,389 | |
| `executor_type == "e2b"` | 29,561 | the `llm` rows have simulated outputs |
| excluded: SmolDataEnvs dataset overlap | 17,233 | by `kaggle_dataset` slug, 122 banned |
| gradable answer | 9,187 | numeric 6,932 / label 2,007 / bool 248 |
| of which ungradable | 3,141 | units, derivations, "not explicitly stated" |
| **available beyond v1** | **7,187** | 4.6x the extract on disk |

**The overlap firewall was wrong in both directions, and `data/jtasks_v3.jsonl` is the corrected
pool.** v1 and v2 dropped any row whose Kaggle dataset appeared in SmolDataEnvs *at all*. That
over-fired, because `train` is not held out: 5,000 rows of the same public Kaggle corpus, and a
jupyter-agent task sharing a table with it measures nothing the ladder has already scored. And it
under-fired, for two reasons that are worth separating because only one is a bug in the code.

*The 122-vs-471 discrepancy is a reporting error, not a matching failure.* The docs describe
SmolDataEnvs as "built from 471 Kaggle datasets", and 471 is exactly the size of its **train**
split. The code banned only the **test** split's 122. `eval` adds 81, and the union over all three
splits is 526. So the two numbers were never two counts of one set; they were two different
splits. Nothing was mis-matched, the exclusion just covered a third of the corpus it claimed to.

*What did miss overlaps is the identity rule.* Kaggle identifies a dataset as `owner/name`, and
both sources spell it that way, so comparing slugs looks right. But the same upload is routinely
re-uploaded under another owner, and jupyter-agent's own `files_used` paths carry the bare name
(`kaggle/input/pokemon/...`), not the owner. Full-slug matching therefore missed **121 tasks in
the shipped v2 pool** that reach a held-out SmolDataEnvs table through a mirror — `abbasit/titanic`
against SmolDataEnvs' `mhouellemont/titanic`, `alopez247/pokemon` against `abcsds/pokemon`, and
five more bare names. Added to **1,548 tasks on an `eval` slug the old rule never banned at all**,
that is **1,669 of v2's 9,187 tasks (18%) built on a table SmolDataEnvs `test` or `eval` has already
scored** — the contamination the firewall existed to prevent.

v3 fires on `test` and `eval` only, and matches on the bare dataset name, so both errors close at
once: a mirror of a held-out table is caught, and a task sharing a table with SmolDataEnvs `train`
is kept and tagged `sde_overlap: "train"` / `shares_table_with_sde_train: true`. The trade is
deliberate and stated in `smol_ladder.jtasks.dataset_key`: a false positive costs one train-pool
task that was never held out, a false negative ships a task whose gold answer is already in the
ladder's own results.

| Stage | v3 count | |
|---|---|---|
| rows in all 103 shards | 51,389 | |
| `executor_type == "e2b"` | 29,561 | the `llm` rows have simulated outputs |
| excluded: table held out by SDE `test`/`eval` | 19,407 | 169 held-out bare names (170 slugs) |
| kept but tagged: shares SDE `train` | 8,911 | train-only, 343 bare names |
| gradable answer | **7,518** | numeric 5,660 / label 1,658 / bool 200 |
| of which ungradable | 2,636 | units, derivations, "not explicitly stated" |
| ladder-grade (`is_ladder_grade`) | 4,217 | 860 of them in the hard families |

v3 is smaller than v2 (7,518 vs 9,187) even though it is *less* strict about which split counts,
because the identity fix is worth more rows than the split change is: 19,407 executed rows are
excluded, against 17,233 before. Every task in v3 is a task that was already in v2, byte-identical
apart from the two new fields, so no existing result is invalidated and no `task_id` moved. v1 and
v2 on disk are untouched; `python -m smol_ladder.jtasks_v2 --v1-compatible` reproduces
`data/jtasks_v2.jsonl` exactly, which is what pins the comparison.

Tags, all pure functions of the row's own text and files, so the pool tags identically on every
rebuild: `op_family` (ordered regex, first match wins), `answer_type`, `nondeterministic` with its
reasons, `ambiguous` with its reasons, `n_files`, `input_bytes`, and `sde_overlap`.

The two flags come straight from the failure analysis, not from taste. `nondeterministic` fires on
model fits, seeds, sampling and train/test splits — the cases where the gold is one draw from a
distribution the question does not pin, so a correct agent scores 0 and no information rung can
fix it. `ambiguous` fires on two independent things: a label answer whose own words never appear
in the question (gold `North America`, agent says `NA`), and a question that defers to a criterion
it never states ("based on correlation analysis", "the threshold that separates ...").

| tag | value |
|---|---|
| `op_family` | ml_fit 2,528 · count 2,306 · agg 1,260 · stat_test 786 · string 730 · filter 708 · argmax 389 · other 276 · groupby 166 · lookup 21 · join 17 |
| `answer_type` | numeric 6,932 · label 2,007 · bool 248 |
| `nondeterministic` | 2,286 (model fit 2,063, random sampling 433) |
| `ambiguous` | 2,179 (label vocabulary 1,610, unstated threshold 449, unjudged comparison 335) |
| `n_files` | 1: 7,562 · 2: 1,196 · 3: 224 · ≥4: 205 |

**Ladder-grade subset** — a task that is not nondeterministic, not ambiguous, and has files:
**5,124 of 9,187** (55.8%). That keeps 1,102 of the 2,000 v1 tasks, so it is a selection and not
a repopulation. Its family mix is count 1,776 · agg 841 · string 617 · filter 544 · stat_test 495
· ml_fit 410, i.e. it deliberately keeps the method-ambiguous families (1,055 hard-family tasks)
and does not tune towards count/agg, because those are the families the ladder has nothing to
disambiguate.

Inputs are priced, not fetched: **8,668 of 9,187 tasks (94.3%) reuse datasets already in the
Kaggle cache**, and the remaining 519 tasks across 148 uncached datasets need ~69 GB. No bulk
download was attempted; sizes come from Kaggle's dataset-view metadata endpoint (~4 KB per
dataset) and are cached in `data/jl_dataset_sizes.json`, because that endpoint rate-limits and a
rebuild without the cache reports a *smaller* download cost each time more lookups 429.

### Synthetic: gold by construction, and the gate it needed

A third source removes the yield gate that limits the other two. We pick a table, instantiate a
*specification* over it, and execute the specification to get the answer, so the gold is correct
by construction and the population is not conditioned on the model. Each task records the ops it
applied, and those ops are also the L2–L4 reference: the rung and the gold come from one
implementation, so they cannot describe different computations.

Two rules borrowed from the openswe pipeline, the only prior art here that took solvability
seriously. **ARBITRARY vs DERIVABLE**: a task is only fair if a competent analyst holding the
question would produce the graded answer; a question whose answer depends on a tie-break we chose
is dropped rather than published. **Prove it three ways**: the answer must be computable, the
question must not name the answer, and an independent execution of the spec must agree.

The third proof is the shipped-file gate, and it is what this source had been missing. It
re-executes each task's own reference against the shipped file, through the real grader, and
refuses to emit a task that does not grade 1.0. Applying it found **two independent defects, and
both of them made a task unpassable rather than merely hard**. The split between them — 38 of the
64 failing L4 tasks to the truncation and 26 to the print/tolerance bug — is from the pre-fix
sweep and cannot be recomputed from what is on disk now, since the 275-task old corpus has been
replaced; what *is* on disk is the 757-trial regrade below, which is the number that matters:

1. **The gold was computed over a different table than the one shipped.**
   `iter_tables()` read each candidate table with `nrows=50_000` and wrote the *untruncated* file
   into `data/inputs`, so on any table over that size the answer was computed over the first
   50,000 rows while the agent summed all of them. That sweep found 58 of the 275 tables over the
   cap. Concretely, `syn_runs_2_sum` asks for `sum(horse_no)` on `runs.csv` (79,447 rows):
   the gold was 345296, the sum over the shipped file is 548631, and the model's 548631 was
   marked wrong.
2. **The gold was printed lossily against a tolerance too tight to survive it.** `_fmt` rendered
   every float as `.6g`, and `_tolerance` read off the *shape* of that string and handed a float
   a 1e-6 relative tolerance. For an answer of more than about seven digits the two are
   inconsistent: the printed gold differs from the true value by more than 1e-6 relative, so the
   grader rejects **the task's own correct answer**. `syn_Crime1_2_sum` asks for `sum(Y)` over
   835 rows; the gold was `31535.6`, the sum is `31535.63530029`, and 3.5e-5 relative error
   against a 1e-6 tolerance is 0.0. None of these tasks is on a table large enough for the
   truncation to apply — `syn_ca_offenses_by_campus_0_mean` has 54 rows and its gold `10.9388`
   likewise excludes the true `10.938775510204081`.

The fix is at the root in both cases: read the shipped file with no row cap, print the answer
exactly with `repr`, and derive the tolerance from the printed string rather than from the op, so
the grader accepts the gold as its own prediction. `verify_shipped()` then enforces it end to
end, and `tests/test_synthetic.py` covers each cause individually.

Regrading the stored trials against the corrected gold (`python -m smol_lader.regrade_gold
--split synthetic`, which reads `result.json` and never writes it) is what exposed the scale.
Re-measured 2026-10-01, quoting the tool:

| rung | trials | pass, old gold | pass, corrected gold | flipped |
|---|---|---|---|---|
| L1 | 275 | 179 | 213 | 44 |
| L1+schema | 275 | 197 | 242 | 57 |
| L2 | 78 | 13 | 73 | 62 |
| L3 | 65 | 1 | 61 | 62 |
| L4 | 64 | 0 | 62 | 62 |

287 of the 757 stored trials moved. The old L3 and L4 curve — 1/65 and 0/64, read at the time as
a strong model result on the hardest rung — is 61/65 and 62/64, so **that result was an artifact
of the gold, not a measurement of the model.** The old rewards are kept on disk unchanged; the
regrade writes a separate `data/runs/regrade_gold_post-goldfix.jsonl` carrying `reward` beside
`reward_old` and the gold each row was graded against, so the before/after table above compares two
golds and not two differently-populated samples.

A correction to what was said here before that regrade was run on the shipped corpus. It was
written that the tool grades "the 136 trials whose id the corrected corpus dropped" against the
superseded gold, which was the case it was built for. On this corpus it does not: the committed id
cache carried **all 275** published ids forward, so every stored trial has a corrected gold and all
757 rows are graded against it (`gold_source` is `corrected` on every row). The superseded-gold
branch is still there and still correct, because a corpus regenerated without
`smol_ladder/synthetic_ids.jsonl` would drop those ids — but that has not happened yet.

That regrade also says what the run can no longer answer. The climb stops at the first pass, so a
task whose pass the corrected gold reverses was sent up rungs it should never have reached, and a
task that now passes earlier never saw the ones in between. **87 rung-trials across 78 tasks were
never run and are owed** (28 at L2, 29 at L3, 30 at L4); no regrade can recover a prediction that
was never produced. The `synthetic` split's own numbers are therefore a measurement of a broken
gold and are superseded by the table above, which is a measurement of a corrected one.

The only `synthetic` figure quoted as a result of the source itself is the corpus it produces:
**6,956 tasks over 42 tables**, every one of which passed the shipped-file gate. That corpus is
almost entirely unmeasured — the stored `data/runs/synthetic` tree holds 275 tasks, so **6,681 of
the 6,956 have no trial at all** — and the rung table above is a regrade of one 275-task run, not
a curve over the corpus.

## Construction rules

1. **Cumulative.** Lk = L(k-1) + a new block. Nothing is reworded or dropped. This is what makes
   the Blackwell ordering hold.
2. **Deterministic control.** L1+schema is generated by a script from the tables, the same for every model.
3. **L2–L4 come from one verified solution per task:** the space-bunny `solution.py` that
   reproduced the gold answer in the offline sandbox (`smol_ladder/gen_solutions.py`). Tasks without
   one get no L2–L4, and we report that count.
4. **No leaked answers.** No hint may contain the gold answer. Three checks, because each
   catches what the others miss:
   - *Text*: normalised substring match, plus every numeric literal in the hint put through the
     dataset's own grader, so the test asks exactly the question the reward will ask. Skipped
     for answers under 4 characters, which match by chance.
   - *Differential*: a hit only counts if L1 does not already have it. The `task_inherent` bucket
     exists for that case, and the comment on it used to explain it with "52/250 test tasks are
     multiple choice, where the answer is one of the options in the question". **That is wrong by
     two orders of magnitude and the claim is withdrawn.** Re-measured on all 250 `test` questions
     on 2026-10-01 across lettered options, "which of the following", the words *options* and
     *candidates*, and the `, or ` alternative shape: 1 task by a strict lettered-options detector
     and 4 by the loosest shape, and all four are open questions that merely enumerate inline
     categories ("Which wetland category (L, P, or R) has the highest mean circularity?"). The
     longest question in the split is 211 characters. `task_inherent` is therefore a property of
     something other than multiple choice — 13 per `ladder_test.log` — and the comment has been
     corrected to say so rather than left asserting a reason we cannot reproduce.
   - *Execution* (`smol_ladder.ladder.leak_free`): strip every print/logging call and every
     docstring from the reference, then **run** the payload offline and re-grade it. A string
     match cannot see a program that names the answer in a label map, a ternary, or a
     threshold. A task whose L4 still grades 1.0 is **excluded** from L2–L4, not repaired.

   Removing only the final top-level print — the obvious first implementation — left 25/181
   (13.8%) of test payloads still emitting the gold answer when run, because agents use prints
   as debug output and emit a leaderboard or a per-value sweep before the final line. The
   current `strip_output` removes output at every AST depth and the oracle re-checks.
5. **One path, not the only path.** The reference solution is one correct θ. Hints describe it,
   so L2+ measures "can the model follow this path", not "is this the only interpretation".
   State this limitation in the write-up.
6. **Held-out only.** Reference solutions come from `test`/`eval`. They never enter training data.

## The funnel is a result, not a footnote

L2–L4 exist only where a strong agent produced a solution that reproduced the gold answer. The
reference count moved and that is a fact about disk today, not a correction: the summarise tool
reports **213/250 = 85%** with a verified reference, because the reference sweep was re-run after
the numbers audit was written, and 181 of those are the tasks whose trials exist. **Every rung
number we report is conditional on that**, so the funnel is reported first and repeated in every
caption:

| Stage | test | What it means |
|---|---|---|
| tasks in split | 250 | |
| reference present in the source | 213 | as of 2026-10-01, per the summarise tool |
| L2 trials actually on disk | 181 | the climb ran before the last 32 references existed, so 8 of the 63 L1 failures with a reference have no rung above L1 |
| L4 excluded by the run-and-grade oracle | unverifiable | `data/ladder/` holds only `_leakcheck`; no per-rung prompts or `meta.json` survive for any split, so this row cannot be recomputed from disk |
| usable ladder tasks | 181 | L2–L4 measurable; L1 and the control measured on all 250 |

Two consequences, stated up front rather than buried:

- The **69 tasks with no usable reference** (250 − 181; of which the legacy run books 28 as
  `not climbable` and 6 as `not scored`) are **not** "tasks where the model needs more information".
  They are tasks where no rung above L1 was ever run, and they get their own row in every table.
  They are never counted as "never passes" on the ladder. Two earlier versions of this file gave
  47 and 69 here and 47 for tasks with no reference at all; that earlier denominator is 37 today
  (250 − 213) and the ladder-measurable one is 69, so the number the confusion was reaching for
  never depended on which was quoted. Both are withdrawn in the audit.
- "Verified" means *the grader accepted this answer*, not *this is the unique correct
  computation*. On tasks with a loose `rtol` a materially different method also grades 1.0. So
  L4 does not certify correctness; it certifies one accepted path. Construction rule 5 already
  said this, and the numbers here are why it matters.

## Known contamination, measured

- **74% of `test` tasks share a source table with a `train` task** (185/250 by `bucket_prefix`),
  and 4 `test` questions appear verbatim in the released `SmolDataEnvs-sft` blob, two with the
  same answer. The split is by question, not by table. Arms fine-tuned on SmolDataEnvs-sft
  therefore see the tables the L4 hint is written about. Every cross-arm rung comparison must
  report this; the clean version partitions the 471 Kaggle datasets (SmolDataEnvs `train`;
  `test` and `eval` use 122 and 81, 526 in union), not the 5,394 tasks. This is the same
  train-is-not-heldout distinction the v3 jupyter-agent firewall turns on.
- Reference solutions are generated by the same model family under evaluation, so "training
  lowered the rung" cannot be separated from "the model recognises its own teacher's code"
  without a shuffled-reference control (a reference from a *different* task, same table).

## Deliberately excluded

- A "partial description" rung. Which part to withhold is a free choice, so two such prompts are
  not comparable. We dropped this rung from the Go ladder for the same reason.
- Rungs that add text a model could use without it being information (motivation, style tips).
  They would break the "each rung is a signal about θ" definition.
