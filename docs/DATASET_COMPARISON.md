# Dataset comparison: SmolDataEnvs vs. our jupyter-agent pool

Measured 2026-10-02 on branch `docs/compare`. Every number here is recomputed by
`smol_ladder/compare_sets.py` and `smol_ladder/judge_quality.py`; the raw output is
`data/compare/overlap.json` and `data/compare/judge.json`.

**Sets compared.** A = SmolDataEnvs `train` (5,000 tasks, 460 Kaggle tables, 5,000 distinct
`source_row_id`s). B = `data/jtasks_v3.jsonl` (7,518), its ladder-grade subset (4,217), and the
2,111 of those that passed L1 in the `ja3` transcript sweep — the set that would actually be
trained on.

---

## Headline

**B is more of the same corpus, not different data — but it is not the same tasks, and the
subset that gets trained on is the most SDE-overlapping part of the pool.**

The four ways to ask "do these overlap" disagree by a factor of five, so the answer depends
entirely on what is matched:

| match on | pool | ladder-grade | ja3-passing (trained on) |
|---|---|---|---|
| Kaggle dataset name | 88.1% | 88.7% | **94.9%** |
| exact input file set | 87.5% | 88.1% | **94.2%** |
| source notebook row (`source_row_id`) | 18.5% | 21.2% | **32.7%** |
| normalised question text | — | 22.5% (934/4,151) | — |

Two things fall out of that table. First, **B shares 88% of its tables with A but only 19% of its
source rows**: most of B's tasks are new questions over tables A has already seen, not new tables.
Second, **the tasks we would train on are the most A-contaminated part of the pool** — the ja3
passing set is 94.9% on shared tables against 88.1% for the pool as a whole. Selection, not
chance, pushed it there: a task the model already solves tends to be a simple task, and simple
tasks concentrate on the crowded, well-covered tables.

---

## 1. Table and notebook overlap

### Tables

Both sides reduce the Kaggle slug to its bare name (`jtasks.dataset_key`), so `owner/a` and
`mirror/a` collide — that is the firewall's own rule and the right one for "same table".

| subject | on an SDE-train table |
|---|---|
| pool (7,518) | 6,625 — **88.1%** |
| ladder-grade (4,217) | 3,740 — **88.7%** |
| ja3-passing (2,111) | 2,003 — **94.9%** |

The reverse direction: **339 of A's 460 train tables (73.7%) also appear in the pool.** So the
overlap is not one-sided — A's table corpus is largely a subset of what B reaches, and B's
extra 1,777 tasks (11.9% of the pool) sit on 121 tables A never touches.

File-set identity is reported alongside because the dataset *name* is metadata and gets it wrong:
`kwullum/fatal-police-shootings-in-the-us` also ships `PercentagePeopleBelowPovertyLevel.csv`, so
a question about poverty rates is correctly attributed but looks mislabelled. The file-set numbers
(87.5% / 88.1% / 94.2%) are within half a point of the name-based ones, so the metadata is not
misleading in aggregate — but the exact-file-set test is the one that matches what the agent
actually sees, and it is the stricter of the two (it requires the whole set to match).

### Source rows

A's rows carry `source_row_id` in raw jupyter-agent form (`0000/324/324276.ipynb_qa_3`) and B's
carry the slugified form (`ja_0000_324_324276.ipynb_qa_3`); `source_row_key` collapses both.

| subject | derives from an SDE-train source row |
|---|---|
| pool (7,518) | 1,392 — **18.5%** |
| ladder-grade (4,217) | 895 — **21.2%** |
| ja3-passing (2,111) | 690 — **32.7%** |

**690 of the 2,111 tasks that would be trained on are literally the same jupyter-agent row as an
SDE-train task** — the same notebook, the same table, frequently the same question. That is not a
statistical resemblance; it is row-level duplication. A+B trained on both will see those 690
questions twice.

---

## 2. Question overlap within shared tables

### Exact match after normalisation

Normalisation lowercases, strips markdown/maths quoting, and collapses punctuation and
whitespace. Over the 4,217 ladder-grade questions (4,151 distinct) against A's 4,907 distinct
questions:

| | count |
|---|---|
| shared question text | **934** |
| …of which on the same table | **921** (98.7%) |
| …of which on a different table | 13 |

934/4,151 = **22.5%** of B's distinct questions are the same string as an A question. The 98.7%
same-table concentration is the point: this is not two collections that happen to phrase similar
questions differently, it is the same question recorded twice.

### Near-duplicates

Thresholds: **token-set Jaccard ≥ 0.8 AND character 5-gram Jaccard ≥ 0.6**, both required. Each
measure fails alone — token Jaccard calls *"How many rows are in the table?"* and *"How many rows
does the table contain?"* identical, while 5-gram Jaccard misses a pure synonym swap. The bar is
read off the measured distribution (177,110 candidate pairs share a content word; the counts fall
off a cliff between 0.9 and 0.6):

| threshold | pairs |
|---|---|
| J ≥ 0.9, N ≥ 0.8 | 37 |
| **J ≥ 0.8, N ≥ 0.6** | **313** (188 on the same table) |
| J ≥ 0.7, N ≥ 0.5 | 841 |
| J ≥ 0.6, N ≥ 0.4 | 2,426 |

No embedding model was used: none is cached on this machine and downloading one was out of scope.
**This is a floor.** Both measures are surface-based, so a synonym swap is missed whenever it is
the only difference — *"average"* against *"mean"* scores 0.5 token Jaccard, *"distinct"* against
*"unique"* scores 0.4. On five-word content sets one swapped word costs a third of the Jaccard, and
no threshold catches that tail without also catching unrelated questions on the same table.

**10 near-duplicate pairs** (all J = 1.0 token Jaccard; `same_table` as marked):

| A (SDE) | B (pool) | same table |
|---|---|---|
| difference in average monthly hours between employees who **left** and those who **stayed** | …between employees who **stayed** and those who **left** | yes |
| what percentage of the **`total_bedrooms` column** contains missing values | what percentage of the **dataset** contains missing values in the `total_bedrooms` column | yes |
| correlation between **median income and median house value** | correlation between **median house value and median income** | yes |
| correlation between `median_income` and `median_house_value` | correlation between median house value and median income | yes |
| highest correlation between any feature and **the** median house value | highest correlation between any feature and median house value | yes |
| how many missing values **were** present in `'Checking account'` before imputation | …**are** present … | no |
| standard deviation of **the** global sales values | standard deviation of global sales values | no |
| correlation between **the** poverty rate and **the** high school graduation rate | correlation between poverty rate and high school graduation rate | yes |
| missing values in the **`open`** column (lowercase) | …the **`Open`** column (capitalised) | no |
| correlation between **the** poverty rate and **the** high school graduation rate | …(article dropped) | yes |

Every one of these is the same question. Seven of the ten differ only in article presence or word
order — the kind of difference a sentence-level normaliser would erase and a token-set measure
cannot. Two are the *same* question across two different uploads (an upper/lower-case column name
difference in the file), which is the mirror case the bare-name rule deliberately over-matches.

### 10 pairs on the same table that are clearly different questions

Sampled one pair per table so ten examples cannot all come from one busy upload.

| table | A (SDE) | B (pool) |
|---|---|---|
| us-baby-names | third most used female name in 2002 | most gender-ambiguous name in 1995 |
| open-exoplanet-catalogue | which habitable-zone exoplanet is nearest Earth | how many stars are Sun-like by mass/radius |
| sf-salaries | median TotalPay for FT employees | how many employees are PROGRAMMER ANALYST |
| twitter-user-gender-classification | which model had highest accuracy on tweet text alone | most common gender at 100% confidence |
| water-consumption…-city | counts per missing-value category | proportion with no missing values |
| german-credit | which gender has higher average credit amount | are male entries more than double female |
| extinct-languages | most common degree of endangerment | which endangerment category has most languages |
| cardataset | is there a positive horsepower/price correlation | MAE of the model's price predictions |
| nips-papers | missing values in `event_type` | which author has most papers |
| video-game-sales-and-ratings | chi-square p-value for genre × completion | genres with declining median sales after 1995 |

Jaccard on these runs 0.00–0.36, median 0.11. This is what "different question, same table"
looks like, and it is the majority of what the 88% table overlap consists of.

### Same question, different gold answer

**66 pairs** (47 on the same table). Fifteen of them, with the interesting ones first:

| question | A gold | B gold | same table |
|---|---|---|---|
| which instructor has the highest number of courses taught | `3` | `Instructor 3` | yes |
| product category (1 or 2) with the higher total sum | `2` | `Product_Category_2` | yes |
| median Axillary nodes detected for deceased patients | `4` | `3` | yes |
| median `total_bedrooms` value used for imputing | `433.0` | `435.0` | yes |
| total unique names + surnames in police killings | `2646` | `4972` | yes |
| proportion of interactions that are completed purchases | `0.008148` | `0.8148` | yes |
| correlation poverty rate vs high school graduation | `-0.86` | `-0.805761` | yes |
| highest correlation of any feature with `median_house_value` | `0.688` | `0.687160` | yes |
| highest correlation between any two variables | `0.904868` | `0.942535` | no |
| highest correlation between any two variables | `0.941047` | `0.942535` | no |
| highest correlation between any two variables | `0.904868` | `0.912127` | no |
| highest absolute correlation between any two features | `0.79` | `0.34` | no |
| highest correlation between any two numerical features | `0.299` | `0.92` | no |
| percentage of missing values in the dataset | `0` | `56.58` | no |
| highest correlation between any two variables | `0.941047` | `0.912127` | no |

Four distinct failure modes, and they are worth separating because they implicate different sets:

1. **Surface-form disagreement** (`3` vs `Instructor 3`, `2` vs `Product_Category_2`). Same answer,
   different string. Under exact-match grading one of these is wrong and the other is right, and
   nothing in the question decides which.
2. **Rounding disagreement** (`-0.86` vs `-0.805761`, `0.688` vs `0.687160`). A is the notebook's
   printed value; B is the computed one. A's tolerance has to be wide enough to accept the exact
   value or it grades its own reference wrong.
3. **Unit/scale disagreement** (`0.008148` vs `0.8148`). A is a fraction, B a percentage. Both
   defensible; a grader that accepts only one turns a correct agent into a failure.
4. **Genuinely different data** (`2646` vs `4972` for unique names, `4` vs `3` for median nodes).
   Same table name, different table vintage. This one is a genuine data defect: one of the two
   golds was computed against a file that is no longer the shipped one.

---

## 3. Question types

One classifier applied to both sides — `op_family`, `nondeterminism_reasons`, `ambiguity_reasons`
and `answer_type` imported from `smol_ladder/jtasks_v2.py`, not reimplemented, so `ml_fit` means
the same thing in both columns.

### Operation family (% of set)

| family | A (5,000) | B pool (7,518) | B ladder-grade (4,217) | B ja3-passing (2,111) |
|---|---|---|---|---|
| count | 23.2% | 25.1% | 34.1% | **39.6%** |
| agg | 19.0% | 13.9% | 16.4% | 19.7% |
| ml_fit | 15.6% | 27.0% | 8.2% | **3.3%** |
| stat_test | 12.1% | 8.7% | 9.7% | 8.3% |
| string | 9.8% | 8.6% | 13.0% | 11.4% |
| filter | 8.2% | 7.8% | 10.6% | 9.7% |
| argmax | 5.8% | 4.2% | 2.3% | 2.5% |
| other | 3.7% | 2.8% | 3.2% | 3.2% |
| groupby | 2.2% | 1.7% | 2.1% | 1.8% |
| lookup | 0.3% | 0.2% | 0.4% | 0.6% |
| join | 0.1% | 0.0% | 0.1% | 0.0% |

The pool is much more ML-heavy than A (27.0% vs 15.6% `ml_fit`), but the ladder-grade filter
removes almost all of it (8.2%) and the ja3 pass removes most of the rest (3.3%). **The trained
set is more arithmetic and less model-fitting than A, not less.** Both `ml_fit` and `stat_test`
questions are ones the ladder's own audit found unanswerable-as-posed, and B has thrown most of
them away before training.

### Determinism and ambiguity

| | A | B pool | B ladder-grade | B ja3-passing |
|---|---|---|---|---|
| nondeterministic | 16.1% | 24.2% | 0.0% | 0.0% |
| ambiguous | 25.9% | 23.8% | 0.0% | 0.0% |
| passes `is_ladder_grade` | **61.4%** (3,072) | 56.1% | 100% | 100% |

**A fails B's own quality bar on 38.6% of its rows** — 803 nondeterministic and 1,293 ambiguous
questions that `is_ladder_grade` would reject outright. B's unfiltered pool is worse than A on
nondeterminism (24.2% vs 16.1%); the ladder-grade filter is what makes it better, not the
underlying data. A train split ships a materially higher share of unanswerable-as-posed questions
than a filtered jupyter-agent pool does.

### Answer type and reward mode

| | A | B pool | B ja3-passing |
|---|---|---|---|
| numeric | 58.1% | 75.3% | **91.8%** |
| exact_short (label) | 28.2% | 22.1% | 5.4% |
| exact_bool | 2.5% | 2.7% | 2.8% |
| **list** | **7.3%** | — | — |
| **flexible** | **3.0%** | — | — |
| **list_csv** | **0.8%** | — | — |

**A has three reward modes B has no equivalent for at all**: `list` (367), `flexible` (152) and
`list_csv` (39) — 558 tasks, 11.2% of A. These are multi-value answers and loosely-graded ones,
which is why the naive "grader tolerance" fixes that help B are not enough for A. B is 91.8% plain
numeric, which makes it easier to grade and easier to overfit to.

### Question length

| | median words | p95 | max |
|---|---|---|---|
| A | 15 | 22 | 61 |
| B pool | 16 | 22 | 33 |
| B ja3-passing | **14** | 20 | 31 |

Nearly identical. A's longer tail (max 61 vs 31) is a handful of multi-part questions.

### Input files and size

| | 1 file | 2 files | 3+ files |
|---|---|---|---|
| A | 88.5% | 8.5% | 3.0% |
| B pool | 85.0% | 10.4% | 4.7% |
| B ja3-passing | 86.1% | 9.8% | 4.1% |

B is modestly more multi-table: **15.1% of the pool and 13.9% of the ja3-passing set read two or
more files, against 11.5% of A.** On size, the two sets are not comparable with what is on disk: the
pool carries `input_bytes` (measured against the Kaggle cache, median 567 KB, p95 60.7 MB, max
897 MB for ladder-grade) and A's rows carry none, because A's tables live in an HF bucket rather
than the local cache. **A's input-size distribution is unmeasured** — see §5.

### Difficulty: A's tiers vs B's tags

These do not align, and there is no mapping between them. A ships a `difficulty_tier`
(easy 28.7% / medium 56.9% / hard 14.4%) and `difficulty_level` 1–4; B ships `edu_score` (4 or 5
only) and the three tags above. `edu_score` is binary in practice and carries almost no
information; `difficulty_tier` is a real ordinal that B has no counterpart for. **A difficulty
comparison between the two sets is not possible with the columns that exist.**

---

## 4. Quality, measured

### (a) LLM-judge rubric

`stealth/space-bunny-alpha`, 150 questions per set, seed 17, **blind**: the two samples are
interleaved with a seeded PRNG into one stream with no set labels, so a judge that drifted stricter
over the stream could not manufacture an A-vs-B difference. Four criteria, yes/no each.

| criterion | A (149 answered) | B ladder-grade (149 answered) |
|---|---|---|
| answerable from the tables | 68.5% [60.6, 75.4] | 69.8% [62.0, 76.6] |
| computation unambiguous | 61.1% [53.1, 68.5] | **55.0%** [47.0, 62.8] |
| answer form determinate | 75.8% [68.4, 82.0] | 73.2% [65.5, 79.6] |
| requires computation | 98.0% [94.2, 99.3] | 98.0% [94.2, 99.3] |

Inter-run agreement, 30 questions re-judged blind (29 answered by both runs):

| criterion | raw agreement | Cohen's κ |
|---|---|---|
| answerable | 0.862 | 0.680 |
| unambiguous | 0.828 | 0.644 |
| determinate | 0.897 | 0.734 |
| requires_compute | 1.000 | 0.000 |

Three of the four criteria are usable (κ 0.64–0.73, substantial agreement). **`requires_compute`
has κ = 0.000 and its rate is not interpretable**: both runs said yes to all 29 items, so chance
agreement is 1.0, no correction is computable, and the "98.0%" figure above is a property of a
saturated criterion, not a measurement. It is reported because it was asked for, and flagged
because it should not be quoted.

**On the four criteria that work, A and B are indistinguishable** — every CI overlaps heavily, and
the one gap (`unambiguous`, 61.1% vs 55.0%) is inside its own confidence intervals. The judge's
verdict contradicts the regex tags in §3, which put B's ambiguity rate at 0% by construction and
A's at 25.9%. The regexes and the judge disagree about what "ambiguous" means: the regexes fire on
surface phrases ("based on correlation analysis", exact-label answers), the judge on whether a
careful reader would actually compute two different things. **The judge is the better instrument
and it says B is not cleaner than A** — which means B's 0% ambiguity rate is a property of the
filter, not of the questions.

### (b) Behavioural evidence already on disk

L1 pass rate, `stealth/space-bunny-alpha`, same agent and rung on both sides:

| | A: SmolDataEnvs `test`, run `v2` | B: ja3 `L1`, pool |
|---|---|---|
| trials | 500 | 3,880 |
| pass rate over trials | **71.4%** | **54.4%** |
| tasks evaluated | 244 | 3,863 |
| tasks passing | 201 (**82.4%**) | 2,110 (**54.6%**) |
| harness failures | 20 (**4.0%**) | 17 (**0.4%**) |

Two cautions on reading that gap. The two sweeps are not the same experiment: `v2` ran 2 samples
per task over 244 `test` tasks with 5 rungs, `ja3` ran 1 sample over 3,863 ladder-grade tasks
with the L1 rung only, and the ja3 ladder-grade pool is 88% on tables A has already seen. Second,
and larger: **the ja3-passing set is defined by having passed**, so its 54.6% is the rate for the
population *before* the pass and the 2,111 that survived are the easier half of it. The number
that compares the two corpora is 54.4% over all ladder-grade tasks vs 71.4% over A's test — and
that gap is partly the pool being harder (more `ml_fit`, more multi-table) and partly A's `test`
split being easier than its `train` split. Neither confound is separable with the runs on disk.

Harness failure rates are good on both sides and not the explanation: 4.0% on A, 0.4% on B.

**Do repeated attempts agree with each other?** From the reference retries
(`data/solutions/{test,eval}/*/attempt_*`, 118 tasks retried after a first failure):

| outcome | test (69) | eval (49) |
|---|---|---|
| attempts disagree with each other | 34 (**49.3%**) | 28 (**57.1%**) |
| agree on the *same wrong* answer | 7 | 7 |
| agree on the gold | 19 | 10 |
| produced no answer at all | 9 | 4 |

**Roughly half of all retried tasks are ones the model answers differently each time.** These are
all SmolDataEnvs tasks, and a model disagreeing with itself across 2–5 attempts on the same table
means the question does not determine the answer — the gold is not a function of the data. The 14
that agree on the *same wrong* value are the stronger signal: the model is stable and the gold
still disagrees, so the gold is not reproducible from what the model can see.

### (c) Trajectory shape

| | SDE-sft (4,677 rows) | our ja3 passing (1,064 transcripts) |
|---|---|---|
| assistant turns (median) | 4 | **6** |
| assistant turns (p95) | 7 | 13 |
| tool calls (median) | 3 | **6** |
| tool calls (p95) | 6 | 14 |
| total chars (median) | 3,505 | — |
| est. tokens (median) | ~876 | ~1,507 |

SDE-sft is uniformly shorter: half as many assistant turns and half as many tool calls. Its own
`n_turns` column agrees — 2,384 of 4,677 rows (51%) are 3-turn trajectories. Our passing
transcripts explore roughly twice as long before submitting.

**Token counts are estimates**, at 4 chars/token over the serialised conversation; no tokenizer
for either model's chat template is available here, so a real count would be a different number
with the same units and less honesty. The *ratio* between the two sets is robust to the constant;
the absolute numbers are not.

**The two sets teach different mechanics, and this is the largest single difference between them.**
SDE-sft computes inline: 4,118 of 4,677 rows (88%) run `python3 -c` and only 648 (14%) write a
heredoc script. Our `run_shell` transcripts do the opposite — they explore over 6 tool calls and
land a standalone `solution.py` (median 546 chars, p95 2,285). So A trains *short inline
computation and submit*, and B's traces train *extended exploration then a written program*. Those
are two different solution strategies, and A+B trains both against the same tasks.

On top of that, the protocols differ: SDE-sft is **bash**-only (write a heredoc, run it), while our
transcripts use a **tool-call** protocol (`run_shell` / `write_solution`). A+B mixes the two
contracts in one training set, which trains format compliance to both at once.
`train/export_sft.py` has the translation path for this (`--source traces`) and it is the
mechanism that would have to be exercised.

---

## 5. Is B different data, or more of the same?

**More of the same, on provenance; different tasks, on content; and the part we would train on is
the most same part.**

B shares 88% of its tables with A and 19% of its source rows. That is the signature of one corpus
re-cut: A took 5,000 questions from the jupyter-agent Kaggle pool, B took 7,518 from the same pool,
and they overlap where the underlying notebooks overlap. B is **not** an independent second source
in any sense that would let a held-out measurement on A be uncontaminated by B.

But at the task level they are substantially different. Only 934 of B's 4,151 distinct questions
(22.5%) are the same string as an A question, and only 313 pairs are near-duplicates by a
conservative lexical bar. **88% shared tables, 22% shared questions** — the other 78% of B's
questions are ones A never asks about the tables A does use.

**What B adds that A lacks:**

- **Questions that survive a quality filter.** 38.6% of A's rows would be rejected by
  `is_ladder_grade` (803 nondeterministic, 1,293 ambiguous); B's ladder-grade set is 0% on both by
  construction. On the judge's criteria the two are indistinguishable, but the regex tags are
  mechanical and re-checkable, and the ambiguity modes they catch (surface-form labels, invented
  thresholds) are documented in `docs/audits/`.
- **A label-free, uniformly-numeric answer distribution.** 91.8% plain numeric against A's 58.1%,
  with no `flexible` or `list` modes that have no tolerance-based fix.
- **Multi-table tasks.** 13.9% of B vs 11.5% of A read two or more files.
- **Extended-exploration trajectories.** A's trajectories compute inline and submit (88% run
  `python3 -c`); B's passing transcripts explore over 6 tool calls before writing a program. B is
  the only one of the two that teaches a model to *look at the data before deciding what to ask*.
- **Measured input sizes.** B carries `input_bytes`; A carries nothing, so A's cost-to-fill is
  unpriced while B's is known.

**What A has that B lacks:**

- **Three reward modes B cannot express at all** — `list`, `flexible`, `list_csv`, 11.2% of A.
- **A difficulty ordinal.** `difficulty_tier` (easy/medium/hard) and `difficulty_level` 1–4.
  B's `edu_score` is 4 or 5 and carries almost nothing.
- **Verified trajectories at scale.** 4,677 real bash-protocol trajectories versus our 1,064
  passing transcripts in a different protocol.
- **121 tables A uses that the pool does not** (460 vs 578 distinct bare names; 339 shared).

### What the overlap means for comparing "SFT on A" with "SFT on A+B"

The arms are A and A+B. Four consequences follow directly from the measurements above, and they
are about *what the comparison can conclude* rather than about which arm is better.

1. **A+B is a re-weighting of A, not an addition to it.** 94.9% of the tasks in the ja3-passing
   set sit on tables A already uses, and 32.7% are the same source row. If the trained-on set is
   the ja3-passing 2,111, then A+B moves ~2,111 tasks into a corpus that already holds 4,673 A
   trajectories over the same 460 tables — of which 690 are row-identical. The effect being
   measured is roughly *how much does re-sampling the same corpus toward model-solvable tasks
   help*, not *does more data help*.

2. **The pass-rate-based selection biases toward A's tables.** The ja3-passing set is 94.9% on
   shared tables versus 88.1% for the unfiltered pool, and 39.6% `count` versus 25.1%. Whatever
   A+B adds, it adds skewed toward easy counting questions on well-covered tables. A difference
   between the arms will partly be a difference in task-type mix, not in data volume — which is
   testable (report `op_family` on both eval sets) and should be, or the result will be read as a
   data effect.

3. **Benchmark contamination is unchanged; training contamination increases.** The `test`/`eval`
   firewall in `jtasks_v2` fires on held-out tables, and it still holds — no `test` or `eval`
   table is in the pool. But the 690 row-identical tasks mean A+B double-trains 690 questions that
   are also in A. That inflates both arms' absolute scores by an unknown amount; it inflates them
   roughly equally, so the *difference* is still interpretable, but neither absolute number is a
   clean generalisation estimate.

4. **The honest framing is a data-mix ablation, not a scaling result.** Because B is a re-cut of
   the same corpus, A vs A+B tests whether a second pass over the same tables — filtered for
   gradability and re-sampled toward tasks the model can solve — buys anything over the first
   pass. That is a real and interesting question. It is not the question "does more data help", and
   a positive result should not be reported as one.

---

## What could not be measured

- **A's input-file sizes.** A's rows carry no byte counts and its tables are in an HF bucket, not
  the local Kaggle cache, so any size comparison with B would be a comparison against nothing.
  `input_bytes` is `None` for all 5,000 A rows in `data/compare/overlap.json`.
- **Embedding-based near-duplicate overlap.** No sentence-embedding model is cached on this
  machine and downloading one was out of scope. The 313-pair count is a lexical **floor**; the
  synonym-swap blind spot is documented and quantified (0.4–0.5 Jaccard for a single-word swap).
- **Whether the 66 disagreements are A's fault or B's.** The four modes are separable by eye, but
  deciding which collection holds the wrong gold requires re-computing against the shipped file,
  which this report did not do.
- **`requires_compute` as a quality signal.** κ = 0.000 — the criterion is saturated and its rate
  is not a measurement. Only three of the four judge criteria are usable.
- **A vs B on difficulty.** A has `difficulty_tier`, B has `edu_score` and three regex tags. There
  is no mapping and no basis for one in the columns that exist.
- **Clean causal comparison of the pass-rate gap.** The `v2` and `ja3` sweeps differ in samples per
  task (2 vs 1), rung set (5 vs 1), task population (244 SDE `test` tasks vs 3,863 ladder-grade),
  and split (`test` vs `train`-derived). No re-run can separate corpus difficulty from split
  difficulty with what is on disk.
- **Whether A+B's extra tasks are *net new skills*.** That needs per-family eval results on both
  arms, which is a training run, not a dataset comparison.

---

## Reproducing

```bash
uv run python -m smol_ladder.compare_sets --out data/compare/overlap.json
uv run --with datasets python -m smol_ladder.judge_quality --n 150 --repeat 30 --seed 17
uv run --with pytest pytest -q tests/test_compare_sets.py tests/test_judge_quality.py
```