"""How much SmolDataEnvs and our jupyter-agent pool overlap, and how they differ.

Both sources were built from the same underlying jupyter-agent Kaggle notebooks, so "do these
overlap" has four different answers depending on what you match on, and they disagree by an
order of magnitude:

- **dataset name** (bare Kaggle slug): ~88% of our pool sits on a table SmolDataEnvs train uses.
- **input file set** (what the agent actually sees): nearly as high, and higher where an upload's
  name does not describe its contents.
- **source notebook row** (`source_row_id`): ~19% is literally the same jupyter-agent row.
- **question text** (normalised): ~22% of our questions are the same string as a SmolDataEnvs one.

Reporting only one of those is how "non-overlapping dataset" gets claimed for two collections cut
from the same tree. This module computes all four, plus one unified set of question-type tags
applied to both sides, plus the lexical near-duplicate structure. It is pure and re-runnable;
nothing here calls a model, so the same inputs always give the same numbers.

**The tags come from `jtasks_v2` by import, not by copy.** `op_family`, `nondeterminism_reasons`
and `ambiguity_reasons` were written against our pool and were never meant to be the only set of
question types; applying them to SmolDataEnvs is what makes the two columns comparable, and a
second implementation of the same regexes would be a second definition that could disagree on a
boundary case. SmolDataEnvs' own `reward_mode` (`flexible`, `list`, `list_csv`) has no
`ANSWER_TYPES` entry, so `answer_type` returns the mode itself rather than guessing a mapping --
that is reported as a finding, not smoothed over.

    uv run python -m smol_ladder.compare_sets --out data/compare/overlap.json
    uv run --with pytest pytest -q tests/test_compare_sets.py
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from smol_ladder.jtasks import dataset_key
from smol_ladder.jtasks_v2 import (
    answer_type,
    is_ladder_grade,
    nondeterminism_reasons,
    op_family,
)
from smol_ladder.tasks import DATA

#: The pool file, the split every ladder-grade number is measured over, and the split that
#: carries a `source_row_id`. Kept as named constants so a report can quote which file it read.
POOL_FILE = "jtasks_v3.jsonl"
POOL_SPLIT = "jupyter-agent-v3"
SDE_SPLIT = "train"
JA3_RUN = "ja3/jupyter-agent-v3"

#: SmolDataEnvs rows carry `source_row_id` in raw jupyter-agent form ("0000/324/324276.ipynb_qa_3")
#: while our task ids carry the slugified form ("ja_0000_324_324276.ipynb_qa_3"). Both are reduced
#: to the same underscored key, which is exactly `jtasks_v2._slug` without the `ja_` prefix.
def source_row_key(value: str | None) -> str:
    """The shared identity of a jupyter-agent row, from either side's spelling of it."""
    text = str(value or "")
    if text.startswith("ja_"):
        text = text[3:]
    return re.sub(r"[^A-Za-z0-9_.-]", "_", text)


# --- question normalisation ---------------------------------------------------------------

#: Question text, reduced to the thing two questions about the same table are allowed to differ
#: by: case, punctuation, whitespace, and the markdown/LaTeX quoting the two collections wrap
#: answers in differently. Deliberately *not* a stemmer or a synonym map: the near-duplicate
#: measure is there to catch paraphrases, and a stemmer would make two genuinely different
#: questions about the same column look like one.
def normalise_question(question: str | None) -> str:
    text = (question or "").lower()
    text = re.sub(r"[$`\\*_>]+", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def tokens(question: str | None) -> frozenset[str]:
    """The content words of a normalised question, for Jaccard."""
    stop = {"the", "a", "an", "of", "in", "is", "are", "for", "to", "and", "or", "on", "what",
            "which", "how", "many", "was", "were", "by", "that", "with", "as", "at", "it", "be",
            "from", "this", "there", "their", "has", "have", "had", "does", "do", "did"}
    return frozenset(w for w in normalise_question(question).split() if w not in stop)


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    """Token-set Jaccard. 1.0 for the same content words in any order, 0.0 for disjoint sets."""
    if not a and not b:
        return 0.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def char_ngrams(text: str, n: int = 5) -> frozenset[str]:
    """Character n-grams over a normalised question, for the order-sensitive measure."""
    padded = f" {normalise_question(text)} "
    if len(padded) < n:
        return frozenset({padded})
    return frozenset(padded[i:i + n] for i in range(len(padded) - n + 1))


def ngram_similarity(a: str, b: str, n: int = 5) -> float:
    """Jaccard over character n-grams. Order-sensitive, so it separates "by country" from
    "country by" where token Jaccard cannot; it is also the measure that survives the questions
    whose difference is one word's inflection."""
    return jaccard(char_ngrams(a, n), char_ngrams(b, n))


#: A pair is a *near-duplicate* when token-set Jaccard >= 0.8 AND character 5-gram Jaccard
#: >= 0.6, and the two measures are both required because each one fails alone: token Jaccard
#: calls "How many rows are in the table?" and "How many rows does the table contain?" identical,
#: while 5-gram Jaccard misses a pure synonym swap that shares almost no characters.
#:
#: The bar is read off the measured distribution rather than guessed. Over all 7,518 pool rows
#: blocked against 5,000 SmolDataEnvs rows, 177,110 candidate pairs share a content word; the
#: counts fall off a cliff between 0.9 and 0.6 and the tail below 0.5 is long and flat, which is
#: what a mixture of "the same question" and "two questions about the same table" looks like:
#:
#:     jaccard >=   0.9   0.8   0.7   0.6      (pairs clearing BOTH bars: 1539 1773 2317 3871)
#:
#: 0.8/0.6 is where the reworded-question tail separates from the mass. `NEAR_DUP_BAND` reports
#: the count at every other threshold, so the headline is never quoted without its sensitivity.
#: No embedding model is used: none is cached on this machine and downloading one was out of scope.
#:
#: **What this measure cannot see.** Both measures are surface-based, so a synonym substitution
#: is missed whenever it is the only difference: "average" against "mean" scores 0.5 token Jaccard
#: and 0.57 on 5-grams, "distinct" against "unique" scores 0.4. On questions this short a content
#: set is five or six words, so one swapped word costs a third of the Jaccard and no threshold
#: catches the tail without also catching unrelated questions about the same table. The measured
#: count below is therefore a *floor* on near-duplicate overlap, and the report quotes it as one.
NEAR_DUP_JACCARD = 0.8
NEAR_DUP_NGRAM = 0.6
#: Strictest first, so the counts it produces rise monotonically and a reader can see the
#: threshold's effect as a single ordered column rather than four unrelated ones.
NEAR_DUP_BAND = ((0.9, 0.8), (0.8, 0.6), (0.7, 0.5), (0.6, 0.4))


def is_near_duplicate(a: str, b: str, jaccard_floor: float = NEAR_DUP_JACCARD,
                      ngram_floor: float = NEAR_DUP_NGRAM) -> tuple[bool, float, float]:
    """`(near, jaccard, ngram)` for one pair. Both measures, so the report can show both."""
    j = jaccard(tokens(a), tokens(b))
    n = ngram_similarity(a, b)
    return (j >= jaccard_floor and n >= ngram_floor), j, n


# --- unified tagging -----------------------------------------------------------------------

@dataclass
class Tagged:
    """A row from either collection, put into one shape so the two can be counted side by side."""

    task_id: str
    question: str
    answer: str
    reward_mode: str
    set_name: str
    table: str = ""
    files: list[str] = field(default_factory=list)
    n_files: int = 0
    input_bytes: int | None = None
    op_family: str = ""
    answer_type: str = ""
    nondeterminism: list[str] = field(default_factory=list)
    ambiguity: list[str] = field(default_factory=list)
    difficulty_tier: str | None = None
    difficulty_level: int | None = None
    edu_score: int | None = None
    source_row: str = ""

    @property
    def q_len_words(self) -> int:
        return len(self.question.split())

    @property
    def q_len_chars(self) -> int:
        return len(self.question)

    @property
    def nondeterministic(self) -> bool:
        return bool(self.nondeterminism)

    @property
    def ambiguous(self) -> bool:
        return bool(self.ambiguity)


def tag_row(row: dict, set_name: str, table: str, files: list[str],
            input_bytes: int | None = None) -> Tagged:
    """Apply the pool's own classifier to any row, whatever collection it came from."""
    question = row.get("question") or ""
    mode = row.get("reward_mode") or ""
    return Tagged(
        task_id=row.get("task_id") or "",
        question=question,
        answer=str(row.get("answer") or ""),
        reward_mode=mode,
        set_name=set_name,
        table=table,
        files=list(files),
        n_files=len(files),
        input_bytes=input_bytes,
        op_family=op_family(question),
        # `answer_type` maps the three modes the pool uses. SmolDataEnvs also ships `flexible`,
        # `list` and `list_csv`, which have no mapping and are returned verbatim -- a distinct
        # reward mode is a real difference between the collections, not a gap to interpolate.
        answer_type=answer_type(mode),
        nondeterminism=nondeterminism_reasons(question),
        ambiguity=_sde_or_pool_ambiguity(row, question),
        difficulty_tier=row.get("difficulty_tier"),
        difficulty_level=row.get("difficulty_level"),
        edu_score=row.get("edu_score"),
        source_row=source_row_key(row.get("source_row_id") or row.get("task_id")),
    )


def _sde_or_pool_ambiguity(row: dict, question: str) -> list[str]:
    from smol_ladder.jtasks_v2 import ambiguity_reasons

    return ambiguity_reasons(question, str(row.get("answer") or ""), row.get("reward_mode") or "")


def load_pool_rows(path: Path | None = None) -> list[dict]:
    src = Path(path or DATA / POOL_FILE)
    return [json.loads(line) for line in src.read_text().splitlines() if line.strip()]


def tag_pool(rows: list[dict]) -> list[Tagged]:
    """The whole v3 pool, tagged. `input_bytes` comes from the row: it was measured against the
    cache when the pool was built, and re-measuring walks 48 GB for a number we already have."""
    return [tag_row(row, "B_pool", dataset_key(row.get("kaggle_dataset_name") or ""),
                    list(row.get("files") or []), row.get("input_bytes"))
            for row in rows]


def tag_sde(rows: list[dict]) -> list[Tagged]:
    """SmolDataEnvs rows, tagged by the same classifier. Their table identity is the bare name,
    the same `dataset_key` the pool's firewall uses, so "shares a table" means the same thing on
    both sides of every count in the report."""
    out = []
    for row in rows:
        files = list(row.get("files") or [])
        out.append(tag_row(row, "A_sde_train", dataset_key(row.get("kaggle_dataset") or ""),
                           files, None))
    return out


def file_key(files: list[str] | None) -> frozenset[str]:
    """A task's input files, lowercased, as a set.

    The second, and more robust, notion of "the same table". `dataset_key` keys on the Kaggle
    upload's name, and a name is metadata that upstream got wrong in places -- the
    `fatal-police-shootings-in-the-us` upload also carries `PercentagePeopleBelowPovertyLevel.csv`,
    so a question about poverty looks mislabelled at the dataset level and correctly labelled at the
    file level. The file set is what the agent actually sees, so it is the honest identity for
    "would this agent see the same data", and it catches the case a dataset name cannot: two
    different uploads shipping the same file.
    """
    return frozenset(Path(f).name.lower() for f in files or [] if f)


def file_overlap(subject: list[Tagged], reference_files: set[frozenset[str]],
                 label: str, subject_name: str) -> Overlap:
    hits = sum(1 for t in subject if file_key(t.files) and file_key(t.files) in reference_files)
    return Overlap(label, subject_name, hits, len(subject))


# --- overlap ------------------------------------------------------------------------------

@dataclass
class Overlap:
    """One containment measurement, with the denominator it was measured against.

    `share` is the share of `subject` whose table/name is in `reference`. Both the numerator and
    the denominator are kept, because "88% overlaps" without "of 4,217 ladder-grade tasks" is the
    kind of number that turns into a claim it cannot carry.
    """

    label: str
    subject: str
    hits: int
    total: int

    @property
    def share(self) -> float:
        return self.hits / self.total if self.total else 0.0

    def as_dict(self) -> dict:
        return {"label": self.label, "subject": self.subject, "hits": self.hits,
                "total": self.total, "share": round(self.share, 4)}


def table_overlap(subject: list[Tagged], reference_tables: set[str], label: str,
                  subject_name: str) -> Overlap:
    hits = sum(1 for t in subject if t.table and t.table in reference_tables)
    return Overlap(label, subject_name, hits, len(subject))


def source_row_overlap(subject: list[Tagged], reference_rows: set[str], label: str,
                       subject_name: str) -> Overlap:
    hits = sum(1 for t in subject if t.source_row and t.source_row in reference_rows)
    return Overlap(label, subject_name, hits, len(subject))


def question_overlap(subject: list[Tagged], reference: list[Tagged]) -> dict:
    """Questions shared between two collections, and how many are on the same table.

    Exact match after `normalise_question`, bucketed by whether the two rows also share a table,
    because "the same question asked of the same table" is a duplicate task and "the same question
    asked of a different table" is a collision that means something quite different.
    """
    by_question: dict[str, list[Tagged]] = defaultdict(list)
    for row in reference:
        key = normalise_question(row.question)
        if key:
            by_question[key].append(row)
    same_table = other_table = 0
    pairs: list[tuple[Tagged, Tagged]] = []
    for row in subject:
        matches = by_question.get(normalise_question(row.question))
        if not matches:
            continue
        # One pair per subject row, against its best match, so a question duplicated inside a
        # collection cannot inflate the count.
        best = max(matches, key=lambda m: (m.table == row.table, m.task_id))
        if best.table == row.table:
            same_table += 1
        else:
            other_table += 1
        pairs.append((row, best))
    return {
        "subject_questions": len({normalise_question(t.question) for t in subject}),
        "reference_questions": len(by_question),
        "shared": same_table + other_table,
        "shared_same_table": same_table,
        "shared_other_table": other_table,
        "pairs": pairs,
    }


def near_duplicates(subject: list[Tagged], reference: list[Tagged], limit: int = 10,
                    exclude_exact: bool = True) -> tuple[list[dict], list[dict], dict]:
    """The closest cross-collection question pairs, best first, plus the threshold sensitivity.

    Returns `(top_pairs, all_pairs, sensitivity)`. `all_pairs` is every candidate that scored at
    or above the *loosest* band entry, with its two measures attached, so a reader can re-cut the
    list at any threshold instead of trusting the one this module picked.

    Blocking by shared content token keeps this linear in the rows rather than quadratic: only
    pairs sharing at least one content word are scored, which is the definition of a
    near-duplicate here. The cost of that choice is stated in the report -- a pure synonym swap
    with no shared content word ("What is the average of sales?" against "What is the mean of
    sales?") shares the column name and is caught, but a question that shares *nothing* with its
    twin is not, and no lexical measure would have found it either.
    """
    index: dict[str, list[tuple[Tagged, frozenset[str], frozenset[str]]]] = defaultdict(list)
    for row in reference:
        toks = tokens(row.question)
        if not toks:
            continue
        grams = char_ngrams(row.question)
        for word in toks:
            index[word].append((row, toks, grams))
    loose_j, loose_n = NEAR_DUP_BAND[-1]
    scored: list[dict] = []
    for row in subject:
        toks = tokens(row.question)
        if not toks:
            continue
        seen: set[str] = set()
        grams = char_ngrams(row.question)
        for word in toks:
            for other, other_toks, other_grams in index.get(word, ()):
                if other.task_id in seen:
                    continue
                seen.add(other.task_id)
                j = jaccard(toks, other_toks)
                n = jaccard(grams, other_grams)
                if j < loose_j or n < loose_n:
                    continue
                exact = (normalise_question(row.question)
                         == normalise_question(other.question))
                if exclude_exact and exact:
                    continue
                scored.append({"subject": row.task_id, "reference": other.task_id,
                               "exact": exact,
                               "jaccard": round(j, 3), "ngram": round(n, 3),
                               "same_table": row.table == other.table,
                               "subject_question": row.question,
                               "reference_question": other.question,
                               "subject_answer": row.answer,
                               "reference_answer": other.answer,
                               "subject_table": row.table, "reference_table": other.table})
    scored.sort(key=lambda d: (-d["jaccard"], -d["ngram"]))
    # `all_pairs` is cut at the *loosest* band entry, which is the cheapest correct filter: pairs
    # below it were already counted as candidates and none can reach any bar, so scoring them
    # again would only slow the sweep down.
    loose_j, loose_n = NEAR_DUP_BAND[-1]
    candidates = [d for d in scored if d["jaccard"] >= loose_j and d["ngram"] >= loose_n]
    sensitivity = {
        f"jaccard>={j},ngram>={n}": sum(1 for d in candidates
                                        if d["jaccard"] >= j and d["ngram"] >= n)
        for j, n in NEAR_DUP_BAND
    }
    sensitivity["scored_candidates"] = len(candidates)
    return scored[:limit], candidates, sensitivity


def disagreements(subject: list[Tagged], reference: list[Tagged],
                  index_by_question: dict[str, list[Tagged]] | None = None) -> list[dict]:
    """The same question, a different gold answer. The interesting failures, not noise.

    A same-text question with two different answers means at least one of: the two rows read
    different tables (a mirror, a different vintage), the gold is one draw from a distribution
    neither question pins, or one of the two collections recorded the notebook's printed value
    rather than the computed one. Each of those is a defect in one of the two sets.
    """
    index = index_by_question or defaultdict(list)
    if not index:
        for row in reference:
            index[normalise_question(row.question)].append(row)
    out = []
    for row in subject:
        for other in index.get(normalise_question(row.question), ()):
            if _answers_agree(row.answer, other.answer):
                continue
            out.append({
                "question": row.question,
                "subject": row.task_id, "reference": other.task_id,
                "subject_set": row.set_name, "reference_set": other.set_name,
                "subject_answer": row.answer, "reference_answer": other.answer,
                "subject_table": row.table, "reference_table": other.table,
                "same_table": row.table == other.table,
                "reward_mode": row.reward_mode,
            })
    return out


def _answers_agree(a: str, b: str) -> bool:
    """Whether two gold answers are the same value. Numeric answers compare as numbers, because
    "88.52" and "88.520" is one answer printed twice and not a disagreement."""
    try:
        return abs(float(a) - float(b)) <= 1e-9
    except (TypeError, ValueError):
        pass
    return normalise_question(a) == normalise_question(b)


def distinct_on_shared_tables(subject: list[Tagged], reference: list[Tagged],
                              limit: int = 10) -> list[dict]:
    """Pairs on the same table whose questions are clearly different questions.

    The counterpart to `near_duplicates`, and the one that answers "how much does B add". It is
    the existence of the table that both collections share, not the question text: two different
    questions over one CSV are two tasks that train two different skills, and counting only the
    near-duplicates would let a shared table read as a shared task.

    Sampling is spread across tables rather than taken from the first ten, so ten examples from
    one busy Kaggle upload cannot stand in for the whole shared-table population.
    """
    reference_by_table: dict[str, list[Tagged]] = defaultdict(list)
    for row in reference:
        if row.table:
            reference_by_table[row.table].append(row)
    found: list[dict] = []
    seen_tables: set[str] = set()
    for row in sorted(subject, key=lambda r: r.task_id):
        if not row.table or row.table in seen_tables:
            continue
        candidates = [c for c in reference_by_table.get(row.table, ())
                      if normalise_question(c.question) != normalise_question(row.question)]
        if not candidates:
            continue
        near, j, n = is_near_duplicate(row.question, candidates[0].question)
        if near:
            continue
        seen_tables.add(row.table)
        found.append({"subject": row.task_id, "reference": candidates[0].task_id,
                      "table": row.table, "jaccard": round(j, 3), "ngram": round(n, 3),
                      "subject_question": row.question,
                      "reference_question": candidates[0].question,
                      "subject_answer": row.answer,
                      "reference_answer": candidates[0].answer})
        if len(found) >= limit:
            break
    return found


# --- distributions ------------------------------------------------------------------------

def distribution(rows: list[Tagged], key) -> dict[str, int]:
    return dict(Counter(key(r) for r in rows).most_common())


def quantile(values: list[float], q: float) -> float:
    """The q-th quantile by the nearest-rank method, so a reported median is a number in the
    data rather than an interpolation between two that never occurred."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[index]


def describe(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    return {"n": len(values), "mean": round(sum(values) / len(values), 2),
            "p25": quantile(values, 0.25), "median": quantile(values, 0.5),
            "p75": quantile(values, 0.75), "p95": quantile(values, 0.95),
            "max": round(max(values), 2)}


# --- behavioural evidence already on disk ---------------------------------------------------

def pass_rate(root: Path, rung: str = "L1") -> dict:
    """L1 pass rate and harness failure rate over a results tree, from `result.json` alone.

    `agent_status != "exit 0"` is a harness failure, not a model failure, and the two are counted
    separately for that reason: a rate computed over all attempted trials charges the model for a
    container that died.
    """
    attempts = passes = clean = 0
    statuses: Counter = Counter()
    per_task: dict[str, list[float]] = defaultdict(list)
    for path in root.glob(f"*/{rung}/**/result.json"):
        try:
            result = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        attempts += 1
        status = result.get("agent_status")
        statuses[str(status)] += 1
        reward = float(result.get("reward") or 0.0)
        if reward >= 1.0:
            passes += 1
        if status == "exit 0":
            clean += 1
            per_task[str(path.relative_to(root).parts[0])].append(reward)
    passing_tasks = sum(1 for rewards in per_task.values() if any(r >= 1.0 for r in rewards))
    return {
        "rung": rung, "trials": attempts, "passes": passes,
        "pass_rate_over_trials": round(passes / attempts, 4) if attempts else 0.0,
        "harness_failures": attempts - clean,
        "harness_failure_rate": round((attempts - clean) / attempts, 4) if attempts else 0.0,
        "tasks_evaluated": len(per_task), "tasks_passing": passing_tasks,
        "task_pass_rate": round(passing_tasks / len(per_task), 4) if per_task else 0.0,
        "statuses": dict(statuses.most_common()),
    }


def retry_agreement(root: Path) -> dict:
    """Do repeated attempts at the same failed task agree with each other?

    The retries in `data/solutions/<split>/<task>/attempt_*` exist precisely because the first
    answer was wrong, so what they measure is not accuracy -- it is whether the gold is a function
    of the table. Attempts that disagree with each other are the tasks where the question does not
    determine the answer, which is a defect in the task; attempts that agree on the *same wrong*
    value are the stronger signal, because there the model is stable and the gold is not
    reproducible from what the model can see.
    """
    tasks = disagree = agree_wrong = agree_right = no_answer = 0
    examples: list[dict] = []
    for task_dir in sorted(root.glob("*")):
        results = []
        for attempt in sorted(task_dir.glob("attempt_*/result.json")):
            try:
                results.append(json.loads(attempt.read_text()))
            except (OSError, ValueError):
                continue
        if not results:
            continue
        tasks += 1
        predictions = {(r.get("prediction") or "").strip() for r in results}
        rewards = [float(r.get("reward") or 0.0) for r in results]
        if len(predictions) == 1 and "" in predictions:
            no_answer += 1
        elif len(predictions) == 1:
            if any(r >= 1.0 for r in rewards):
                agree_right += 1
            else:
                agree_wrong += 1
                if len(examples) < 15:
                    examples.append({"task_id": results[0].get("task_id"),
                                     "attempts": len(results),
                                     "agreed_prediction": next(iter(predictions)),
                                     "gold": results[0].get("gold")})
        else:
            disagree += 1
            if len(examples) < 15:
                examples.append({"task_id": results[0].get("task_id"),
                                 "attempts": len(results),
                                 "distinct_predictions": len(predictions),
                                 "example": sorted(predictions)[:3]})
    return {"tasks_retried": tasks, "attempts_disagree": disagree,
            "attempts_agree_on_same_wrong_answer": agree_wrong,
            "attempts_agree_on_gold": agree_right, "tasks_with_no_prediction": no_answer,
            "disagreement_rate": round(disagree / tasks, 4) if tasks else 0.0,
            "examples": examples}


def trajectory_shape(root: Path, rung: str = "L1", limit: int | None = None) -> dict:
    """Turns, tool calls, code length and a token estimate per passing transcript.

    The token count is an *estimate* (`CHARS_PER_TOKEN` on the serialised conversation), and it is
    labelled one everywhere it is reported: no tokenizer is available for either model's actual
    chat template here, so a real count would be a different number with the same units and less
    honesty.
    """
    CHARS_PER_TOKEN = 4.0
    shapes: list[dict] = []
    transcripts = sorted(root.glob(f"*/{rung}/transcript.json"))
    if limit:
        transcripts = transcripts[:limit]
    for path in transcripts:
        directory = path.parent
        result_path = directory / "result.json"
        if not result_path.exists():
            continue
        try:
            if float(json.loads(result_path.read_text()).get("reward") or 0.0) < 1.0:
                continue
            messages = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if not isinstance(messages, list):
            continue
        assistants = [m for m in messages if m.get("role") == "assistant"]
        calls = sum(len(m.get("tool_calls") or []) for m in assistants)
        code = 0
        for message in assistants:
            for call in message.get("tool_calls") or []:
                name = (call.get("function") or {}).get("name")
                if name in {"write_solution", "submit"}:
                    try:
                        code += len(json.loads(
                            (call["function"].get("arguments") or "{}")).get("code") or "")
                    except (ValueError, KeyError, TypeError):
                        continue
        program = directory / "solution.py"
        if program.exists():
            code = max(code, len(program.read_text(errors="replace")))
        chars = sum(len(str(m.get("content") or "")) for m in messages)
        for message in assistants:
            for call in message.get("tool_calls") or []:
                chars += len(str((call.get("function") or {}).get("arguments") or ""))
        shapes.append({
            "task_id": directory.parent.name,
            "messages": len(messages),
            "assistant_turns": len(assistants),
            "tool_calls": calls,
            "code_chars": code,
            "total_chars": chars,
            "est_tokens": round(chars / CHARS_PER_TOKEN),
        })
    return {
        "n": len(shapes),
        "messages": describe([s["messages"] for s in shapes]),
        "assistant_turns": describe([s["assistant_turns"] for s in shapes]),
        "tool_calls": describe([s["tool_calls"] for s in shapes]),
        "code_chars": describe([s["code_chars"] for s in shapes]),
        "est_tokens": describe([s["est_tokens"] for s in shapes]),
        "per_task": shapes,
    }


def sft_trajectory_shape(rows: list[dict]) -> dict:
    """Turns, tool calls and computation style over upstream's SFT trajectories.

    The counterpart to `trajectory_shape`, for the `messages` + `tools` rows of
    `FineEnvs/SmolDataEnvs-sft`. Two things need care:

    - The tool-call arguments arrive **already parsed as a dict**, not as a JSON string, so the
      string branch is the rare fallback rather than the norm.
    - `code_chars` counts heredoc bodies, and it is near zero for most rows *by construction*:
      upstream's agent computes inline with `python3 -c` rather than writing a script. That is why
      `inline_python_rows` is reported next to it -- a near-zero code length here means a different
      solving strategy, not a different parse.
    """
    HEREDOC = re.compile(r"<<-?\s*['\"]?([A-Za-z_][A-Za-z0-9_]*)['\"]?\s*\n(.*?)\n\s*\1\b",
                         re.S)
    shapes: list[dict] = []
    inline = heredoc = 0
    for row in rows:
        messages = row.get("messages") or []
        assistants = [m for m in messages if m.get("role") == "assistant"]
        calls = 0
        code = 0
        commands: list[str] = []
        for message in assistants:
            for call in message.get("tool_calls") or []:
                calls += 1
                arguments = (call.get("function") or {}).get("arguments")
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except ValueError:
                        arguments = {}
                if not isinstance(arguments, dict):
                    continue
                command = arguments.get("command") or ""
                commands.append(command)
                code += sum(len(m.group(2)) for m in HEREDOC.finditer(command))
        if any("python3 -c" in c or "python -c" in c for c in commands):
            inline += 1
        if "<<" in "\n".join(commands):
            heredoc += 1
        chars = sum(len(str(m.get("content") or "")) for m in messages)
        for command in commands:
            chars += len(command)
        shapes.append({"task_id": row.get("task_id") or "", "messages": len(messages),
                       "assistant_turns": len(assistants), "tool_calls": calls,
                       "code_chars": code, "total_chars": chars,
                       "est_tokens": round(chars / 4.0)})
    return {
        "n": len(shapes),
        "messages": describe([s["messages"] for s in shapes]),
        "assistant_turns": describe([s["assistant_turns"] for s in shapes]),
        "tool_calls": describe([s["tool_calls"] for s in shapes]),
        "code_chars": describe([s["code_chars"] for s in shapes]),
        "est_tokens": describe([s["est_tokens"] for s in shapes]),
        "inline_python_rows": inline,
        "heredoc_rows": heredoc,
    }


# --- report assembly ----------------------------------------------------------------------

def summarise_set(name: str, rows: list[Tagged]) -> dict:
    return {
        "set": name,
        "tasks": len(rows),
        "op_family": distribution(rows, lambda r: r.op_family),
        "answer_type": distribution(rows, lambda r: r.answer_type),
        "reward_mode": distribution(rows, lambda r: r.reward_mode),
        "nondeterministic": sum(r.nondeterministic for r in rows),
        "nondeterministic_rate": round(sum(r.nondeterministic for r in rows) / len(rows), 4)
        if rows else 0.0,
        "ambiguous": sum(r.ambiguous for r in rows),
        "ambiguous_rate": round(sum(r.ambiguous for r in rows) / len(rows), 4) if rows else 0.0,
        "n_files": distribution(rows, lambda r: r.n_files),
        "question_words": describe([r.q_len_words for r in rows]),
        "question_chars": describe([r.q_len_chars for r in rows]),
        "input_bytes": describe([float(r.input_bytes) for r in rows if r.input_bytes]),
        "difficulty_tier": distribution(rows, lambda r: r.difficulty_tier or "none"),
        "edu_score": distribution(rows, lambda r: r.edu_score if r.edu_score is not None else "none"),
        "ladder_grade": sum(1 for r in rows if is_ladder_grade({
            "nondeterministic": r.nondeterministic, "ambiguous": r.ambiguous,
            "n_files": r.n_files})),
    }


def collect(data: Path | None = None, data_root: Path | None = None,
            split_loader=None) -> dict:
    """Everything the report quotes, from the four sources, in one dict.

    `split_loader` is a parameter so the whole pipeline can be run against a fixture: without it
    the only way to test that the table, source-row and question overlaps agree with each other is
    to have the real 5,000-row split on disk and trust the arithmetic.
    """
    if split_loader is None:
        from smol_ladder.tasks import load_split as split_loader

    data_root = data_root or DATA
    pool_rows = load_pool_rows(data_root / POOL_FILE)
    pool = tag_pool(pool_rows)
    ladder = [t for t in pool if is_ladder_grade({
        "nondeterministic": t.nondeterministic, "ambiguous": t.ambiguous,
        "n_files": t.n_files})]
    sde = tag_sde(split_loader(SDE_SPLIT))
    ja3_root = data_root / "runs" / JA3_RUN
    # `result.json`'s own name is "result.json"; the task id is the *directory* two levels up.
    # Taking `p.name` here yields a set of thirty-eight hundred copies of the same string, which
    # intersects no task id and silently reports the ja3-passing set as empty -- a plausible-looking
    # row of zeros rather than an error.
    passing = sorted({p.parent.parent.name for p in ja3_root.glob("*/L1/result.json")
                      if _reward_of(p) >= 1.0})
    passing_set = set(passing)
    passing_tags = [t for t in ladder if t.task_id in passing_set]
    if not passing_tags and passing_set:
        raise RuntimeError(
            f"{len(passing_set)} tasks passed L1 under {JA3_RUN} but none is in the ladder-grade "
            f"pool; the sweep and the pool disagree about the population")

    sde_tables = {t.table for t in sde if t.table}
    sde_rows = {t.source_row for t in sde if t.source_row}
    sde_file_sets = {file_key(t.files) for t in sde if t.files}
    report: dict = {
        "counts": {
            "A_sde_train_tasks": len(sde),
            "A_sde_train_tables": len(sde_tables),
            "A_sde_source_rows": len(sde_rows),
            "B_pool_tasks": len(pool),
            "B_ladder_grade": len(ladder),
            "B_ja3_l1_passed": len(passing),
            "B_ja3_passing_within_ladder_grade": len(passing_tags),
        },
        "table_overlap": [
            table_overlap(pool, sde_tables, "pool on SDE-train table", "B_pool").as_dict(),
            table_overlap(ladder, sde_tables, "ladder-grade on SDE-train table",
                          "B_ladder_grade").as_dict(),
            table_overlap(passing_tags, sde_tables, "ja3-passing on SDE-train table",
                          "B_ja3_passing").as_dict(),
        ],
        "sde_table_coverage": {
            "sde_train_tables": len(sde_tables),
            "covered_by_pool": len(sde_tables & {t.table for t in pool}),
            "share": round(len(sde_tables & {t.table for t in pool}) / len(sde_tables), 4)
            if sde_tables else 0.0,
        },
        "file_overlap": [
            file_overlap(pool, sde_file_sets, "pool on an exact SDE-train file set",
                         "B_pool").as_dict(),
            file_overlap(ladder, sde_file_sets, "ladder-grade on an exact SDE-train file set",
                         "B_ladder_grade").as_dict(),
            file_overlap(passing_tags, sde_file_sets,
                         "ja3-passing on an exact SDE-train file set",
                         "B_ja3_passing").as_dict(),
        ],
        "source_row_overlap": [
            source_row_overlap(pool, sde_rows, "pool task from a SDE-train source row",
                               "B_pool").as_dict(),
            source_row_overlap(ladder, sde_rows, "ladder-grade from a SDE-train source row",
                               "B_ladder_grade").as_dict(),
            source_row_overlap(passing_tags, sde_rows, "ja3-passing from a SDE-train source row",
                               "B_ja3_passing").as_dict(),
        ],
        "question_overlap": {},
        "sets": {},
        "behavioural": {},
        "judge": {},
    }
    for name, rows in (("A_sde_train", sde), ("B_pool", pool), ("B_ladder_grade", ladder),
                       ("B_ja3_passing", passing_tags)):
        report["sets"][name] = summarise_set(name, rows)

    overlap = question_overlap(ladder, sde)
    report["question_overlap"] = {k: v for k, v in overlap.items() if k != "pairs"}
    top, all_near, sensitivity = near_duplicates(ladder, sde, limit=200)
    at_bar = f"jaccard>={NEAR_DUP_JACCARD},ngram>={NEAR_DUP_NGRAM}"
    # `all_near` is cut at the *loosest* band entry, so counting "near-duplicates on the same
    # table" over it would report the 0.6/0.4 count next to a 0.8/0.6 headline. The same threshold
    # is applied to both, or the pair of numbers is not about the same thing.
    report["question_overlap"]["near_duplicate_pairs"] = sensitivity[at_bar]
    report["question_overlap"]["near_duplicate_same_table"] = sum(
        1 for n in all_near if n["same_table"]
        and n["jaccard"] >= NEAR_DUP_JACCARD and n["ngram"] >= NEAR_DUP_NGRAM)
    report["question_overlap"]["near_duplicate_sensitivity"] = sensitivity
    report["near_duplicate_examples"] = top[:10]
    report["distinct_same_table_examples"] = distinct_on_shared_tables(pool, sde, limit=10)

    dis = disagreements(ladder, sde)
    report["disagreements"] = {"count": len(dis), "same_table": sum(d["same_table"] for d in dis),
                               "examples": dis[:15]}

    report["behavioural"] = {
        "sde_test_v2_L1": pass_rate(data_root / "runs" / "v2" / "test"),
        "ja3_v3_L1": pass_rate(ja3_root),
        "reference_retries_test": retry_agreement(data_root / "solutions" / "test"),
        "reference_retries_eval": retry_agreement(data_root / "solutions" / "eval"),
    }
    return report


def _reward_of(path: Path) -> float:
    try:
        return float(json.loads(path.read_text()).get("reward") or 0.0)
    except (OSError, ValueError):
        return 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=DATA / "compare" / "overlap.json")
    ap.add_argument("--limit-transcripts", type=int, default=2000)
    ap.add_argument("--sft", action="store_true",
                    help="also measure the SmolDataEnvs-sft trajectories; needs `datasets`")
    args = ap.parse_args()
    report = collect()
    report["trajectory_shape_ja3_passing"] = trajectory_shape(
        DATA / "runs" / JA3_RUN, limit=args.limit_transcripts)
    if args.sft:
        from train.export_sft import load_sft_rows

        report["trajectory_shape_sde_sft"] = sft_trajectory_shape(load_sft_rows())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps({k: v for k, v in report.items()
                      if k in ("counts", "table_overlap", "sde_table_coverage",
                               "source_row_overlap", "question_overlap")}, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()