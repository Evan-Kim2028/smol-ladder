"""The pure half of `compare_sets`: normalisation, similarity, overlap arithmetic, tagging.

Every function under test here is a function of its arguments alone -- no data directory, no
network, no model. That is the property that makes the numbers in `docs/DATASET_COMPARISON.md`
reproducible from a fixture rather than from whatever happened to be on disk that day, so the
tests pin the arguments' behaviour rather than the corpus's.

    uv run --with pytest pytest -q tests/test_compare_sets.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from smol_ladder.compare_sets import (
    NEAR_DUP_BAND,
    NEAR_DUP_JACCARD,
    NEAR_DUP_NGRAM,
    Overlap,
    Tagged,
    _answers_agree,
    _reward_of,
    char_ngrams,
    collect,
    describe,
    disagreements,
    distinct_on_shared_tables,
    distribution,
    file_key,
    file_overlap,
    is_near_duplicate,
    jaccard,
    near_duplicates,
    ngram_similarity,
    normalise_question,
    pass_rate,
    quantile,
    question_overlap,
    retry_agreement,
    sft_trajectory_shape,
    source_row_key,
    summarise_set,
    table_overlap,
    tag_pool,
    tag_row,
    tag_sde,
    tokens,
    trajectory_shape,
)


def sde(task_id, question, answer, table, files=("t.csv",), **extra):
    row = {"task_id": task_id, "question": question, "answer": answer,
           "reward_mode": "numeric", "kaggle_dataset": f"owner/{table}", "files": list(files)}
    row.update(extra)
    return row


def pool_row(task_id, question, answer, table, files=("t.csv",), **extra):
    row = {"task_id": f"ja_{task_id}", "question": question, "answer": answer,
           "reward_mode": "numeric", "kaggle_dataset_name": f"owner/{table}",
           "files": list(files)}
    row.update(extra)
    return row


# --- normalisation -----------------------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("What is the mean?", "what is the mean"),
    ("What is the MEAN?", "what is the mean"),
    ("  What   is the\nmean?  ", "what is the mean"),
    ("What's the mean?", "what s the mean"),
    ("What is the `mean`?", "what is the mean"),
    ("What is the $x$ value?", "what is the x value"),
    ("$1,234.56?", "1 234 56"),
    ("", ""),
    (None, ""),
])
def test_normalise_question(raw, expected):
    """Case, punctuation, spacing and the markdown/maths quoting are noise; digits and word order
    are not. "$1,234.56?" normalising to "1 234 56" is what makes two questions about the same
    number the same string."""
    assert normalise_question(raw) == expected


def test_normalisation_is_idempotent():
    once = normalise_question("What is the MEAN of `Total Sales`?")
    assert normalise_question(once) == once


def test_tokens_drops_stopwords_only():
    assert tokens("How many rows are there in the table") == {"rows", "table"}


def test_tokens_empty_question_is_empty():
    assert tokens("") == frozenset()
    assert tokens("the of a") == frozenset()


# --- similarity --------------------------------------------------------------------------

def test_jaccard_is_set_overlap():
    assert jaccard(frozenset("a b c".split()), frozenset("b c d".split())) == pytest.approx(0.5)
    assert jaccard(frozenset("a b".split()), frozenset("a b".split())) == 1.0
    assert jaccard(frozenset("a".split()), frozenset("b".split())) == 0.0


def test_jaccard_of_two_empty_sets_is_zero_not_one():
    """Two empty questions are not near-duplicates of each other. Returning 1.0 would make every
    unparseable row in both collections a mutual duplicate."""
    assert jaccard(frozenset(), frozenset()) == 0.0


def test_ngrams_are_order_sensitive():
    """Token Jaccard cannot see word order; 5-gram Jaccard can, which is why both are reported."""
    a, b = "sales by country per year", "by sales country per year"
    assert jaccard(tokens(a), tokens(b)) == 1.0
    assert ngram_similarity(a, b) < 1.0


def test_ngrams_of_short_text_is_one_gram():
    assert char_ngrams("ab") == frozenset({" ab "})


@pytest.mark.parametrize("a,b,expected", [
    ("How many rows are in the table?", "How many rows are in the table?", True),
    ("What is the total sales for each country in 2015?",
     "What are the total sales for each country in 2015?", True),
    ("How many rows are in the table?", "What is the mean of the sales column?", False),
])
def test_is_near_duplicate(a, b, expected):
    near, _, _ = is_near_duplicate(a, b)
    assert near is expected


def test_both_measures_are_required_not_either():
    """Both bars must be cleared, and each one independently can veto. The threshold is passed
    explicitly rather than read off a sentence pair whose scores would move if the stopword list
    ever changed -- what is being pinned is the rule, not two numbers."""
    a = "What is the total sales for each country in 2015 and 2016?"
    b = "What are the total sales for each country in 2015?"
    j, n = jaccard(tokens(a), tokens(b)), ngram_similarity(a, b)
    # The n-gram bar alone vetoes this pair even though token Jaccard clears easily.
    assert j >= NEAR_DUP_JACCARD
    assert is_near_duplicate(a, b, ngram_floor=0.7)[0] is False
    assert is_near_duplicate(a, b, ngram_floor=0.6)[0] is True   # the bar this pair actually clears
    # The token bar alone vetoes a pair that shares long spans but says something else.
    other = "What are the total sales for each country in 2015?"
    assert ngram_similarity(a, other) >= NEAR_DUP_NGRAM
    assert is_near_duplicate("How many rows are in the table?",
                             "What is the mean of the sales column?",
                             jaccard_floor=0.95, ngram_floor=0.0)[0] is False


def test_a_word_insertion_does_not_clear_the_bar():
    """One inserted content word is a *different* question, not a near-duplicate. At 0.75 token
    Jaccard it sits under the bar by construction, which is the distinction the 0.8 threshold
    exists to draw; see the measured band in `compare_sets`."""
    a = "What is the mean of the sales column?"
    b = "What is the mean of the total sales column?"
    assert jaccard(tokens(a), tokens(b)) == pytest.approx(0.75)
    assert is_near_duplicate(a, b)[0] is False


def test_a_synonym_swap_is_a_documented_blind_spot():
    """The measure cannot see a synonym substitution, and this pins that as a known limit rather
    than a bug to be discovered later: on a five-word content set, swapping one word for its
    synonym drops token Jaccard to 0.5. The report quotes the near-duplicate count as a floor
    because of exactly this."""
    a = "What is the average of the sales column?"
    b = "What is the mean of the sales column?"
    assert jaccard(tokens(a), tokens(b)) < NEAR_DUP_JACCARD
    assert is_near_duplicate(a, b)[0] is False


def test_thresholds_are_stated_not_derived():
    assert (NEAR_DUP_JACCARD, NEAR_DUP_NGRAM) == (0.8, 0.6)


# --- source row identity ------------------------------------------------------------------

@pytest.mark.parametrize("value,expected", [
    ("0000/324/324276.ipynb_qa_3", "0000_324_324276.ipynb_qa_3"),
    ("ja_0000_324_324276.ipynb_qa_3", "0000_324_324276.ipynb_qa_3"),
    ("0000_324_324276.ipynb_qa_3", "0000_324_324276.ipynb_qa_3"),
])
def test_source_row_key_collapses_both_spellings(value, expected):
    """The two collections spell the same jupyter-agent row differently -- SmolDataEnvs keeps the
    slashes, our pool slugs them for a directory-safe id. This is the function that makes them
    comparable."""
    assert source_row_key(value) == expected


def test_source_row_key_of_nothing_is_empty():
    assert source_row_key(None) == "" and source_row_key("") == ""


# --- overlap arithmetic -------------------------------------------------------------------

def test_overlap_share_and_serialisation():
    o = Overlap("label", "subject", 3, 4)
    assert o.share == pytest.approx(0.75)
    assert o.as_dict() == {"label": "label", "subject": "subject", "hits": 3, "total": 4,
                           "share": 0.75}


def test_overlap_of_empty_subject_is_zero_not_a_crash():
    assert Overlap("l", "s", 0, 0).share == 0.0


def test_table_overlap_matches_on_bare_name():
    """`owner/a` and `mirror/a` are the same table for the firewall's purposes, so a
    row whose table only appears under another owner still counts."""
    reference = tag_sde([sde("a1", "q1", "1", "sales")])
    subject = tag_pool([pool_row("b1", "q1", "1", "sales")])
    hit = table_overlap(subject, {t.table for t in reference}, "l", "s")
    assert hit.hits == 1 and hit.total == 1


def test_table_overlap_ignores_a_missing_table():
    subject = tag_pool([pool_row("b1", "q1", "1", "")])
    assert table_overlap(subject, {"sales"}, "l", "s").hits == 0


def test_file_key_lowercases_and_takes_basenames():
    assert file_key(["/a/B/C.Sales.csv", "d.csv"]) == frozenset({"c.sales.csv", "d.csv"})
    assert file_key([]) == frozenset()
    assert file_key(None) == frozenset()


def test_file_overlap_matches_on_the_exact_file_set_not_the_dataset_name():
    """The dataset name is metadata and upstream got it wrong in places: the
    `fatal-police-shootings-in-the-us` upload also ships `PercentagePeopleBelowPovertyLevel.csv`,
    so a poverty question is mislabelled at the dataset level and correctly labelled at the file
    level. The file set is what the agent actually sees."""
    reference = tag_sde([sde("a1", "poverty?", "1", "fatal-police-shootings",
                             files=["PercentagePeopleBelowPovertyLevel.csv"])])
    # A different owner, a different upload name, the same file.
    subject = tag_pool([pool_row("b1", "poverty?", "1", "some-mirror",
                                 files=["percentagepeoplebelowpovertylevel.csv"])])
    assert table_overlap(subject, {t.table for t in reference}, "l", "s").hits == 0
    assert file_overlap(subject, {file_key(t.files) for t in reference}, "l", "s").hits == 1


def test_file_overlap_requires_the_whole_set_not_a_subset():
    """One shared file is not the same table: a task reading one CSV out of five sees different
    data from a task reading all five, so the match is on the set."""
    reference = tag_sde([sde("a1", "q", "1", "t", files=["a.csv", "b.csv"])])
    subject = tag_pool([pool_row("b1", "q", "1", "t", files=["a.csv"])])
    assert file_overlap(subject, {file_key(t.files) for t in reference}, "l", "s").hits == 0


def test_file_overlap_ignores_a_task_with_no_files():
    subject = tag_pool([pool_row("b1", "q", "1", "t", files=[])])
    assert file_overlap(subject, {frozenset({"a.csv"})}, "l", "s").hits == 0


def test_source_row_overlap_counts_shared_rows():
    reference = tag_sde([sde("a1", "q1", "1", "sales",
                             source_row_id="0000/324/324276.ipynb_qa_3")])
    subject = tag_pool([pool_row("0000_324_324276.ipynb_qa_3", "q1", "1", "sales")])
    # `tag_row` falls back to the task_id for `source_row`, and the pool id carries the `ja_`
    # prefix that `source_row_key` strips.
    assert table_overlap(subject, {t.table for t in reference}, "l", "s").hits == 1
    assert subject[0].source_row in {t.source_row for t in reference}


def test_question_overlap_buckets_by_table():
    reference = tag_sde([
        sde("a1", "What is the mean of sales?", "1", "sales"),
        sde("a2", "How many rows are in the table?", "2", "people"),
    ])
    subject = tag_pool([
        pool_row("b1", "What is the mean of `sales`?", "1", "sales"),     # same question, same table
        pool_row("b2", "how many rows are in the TABLE?", "2", "people"),   # case-only difference
        pool_row("b3", "What is the mean of sales?", "1", "people"),       # same question, other table
        pool_row("b4", "Something else entirely?", "3", "sales"),          # no match
    ])
    out = question_overlap(subject, reference)
    assert out["shared"] == 3
    assert out["shared_same_table"] == 2
    assert out["shared_other_table"] == 1


def test_question_overlap_counts_each_subject_row_once():
    """A question duplicated inside the reference collection must not multiply the subject's
    count -- otherwise a corpus with three copies of one SDE row reports three overlaps."""
    reference = tag_sde([sde("a1", "q?", "1", "sales"), sde("a2", "q?", "1", "sales"),
                         sde("a3", "q?", "1", "sales")])
    subject = tag_pool([pool_row("b1", "q?", "1", "sales")])
    assert question_overlap(subject, reference)["shared"] == 1


def test_near_duplicates_are_ranked_and_exclude_exact_matches():
    reference = tag_sde([sde("a1", "What is the total sales for each country in 2015?", "1",
                             "sales")])
    subject = tag_pool([
        pool_row("b1", "What is the total sales for each country in 2015?", "1", "sales"),
        pool_row("b2", "What are the total sales for each country in 2015?", "1", "sales"),
    ])
    top, every, sensitivity = near_duplicates(subject, reference, limit=10)
    assert [d["subject"] for d in every] == ["ja_b2"]
    assert top[0]["subject"] == "ja_b2"
    assert top[0]["jaccard"] >= NEAR_DUP_JACCARD and top[0]["ngram"] >= NEAR_DUP_NGRAM
    # The sensitivity table is reported alongside the headline so the threshold is visible as a
    # choice. NEAR_DUP_BAND runs strictest-first, so the counts must rise monotonically toward the
    # looser bar -- a band that reported *fewer* pairs at a looser threshold than at a stricter one
    # would mean the threshold is not being applied.
    counts = [sensitivity[f"jaccard>={j},ngram>={n}"] for j, n in NEAR_DUP_BAND]
    assert counts == sorted(counts)
    assert counts[-1] == 1 and sensitivity["scored_candidates"] == 1


def test_near_duplicates_keep_exact_matches_when_asked():
    reference = tag_sde([sde("a1", "What is the mean of the sales column?", "1", "sales")])
    subject = tag_pool([pool_row("b1", "What is the mean of the sales column?", "1", "sales")])
    _, every, _ = near_duplicates(subject, reference, exclude_exact=False)
    assert len(every) == 1 and every[0]["exact"] is True


def test_near_duplicates_score_one_pair_per_reference_row():
    """A subject row sharing three content words with the same reference row must still produce
    one pair, not three -- the blocking index visits a reference row once per shared word."""
    reference = tag_sde([sde("a1", "What is the mean of the total sales column?", "1", "sales")])
    subject = tag_pool([pool_row("b1", "What is the mean of the sales column?", "1", "sales")])
    _, every, _ = near_duplicates(subject, reference)
    assert len(every) == 1


def test_near_duplicates_skip_questions_with_no_content_words():
    reference = tag_sde([sde("a1", "the of a", "1", "sales")])
    subject = tag_pool([pool_row("b1", "the of a", "1", "sales")])
    assert near_duplicates(subject, reference, exclude_exact=False)[1] == []


def test_near_duplicates_report_the_same_table_flag():
    """A near-duplicate on the same table is a repeated task; the same question on two different
    tables is a coincidence, and the report needs to be able to tell them apart."""
    reference = tag_sde([sde("a1", "What is the mean of the sales column?", "1", "sales")])
    same = tag_pool([pool_row("b1", "What is the mean of the total sales column?", "1", "sales")])
    other = tag_pool([pool_row("b2", "What is the mean of the total sales column?", "1", "people")])
    assert near_duplicates(same, reference)[1][0]["same_table"] is True
    assert near_duplicates(other, reference)[1][0]["same_table"] is False


# --- distinct questions on a shared table ----------------------------------------------------

def test_distinct_on_shared_tables_pairs_questions_that_are_not_the_same():
    reference = tag_sde([sde("a1", "What is the mean of the sales column?", "1", "sales")])
    subject = tag_pool([pool_row("b1", "Which country has the highest total sales?", "2", "sales")])
    out = distinct_on_shared_tables(subject, reference)
    assert len(out) == 1 and out[0]["table"] == "sales"
    assert out[0]["jaccard"] < NEAR_DUP_JACCARD


def test_distinct_on_shared_tables_skips_a_near_duplicate():
    reference = tag_sde([sde("a1", "What is the total sales for each country in 2015?", "1",
                             "sales")])
    subject = tag_pool([pool_row("b1", "What are the total sales for each country in 2015?", "1",
                                 "sales")])
    assert distinct_on_shared_tables(subject, reference) == []


def test_distinct_on_shared_tables_spreads_across_tables():
    """Ten examples from one busy upload would not represent the shared-table population, so one
    example per table is taken before a second."""
    reference = tag_sde([sde(f"a{i}", f"Question number {i} here?", str(i), f"t{i}")
                         for i in range(3)])
    subject = tag_pool([pool_row(f"b{i}", f"A completely unrelated question {i}?", "9", f"t{i}")
                        for i in range(3)])
    out = distinct_on_shared_tables(subject, reference, limit=10)
    assert len({d["table"] for d in out}) == 3


def test_distinct_on_shared_tables_ignores_tables_only_one_side_has():
    reference = tag_sde([sde("a1", "What is the mean of sales?", "1", "sales")])
    subject = tag_pool([pool_row("b1", "Which country sold most?", "2", "people")])
    assert distinct_on_shared_tables(subject, reference) == []


# --- disagreements ------------------------------------------------------------------------

@pytest.mark.parametrize("a,b,agree", [
    ("88.52", "88.52", True),
    ("88.52", "88.520", True),
    ("1", "1.0", True),
    ("Web Development", "web development", True),
    ("88.52", "88.53", False),
    ("Web Development", "Web Design", False),
    ("", "1", False),
])
def test_answers_agree(a, b, agree):
    assert _answers_agree(a, b) is agree


def test_disagreements_are_the_same_question_with_two_golds():
    reference = tag_sde([sde("a1", "What is the mean of sales?", "10", "sales")])
    subject = tag_pool([pool_row("b1", "What is the mean of sales?", "12", "sales")])
    out = disagreements(subject, reference)
    assert len(out) == 1
    assert out[0]["subject_answer"] == "12" and out[0]["reference_answer"] == "10"
    assert out[0]["same_table"] is True


def test_disagreements_ignore_two_printings_of_one_number():
    reference = tag_sde([sde("a1", "What is the mean of sales?", "10", "sales")])
    subject = tag_pool([pool_row("b1", "What is the mean of sales?", "10.0", "sales")])
    assert disagreements(subject, reference) == []


# --- unified tagging ----------------------------------------------------------------------

def test_tag_row_applies_the_pool_classifier_to_an_sde_row():
    """The classifier is the pool's, imported not copied, so `ml_fit` means the same thing on
    both sides. A SmolDataEnvs accuracy question must come out `ml_fit`."""
    tagged = tag_row(sde("a1", "What accuracy did the random forest achieve?", "0.9",
                         "sales"), "A_sde_train", "sales", ["t.csv"])
    assert tagged.op_family == "ml_fit"
    assert "model fit" in tagged.nondeterminism
    assert tagged.n_files == 1


def test_answer_type_maps_the_pool_modes_and_leaves_the_rest_alone():
    """SmolDataEnvs ships `flexible`, `list` and `list_csv`, which the pool's ANSWER_TYPES table
    does not cover. Returning the mode verbatim keeps that difference visible; guessing a mapping
    would hide it."""
    assert tag_row(sde("a", "q", "1", "t", ), "A", "t", ["f"]).answer_type == "numeric"
    flexible = dict(reward_mode="list")
    assert tag_row({"task_id": "a", "question": "q", "answer": "a, b",
                    **flexible}, "A", "t", ["f"]).answer_type == "list"


def test_source_row_falls_back_to_the_task_id():
    """A pool row has no `source_row_id` column, so the identity is its own id. Without the
    fallback the pool's side of the source-row count would be identically zero."""
    assert tag_pool([pool_row("b1", "q", "1", "sales")])[0].source_row == "b1"


def test_tag_pool_carries_the_measured_input_size():
    tagged = tag_pool([pool_row("b1", "q", "1", "sales", input_bytes=1234)])
    assert tagged[0].input_bytes == 1234 and tagged[0].input_bytes is not None


def test_tagged_lengths_are_words_and_characters_of_the_raw_question():
    tagged = Tagged(task_id="t", question="How many rows?", answer="1", reward_mode="numeric",
                    set_name="X")
    assert tagged.q_len_words == 3 and tagged.q_len_chars == 14


# --- distributions ------------------------------------------------------------------------

def test_quantile_is_a_value_in_the_data():
    values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert quantile(values, 0.5) in values
    assert quantile(values, 0.5) == 5
    assert quantile([], 0.5) == 0.0
    assert quantile([7], 0.95) == 7


def test_describe_reports_the_fields_the_report_quotes():
    out = describe([1, 2, 3])
    assert out["n"] == 3 and out["median"] == 2 and out["max"] == 3.0
    assert describe([]) == {"n": 0}


def test_distribution_is_most_common_first():
    assert distribution([Tagged("a", "q", "1", "m", "X"), Tagged("b", "q", "1", "m", "X")],
                        lambda r: "same") == {"same": 2}


def test_summarise_set_reports_the_three_rates_side_by_side():
    rows = tag_sde([
        sde("a1", "How many rows are there?", "1", "sales"),
        sde("a2", "What accuracy did the random forest reach?", "2", "sales"),
    ])
    out = summarise_set("A", rows)
    assert out["tasks"] == 2
    assert out["nondeterministic_rate"] == 0.5
    assert out["op_family"]["count"] == 1 and out["op_family"]["ml_fit"] == 1
    assert out["ladder_grade"] == 1


# --- behavioural ---------------------------------------------------------------------------

def test_pass_rate_separates_harness_failures_from_model_failures(tmp_path):
    """A container that timed out is not the model getting it wrong, so the two counts are
    separate and the rate over clean trials is computable from them."""
    def trial(name, reward, status):
        directory = tmp_path / name / "L1"
        directory.mkdir(parents=True)
        (directory / "result.json").write_text(
            json.dumps({"task_id": name, "reward": reward, "agent_status": status}))
    trial("t1", 1.0, "exit 0")
    trial("t2", 0.0, "exit 0")
    trial("t3", 0.0, "timeout")
    out = pass_rate(tmp_path)
    assert out["trials"] == 3 and out["passes"] == 1
    assert out["harness_failures"] == 1
    assert out["harness_failure_rate"] == pytest.approx(1 / 3, abs=1e-3)
    assert out["task_pass_rate"] == 0.5


def test_pass_rate_skips_a_three_deep_sample_directory(tmp_path):
    """`--samples K` nests the extra trials one level deeper; a glob that stopped at `*/L1/` sees
    only sample 0 and would report a pass rate for a fraction of the run."""
    deep = tmp_path / "t1" / "L1" / "s1"
    deep.mkdir(parents=True)
    (deep / "result.json").write_text(
        json.dumps({"task_id": "t1", "reward": 1.0, "agent_status": "exit 0"}))
    assert pass_rate(tmp_path)["trials"] == 1


def test_pass_rate_of_an_empty_tree_is_zero_not_a_division_error(tmp_path):
    out = pass_rate(tmp_path / "nothing")
    assert out["trials"] == 0 and out["pass_rate_over_trials"] == 0.0


def _write_trial(root: Path, task_id: str, reward: float):
    directory = root / task_id / "L1"
    directory.mkdir(parents=True)
    (directory / "result.json").write_text(json.dumps(
        {"task_id": task_id, "reward": reward, "agent_status": "exit 0"}))


def test_passing_task_ids_come_from_the_directory_not_the_file_name(tmp_path):
    """`result.json`'s name is `result.json` in every trial, so an id read from the file name is
    the same string thousands of times, matches no task id, and reports the passing set as empty.
    This walks the same three lines `collect` uses."""
    for task_id in ("t1", "t2", "t3"):
        _write_trial(tmp_path, task_id, 1.0)
    _write_trial(tmp_path, "t4", 0.0)
    names = {p.name for p in tmp_path.glob("*/L1/result.json")}
    assert names == {"result.json"}
    passing = sorted({p.parent.parent.name for p in tmp_path.glob("*/L1/result.json")
                      if _reward_of(p) >= 1.0})
    assert passing == ["t1", "t2", "t3"]


def test_collect_wires_the_populations_together(tmp_path):
    """The end-to-end shape on a fixture: a pool, an SDE split and a results tree, and the report
    tying them by table, source row and question. Every count is checkable by hand here, which is
    the point -- the module's job is arithmetic nobody has to trust."""
    data = tmp_path
    (data / "jtasks_v3.jsonl").write_text("\n".join(json.dumps(r) for r in [
        pool_row("b1", "How many rows are in the table?", "10", "sales", n_files=1),
        pool_row("b2", "Which country sold the most?", "3", "sales", n_files=1),
        pool_row("b3", "How many rows are in the table?", "12", "people", n_files=1),
    ]) + "\n")
    for task_id, reward in (("ja_b1", 1.0), ("ja_b2", 0.0)):
        _write_trial(data / "runs" / "ja3" / "jupyter-agent-v3", task_id, reward)

    class Split:
        def __init__(self, rows):
            self.rows = rows

        def __call__(self, split):
            return self.rows

    sde_rows = [sde("a1", "How many rows are in the table?", "11", "sales")]
    report = collect(data_root=data, split_loader=Split(sde_rows))

    assert report["counts"]["B_pool_tasks"] == 3
    assert report["counts"]["B_ja3_l1_passed"] == 1
    assert report["counts"]["B_ja3_passing_within_ladder_grade"] == 1
    # b1 and b2 are on a table SDE train uses; b3 is not.
    pool_table = {o["label"]: o for o in report["table_overlap"]}
    assert pool_table["pool on SDE-train table"]["hits"] == 2
    # Only b1 passed L1, and it is the one that shares the table.
    assert {o["hits"] for o in report["table_overlap"] if o["subject"] == "B_ja3_passing"} == {1}
    # The shared question text is recognised, once per pool row that asks it: b1 and b3 both ask
    # "How many rows are in the table?", and only b1 is on the table SDE uses.
    assert report["question_overlap"]["shared"] == 2
    assert report["question_overlap"]["shared_same_table"] == 1
    assert report["question_overlap"]["shared_other_table"] == 1
    # The same question with a different gold is a disagreement, not an overlap. Both pool rows ask
    # what SDE answers 11; b1 says 10 and b3 says 12, so there are two disagreements, and only b1's
    # is on the same table.
    assert report["disagreements"]["count"] == 2
    assert report["disagreements"]["same_table"] == 1
    answers = sorted(d["subject_answer"] for d in report["disagreements"]["examples"])
    assert answers == ["10", "12"]


def test_reward_of_a_corrupt_file_is_zero(tmp_path):
    path = tmp_path / "result.json"
    path.write_text("{not json")
    assert _reward_of(path) == 0.0


def test_retry_agreement_buckets_the_three_outcomes(tmp_path):
    def attempts(name, predictions, gold="9"):
        for index, prediction in enumerate(predictions, start=1):
            directory = tmp_path / name / f"attempt_{index}"
            directory.mkdir(parents=True)
            reward = 1.0 if prediction == gold else 0.0
            (directory / "result.json").write_text(json.dumps(
                {"task_id": name, "prediction": prediction, "gold": gold, "reward": reward,
                 "attempt": index}))
    attempts("agrees_wrong", ["4", "4"])
    attempts("agrees_right", ["9", "9"])
    attempts("disagrees", ["4", "5"])
    attempts("silent", ["", ""])
    out = retry_agreement(tmp_path)
    assert out["tasks_retried"] == 4
    assert out["attempts_agree_on_same_wrong_answer"] == 1
    assert out["attempts_agree_on_gold"] == 1
    assert out["attempts_disagree"] == 1
    assert out["tasks_with_no_prediction"] == 1


def test_retry_agreement_of_a_tree_with_no_attempts_is_zero(tmp_path):
    assert retry_agreement(tmp_path)["tasks_retried"] == 0


def test_trajectory_shape_counts_turns_calls_and_code(tmp_path):
    directory = tmp_path / "t1" / "L1"
    directory.mkdir(parents=True)
    (directory / "result.json").write_text(
        json.dumps({"task_id": "t1", "reward": 1.0, "agent_status": "exit 0"}))
    (directory / "transcript.json").write_text(json.dumps([
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "a", "tool_calls": [
            {"id": "1", "function": {"name": "run_shell", "arguments": "{}"}}]},
        {"role": "tool", "content": "o", "tool_call_id": "1"},
        {"role": "assistant", "content": "a2", "tool_calls": [
            {"id": "2", "function": {"name": "write_solution",
                                     "arguments": json.dumps({"code": "print(1)"})}}]},
    ]))
    out = trajectory_shape(tmp_path)
    assert out["n"] == 1
    assert out["assistant_turns"]["median"] == 2
    assert out["tool_calls"]["median"] == 2
    assert out["code_chars"]["median"] == 8  # "print(1)"


def test_trajectory_shape_keeps_only_passing_transcripts(tmp_path):
    """A trajectory is only comparable as training data if the run actually produced the gold
    answer; the shape of a failed run is the shape of a wrong answer."""
    directory = tmp_path / "t1" / "L1"
    directory.mkdir(parents=True)
    (directory / "result.json").write_text(
        json.dumps({"task_id": "t1", "reward": 0.0, "agent_status": "exit 0"}))
    (directory / "transcript.json").write_text(json.dumps([{"role": "assistant", "content": "a"}]))
    assert trajectory_shape(tmp_path)["n"] == 0


# --- SmolDataEnvs-sft trajectory shape -------------------------------------------------------

def _bash_call(arguments):
    return {"id": "1", "type": "function",
            "function": {"name": "bash", "arguments": arguments}}


def test_sft_shape_reads_arguments_that_are_already_a_dict():
    """Upstream's parquet stores `arguments` parsed, not as a JSON string. A parser that only
    handles the string form counts zero tool calls for the entire dataset and reports it as
    confidently as it would report a real measurement."""
    rows = [{"task_id": "s1", "messages": [
        {"role": "system", "content": "s"},
        {"role": "assistant", "content": "a", "tool_calls": [
            _bash_call({"command": "python3 -c 'print(1)'"})]},
    ]}]
    out = sft_trajectory_shape(rows)
    assert out["n"] == 1 and out["tool_calls"]["median"] == 1
    assert out["inline_python_rows"] == 1


def test_sft_shape_also_handles_arguments_as_a_json_string():
    rows = [{"task_id": "s1", "messages": [
        {"role": "assistant", "content": "a",
         "tool_calls": [_bash_call(json.dumps({"command": "ls /input"}))]},
    ]}]
    assert sft_trajectory_shape(rows)["tool_calls"]["median"] == 1


def test_sft_shape_ignores_unparseable_arguments_instead_of_raising():
    rows = [{"task_id": "s1", "messages": [
        {"role": "assistant", "content": "a",
         "tool_calls": [_bash_call("{not json"), _bash_call(None)]},
    ]}]
    assert sft_trajectory_shape(rows)["tool_calls"]["median"] == 2


def test_sft_shape_counts_heredoc_bodies_as_code():
    command = "cat > s.py << 'EOF'\nprint(1)\nprint(2)\nEOF\npython3 s.py"
    rows = [{"task_id": "s1", "messages": [
        {"role": "assistant", "content": "a", "tool_calls": [_bash_call({"command": command})]},
    ]}]
    out = sft_trajectory_shape(rows)
    assert out["code_chars"]["median"] == len("print(1)\nprint(2)")
    assert out["heredoc_rows"] == 1 and out["inline_python_rows"] == 0


def test_sft_shape_separates_inline_computation_from_written_programs():
    """The headline mechanical difference between the two collections. Upstream computes inline
    and submits; our transcripts explore and then write a program. A near-zero code length on the
    SFT side is a different strategy, not a parse failure, and the two counters have to be
    reported together for that to be visible."""
    inline = [{"task_id": "s1", "messages": [
        {"role": "assistant", "content": "a",
         "tool_calls": [_bash_call({"command": "python3 -c 'print(1)'"})]}]}]
    script = [{"task_id": "s2", "messages": [
        {"role": "assistant", "content": "a", "tool_calls": [
            _bash_call({"command": "cat > s.py << 'EOF'\nimport pandas as pd\nEOF"})]}]}]
    assert sft_trajectory_shape(inline)["inline_python_rows"] == 1
    assert sft_trajectory_shape(script)["heredoc_rows"] == 1
    assert sft_trajectory_shape(script)["code_chars"]["median"] > 0
    assert sft_trajectory_shape(inline)["code_chars"]["median"] == 0


def test_sft_shape_of_a_row_with_no_messages():
    """One row with zero turns is described as `n: 1, median: 0`, not as an empty measurement --
    `describe` counts rows, and the median of one zero is zero."""
    out = sft_trajectory_shape([{"task_id": "s1"}])
    assert out["n"] == 1
    assert out["assistant_turns"]["n"] == 1 and out["assistant_turns"]["median"] == 0
    assert out["tool_calls"]["median"] == 0 and out["code_chars"]["median"] == 0