"""Tags on the jupyter-agent pool, and the ladder-grade subset they define.

Every tag is a pure function of the row's own text (`question`, `answer`, `reward_mode`,
`files`) or of the task's input files as they sit on disk. Nothing here consults a model
result, a pass rate, or a random number generator, so the same pool always gets the same
tags — otherwise a rebuild would silently reshuffle the subset a run was measured on.

The tags exist because the audit of this source found that the model's failures are not
mostly reasoning failures: `stat_test` and ML-fit questions depend on an unstated seed or
split, label answers depend on an exact surface form the question never pins, and large
tables cost 8-10 points on every source. None of those are visible in a row that carries
only a question and an answer, so the pool gets tags and a subset definition built on them.

    uv run --with pytest pytest -q tests/test_jtasks_v2.py
"""

from __future__ import annotations

import json

import pytest

from smol_ladder.jtasks_v2 import (
    FAMILIES,
    HARD_FAMILIES,
    cache_index,
    download_estimate,
    input_bytes,
    is_ladder_grade,
    load_size_cache,
    save_size_cache,
    tag,
    tags_for_rows,
)


def row(question: str, answer: str = "3", mode: str = "numeric", files=("t.csv",)):
    return {
        "task_id": "ja_x",
        "question": question,
        "answer": answer,
        "reward_mode": mode,
        "files": list(files),
    }


# --- operation family ------------------------------------------------------------------

@pytest.mark.parametrize("question,expected", [
    ("How many rows have a non-null rating?", "count"),
    ("What is the number of distinct airlines in the table?", "count"),
    ("What is the mean salary across all rows?", "agg"),
    ("Which publisher has the highest total sales?", "agg"),
    ("What is the most common gender among new coders?", "argmax"),
    ("For each country, what is the median age?", "groupby"),
    ("Give a breakdown of sales by region.", "groupby"),
    ("Considering only rows from 2015, what is the average price?", "filter"),
    ("What is the p-value from the one-way chi-square test of gender?", "stat_test"),
    ("What is the accuracy of a Random Forest in predicting readmission?", "ml_fit"),
    ("After tuning the KNN parameters, what was the best accuracy?", "ml_fit"),
    ("What is the Pearson correlation between sepal length and petal width?", "stat_test"),
    ("How many records in the joined table match both sources?", "join"),
    ("What is the value of column X in row where id is 7?", "lookup"),
])
def test_operation_family_is_ordered_first_match_wins(question, expected):
    assert tag(row(question))["op_family"] == expected


def test_every_family_is_a_known_key():
    names = [name for name, _ in FAMILIES]
    assert "other" not in names, "other is the fallback, not a pattern"
    assert set(HARD_FAMILIES) <= set(names)


# --- answer type ----------------------------------------------------------------------

def test_answer_type_follows_the_grader_mode():
    assert tag(row("How many rows?", "3"))["answer_type"] == "numeric"
    assert tag(row("Which one?", "Wii Sports", "exact_short"))["answer_type"] == "label"
    assert tag(row("Is it true?", "yes", "exact_bool"))["answer_type"] == "bool"


# --- nondeterminism -------------------------------------------------------------------

@pytest.mark.parametrize("question", [
    "What is the accuracy of a Gaussian Naive Bayes model predicting readmission?",
    "How many benign cases were misclassified as malignant by the classifier?",
    "What is the maximum F1 score across the model panel?",
    "What was the best accuracy during the KNN parameter tuning process?",
    "Using train_test_split, how many rows land in the test set?",
    "With random sampling, what is the sample mean of the column?",
])
def test_model_fit_and_sampling_questions_are_flagged_nondeterministic(question):
    assert tag(row(question))["nondeterministic"] is True


@pytest.mark.parametrize("question", [
    "How many rows have a rating above 3?",
    "What is the mean of the Total column?",
    "For each region, what is the total sales?",
])
def test_plain_aggregations_are_not_nondeterministic(question):
    assert tag(row(question))["nondeterministic"] is False


def test_nondeterminism_is_a_reason_list_not_a_bare_bool():
    """The flag has to be auditable: which phrase fired is the difference between a tag the
    team can argue with and a tag they have to trust."""
    reasons = tag(row("What is the accuracy of a random forest after cross-validation?"))[
        "nondeterminism_reasons"]
    assert reasons
    assert any("fit" in r or "model" in r or "sample" in r for r in reasons)


# --- ambiguity ------------------------------------------------------------------------

def test_a_label_answer_the_question_never_pins_is_ambiguous():
    """The audit's grader-strictness failures were mostly label vocabulary: gold
    `North America`, prediction `NA`. The question never says which spelling to use."""
    assert tag(row("Which region had the most listings?", "North America", "exact_short"))[
        "ambiguous"] is True


def test_a_label_answer_the_question_pins_verbatim_is_not_ambiguous():
    assert tag(row("Is the region North America or South America?", "North America",
                   "exact_short"))["ambiguous"] is False


@pytest.mark.parametrize("question", [
    "What is the total stat threshold that separates legendary from pseudo-legendary Pokemon?",
    "How many features remain after removing redundant features based on correlation analysis?",
    "In which year did mixtapes become competitive with albums, based on release counts?",
])
def test_questions_that_defer_to_an_unnamed_criterion_are_ambiguous(question):
    """The 'arbitrary task' category in the audit: the reference invented a constant."""
    assert tag(row(question))["ambiguous"] is True


def test_numeric_answers_are_never_ambiguous_by_vocabulary():
    """Exact-match surface risk is a property of labels; a number has one spelling."""
    assert tag(row("How many rows match the filter?", "453"))["ambiguous"] is False


# --- files and size -------------------------------------------------------------------

def test_file_count_and_list_are_recorded():
    tags = tag(row("How many rows?", files=("a.csv", "b.csv")))
    assert tags["n_files"] == 2
    assert tags["files"] == ["a.csv", "b.csv"]


def test_input_bytes_comes_from_the_task_s_own_named_files(tmp_path):
    (tmp_path / "t.csv").write_text("x\n1\n")
    index = cache_index(tmp_path)
    row_ = row("How many rows?", files=("t.csv",))
    assert input_bytes(row_, index) == (tmp_path / "t.csv").stat().st_size


def test_input_bytes_ignores_a_file_the_task_does_not_name(tmp_path):
    """The agent is handed only the files the task names, so the size must be those files."""
    (tmp_path / "t.csv").write_text("x\n")
    (tmp_path / "other.csv").write_text("y\n" * 10_000)
    index = cache_index(tmp_path)
    assert input_bytes(row("How many rows?", files=("t.csv",)), index) == 2


def test_input_bytes_is_none_when_a_named_file_is_absent(tmp_path):
    index = cache_index(tmp_path)
    assert input_bytes(row("How many rows?", files=("missing.csv",)), index) is None


def test_cache_index_ignores_dotfiles(tmp_path):
    (tmp_path / ".complete").write_text("[]")
    (tmp_path / "t.csv").write_text("x\n")
    assert set(cache_index(tmp_path)) == {"t.csv"}


# --- ladder-grade subset --------------------------------------------------------------

def test_ladder_grade_excludes_nondeterministic_and_ambiguous_tasks():
    assert not is_ladder_grade(tag(row("What is the accuracy of a random forest classifier?")))
    assert not is_ladder_grade(tag(row("Which region had the most listings?", "North America",
                                       "exact_short")))
    assert not is_ladder_grade(tag(row("What is the total stat threshold that separates "
                                       "legendary Pokemon?")))


def test_ladder_grade_keeps_deterministic_underspecified_families():
    """The families that need a method choice are the ones worth keeping: that is the
    population whose answer the information rungs can actually disambiguate."""
    for question in ("For each country, what is the median age?",
                     "What is the p-value from the chi-square test of gender?",
                     "How many rows are in the joined table?"):
        assert is_ladder_grade(tag(row(question))), question


def test_ladder_grade_keeps_a_label_answer_the_question_pins():
    tags = tag(row("Is the region North America or South America?", "North America",
                   "exact_short"))
    assert is_ladder_grade(tags)


def test_ladder_grade_excludes_a_task_with_no_files():
    tags = tag(row("How many rows?", files=()))
    assert not is_ladder_grade(tags)


def test_tags_for_rows_is_deterministic_and_keeps_task_ids():
    rows = [row("How many rows?", files=("a.csv",)),
            row("What is the mean of x?", files=("b.csv",))]
    rows[0]["task_id"], rows[1]["task_id"] = "ja_a", "ja_b"
    first = tags_for_rows(rows)
    second = tags_for_rows(rows)
    assert first == second
    assert [t["task_id"] for t in first] == ["ja_a", "ja_b"]


def test_tags_never_drop_a_field_the_v1_row_carried():
    """A v2 row is a v1 row plus tags. If a tag pass drops `atol` or `kaggle_dataset_name`
    the row stops being loadable by the grader, and the failure only shows up at trial time."""
    base = {"task_id": "ja_a", "question": "q", "answer": "3", "reward_mode": "numeric",
            "atol": 1e-4, "rtol": 1e-4, "files": ["t.csv"], "source": "s",
            "kaggle_dataset_name": "o/d", "edu_score": 5}
    tagged = tag(base)
    assert set(base) <= set(tagged)
    assert tagged["answer"] == "3"
    assert tagged["atol"] == 1e-4
    assert json.loads(json.dumps(tagged)) == tagged


# --- download estimate -----------------------------------------------------------------

def test_estimate_counts_a_task_covered_only_when_every_file_is(tmp_path):
    (tmp_path / "a.csv").write_text("a")
    index = cache_index(tmp_path)
    rows = [{"files": ["a.csv"], "kaggle_dataset_name": "o/d"},
            {"files": ["a.csv", "gone.csv"], "kaggle_dataset_name": "o/e"}]
    est = download_estimate(rows, index, {})
    assert est["tasks_inputs_cached"] == 1
    assert est["tasks_needing_download"] == 1
    assert est["datasets_covered_by_cache"] == 1
    assert est["datasets_needing_download"] == 1


def test_an_unpriced_dataset_is_excluded_from_the_gb_total_not_counted_as_free(tmp_path):
    """Kaggle's metadata endpoint rate-limits, so some datasets come back unmeasurable. The
    GB figure must be a floor over what was measured, with the gap reported, or a 429 would
    read as a cheaper pool than the one that exists."""
    (tmp_path / "a.csv").write_text("a")
    index = cache_index(tmp_path)
    rows = [{"files": ["gone1.csv"], "kaggle_dataset_name": "o/big"},
            {"files": ["gone2.csv"], "kaggle_dataset_name": "o/unknown"}]
    est = download_estimate(rows, index, {"o/big": 5_000_000_000})
    assert est["download_gb"] == 5.0
    assert est["datasets_priced"] == 1
    assert est["datasets_unpriced"] == 1
    assert est["tasks_needing_download"] == 2


def test_size_cache_keeps_old_measurements_when_a_lookup_fails(tmp_path):
    cache = tmp_path / "sizes.json"
    save_size_cache(cache, {"o/a": 100, "o/b": None})
    assert load_size_cache(cache) == {"o/a": 100}
    save_size_cache(cache, {"o/c": 200, "o/a": None})
    assert load_size_cache(cache) == {"o/a": 100, "o/c": 200}


def test_size_cache_of_a_missing_or_corrupt_file_is_empty(tmp_path):
    assert load_size_cache(tmp_path / "nope.json") == {}
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert load_size_cache(bad) == {}