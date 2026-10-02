"""The pure half of the quality judge: verdict parsing, Wilson intervals, kappa, blind sampling.

The judge calls a model, so none of that is tested here -- these are the functions that decide
what a model's answer *means* and what a number means, and both are places where a quiet wrong
answer would move every rate in `docs/DATASET_COMPARISON.md` without looking wrong.

    uv run --with pytest pytest -q tests/test_judge_quality.py
"""

from __future__ import annotations

import pytest

from smol_ladder.judge_quality import (
    CRITERIA,
    Judgement,
    agreement,
    cohens_kappa,
    criterion_rate,
    files_description,
    judge_one,
    parse_verdict,
    sample_blind,
    user_prompt,
    wilson_interval,
)


# --- parsing ---------------------------------------------------------------------------------

def test_parse_verdict_reads_a_clean_object():
    parsed = parse_verdict('{"answerable": true, "unambiguous": false, "determinate": true,'
                           ' "requires_compute": true, "reason": "needs a sum"}')
    assert parsed["answerable"] is True
    assert parsed["unambiguous"] is False
    assert parsed["requires_compute"] is True
    assert parsed["reason"] == "needs a sum"
    assert parsed["error"] == ""


def test_parse_verdict_survives_a_code_fence_and_preamble():
    """The model wraps its JSON in prose often enough that a parser which does not look inside
    would score the whole sample as unanswered."""
    text = 'Sure, here is my assessment:\n```json\n{"answerable": true, "unambiguous": true,'\
           ' "determinate": true, "requires_compute": false, "reason": "lookup"}\n```'
    assert parse_verdict(text)["answerable"] is True


@pytest.mark.parametrize("text,reason", [
    ("no json at all", "no JSON object"),
    ("{not valid json}", "invalid JSON"),
])
def test_parse_verdict_reports_why_it_gave_up(text, reason):
    parsed = parse_verdict(text)
    assert all(parsed[c] is None for c in CRITERIA)
    assert reason in parsed["error"]


def test_parse_verdict_accepts_a_stringified_boolean():
    parsed = parse_verdict('{"answerable": "true", "unambiguous": "false", "determinate": "true",'
                           ' "requires_compute": "false", "reason": "r"}')
    assert parsed["answerable"] is True and parsed["unambiguous"] is False


def test_parse_verdict_leaves_a_missing_criterion_unanswered():
    """A criterion the judge did not answer must stay None, not become False: scoring an
    unanswered question as a failure biases every rate downward by the model's formatting
    mistakes, which look exactly like data quality."""
    parsed = parse_verdict('{"answerable": true, "reason": "only one answered"}')
    assert parsed["answerable"] is True
    assert parsed["unambiguous"] is None


def test_parse_verdict_rejects_a_non_boolean_value():
    parsed = parse_verdict('{"answerable": "maybe", "unambiguous": 1, "determinate": null,'
                           ' "requires_compute": [], "reason": ""}')
    assert all(parsed[c] is None for c in CRITERIA)


def test_parse_verdict_of_nothing_is_all_none():
    parsed = parse_verdict("")
    assert all(parsed[c] is None for c in CRITERIA)


# --- Wilson intervals -------------------------------------------------------------------------

def test_wilson_interval_brackets_the_rate():
    low, high = wilson_interval(80, 100)
    assert low < 0.80 < high


def test_wilson_interval_stays_inside_zero_and_one():
    """The normal approximation runs past 1.0 at these rates, which is what the Wilson interval
    exists to prevent; a CI reported as [0.79, 1.04] is a CI nobody can read."""
    low, high = wilson_interval(148, 150)
    assert 0.0 <= low < high <= 1.0
    low, high = wilson_interval(0, 150)
    assert low == 0.0 and 0.0 < high < 1.0


def test_wilson_interval_of_no_answers_is_degenerate_not_an_error():
    assert wilson_interval(0, 0) == (0.0, 0.0)


def test_wilson_interval_narrows_as_n_grows():
    narrow = wilson_interval(120, 150)
    wide = wilson_interval(12, 15)
    assert (narrow[1] - narrow[0]) < (wide[1] - wide[0])


# --- Cohen's kappa ----------------------------------------------------------------------------

def test_kappa_is_one_for_perfect_agreement():
    assert cohens_kappa([(True, True), (True, True), (False, False), (False, False)]) == 1.0


def test_kappa_is_roughly_the_disagreement_rate_when_marginals_are_balanced():
    """With both runs balanced 50/50, chance agreement is 0.5 and kappa is very nearly the raw
    agreement itself. The reference value that makes the 0.8 above legible: 10% disagreement with
    balanced marginals is a kappa of 0.8, not a kappa of 0.9."""
    tracked = ([(True, True)] * 45 + [(False, False)] * 45
               + [(True, False)] * 5 + [(False, True)] * 5)
    assert sum(1 for a, b in tracked if a == b) / len(tracked) == pytest.approx(0.90)
    assert cohens_kappa(tracked) == pytest.approx(0.80)


def test_kappa_is_near_zero_for_chance_agreement():
    """The case the raw-agreement number hides: a judge that says yes 90% of the time and flips a
    coin on the rest agrees with itself 0.90 of the time, and its kappa says the rest is noise."""
    pairs = [(True, True)] * 90 + [(True, False)] * 10
    assert cohens_kappa(pairs) < 0.25


def test_kappa_of_no_pairs_is_zero():
    assert cohens_kappa([]) == 0.0


def test_kappa_does_not_divide_by_zero_when_both_runs_are_constant_and_identical():
    """Both runs always answering the same way is a degenerate run, not an undefined ratio. Chance
    agreement is 1.0, the formula's denominator is 0, and the honest answer is "no correction can
    be computed" -- reported as 0.0, so the criterion is flagged rather than given a wild number."""
    assert cohens_kappa([(True, True)] * 30) == 0.0


def test_kappa_is_minus_one_when_the_runs_are_exact_opposites():
    """Worse than chance is a real and diagnosable outcome -- a judge whose polarity flipped
    between runs -- and kappa has to be able to say so rather than clamping at zero."""
    assert cohens_kappa([(True, False), (False, True)]) == -1.0


def test_kappa_is_zero_when_one_run_is_entirely_yes():
    """The other half of the degenerate case: one run said "no" to nothing across all 100 items,
    which is exactly what a near-saturated criterion produces, and the run must not divide by zero."""
    assert cohens_kappa([(True, False)] * 100) == 0.0


def test_kappa_is_zero_when_the_two_runs_agree_exactly_at_chance():
    """Two independent runs on 100 items with balanced marginals: they agree on half the items,
    which is exactly what chance predicts, so kappa is 0 and the raw number says nothing."""
    pairs = ([(True, True)] * 25 + [(True, False)] * 25
             + [(False, True)] * 25 + [(False, False)] * 25)
    raw = sum(1 for a, b in pairs if a == b) / len(pairs)
    assert raw == pytest.approx(0.5)
    assert cohens_kappa(pairs) == pytest.approx(0.0)


def test_kappa_separates_a_shared_yes_tendency_from_sustained_agreement():
    """The reason kappa is reported beside the raw number. Run 1 answers yes on all 100 items; run
    2 answers yes on 90. The two runs agree on 90 items -- a bare 0.90 that reads as a good
    number -- while kappa sits at zero, because the first run's total yes-tendency already predicts
    90 of those 90."""
    first = [True] * 100
    second = [True] * 90 + [False] * 10
    pairs = list(zip(first, second))
    raw = sum(1 for a, b in pairs if a == b) / len(pairs)
    assert raw == pytest.approx(0.90)
    assert cohens_kappa(pairs) == pytest.approx(0.0)


# --- rates ------------------------------------------------------------------------------------

def judgement(task_id, **criteria):
    return Judgement(task_id=task_id, set_name="X",
                     **{c: criteria.get(c) for c in CRITERIA})


def test_criterion_rate_counts_yes_and_no_against_the_answered_denominator():
    rows = [judgement("t1", answerable=True), judgement("t2", answerable=False)]
    out = criterion_rate(rows, "answerable")
    assert out["yes"] == 1 and out["no"] == 1 and out["n"] == 2 and out["rate"] == 0.5


def test_criterion_rate_excludes_unanswered_from_the_denominator():
    rows = [judgement("t1", answerable=True), judgement("t2", answerable=True),
            judgement("t3")]
    out = criterion_rate(rows, "answerable")
    assert out["n"] == 2 and out["rate"] == 1.0


def test_criterion_rate_of_an_empty_set():
    assert criterion_rate([], "answerable") == {"criterion": "answerable", "yes": 0, "no": 0,
                                                "n": 0, "rate": 0.0, "ci95": [0.0, 0.0]}


def test_criterion_rate_carries_a_ci():
    out = criterion_rate([judgement(f"t{i}", answerable=i < 10) for i in range(20)],
                         "answerable")
    low, high = out["ci95"]
    assert low < 0.5 < high


def test_agreement_pairs_two_runs_by_task_id():
    first = [judgement("t1", answerable=True), judgement("t2", answerable=True)]
    second = [judgement("t2", answerable=False), judgement("t1", answerable=True)]
    out = agreement(first, second, "answerable")
    assert out["n"] == 2 and out["raw_agreement"] == 0.5


def test_agreement_skips_an_id_the_second_run_never_judged():
    first = [judgement("t1", answerable=True), judgement("t2", answerable=True)]
    second = [judgement("t1", answerable=True)]
    assert agreement(first, second, "answerable")["n"] == 1


def test_agreement_of_nothing_measured():
    assert agreement([], [], "answerable") == {"criterion": "answerable", "n": 0,
                                               "raw_agreement": 0.0, "cohens_kappa": 0.0}


# --- blind sampling ------------------------------------------------------------------------------

def rows(*ids):
    return [{"task_id": i, "question": f"question {i}?", "files": ["t.csv"]} for i in ids]


def test_sample_blind_takes_n_per_set():
    picks = sample_blind({"A": rows(*[f"a{i}" for i in range(50)]),
                          "B": rows(*[f"b{i}" for i in range(50)])}, n=10, seed=17)
    assert len(picks) == 20
    assert sum(1 for name, _ in picks if name == "A") == 10
    assert sum(1 for name, _ in picks if name == "B") == 10


def test_sample_blind_takes_everything_when_the_set_is_smaller_than_n():
    picks = sample_blind({"A": rows("a1", "a2")}, n=150, seed=17)
    assert len(picks) == 2


def test_sample_blind_is_seed_reproducible():
    sets = {"A": rows(*[f"a{i}" for i in range(40)]), "B": rows(*[f"b{i}" for i in range(40)])}
    assert sample_blind(sets, 10, 17) == sample_blind(sets, 10, 17)


def test_a_different_seed_draws_a_different_sample():
    sets = {"A": rows(*[f"a{i}" for i in range(60)]), "B": rows(*[f"b{i}" for i in range(60)])}
    assert sample_blind(sets, 10, 17) != sample_blind(sets, 10, 99)


def test_sample_blind_interleaves_the_sets_rather_than_grouping_them():
    """The blind requirement in one assertion: a judge that got stricter as the stream went on
    would otherwise show up as a difference between A and B purely from ordering."""
    sets = {"A": rows(*[f"a{i}" for i in range(40)]), "B": rows(*[f"b{i}" for i in range(40)])}
    names = [name for name, _ in sample_blind(sets, 20, 17, shuffle=True)]
    assert names != ["A"] * 20 + ["B"] * 20
    # Both sets must appear in both halves, so neither can be a prefix of the stream.
    assert "B" in names[:20] and "A" in names[20:]


def test_sample_blind_grouped_order_is_available_for_the_repeat_pass():
    sets = {"A": rows("a1", "a2"), "B": rows("b1", "b2")}
    names = [name for name, _ in sample_blind(sets, 5, 17, shuffle=False)]
    assert names == ["A", "A", "B", "B"]


def test_sample_blind_of_an_empty_set_contributes_nothing():
    assert sample_blind({"A": [], "B": rows("b1")}, n=5, seed=17) == [("B", rows("b1")[0])]


# --- prompt construction ----------------------------------------------------------------------

def test_user_prompt_carries_the_question_and_the_files():
    prompt = user_prompt({"task_id": "t", "question": "What is the mean?",
                          "files": ["a.csv", "b.csv"]})
    assert "a.csv, b.csv" in prompt and "What is the mean?" in prompt


def test_files_description_truncates_a_long_file_list():
    description = files_description({"files": [f"f{i}.csv" for i in range(20)]})
    assert "f5.csv" in description and "20 files total" in description


def test_files_description_of_a_row_with_no_files_says_so():
    assert files_description({"files": []}) == "no files listed"
    assert files_description({}) == "no files listed"


# --- failure handling ---------------------------------------------------------------------------

def test_a_failed_call_becomes_an_errored_row_not_an_exception():
    """One lost OpenRouter call must cost one sample, not the run: `judge_one` is called from a
    thread pool and an exception there would abort the whole sweep."""
    class Boom:
        def __getattr__(self, name):
            raise RuntimeError("no route")

    result = judge_one({"task_id": "t", "question": "q", "files": []}, "A", ep=Boom(),
                       attempts=1)
    assert all(getattr(result, c) is None for c in CRITERIA)
    assert "no route" in result.error


def test_a_transport_failure_is_retried_up_to_the_attempt_budget():
    """`call_model` is what raises, and it is retried: a transient 429 is a lost sample, not a
    lost sweep."""
    import smol_ladder.judge_quality as jq

    calls = []
    original = jq.call_model

    def failing(*args, **kwargs):
        calls.append(1)
        raise RuntimeError("no route")

    jq.call_model = failing
    try:
        result = jq.judge_one({"task_id": "t", "question": "q", "files": []}, "A",
                               attempts=3)
    finally:
        jq.call_model = original
    assert len(calls) == 3
    assert result.answerable is None and "no route" in result.error


def _completion(content, finish_reason):
    return {"choices": [{"message": {"content": content}, "finish_reason": finish_reason}]}


def test_a_truncated_answer_is_retried_with_a_larger_budget():
    """`space-bunny-alpha` reasons in a hidden block and returns `content: None` with
    `finish_reason: "length"` when the cap is too small. Left unretried that drops exactly the
    questions the model spent longest on -- the long and hard ones -- so the retry doubles the cap
    until the model actually answers."""
    import smol_ladder.judge_quality as jq

    seen = []
    original = jq.call_model

    def fake(messages, model, tools, ep=None, max_tokens=None):
        seen.append(max_tokens)
        if len(seen) == 1:
            return _completion(None, "length")
        return _completion('{"answerable": true, "unambiguous": true, "determinate": true,'
                           ' "requires_compute": true, "reason": "ok"}', "stop")

    jq.call_model = fake
    try:
        result = jq.judge_one({"task_id": "t", "question": "q", "files": []}, "A")
    finally:
        jq.call_model = original
    assert len(seen) == 2 and seen[1] > seen[0]
    assert result.answerable is True and result.error == ""


def test_a_truncation_that_never_clears_is_recorded_as_an_error():
    """Retrying must not loop forever on a model that will never produce JSON: after the budget
    the row is an error, so it is visible in the report rather than scored as a failure."""
    import smol_ladder.judge_quality as jq

    original = jq.call_model
    jq.call_model = lambda *a, **k: _completion(None, "length")
    try:
        result = jq.judge_one({"task_id": "t", "question": "q", "files": []}, "A", attempts=2)
    finally:
        jq.call_model = original
    assert result.answerable is None and result.error


def test_an_unparseable_answer_that_was_not_truncated_is_not_retried():
    """A wrong-but-complete answer is a judgement about the judge's formatting, not a budget
    problem, so spending three calls on it would just triple the cost of the same mistake."""
    import smol_ladder.judge_quality as jq

    calls = []
    original = jq.call_model

    def fake(*args, **kwargs):
        calls.append(1)
        return _completion("I think it is fine.", "stop")

    jq.call_model = fake
    try:
        result = jq.judge_one({"task_id": "t", "question": "q", "files": []}, "A")
    finally:
        jq.call_model = original
    assert len(calls) == 1
    assert result.answerable is None and "no JSON" in result.error