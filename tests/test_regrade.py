"""The ANSWER:-prefix defect: our prompt never asked for the prefix, and our grader never
strips it, so a right answer died on a label."""

from smol_ladder.grade import grade
from smol_ladder.ladder import PROMPT
from smol_ladder.or_agent import SYSTEM
from smol_ladder.regrade import REREDUCED_PREFIX, regrade_prediction, report


ROW = {"task_id": "t1", "answer": "Size(sqf)", "reward_mode": "exact_short",
       "atol": 0.0, "rtol": 0.0}


def test_the_upstream_grader_does_not_strip_the_prefix_so_we_must_not_either():
    """The fact the whole decision turns on, checked against the shipped grader rather than
    against a comment about it.

    SmolDataEnvs' own grader normalises case and whitespace and nothing else, and its Harbor
    verifier pipes /workdir/answer.txt into it verbatim. So upstream has no normalisation pass
    of its own that we would be duplicating, and a prediction that grades 0.0 upstream grades
    0.0 here. Making ours stricter than the benchmark would make our pass rate measure our
    grader, so the fix has to be in the prompt.
    """
    from smol_ladder.grade import _grader

    assert _grader()._normalize("ANSWER: Size(sqf)") != _grader()._normalize("Size(sqf)")
    assert grade(ROW, "ANSWER: Size(sqf)") == 0.0


def test_our_prompts_ask_for_the_bare_value_and_forbid_any_label():
    """The half of the fix that changes model behaviour, pinned where it lives.

    Both prompts already said "the final answer is just the value". That left the common
    reporting idiom ("ANSWER: 42") readable as compliant, and 15 stored test predictions took
    it. The instruction now names the line's shape and rules out anything before the value.
    """
    for prompt in (PROMPT, SYSTEM):
        assert "ANSWER:" in prompt.upper(), "the prompt must name the idiom it is ruling out"
        assert '"Answer: 42" grades as 0' in prompt, prompt
    assert "graded on its own" in PROMPT, "the ladder prompt must say the line is what counts"


def test_regrade_counts_a_prefixed_prediction_without_pretending_the_grader_accepted_it():
    """regrade is a measurement tool, so it reports both numbers and never overwrites one."""
    assert REREDUCED_PREFIX.sub("", "ANSWER: Size(sqf)").strip() == "Size(sqf)"
    assert grade(ROW, "ANSWER: Size(sqf)") == 0.0
    assert regrade_prediction(ROW, "ANSWER: Size(sqf)") == (0.0, 1.0)


def test_regrade_leaves_a_prediction_it_has_no_opinion_about_alone():
    """With no label there is nothing to remove, so the two numbers must agree.

    A normalised column that could rise without the prefix being present would book a fix that
    was never applied.
    """
    assert regrade_prediction(ROW, "São Paulo") == (0.0, 0.0)
    assert regrade_prediction(ROW, "ANSWER: United States") == (0.0, 0.0)


def test_report_separates_the_graded_out_prefixes_from_the_harness_failures():
    """A task with no prediction is not a surface-form failure, and a blend hides both.

    The audit read 63 L1 failures on test and split them by hand. A report that counts a timeout
    and a label prefix in one row reproduces exactly the confusion the audit had to undo. The
    stored `reward` is never read, so a stale value in a result.json cannot skew the counts.
    """
    def result(task_id, prediction, reward=0.0, status="exit 0"):
        return {"task_id": task_id, "prediction": prediction, "reward": reward,
                "agent_status": status}

    runs = {"test": {
        "p1": {"L1": result("p1", "ANSWER: yes")},         # prefixed, only the label lost
        "p2": {"L1": result("p2", "ANSWER: maybe")},    # prefixed, wrong either way
        "p3": {"L1": result("p3", "F")},                  # wrong label vocabulary
        "p4": {"L1": result("p4", "", status="timeout")},  # never finished
        "p5": {"L1": result("p5", "")},                   # finished with nothing to grade
        "p6": {"L1": result("p6", "yes")},                # clean pass
    }}
    rows = {
        "p1": {"task_id": "p1", "answer": "yes", "reward_mode": "exact_bool",
               "atol": 0.0, "rtol": 0.0},
        "p2": {"task_id": "p2", "answer": "no", "reward_mode": "exact_bool",
               "atol": 0.0, "rtol": 0.0},
        "p3": {"task_id": "p3", "answer": "Female", "reward_mode": "exact_short",
               "atol": 0.0, "rtol": 0.0},
        "p4": {"task_id": "p4", "answer": "CNN", "reward_mode": "exact_short",
               "atol": 0.0, "rtol": 0.0},
        "p5": {"task_id": "p5", "answer": "Temple", "reward_mode": "exact_short",
               "atol": 0.0, "rtol": 0.0},
        "p6": {"task_id": "p6", "answer": "yes", "reward_mode": "exact_bool",
               "atol": 0.0, "rtol": 0.0},
    }

    got = report(runs["test"], rows)["rungs"]["L1"]

    assert got["graded"] == 4, got                  # p1, p2, p3, p6 produced a line to grade
    assert got["prefixed"] == 2, got                # p1, p2
    assert got["passed_strict"] == 1                # only p6
    assert got["passed_prefix_stripped"] == 2       # p2 joins p6; p1 stays wrong
    assert got["harness_failures"] == 1             # p4 timed out
    assert got["no_prediction"] == 1                # p5 finished with nothing to grade