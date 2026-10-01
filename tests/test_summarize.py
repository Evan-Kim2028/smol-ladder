"""The summary's tables must be partitions and disjoint, because the old ones were neither.

Every test here builds a synthetic result tree, so it runs in milliseconds and touches nothing
on disk. Two of the cases are the exact shapes the old code got wrong — a control rescue with no
L2 result, and a task with no result at all — and there is one test that drives the *old* bucket
names through the new partition, so the double-count cannot come back under a rename.

The trees are lists of samples per (task, rung), since that is what the runner writes now; a
one-element list is the k=1 case the old tests were written for.

    uv run --with pytest pytest -q tests/test_summarize.py
"""

import pytest

from smol_ladder import summarize as S

Control = "L1+schema"          # the control's name in a prompt, the spelling the old code used


def trial(reward=0.0, status="exit 0", hash_=None):
    """One sample. The hash defaults to None so these trees exercise the no-provenance path."""
    return {"reward": reward, "agent_status": status, "prediction": str(reward),
            **({"prompt_sha256": hash_} if hash_ else {})}


def old_hist(runs, has_reference):
    """summarize.py as it stood before the rewrite, verbatim in the part that was wrong.

    Kept here as the specification of the failure: a task the control rescued incremented two
    buckets, and a task with no L1 result incremented none. Several tests below feed the same tree
    through this and through the real partition, so the old numbers stay pinned without pinning
    the old code.
    """
    hist = {k: 0 for k in ["L1", Control, "L2", "L3", "L4", "never", "no reference"]}
    for task, rungs in runs.items():
        if "L1" not in rungs:
            continue
        if rungs["L1"][0]["reward"] >= 1.0:
            hist["L1"] += 1
            continue
        if not has_reference(task):
            hist["no reference"] += 1
            continue
        if (rungs.get(Control) or [{}])[0].get("reward", 0.0) >= 1.0:
            hist[Control] += 1                      # no `continue`: the double-count
        first = next((r for r in ["L2", "L3", "L4"]
                      if (rungs.get(r) or [{}])[0].get("reward", 0.0) >= 1.0), None)
        hist[first or "never"] += 1
    return hist


# --- the partition -------------------------------------------------------------------------

def test_a_control_rescue_lands_in_exactly_one_bucket():
    """Failed L1, rescued by the control, no rung above L1 on disk.

    This is the shape that made 250 tasks summarise to 266: the old histogram booked the task
    under the control and again under `never`.
    """
    runs = {"t": {"L1": [trial(0.0)], Control: [trial(1.0)]}}
    assert old_hist(runs, lambda _t: True) == {"L1": 0, Control: 1, "L2": 0, "L3": 0, "L4": 0,
                                               "never": 1, "no reference": 0}
    assert S.partition(runs, lambda _t: True)["never"] == 1
    assert sum(S.partition(runs, lambda _t: True).values()) == 1


def test_the_control_is_not_a_bucket_of_the_partition():
    runs = {"t": {"L1": [trial(0.0)], Control: [trial(1.0)], "L2": [trial(1.0)]}}
    assert S.first_passing_rung(runs["t"], has_reference=True) == "L2"
    assert Control not in S.BUCKETS


def test_a_control_rescue_and_an_l2_pass_do_not_double_count():
    runs = {"a": {"L1": [trial(0.0)], Control: [trial(1.0)], "L2": [trial(1.0)]},
            "b": {"L1": [trial(1.0)]},
            "c": {"L1": [trial(0.0)], Control: [trial(1.0)]}}
    hist = S.partition(runs, lambda _t: True)
    assert hist == {"L1": 1, "L2": 1, "L3": 0, "L4": 0, "never": 1,
                    "not climbable (no reference)": 0,
                    "not scored (every trial was a harness failure)": 0,
                    "not attempted": 0}
    assert sum(hist.values()) == 3


def test_first_passing_rung_takes_the_lowest_rung_that_passed():
    runs = {"t": {"L1": [trial(0.0)], "L2": [trial(0.0)], "L3": [trial(1.0)], "L4": [trial(1.0)]}}
    assert S.first_passing_rung(runs["t"], has_reference=True) == "L3"


def test_never_and_not_climbable_are_different_findings():
    """`never` means every climbable rung was tried and failed; no reference means none was tried."""
    climbed = {"L1": [trial(0.0)], "L2": [trial(0.0)], "L3": [trial(0.0)], "L4": [trial(0.0)]}
    assert S.first_passing_rung(climbed, has_reference=True) == "never"
    assert S.first_passing_rung({"L1": [trial(0.0)]}, has_reference=False) == \
        "not climbable (no reference)"
    tree = {"a": climbed, "b": {"L1": [trial(0.0)]}}
    assert old_hist(tree, lambda t: t == "a")["no reference"] == 1
    assert old_hist(tree, lambda t: t == "a")["never"] == 1   # the two findings are conflated


def test_a_task_no_rung_was_run_on_is_not_booked_as_a_failure():
    """The old loop `continue`d on a missing L1 result, so the task dropped out of the histogram
    and the sum came up short instead of over. It belongs in `not attempted`."""
    runs = {"t": {}}
    assert sum(old_hist(runs, lambda _t: True).values()) == 0   # dropped on the floor
    assert S.partition(runs, lambda _t: True)["not attempted"] == 1


def test_the_partition_sums_to_the_task_count_on_a_mixed_tree():
    runs = {
        "passes": {"L1": [trial(1.0)]},
        "climbs": {"L1": [trial(0.0)], Control: [trial(0.0)], "L2": [trial(1.0)]},
        "hard": {"L1": [trial(0.0)], Control: [trial(1.0)], "L2": [trial(0.0)], "L3": [trial(1.0)]},
        "never": {"L1": [trial(0.0)], "L2": [trial(0.0)]},
        "crashed": {"L1": [trial(0.0, status="timeout")]},
        "unattempted": {},
    }
    refs = {"passes", "climbs", "hard", "never"}
    report = S.summarise("t", runs, lambda task: task in refs)
    assert sum(report["first_passing_rung"].values()) == len(runs)
    # "crashed" is neither an L1 pass nor an L1 failure: its only trial was a harness failure,
    # and booking it either way would put the harness's health in the model's pass rate.
    assert report["first_passing_rung"] == {"L1": 1, "L2": 1, "L3": 1, "L4": 0, "never": 1,
                                            "not climbable (no reference)": 0,
                                            "not scored (every trial was a harness failure)": 1,
                                            "not attempted": 1}


# --- the control ---------------------------------------------------------------------------

def test_control_is_reported_apart_from_the_ladder():
    runs = {"a": {"L1": [trial(0.0)], Control: [trial(1.0)]},
            "b": {"L1": [trial(0.0)], Control: [trial(0.0)]},
            "c": {"L1": [trial(0.0)], Control: [trial(1.0)]},
            "d": {"L1": [trial(1.0)]}}
    block = S.control_block(runs, lambda t: t in {"a", "b"})
    assert block["attempted"] == 3
    assert block["rescued"] == 2
    assert block["with reference"] == {"attempted": 2, "rescued": 1}
    assert block["without reference"] == {"attempted": 1, "rescued": 1}
    assert block["harness_failures"] == 0


def test_control_harness_failures_are_counted_not_scored():
    runs = {"a": {"L1": [trial(0.0)], Control: [trial(0.0, status="timeout")]}}
    block = S.control_block(runs, lambda _t: True)
    assert block["rescued"] == 0
    assert block["harness_failures"] == 1
    assert block["attempted"] == 1


def test_the_control_rate_is_over_attempted_not_over_l1_failures():
    """The control is gated on nothing, so its denominator is its own trial count."""
    runs = {"a": {"L1": [trial(1.0)], Control: [trial(1.0)]},
            "b": {"L1": [trial(0.0)], Control: [trial(0.0)]},
            "c": {"L1": [trial(0.0)], Control: [trial(0.0)]}}
    block = S.control_block(runs, lambda _t: True)
    assert block["attempted"] == 3 and block["rescued"] == 1


# --- per-rung counts -----------------------------------------------------------------------

def test_per_rung_counts_separate_harness_failures_from_model_failures():
    runs = {"a": {"L1": [trial(1.0)]}, "b": {"L1": [trial(0.0)]},
            "c": {"L1": [trial(0.0, status="timeout")]},
            "d": {"L1": [trial(0.0, status="error: TimeoutError")]}}
    block = S.rung_stats(runs)["L1"]
    assert (block["tasks"], block["harness_failures"], block["trials_scored"]) == (4, 2, 2)
    assert S.rung_stats(runs)["L4"]["tasks"] == 0


def test_every_bucket_is_present_even_when_zero():
    hist = S.partition({"a": {"L1": [trial(1.0)]}}, lambda _t: True)
    assert list(hist) == S.BUCKETS
    assert hist["L4"] == 0


def test_a_pass_needs_reward_one():
    assert S.first_passing_rung({"L1": [trial(1.0)]}, has_reference=True) == "L1"
    assert S.first_passing_rung({"L1": [trial(0.999)]}, has_reference=True) == "never"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))