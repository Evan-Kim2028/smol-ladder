"""The analyses the v2 write-up needs and the summary did not have.

The headline table is per rung over the tasks scored at that rung, and that is the wrong
denominator for a ladder curve: L2-L4 are gated on a verified reference, so they run on 213 of
250 tasks while L1 runs on 250, and a curve read off two populations is not a curve. So the
common-set table here restricts every rung to the same tasks, and reports the tasks outside it
separately instead of folding them in.

The rest are the questions a pass-rate curve cannot answer on its own: whether the control beats
the plain prompt, whether two samples of the same cell agree, whether the hint hand (a model or
the AST fallback) matters, and -- the one that decides whether any of this says anything --
how much headroom the hints have left once the tasks the model already solves are removed.
"""

import pytest

from smol_ladder import summarize as S

HASH = "a" * 64


def trial(reward=0.0, status="exit 0", **extra):
    return {"reward": reward, "agent_status": status, "prediction": str(reward),
            "prompt_sha256": HASH, **extra}


def tree(**rungs):
    return {task: {rung: list(ts) for rung, ts in per.items()} for task, per in rungs.items()}


# --- (a) the common set ------------------------------------------------------------------------

def test_every_rung_is_reported_on_the_same_task_set():
    """The comparison the per-rung table cannot make. L2 is gated on a reference and L1 is not,
    so a curve read off their own denominators compares two populations."""
    runs = tree(
        climbable={"L1": [trial(0.0)], "L2": [trial(1.0)], "L3": [trial(1.0)], "L4": [trial(1.0)]},
        unreferenced={"L1": [trial(1.0)], S.CONTROL: [trial(1.0)]},
    )
    common = S.common_set(runs, lambda t: t == "climbable")
    assert common["L1"]["scored_tasks"] == 1
    assert common["L2"]["scored_tasks"] == 1
    # L1 on the common set is 0.0 and on its own 250 it would be 0.5 -- the whole point.
    assert common["L1"]["mean_pass_probability"] == 0.0
    assert S.rung_stats(runs)["L1"]["mean_pass_probability"] == 0.5


def test_the_tasks_outside_the_common_set_are_reported_on_their_own():
    """Dropping them silently would hide 37 tasks and the control's whole behaviour there."""
    runs = tree(
        climbable={"L1": [trial(0.0)], "L2": [trial(1.0)]},
        unreferenced={"L1": [trial(1.0)], S.CONTROL: [trial(1.0)]},
    )
    outside = S.common_set(runs, lambda t: t == "climbable", inside=False)
    assert outside["L1"]["scored_tasks"] == 1
    assert outside["L1"]["mean_pass_probability"] == 1.0
    # L2 was never run on the unreferenced task, so the outside set has no L2 at all.
    assert outside["L2"]["scored_tasks"] == 0


def test_a_rung_run_on_no_task_in_the_set_carries_nan_not_a_zero():
    """A rung with no measurements must not read as 0%: the control outside the referenced set is
    exactly this, and a zero would say the model failed every one of them."""
    runs = tree(climbable={"L1": [trial(1.0)]})
    common = S.common_set(runs, lambda t: t == "climbable")
    assert common["L4"]["scored_tasks"] == 0
    assert common["L4"]["mean_pass_probability"] != common["L4"]["mean_pass_probability"]


def test_the_two_sets_partition_the_tasks():
    runs = tree(a={"L1": [trial(1.0)]}, b={"L1": [trial(0.0)], "L2": [trial(1.0)]})
    inside = S.common_set(runs, lambda t: t == "b", inside=True)
    outside = S.common_set(runs, lambda t: t == "b", inside=False)
    assert inside["L1"]["scored_tasks"] + outside["L1"]["scored_tasks"] == 2


# --- (b) the paired control effect ---------------------------------------------------------------

def test_the_control_effect_is_paired_per_task():
    """Paired, because the control and L1 run on the same tasks: the question is what the schema
    dump added to a given task, not whether two independent rates differ."""
    runs = tree(
        rescued={"L1": [trial(0.0)], S.CONTROL: [trial(1.0)]},
        lost={"L1": [trial(1.0)], S.CONTROL: [trial(0.0)]},
        same={"L1": [trial(1.0)], S.CONTROL: [trial(1.0)]},
        unpaired={"L1": [trial(1.0)]},
    )
    effect = S.control_effect(runs)
    assert effect["paired"] == 3
    assert effect["rescued"] == 1 and effect["hurt"] == 1
    assert effect["mean_delta"] == pytest.approx(0.0)
    assert effect["excluded_unpaired"] == 1


def test_the_control_effect_carries_a_sign_test_and_an_interval():
    runs = tree(**{f"t{i}": {"L1": [trial(0.0)], S.CONTROL: [trial(1.0)]} for i in range(8)})
    effect = S.control_effect(runs)
    assert effect["discordant"] == 8
    assert effect["p_value"] == pytest.approx(2 * 0.5 ** 8)
    low, high = effect["ci95"]
    assert low <= effect["mean_delta"] <= high


def test_a_control_with_no_discordant_task_reports_no_p_value():
    """Every task did the same thing either way. A sign test over zero discordant pairs is
    undefined, and p=1.0 would read as a measurement of no effect."""
    runs = tree(**{f"t{i}": {"L1": [trial(1.0)], S.CONTROL: [trial(1.0)]} for i in range(4)})
    assert S.control_effect(runs)["p_value"] is None


def test_a_harness_failure_does_not_become_a_control_regression():
    """A crashed control trial is not evidence the schema dump hurt. Dropping the task from the
    pair keeps the harness's health out of the control's effect."""
    runs = tree(a={"L1": [trial(0.0)], S.CONTROL: [trial(0.0, status="timeout")]},
                b={"L1": [trial(0.0)], S.CONTROL: [trial(1.0)]})
    effect = S.control_effect(runs)
    assert effect["paired"] == 1 and effect["rescued"] == 1 and effect["hurt"] == 0


# --- (c) rerun consistency -----------------------------------------------------------------------

def test_consistency_is_the_share_of_tasks_whose_samples_agree():
    """At k=2 two samples either agree or they do not, and the share that disagree is the noise
    floor every pass probability sits on."""
    runs = tree(agree={"L1": [trial(1.0), trial(1.0)]},
                disagree={"L1": [trial(1.0), trial(0.0)]},
                single={"L1": [trial(1.0)]})
    block = S.consistency(runs)["L1"]
    assert block["tasks"] == 2, "a single-sample task cannot agree with itself"
    assert block["agree"] == 1 and block["disagree"] == 1
    assert block["agreement"] == pytest.approx(0.5)


def test_consistency_needs_at_least_two_scored_samples():
    """A task whose other sample crashed has one observation, and calling that agreement would
    credit the harness for a measurement it never made."""
    runs = tree(a={"L1": [trial(1.0), trial(1.0)]},
                b={"L1": [trial(1.0), trial(0.0, status="timeout")]})
    assert S.consistency(runs)["L1"]["tasks"] == 1


# --- (d) the hint source ------------------------------------------------------------------------

def test_the_hint_source_split_separates_a_model_hint_from_the_ast_fallback():
    """L2-L4 text comes from a validated model hint or, where there is none, from the AST
    extractor. They are different instruments measuring the same rung, and a curve that blends
    them is averaging over the instrument as well as the rung."""
    runs = tree(
        hinted={"L2": [trial(1.0, hint_source="llm")], "L3": [trial(1.0, hint_source="llm")]},
        fallback={"L2": [trial(0.0, hint_source="ast")], "L3": [trial(0.0, hint_source="ast")]},
    )
    split = S.hint_source_split(runs)["L2"]["hands"]
    assert split["llm"]["scored_tasks"] == 1
    assert split["llm"]["mean_pass_probability"] == 1.0
    assert split["ast"]["mean_pass_probability"] == 0.0


def test_a_task_with_no_hint_source_is_booked_under_none_not_dropped():
    """gen_refs calls once() without a hint split, so results exist that never say which hand
    wrote the text. Dropping them would quietly shrink the denominator."""
    runs = tree(a={"L2": [trial(1.0)]})
    assert S.hint_source_split(runs)["L2"]["hands"]["none"]["scored_tasks"] == 1


# --- (e) the ceiling ----------------------------------------------------------------------------

def test_the_ceiling_counts_the_referenced_tasks_that_already_pass_l1_twice():
    """A task the model solves from the question alone has no headroom for a hint, so it cannot
    contribute to a hint effect however many rungs it passes."""
    runs = tree(
        solved={"L1": [trial(1.0), trial(1.0)], "L2": [trial(1.0), trial(1.0)]},
        flaky={"L1": [trial(1.0), trial(0.0)], "L2": [trial(1.0), trial(1.0)]},
        failed={"L1": [trial(0.0), trial(0.0)], "L2": [trial(1.0), trial(1.0)]},
    )
    ceiling = S.ceiling(runs)
    assert ceiling["passed_l1_both"] == 1
    assert ceiling["failed_l1_at_least_once"] == 2
    assert ceiling["headroom"] == pytest.approx(2 / 3)


def test_a_task_solved_on_one_l1_sample_of_two_still_has_headroom():
    """Half the point of k=2. A task that passed once has not demonstrated the skill, and
    counting it as ceiling would shrink the very set the hint effect is measured on."""
    runs = tree(solved={"L1": [trial(1.0), trial(1.0)]},
                flaky={"L1": [trial(1.0), trial(0.0)]})
    assert S.ceiling(runs)["passed_l1_both"] == 1
    assert S.ceiling(runs)["failed_l1_at_least_once"] == 1


def test_the_curve_restricted_to_tasks_with_headroom_drops_the_solved_tasks():
    """The number that says whether a hint can do anything at all. On a set where the model
    already passes L1 in both samples, every rung reads as a pass and the curve is flat by
    construction."""
    runs = tree(
        solved={"L1": [trial(1.0), trial(1.0)], "L2": [trial(1.0), trial(1.0)]},
        hard={"L1": [trial(0.0), trial(0.0)], "L2": [trial(0.0), trial(0.0)],
              "L3": [trial(1.0), trial(1.0)]},
    )
    restricted = S.headroom_curve(runs)
    assert restricted["L1"]["scored_tasks"] == 1
    assert restricted["L1"]["mean_pass_probability"] == 0.0
    assert restricted["L3"]["mean_pass_probability"] == 1.0


def test_the_ceiling_ignores_a_task_with_one_l1_sample():
    """One sample cannot establish that the model always gets it, so it is neither certain
    ceiling nor certain headroom -- it is unmeasured and stays out of both counts."""
    runs = tree(single={"L1": [trial(1.0)]}, solved={"L1": [trial(1.0), trial(1.0)]})
    ceiling = S.ceiling(runs)
    assert ceiling["passed_l1_both"] == 1
    assert ceiling["failed_l1_at_least_once"] == 0
    assert ceiling["unmeasured"] == 1


# --- the report carries all of it ---------------------------------------------------------------

def test_the_report_carries_every_analysis():
    runs = tree(a={"L1": [trial(1.0), trial(1.0)], "L2": [trial(1.0), trial(1.0)],
                   S.CONTROL: [trial(1.0), trial(1.0)]},
                b={"L1": [trial(0.0), trial(0.0)], "L2": [trial(0.0), trial(0.0)]})
    report = S.summarise("test", runs, lambda t: t == "a")
    assert {"common_set", "outside_common_set", "control_effect", "consistency",
            "hint_source", "ceiling", "headroom_curve"} <= set(report)
    assert report["common_set"]["L1"]["scored_tasks"] == 1
    assert report["outside_common_set"]["L1"]["scored_tasks"] == 1
    assert report["ceiling"]["passed_l1_both"] == 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
