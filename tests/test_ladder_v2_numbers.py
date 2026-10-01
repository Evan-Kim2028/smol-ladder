"""Every documented number in LADDER.md's v2 section is a test.

The v2 numbers were quoted by hand into the write-up from one run of the summarise tool, which is
exactly how the withdrawn 92% claim happened: a figure copied out of a log into prose, with no
artefact that can be recomputed when the tool changes. These tests do not recompute the whole run
-- they pin the arithmetic each published number rests on, so a change to the denominators, the
CI, the pairing or the headroom set moves the documented number instead of silently leaving it.

The run itself is not checked in (data/ is git-ignored and the tree is tens of GB), so the tables
in LADDER.md carry the exact command that produced them, and the claims that are not arithmetic
-- the bias in the referenced set, the ceiling -- are argued in prose and restated here as the
invariants that make them true.
"""

import pytest

from smol_ladder import summarize as S

HASH = "a" * 64
FINGERPRINT = "b" * 64


def trial(reward=0.0, status="exit 0", **extra):
    return {"reward": reward, "agent_status": status, "prediction": str(reward),
            "prompt_sha256": HASH, **extra}


def tree(**rungs):
    return {task: {rung: list(ts) for rung, ts in per.items()} for task, per in rungs.items()}


# --- the denominators the v2 tables are quoted on ----------------------------------------------

def test_l2_is_gated_on_a_reference_so_the_two_populations_really_do_differ():
    """The premise of the common-set table. If L1 and L2 were ever run on the same tasks this
    would stop being why the table exists, and the doc's claim that "L1 over 250 is not
    comparable with L2 over 213" would need withdrawing."""
    runs = tree(referenced={"L1": [trial(0.0)], "L2": [trial(1.0)]},
                unreferenced={"L1": [trial(0.0)]})
    report = S.summarise("test", runs, lambda t: t == "referenced")
    assert report["common_set"]["L1"]["scored_tasks"] == 1
    assert report["outside_common_set"]["L1"]["scored_tasks"] == 1
    assert report["outside_common_set"]["L2"]["scored_tasks"] == 0
    assert report["rungs"]["L1"]["tasks"] == 2


def test_the_common_set_curve_is_one_population_end_to_end():
    """Every rung in the common-set table must carry the same denominator, or it is the
    per-rung table with extra steps."""
    runs = tree(**{f"t{i}": {"L1": [trial(0.0), trial(1.0)], "L2": [trial(1.0), trial(1.0)],
                              "L3": [trial(1.0), trial(1.0)], "L4": [trial(1.0), trial(1.0)],
                              S.CONTROL: [trial(1.0), trial(1.0)]} for i in range(6)})
    common = S.common_set(runs, lambda _t: True)
    denominators = {common[rung]["scored_tasks"] for rung in S.ALL}
    assert denominators == {6}


def test_a_rung_harness_failure_leaves_the_denominator_it_leaves():
    """v2 quotes 452/500 L1 trials scored. The arithmetic is 500 - 48, so a harness failure has
    to cost the trial its place in the denominator and not merely its numerator."""
    runs = tree(**{"good": {"L1": [trial(1.0), trial(1.0)]},
                   "crashed": {"L1": [trial(0.0, verify_status="verify timeout"),
                                      trial(0.0, status="timeout")]}})
    block = S.rung_stats(runs)["L1"]
    assert (block["trials"], block["trials_scored"], block["harness_failures"]) == (4, 2, 2)
    assert block["scored_tasks"] == 1


# --- the ceiling argument, which is the one that cuts against the project ------------------------

def test_a_task_that_passes_l1_in_both_samples_cannot_show_a_hint_effect():
    """The bias the v2 section states against our own interest. A task the model solves from the
    question alone is 1.0 at every rung, so including it in the curve adds a constant and can
    only flatten the rungs' apparent effect."""
    runs = tree(
        ceiling={"L1": [trial(1.0), trial(1.0)], "L2": [trial(1.0), trial(1.0)]},
        rescued={"L1": [trial(0.0), trial(0.0)], "L2": [trial(1.0), trial(1.0)]},
    )
    full = S.rung_stats(runs)
    headroom = S.headroom_curve(runs)
    # Blended, the L2 gain over L1 is half what it is where a hint can act.
    blended = full["L2"]["mean_pass_probability"] - full["L1"]["mean_pass_probability"]
    restricted = (headroom["L2"]["mean_pass_probability"]
                  - headroom["L1"]["mean_pass_probability"])
    assert restricted > blended
    assert S.ceiling(runs)["passed_l1_both"] == 1


def test_the_ceiling_and_the_headroom_set_partition_the_measured_tasks():
    """Both counts are quoted side by side, so they have to account for the same tasks. A task
    measured once belongs to neither, and the doc's "and X measured once" row is that remainder."""
    runs = tree(a={"L1": [trial(1.0), trial(1.0)]},
                b={"L1": [trial(0.0), trial(0.0)]},
                c={"L1": [trial(1.0)]},
                d={"L1": [trial(0.0, status="timeout")]})
    block = S.ceiling(runs)
    assert (block["passed_l1_both"], block["failed_l1_at_least_once"],
            block["unmeasured"]) == (1, 1, 1)
    # d contributed no scored sample, so it is not in any of the three.
    assert sum(block[k] for k in ("passed_l1_both", "failed_l1_at_least_once",
                                  "unmeasured")) == 3


# --- the paired control effect -------------------------------------------------------------------

def test_the_control_effect_is_a_difference_of_pass_probabilities_not_of_task_counts():
    """v2's k=2 means a task's arm value is a fraction, so the effect is the mean of per-task
    differences. Summing passes and dividing by task counts would weight a task by its samples."""
    runs = tree(
        half={"L1": [trial(0.0), trial(1.0)], S.CONTROL: [trial(1.0), trial(1.0)]},
        zero={"L1": [trial(0.0), trial(0.0)], S.CONTROL: [trial(0.0), trial(0.0)]},
    )
    effect = S.control_effect(runs)
    # half: L1 scores 1 of 2, control 2 of 2, so +0.5. zero: 0.0. The mean of the per-task
    # deltas is 0.25; a pooled pass count would have read (3 control passes - 1 L1 pass) over
    # 4 trials = 0.5, which is the weighting this avoids.
    assert effect["mean_delta"] == pytest.approx(0.25)
    assert effect["paired"] == 2 and effect["discordant"] == 1


def test_the_control_effect_excludes_the_control_from_the_ladder_curve():
    """The control adds no information, so it is not a rung and a control rescue must not be
    counted as L2 saving the task. It appears in this one table and the `control` block."""
    runs = tree(a={"L1": [trial(0.0)], S.CONTROL: [trial(1.0)], "L2": [trial(0.0)]})
    assert S.control_effect(runs)["rescued"] == 1
    assert S.monotonicity(runs)["L1->L2"]["violations"] == 0, "the control leaked into a pair"
    assert S.CONTROL not in S.BUCKETS


# --- consistency ------------------------------------------------------------------------------

def test_consistency_is_reported_for_every_rung_not_just_l1():
    """The v2 table has a row per rung. A consistency figure at L1 alone would leave the reader
    unable to tell whether a rung's width is the model or the rerun."""
    runs = tree(**{f"t{i}": {"L1": [trial(1.0), trial(0.0)], "L2": [trial(1.0), trial(1.0)]}
                   for i in range(4)})
    block = S.consistency(runs)
    assert block["L1"]["agreement"] == 0.0
    assert block["L2"]["agreement"] == 1.0


# --- the hint-source split ----------------------------------------------------------------------

def test_a_rung_with_only_one_hint_hand_reports_that_hand_and_not_a_split():
    """v2's 386 llm-hint and 40 AST-fallback tasks are read as two populations. When one of
    them is empty the table must show the empty one, because "the AST rungs scored the same" is a
    different claim from "the AST rungs were not run"."""
    runs = tree(a={"L2": [trial(1.0, hint_source="llm")]})
    split = S.hint_source_split(runs)["L2"]
    assert set(split["hands"]) == {"llm"}
    assert split["hands"]["llm"]["scored_tasks"] == 1
    assert split["trials"] == 1


def test_the_hint_split_counts_harness_failures_so_the_two_arms_add_up_to_the_rung():
    runs = tree(
        hinted={"L2": [trial(1.0, hint_source="llm")]},
        fallen={"L2": [trial(0.0, hint_source="ast"),
                       trial(0.0, verify_status="verify timeout", hint_source="ast")]},
    )
    split = S.hint_source_split(runs)["L2"]
    assert split["hands"]["llm"]["scored_tasks"] == 1
    assert split["hands"]["ast"]["scored_tasks"] == 1, "a crash must not vanish from the arm's denominator"
    assert split["harness_failures"] == 1
    assert split["trials"] == 3


# --- the recovery the re-verification bought -----------------------------------------------------

def test_a_recovered_trial_enters_the_denominator_it_was_missing_from():
    """The whole point of re-verifying. Before, these trials were harness failures and left every
    pass rate; after, they are scored observations, which is what makes v2's denominators bigger
    than the first pass's."""
    crashed = {"a": {"L1": [{"reward": 0.0, "agent_status": "exit 0",
                             "verify_status": "verify timeout"}]}}
    recovered = {"a": {"L1": [{"reward": 0.0, "agent_status": "exit 0",
                              "verify_status": "verify timeout",
                              "reverify": {"new_verify_status": "exit 0", "reward": 1.0,
                                           "prediction": "42"}}]}}
    before = S.rung_stats(crashed)["L1"]
    after = S.rung_stats(recovered)["L1"]
    assert (before["trials_scored"], before["harness_failures"]) == (0, 1)
    assert (after["trials_scored"], after["harness_failures"]) == (1, 0)
    assert after["mean_pass_probability"] == 1.0


def test_a_trial_that_failed_twice_is_still_out_and_is_still_counted_as_attempted():
    """Re-verification is not guaranteed to work, and the summary has to say so rather than
    quietly re-including it. `attempts` is the number the doc quotes against `recovered`."""
    runs = {"a": {"L1": [{"reward": 0.0, "agent_status": "exit 0",
                          "verify_status": "verify timeout",
                          "reverify": {"new_verify_status": "verify timeout", "reward": 0.0}}]}}
    report = S.summarise("test", runs, lambda _t: True)
    assert report["reverified"] == {"recovered": 0, "still_failing": 1, "attempts": 1}
    assert report["rungs"]["L1"]["harness_failures"] == 1
    assert report["rungs"]["L1"]["tasks"] == 1


# --- what the v2 numbers must never be read as ---------------------------------------------------

def test_monotonicity_is_reported_per_adjacent_pair_and_not_as_a_single_verdict():
    """v2's L1->L2 result is significant and its L2->L3 result is not. Collapsing them into "the
    ladder is monotone" would discard the only finding in the table."""
    runs = tree(
        rising={"L1": [trial(0.0)], "L2": [trial(1.0)], "L3": [trial(1.0)]},
        falling={"L1": [trial(1.0)], "L2": [trial(0.0)], "L3": [trial(0.0)]},
    )
    mono = S.monotonicity(runs)
    assert mono["L1->L2"]["violations"] == 1 and mono["L1->L2"]["p_value"] is not None
    assert set(mono) == {"L1->L2", "L2->L3", "L3->L4"}
    assert S.CONTROL not in "".join(mono)


def test_a_single_violation_is_never_called_monotone_on_its_own():
    """The doc says the drops are findings, not noise. A run with zero violations reports no
    p-value at all, because a sign test over no discordant pairs is undefined and p=1.0 would
    read as a measurement."""
    runs = tree(**{f"t{i}": {"L1": [trial(1.0)], "L2": [trial(1.0)]} for i in range(3)})
    assert S.monotonicity(runs)["L1->L2"]["p_value"] is None


@pytest.mark.parametrize("rung", S.ALL)
def test_every_rung_appears_in_the_common_set_table(rung):
    """The doc prints one row per rung, control included. A rung missing from the table would be
    a rung the reader has no number for, on the one table that is comparable."""
    runs = tree(a={"L1": [trial(0.0)], "L2": [trial(0.0)], "L3": [trial(0.0)], "L4": [trial(0.0)],
                   S.CONTROL: [trial(0.0)]})
    assert rung in S.common_set(runs, lambda _t: True)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
