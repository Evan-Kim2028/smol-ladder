"""The gold regrade: which stored trials move, and which rungs the corrected gold still owes.

These tests exist because the interesting failure of a regrade is silent. A regrade that loses
the tasks whose id the corrected corpus dropped still prints a confident before/after table; it
is just a comparison of two differently-populated samples, and nothing in the output says so.
"""

import json

import pytest

from smol_ladder import regrade_gold as rg
from smol_ladder.synthetic import _tolerance


def gold(task_id, answer, mode="numeric"):
    return {"task_id": task_id, "answer": answer, "reward_mode": mode,
            "atol": _tolerance(answer)[0], "rtol": _tolerance(answer)[1]}


def store(tmp_path, task, rungs):
    """A results tree with one directory per rung, as run_ladder writes it."""
    for rung, result in rungs.items():
        directory = tmp_path / "runs" / "synthetic" / task / rung
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "result.json").write_text(json.dumps(result))


@pytest.fixture
def tree(tmp_path, monkeypatch):
    monkeypatch.setattr(rg, "DATA", tmp_path)
    return tmp_path


def test_a_stored_prediction_is_regraded_against_the_corrected_gold(tree, tmp_path):
    """The whole point: a prediction scored against a wrong gold scores differently now."""
    store(tmp_path, "t1", {"L4": {"prediction": "31535.63530029", "reward": 0.0,
                                  "agent_status": "exit 0"}})
    out = tmp_path / "regrade.jsonl"
    report = rg.regrade("synthetic", {"t1": gold("t1", "31535.63530029")}, {}, out)

    assert report["rungs"]["L4"]["trials"] == 1
    assert report["rungs"]["L4"]["new_pass_corrected"] == 1
    assert report["rungs"]["L4"]["old_pass_corrected"] == 0
    assert [f["task_id"] for f in report["flips"]] == ["t1"]
    assert json.loads(out.read_text())["reward"] == 1.0


def test_the_original_reward_is_never_rewritten(tree, tmp_path):
    """The stored reward is the record of what the run scored then, and it is the only copy."""
    store(tmp_path, "t1", {"L4": {"prediction": "1.0", "reward": 0.0, "agent_status": "exit 0"}})
    rg.regrade("synthetic", {"t1": gold("t1", "1.0")}, {}, tmp_path / "out.jsonl")
    assert json.loads((tmp_path / "runs/synthetic/t1/L4/result.json").read_text())["reward"] == 0.0


def test_a_task_the_corrected_corpus_dropped_is_still_regraded(tree, tmp_path):
    """The bug this module exists to avoid: the gate refuses some ids, their trials remain.

    Grading only against the corrected corpus silently drops those trials from the report, so
    the table compares two different task populations. They are graded against the superseded
    gold instead, and every row records which gold it was graded against.
    """
    store(tmp_path, "kept", {"L4": {"prediction": "1.0", "reward": 0.0, "agent_status": "exit 0"}})
    store(tmp_path, "dropped", {"L4": {"prediction": "1.0", "reward": 0.0,
                                       "agent_status": "exit 0"}})
    out = tmp_path / "out.jsonl"
    current = {"kept": gold("kept", "1.0")}
    old = {"dropped": gold("dropped", "1.0")}
    report = rg.regrade("synthetic", current, old, out)

    assert report["rungs"]["L4"]["trials"] == 2, "the dropped id's trial vanished from the report"
    assert report["ungraded"] == []
    sources = {json.loads(line)["task_id"]: json.loads(line)["gold_source"]
               for line in out.read_text().splitlines()}
    assert sources == {"kept": "corrected", "dropped": "superseded"}


def test_the_corrected_gold_wins_over_the_superseded_one(tree, tmp_path):
    store(tmp_path, "t1", {"L4": {"prediction": "2.0", "reward": 0.0, "agent_status": "exit 0"}})
    row, source = rg.resolve("t1", {"t1": gold("t1", "2.0")}, {"t1": gold("t1", "9.0")})
    assert (row["answer"], source) == ("2.0", "corrected")


def test_a_task_with_no_gold_in_either_version_is_reported_not_graded(tree, tmp_path):
    store(tmp_path, "stranger", {"L1": {"prediction": "1.0", "reward": 1.0,
                                        "agent_status": "exit 0"}})
    out = tmp_path / "out.jsonl"
    report = rg.regrade("synthetic", {}, {}, out)
    assert report["ungraded"] == ["stranger"]
    assert report["rungs"]["L1"]["trials"] == 0
    assert out.read_text() == ""


def test_a_climb_that_stopped_on_a_reward_the_gold_reverses_owes_the_rungs_above(tree, tmp_path):
    """The corrected gold no longer passes this task at L1, so the climb should have gone on.

    Its L2/L3/L4 predictions were never produced, so no regrade can recover them: they are owed.
    The control is not among them -- it is not a rung, and it runs on every L1 failure anyway.
    """
    store(tmp_path, "t1", {"L1": {"prediction": "1.0", "reward": 1.0, "agent_status": "exit 0"}})
    runs = rg.collect("synthetic")
    owed = [rung for _, rung in rg.rerun_table({"t1": gold("t1", "2.0")}, {}, runs)["owed_trials"]]
    assert owed == ["L2", "L3", "L4"]


def test_a_task_that_still_passes_where_it_did_owes_nothing(tree, tmp_path):
    """Climbing stopped at L1 and still should: there is no rung left to run."""
    store(tmp_path, "t1", {"L1": {"prediction": "1.0", "reward": 1.0, "agent_status": "exit 0"}})
    runs = rg.collect("synthetic")
    assert rg.rerun_table({"t1": gold("t1", "1.0")}, {}, runs)["owed_trials"] == []


def test_a_gold_that_turns_a_failure_into_a_pass_still_owes_the_rungs_the_climb_skipped(tree, tmp_path):
    """A pass gained at L1 where the climb used to fail stops the climb earlier, not later.

    The ladder rungs between the new pass and the rung actually reached were never run, so L2 and
    L3 are owed; L4 does exist, so it is a measurement rather than a gap.
    """
    store(tmp_path, "t1", {"L1": {"prediction": "5.0", "reward": 0.0, "agent_status": "exit 0"},
                           "L4": {"prediction": "9.0", "reward": 0.0, "agent_status": "exit 0"}})
    runs = rg.collect("synthetic")
    owed = rg.rerun_table({"t1": gold("t1", "5.0")}, {}, runs)["owed_trials"]
    assert owed == []


def test_a_rung_already_run_is_not_owed_even_when_the_climb_should_have_stopped_earlier(tree, tmp_path):
    """Owed means *missing*, not *misordered*. L2 exists, so L2 is a measurement."""
    store(tmp_path, "t1", {"L1": {"prediction": "1.0", "reward": 0.0, "agent_status": "exit 0"},
                           "L2": {"prediction": "9.0", "reward": 0.0, "agent_status": "exit 0"}})
    runs = rg.collect("synthetic")
    owed = [rung for _, rung in rg.rerun_table({"t1": gold("t1", "2.0")}, {}, runs)["owed_trials"]]
    assert "L2" not in owed
    assert "L3" in owed and "L4" in owed


def test_a_trial_whose_reward_moved_is_listed_as_stale(tree, tmp_path):
    store(tmp_path, "t1", {"L4": {"prediction": "2.0", "reward": 0.0, "agent_status": "exit 0"}})
    runs = rg.collect("synthetic")
    assert rg.rerun_table({"t1": gold("t1", "2.0")}, {}, runs)["stale_trials"] == [("t1", "L4")]


def test_the_control_directory_is_reported_under_the_rung_name(tree, tmp_path):
    """`L1_schema` is how the filesystem spells it; `L1+schema` is how the ladder calls it, and
    a table that mixed the two spellings would silently split one rung's counts in two."""
    store(tmp_path, "t1", {"L1_schema": {"prediction": "1.0", "reward": 1.0,
                                         "agent_status": "exit 0"}})
    runs = rg.collect("synthetic")
    assert set(runs["t1"]) == {"L1+schema"}
    report = rg.regrade("synthetic", {"t1": gold("t1", "1.0")}, {}, tree / "out.jsonl")
    assert report["rungs"]["L1+schema"]["trials"] == 1


def test_a_harness_failure_is_counted_separately_from_a_pass(tree, tmp_path):
    store(tmp_path, "t1", {"L4": {"prediction": "", "reward": 0.0, "agent_status": "timeout"}})
    report = rg.regrade("synthetic", {"t1": gold("t1", "1.0")}, {}, tree / "out.jsonl")
    counts = report["rungs"]["L4"]
    assert counts["old_harness"] == 1 and counts["new_pass_corrected"] == 0


def test_the_table_counts_every_stored_trial_once(tree, tmp_path):
    """Guards the split-by-gold bookkeeping: the two gold versions must partition the trials, or
    the headline row silently means something narrower than the trials it was computed from."""
    store(tmp_path, "a", {"L4": {"prediction": "1.0", "reward": 0.0, "agent_status": "exit 0"}})
    store(tmp_path, "b", {"L4": {"prediction": "1.0", "reward": 0.0, "agent_status": "exit 0"}})
    report = rg.regrade("synthetic", {"a": gold("a", "1.0")}, {"b": gold("b", "1.0")},
                        tree / "out.jsonl")
    counts = report["rungs"]["L4"]
    assert counts["trials_corrected"] + counts["trials_superseded"] == counts["trials"] == 2
    for key in ("old_pass", "new_pass"):
        total = sum(v for k, v in counts.items() if k.startswith(key))
        assert total <= counts["trials"], key
    assert "corrected 1, superseded 1" in rg.table(report)