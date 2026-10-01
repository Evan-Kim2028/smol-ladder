"""Re-verify the trials whose offline grading pass failed, without touching the model or the tree.

The v2 run lost 82 trials to `verify timeout` and 8 to `verify exit 1` -- the sealed offline
pass, not the agent, ran out of time while the machine was at load average 60-80 with three
sweeps and two test suites on it. `solution.py` is on disk for every one of them, so the
question is not whether the model wrote a program but whether that program grades, and the
grading needs no model at all.

So this tool re-runs the sealed pass on a generous deadline, four at a time, `nice`d, and
records the outcome in a NEW `reverify.json` beside each trial. It never writes `result.json`:
the run on disk is the measurement that was made, and overwriting it would destroy the
evidence that the first pass failed under load. The summary prefers a successful
re-verification and reports how many trials it recovered, so the recovery is visible in the
numbers rather than silently folded into them.

    uv run python -m smol_ladder.reverify --run-tag v2 --split test
    uv run python -m smol_ladder.reverify --run-tag v2 --split test --dry-run
"""

import json
import subprocess
from pathlib import Path

import pytest

import smol_ladder.run_ladder as runner_module

ROW = {"task_id": "t1", "question": "Q?", "files": ["t.csv"], "answer": "42",
       "reward_mode": "numeric", "atol": 0.0, "rtol": 0.0}


def trial_dir(root, task, rung, sample=0, **result):
    """Write a trial under `root` (a results tree) and return its directory.

    Sample 0 is the rung dir itself and sample k is <rung>/s<k>, which is the layout the runner
    writes and the one `candidates` has to walk.
    """
    path = root / task / rung if sample == 0 else root / task / rung / f"s{sample}"
    path.mkdir(parents=True, exist_ok=True)
    base = {"task_id": task, "agent_status": "exit 0", "reward": 0.0, "prediction": "",
            "prompt_sha256": "a" * 64, "rung": rung, "sample": sample}
    base.update(result)
    (path / "result.json").write_text(json.dumps(base))
    return path


def tree(tmp_path):
    return tmp_path / "runs" / "test"


def tables(tmp_path):
    """A task's input directory, so the pass has something to bind-mount."""
    inputs = tmp_path / "in"
    inputs.mkdir(exist_ok=True)
    (inputs / "t.csv").write_text("a\n1\n")
    return inputs


def with_program(path, code="print(42)\n"):
    (path / "solution.py").write_text(code)
    return path


# --- which trials are eligible ----------------------------------------------------------------

def test_a_trial_whose_graded_cleanly_is_not_reverified(tmp_path):
    """Nothing is wrong with it. Re-running a pass that already produced a prediction would only
    risk turning a measurement into a re-measurement."""
    from smol_ladder import reverify

    path = trial_dir(tree(tmp_path), "a", "L1")
    with_program(path)
    assert reverify.candidates(tree(tmp_path)) == []


def test_a_trial_whose_offline_pass_timed_out_is_eligible(tmp_path):
    from smol_ladder import reverify

    path = trial_dir(tree(tmp_path), "a", "L1", verify_status="verify timeout")
    with_program(path)
    assert [t.path for t in reverify.candidates(tree(tmp_path))] == [path]


def test_a_trial_whose_offline_pass_crashed_is_eligible(tmp_path):
    from smol_ladder import reverify

    path = trial_dir(tree(tmp_path), "a", "L1", verify_status="verify exit 1")
    with_program(path)
    assert len(reverify.candidates(tree(tmp_path))) == 1


def test_a_trial_with_no_program_is_not_eligible(tmp_path):
    """The agent never wrote one, so there is nothing to re-run. Re-grading nothing would book a
    harness failure as recovered."""
    from smol_ladder import reverify

    trial_dir(tree(tmp_path), "a", "L1", verify_status="verify timeout")
    assert reverify.candidates(tree(tmp_path)) == []


def test_an_agent_timeout_is_not_eligible(tmp_path):
    """The model loop itself ran out of time, and the only fix for that is the model. This tool
    makes no model calls, so it must not pretend to recover one."""
    from smol_ladder import reverify

    path = trial_dir(tree(tmp_path), "a", "L1", agent_status="timeout", verify_status="verify timeout")
    with_program(path)
    assert reverify.candidates(tree(tmp_path)) == []


def test_each_candidate_carries_its_own_rung(tmp_path):
    """The report is grouped by rung, and a task row cannot supply it: the same task was run at
    five rungs, so grouping by the row would file every trial under one bucket."""
    from smol_ladder import reverify

    a = with_program(trial_dir(tree(tmp_path), "a", "L1", verify_status="verify timeout"))
    b = with_program(trial_dir(tree(tmp_path), "a", "L2", verify_status="verify timeout"))
    assert {t.path: t.rung for t in reverify.candidates(tree(tmp_path))} == {a: "L1", b: "L2"}


def test_every_sample_of_a_task_is_eligible_separately(tmp_path):
    """Samples live one level deeper. A pattern that stopped at */L1/result.json would find one
    sample of a two-sample cell and silently re-verify the wrong thing."""
    from smol_ladder import reverify

    a = with_program(trial_dir(tree(tmp_path), "a", "L1", 0, verify_status="verify timeout"))
    b = with_program(trial_dir(tree(tmp_path), "a", "L1", 1, verify_status="verify timeout"))
    assert {t.path for t in reverify.candidates(tree(tmp_path))} == {a, b}


# --- the outcome lands beside the trial, never in it -------------------------------------------

def test_the_outcome_is_written_beside_the_result_and_the_result_is_untouched(tmp_path):
    """result.json is the run. This tool's whole claim is that it is evidence, so it must not
    write to it -- the only new file is reverify.json."""
    from smol_ladder import reverify

    path = trial_dir(tree(tmp_path), "a", "L1", verify_status="verify timeout")
    with_program(path)
    before = (path / "result.json").read_text()

    reverify.verify_trial(reverify.Trial(path=path, row=dict(ROW), inputs=tables(tmp_path)), timeout=60)

    assert (path / "result.json").read_text() == before
    record = json.loads((path / "reverify.json").read_text())
    assert record["old_verify_status"] == "verify timeout"
    assert record["new_verify_status"] == "exit 0"
    assert record["prediction"] == "42"
    assert record["reward"] == 1.0


def test_the_record_names_the_deadline_it_used(tmp_path):
    """A re-verification that passed on a 900s deadline is a weaker claim than one that passed
    on 180s, and the number is what a reader needs to judge it."""
    from smol_ladder import reverify

    path = trial_dir(tree(tmp_path), "a", "L1", verify_status="verify timeout")
    with_program(path)
    reverify.verify_trial(reverify.Trial(path=path, row=dict(ROW), inputs=tables(tmp_path)), timeout=61)
    assert json.loads((path / "reverify.json").read_text())["timeout_seconds"] == 61


def test_a_pass_is_recorded_with_the_graded_prediction_not_the_stored_one(tmp_path):
    """The stored prediction is "" -- a timed-out pass grades nothing. The new one is the value
    the program actually printed when re-run, which is the only reason to re-run it."""
    from smol_ladder import reverify

    path = trial_dir(tree(tmp_path), "a", "L1", verify_status="verify timeout", prediction="")
    with_program(path)
    reverify.verify_trial(reverify.Trial(path=path, row=dict(ROW), inputs=tables(tmp_path)), timeout=60)
    record = json.loads((path / "reverify.json").read_text())
    assert record["old_prediction"] == "" and record["prediction"] == "42"


def test_a_second_re_verification_replaces_the_first(tmp_path, monkeypatch):
    """The first attempt ran under load and the second did not. Keeping both would leave a reader
    unable to say which is the current reading, so the later one wins and the record says so."""
    from smol_ladder import reverify

    path = trial_dir(tree(tmp_path), "a", "L1", verify_status="verify timeout")
    with_program(path)
    trial = reverify.Trial(path=path, row=dict(ROW), inputs=tables(tmp_path))
    reverify.verify_trial(trial, timeout=60)
    (path / "reverify.json").write_text(json.dumps(
        {"new_verify_status": "verify timeout", "attempts": 1, "reward": 0.0}))
    reverify.verify_trial(trial, timeout=60)
    record = json.loads((path / "reverify.json").read_text())
    assert record["attempts"] == 2 and record["new_verify_status"] == "exit 0"


def test_a_program_that_still_times_out_is_recorded_as_still_failing(tmp_path, monkeypatch):
    """A second timeout is a real answer -- this program is slow -- and it must be recorded as a
    harness failure rather than a model failure, exactly as the first one was."""
    from smol_ladder import reverify

    path = trial_dir(tree(tmp_path), "a", "L1", verify_status="verify timeout")
    with_program(path, "import time\ntime.sleep(30)\n")
    monkeypatch.setattr(runner_module, "_run_jailed", _raising_runner())
    reverify.verify_trial(reverify.Trial(path=path, row=dict(ROW), inputs=tables(tmp_path)), timeout=2)
    record = json.loads((path / "reverify.json").read_text())
    assert record["new_verify_status"] == "verify timeout"
    assert record["prediction"] == "" and record["reward"] == 0.0


def test_a_program_that_crashes_is_recorded_as_a_crashed_pass(tmp_path, monkeypatch):
    from smol_ladder import reverify

    path = trial_dir(tree(tmp_path), "a", "L1", verify_status="verify timeout")
    with_program(path, "raise SystemExit(3)\n")
    monkeypatch.setattr(runner_module, "_run_jailed", _returning_runner(returncode=1))
    reverify.verify_trial(reverify.Trial(path=path, row=dict(ROW), inputs=tables(tmp_path)), timeout=60)
    assert json.loads((path / "reverify.json").read_text())["new_verify_status"] == "verify exit 1"


def _raising_runner():
    """Stands in for the sealed pass and reports a timeout, exactly as the first pass did."""
    def fake(cmd, cwd, env, timeout):
        raise subprocess.TimeoutExpired(cmd, timeout)

    return fake


def _returning_runner(returncode):
    def fake(cmd, cwd, env, timeout):
        return subprocess.CompletedProcess(cmd, returncode, b"", b"boom")

    return fake


# --- the real pass, end to end -----------------------------------------------------------------

def test_a_program_that_prints_the_gold_answer_is_recovered(tmp_path):
    """The real bubblewrap pass, on a real table. This is the case the whole tool exists for:
    a program the model did write, which the first pass never got the machine to grade."""
    from smol_ladder import reverify

    path = trial_dir(tree(tmp_path), "a", "L1", verify_status="verify timeout")
    with_program(path, "print(42)\n")

    reverify.verify_trial(reverify.Trial(path=path, row=dict(ROW), inputs=tables(tmp_path)),
                          timeout=120)

    record = json.loads((path / "reverify.json").read_text())
    assert record["new_verify_status"] == "exit 0", record
    assert record["prediction"] == "42" and record["reward"] == 1.0


def test_the_pass_runs_niced_and_bounded(tmp_path, monkeypatch):
    """The whole reason for the concurrency and nice settings: the machine that timed out the
    first pass is still running another sweep. A re-verification that competes with it would
    reproduce the failure it is here to repair."""
    import smol_ladder.reverify as reverify

    seen: dict = {}

    def spy(cmd, cwd, env, timeout):
        seen["cmd"] = cmd
        seen["timeout"] = timeout
        import subprocess as sp
        return sp.CompletedProcess(cmd, 0, b"42\n", b"")

    monkeypatch.setattr(runner_module, "_run_jailed", spy)
    path = trial_dir(tree(tmp_path), "a", "L1", verify_status="verify timeout")
    with_program(path)
    reverify.verify_trial(reverify.Trial(path=path, row=dict(ROW), inputs=tables(tmp_path)), timeout=123)
    assert seen["cmd"][0] == "nice"
    assert "15" in seen["cmd"][:3]
    assert seen["timeout"] == 123


def test_a_trial_whose_tables_are_missing_is_recorded_not_crashed(tmp_path, monkeypatch):
    """A task whose input cache was never fetched cannot be re-verified. It has to come out as a
    stated reason, not an exception that takes the whole pass down with it."""
    import smol_ladder.reverify as reverify

    path = trial_dir(tree(tmp_path), "a", "L1", verify_status="verify timeout")
    with_program(path)
    record = reverify.verify_trial(reverify.Trial(path=path, row=dict(ROW), inputs=None), timeout=5)
    assert record["new_verify_status"] == "no inputs"
    assert record["recovered"] is False


# --- the summary's side of it -------------------------------------------------------------------

def test_the_summary_counts_a_recovered_trial_as_scored(tmp_path):
    """Without this the recovery is invisible and the harness failure stands forever, which is the
    outcome the run actually had."""
    from smol_ladder import summarize as S

    trial = {"reward": 0.0, "agent_status": "exit 0", "verify_status": "verify timeout",
             "reverify": {"new_verify_status": "exit 0", "reward": 1.0, "prediction": "42"}}
    assert S._finished(trial) is True
    assert S._passed(trial) is True


def test_a_re_verification_that_still_failed_leaves_the_trial_a_harness_failure(tmp_path):
    from smol_ladder import summarize as S

    trial = {"reward": 0.0, "agent_status": "exit 0", "verify_status": "verify timeout",
             "reverify": {"new_verify_status": "verify timeout", "reward": 0.0}}
    assert S._finished(trial) is False
    assert S._passed(trial) is False


def test_a_re_verification_does_not_override_a_trial_that_graded_cleanly(tmp_path):
    """Only a failed pass is re-verified, so a reverify record on a clean trial is stale. The
    clean measurement wins rather than being overwritten by a later reading."""
    from smol_ladder import summarize as S

    trial = {"reward": 1.0, "agent_status": "exit 0",
             "reverify": {"new_verify_status": "verify timeout", "reward": 0.0}}
    assert S._finished(trial) is True and S._passed(trial) is True


def test_the_summary_collects_the_reverify_records_off_disk(tmp_path, monkeypatch):
    """Reading the side file is what makes the number reportable at all."""
    from smol_ladder import summarize as S

    monkeypatch.setattr(S, "DATA", tmp_path / "data")
    root = tmp_path / "data" / "runs" / "test"
    for name, record in (("a", {"new_verify_status": "exit 0", "reward": 1.0}),
                         ("b", {"new_verify_status": "verify timeout", "reward": 0.0})):
        path = root / name / "L1"
        path.mkdir(parents=True)
        (path / "result.json").write_text(json.dumps(
            {"reward": 0.0, "agent_status": "exit 0", "verify_status": "verify timeout"}))
        (path / "reverify.json").write_text(json.dumps(record))
    runs = S.collect("test")
    assert runs["a"]["L1"][0]["reverify"]["reward"] == 1.0
    assert runs["b"]["L1"][0]["reverify"]["new_verify_status"] == "verify timeout"


def test_the_report_says_how_many_trials_were_recovered(tmp_path):
    """Only trials that were actually re-verified are counted. A task whose first pass failed and
    was never revisited is still a harness failure, and counting it as an attempt that failed
    would report a recovery rate over a denominator the tool never touched."""
    from smol_ladder import summarize as S

    runs = {"a": {"L1": [{"reward": 0.0, "agent_status": "exit 0",
                          "verify_status": "verify timeout",
                          "reverify": {"new_verify_status": "exit 0", "reward": 1.0}}]},
            "b": {"L1": [{"reward": 0.0, "agent_status": "exit 0",
                          "verify_status": "verify timeout",
                          "reverify": {"new_verify_status": "verify timeout", "reward": 0.0}}]},
            "c": {"L1": [{"reward": 0.0, "agent_status": "exit 0",
                          "verify_status": "verify timeout"}]}}
    report = S.summarise("test", runs, lambda _t: True)
    assert report["reverified"] == {"recovered": 1, "still_failing": 1, "attempts": 2}
    # a is scored now, b and c are not: only one trial came back from the pass.
    assert report["rungs"]["L1"]["harness_failures"] == 2
    assert report["rungs"]["L1"]["trials_scored"] == 1


# --- the pass over a whole tree ------------------------------------------------------------------

def test_run_does_not_touch_a_tree_with_nothing_to_re_verify(tmp_path, monkeypatch):
    """`run` is the path a person types, and the ThreadPoolExecutor it builds only runs when there
    is work. A type error in that construction reached a real run on v2 before any test hit it,
    because every test called verify_trial directly."""
    from smol_ladder import reverify

    monkeypatch.setattr(reverify, "source_for", lambda split: ([], lambda row: Path("/nonexistent")))
    monkeypatch.setattr(reverify, "DATA", tmp_path)
    (tree(tmp_path)).mkdir(parents=True)
    got = reverify.run("test", None, workers=4, timeout=1)
    assert got == {"candidates": 0, "recovered": 0, "still_failing": 0, "rungs": {}}


def test_run_counts_recoveries_by_rung(tmp_path, monkeypatch):
    from smol_ladder import reverify

    rows = [dict(ROW, task_id="a"), dict(ROW, task_id="b")]
    monkeypatch.setattr(reverify, "source_for",
                        lambda split: (rows, lambda row: table_for(row["task_id"], tmp_path)))
    monkeypatch.setattr(reverify, "DATA", tmp_path)
    with_program(trial_dir(tree(tmp_path), "a", "L1", verify_status="verify timeout"))
    with_program(trial_dir(tree(tmp_path), "b", "L2", verify_status="verify timeout"))

    got = reverify.run("test", None, workers=2, timeout=120)
    assert got["candidates"] == 2
    assert got["recovered"] == 2, got
    assert got["rungs"]["L1"] == {"candidates": 1, "recovered": 1, "still_failing": 0}
    assert got["rungs"]["L2"] == {"candidates": 1, "recovered": 1, "still_failing": 0}


def table_for(task_id, tmp_path):
    inputs = tmp_path / f"in-{task_id}"
    inputs.mkdir(exist_ok=True)
    (inputs / "t.csv").write_text("a\n1\n")
    return inputs


def test_a_dry_run_writes_nothing(tmp_path, monkeypatch):
    from smol_ladder import reverify

    rows = [{"task_id": "a"}]
    monkeypatch.setattr(reverify, "source_for", lambda split: (rows, lambda row: None))
    monkeypatch.setattr(reverify, "DATA", tmp_path)
    path = with_program(trial_dir(tree(tmp_path), "a", "L1", verify_status="verify timeout"))

    got = reverify.run("test", None, dry_run=True)
    assert got["candidates"] == 1 and got["dry_run"] is True
    assert not (path / "reverify.json").exists()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
