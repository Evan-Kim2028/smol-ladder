"""The retry path in gen_refs: what counts as a reference, and what is evidence.

The bug this pins down: reference() short-circuited on agent_status == "exit 0", so a task whose
solution ran cleanly and printed the wrong answer was treated as final and could never be
re-attempted. The exit status says the *agent* finished; only reward >= 1.0 says the solution
reproduced the gold answer, and that is the only thing a reference is.
"""

import json
import subprocess
import time
from pathlib import Path

import pytest

from smol_ladder import gen_refs
from smol_ladder.ladder import read_source
from test_runner import grading_pass_argv


def _row(task_id="t1", answer="42"):
    return {"task_id": task_id, "question": "Q?", "files": ["t.csv"], "answer": answer,
            "reward_mode": "numeric", "atol": 0.0, "rtol": 0.0}


def _fake_solver(predictions, tmp_path, monkeypatch, statuses=None):
    """Stand in for the model loop, keeping every attempt's prediction for inspection.

    Only the model loop is stubbed: the offline grading pass runs the real bubblewrap, so the
    prediction that gets graded is still whatever solution.py printed when re-run. `predictions`
    is consumed one entry per attempt, and the last entry repeats if the sweep asks for more.
    """
    import smol_ladder.run_ladder as runner

    real_run = runner._run_jailed
    seen: list[str] = []
    statuses = statuses or ["exit 0"] * len(predictions)
    state = {"n": 0}

    def fake_run(cmd, cwd, env, timeout):
        if cmd == grading_pass_argv(Path(cwd), Path(cwd)):
            return real_run(cmd, cwd, env, timeout)
        i = min(state["n"], len(predictions) - 1)
        state["n"] += 1
        seen.append(str(cwd))
        status = statuses[min(i, len(statuses) - 1)]
        if status == "timeout":
            # once() catches TimeoutExpired and records "timeout", so raise the real thing
            # rather than a nonzero exit: an exit code is a different failure and the task
            # classifier in the report distinguishes them.
            raise subprocess.TimeoutExpired(cmd, timeout)
        if status == "exit 0":
            (Path(cwd) / "solution.py").write_text(f"print({predictions[i]!r})\n")
            return real_run(["bash", "-c", "echo ok"], cwd, env, timeout)
        (Path(cwd) / "solution.py").write_text("print('partial')\n")
        return real_run(["bash", "-c", f"exit {status.split()[-1]}"], cwd, env, timeout)

    monkeypatch.setattr(runner, "_run_jailed", fake_run)
    monkeypatch.setenv("SMOL_LADDER_SCRATCH", str(tmp_path / "scratch"))
    return seen


@pytest.fixture
def home(tmp_path, monkeypatch):
    """Point both DATA references at a tmpdir so a test never writes under the real data/ tree.

    gen_refs writes to its own DATA and read_source reads ladder's; they are separate bindings of
    the same path, so patching one and not the other would have the test write into the real
    tree and read from tmp (or the reverse), which passes for the wrong reason.
    """
    import smol_ladder.ladder as ladder

    fake = tmp_path / "data"
    (fake / "solutions").mkdir(parents=True)
    monkeypatch.setattr(gen_refs, "DATA", fake)
    monkeypatch.setattr(ladder, "DATA", fake)
    monkeypatch.setattr(gen_refs, "source_for", lambda split: ([], None))
    return fake


def test_a_clean_exit_with_the_wrong_answer_is_retried(tmp_path, monkeypatch, home):
    """The regression. agent_status "exit 0" only means the agent finished, not that it was
    right: a solution that runs and prints the wrong value must still be re-attempted."""
    _fake_solver(["41", "42"], tmp_path, monkeypatch)
    row = _row()
    inputs = tmp_path / "in"
    inputs.mkdir()

    first = gen_refs.reference(row, "test", "m", lambda r: inputs, 5, attempts=1)
    assert first["reward"] == 0.0 and first["keep"] is False

    # attempts=2 must run again rather than return the cached failure
    second = gen_refs.reference(row, "test", "m", lambda r: inputs, 5, attempts=2)
    assert second["reward"] == 1.0, second["prediction"]
    assert second["keep"] is True


def test_a_verified_reference_is_never_re_attempted(tmp_path, monkeypatch, home):
    _fake_solver(["42"], tmp_path, monkeypatch)
    row = _row()
    inputs = tmp_path / "in"
    inputs.mkdir()

    got = gen_refs.reference(row, "test", "m", lambda r: inputs, 5, attempts=3)
    assert got["keep"] is True
    # A second sweep, whatever the attempt budget, must return the verified result and not
    # roll the dice again: a reference we already verified is not a lottery.
    again = gen_refs.reference(row, "test", "m", lambda r: inputs, 5, attempts=3)
    assert again["reward"] == 1.0
    assert read_source(row, "test") is not None


def test_each_attempt_gets_its_own_directory_and_none_is_overwritten(tmp_path, monkeypatch,
                                                                    home):
    """Attempt history is evidence. A later attempt may fail where an earlier one passed, and
    that difference is the whole point of keeping both."""
    _fake_solver(["41", "40", "42"], tmp_path, monkeypatch)
    row = _row()
    inputs = tmp_path / "in"
    inputs.mkdir()

    got = gen_refs.reference(row, "test", "m", lambda r: inputs, 5, attempts=3)

    task = home / "solutions" / "test" / "t1"
    # attempt 0 is the flat legacy slot; the fresh ones are numbered subdirectories
    dirs = sorted(p.name for p in task.iterdir() if p.is_dir())
    assert dirs == ["attempt_1", "attempt_2", "attempt_3"], dirs
    for name in ("attempt_1", "attempt_2", "attempt_3"):
        assert (task / name / "result.json").exists(), name
    # each attempt kept its own prediction
    preds = {name: json.loads((task / name / "result.json").read_text())["prediction"]
             for name in ("attempt_1", "attempt_2", "attempt_3")}
    assert preds == {"attempt_1": "41", "attempt_2": "40", "attempt_3": "42"}


def test_a_failed_attempt_records_prediction_and_gold(tmp_path, monkeypatch, home):
    """A near-miss is only actionable if the record shows what was said and what was wanted."""
    _fake_solver(["41"], tmp_path, monkeypatch)
    row = _row(answer="42")
    inputs = tmp_path / "in"
    inputs.mkdir()

    gen_refs.reference(row, "test", "m", lambda r: inputs, 5, attempts=1)
    r = json.loads((home / "solutions" / "test" / "t1" / "attempt_1" / "result.json").read_text())
    assert r["prediction"] == "41"
    assert r["gold"] == "42"
    assert r["reward"] == 0.0


def test_attempt_metadata_is_recorded(tmp_path, monkeypatch, home):
    _fake_solver(["42"], tmp_path, monkeypatch)
    row = _row()
    inputs = tmp_path / "in"
    inputs.mkdir()

    gen_refs.reference(row, "test", "stealth/space-bunny-alpha", lambda r: inputs, 5, attempts=1)
    r = json.loads((home / "solutions" / "test" / "t1" / "attempt_1" / "result.json").read_text())
    assert r["attempt"] == 1
    assert r["model"] == "stealth/space-bunny-alpha"
    assert r["split"] == "test"
    assert isinstance(r["started_at"], (int, float)), r
    assert r["git_commit"], "a reference must record the commit it was produced from"
    assert r["agent_status"] == "exit 0"


def test_legacy_flat_layout_is_read_as_attempt_zero(tmp_path, monkeypatch, home):
    """gen_solutions.py wrote result.json straight into the task directory. Those must still
    count, or the 69 test and 49 eval tasks with a cached failure would look brand new."""
    task = home / "solutions" / "test" / "t1"
    task.mkdir(parents=True)
    (task / "result.json").write_text(json.dumps({"reward": 0.0, "agent_status": "exit 0"}))
    (task / "solution.py").write_text("print(41)\n")

    inputs = tmp_path / "in"
    inputs.mkdir()
    _fake_solver(["42"], tmp_path, monkeypatch)

    got = gen_refs.reference(_row(), "test", "m", lambda r: inputs, 5, attempts=2)
    assert got["reward"] == 1.0
    # the retry is numbered after the flat slot, so it lands beside the legacy attempt
    assert (task / "attempt_1" / "result.json").exists()
    # and the flat result.json now names the attempt that earned the reference
    promoted = json.loads((task / "result.json").read_text())
    assert promoted["reward"] == 1.0
    assert promoted["reference_attempt"] == 1


def test_legacy_verified_reference_is_not_re_attempted(tmp_path, monkeypatch, home):
    """A flat task dir that already verified is a reference, whatever the caller asked for."""
    task = home / "solutions" / "test" / "t1"
    task.mkdir(parents=True)
    (task / "result.json").write_text(json.dumps({"reward": 1.0, "agent_status": "exit 0",
                                                  "keep": True}))
    (task / "solution.py").write_text("print(42)\n")

    seen = _fake_solver(["40"], tmp_path, monkeypatch)
    inputs = tmp_path / "in"
    inputs.mkdir()

    got = gen_refs.reference(_row(), "test", "m", lambda r: inputs, 5, attempts=3)
    assert got["reward"] == 1.0 and got["attempts_made"] == 0
    assert seen == [], "a verified reference was re-attempted"


def test_promoted_reference_is_what_read_source_returns(tmp_path, monkeypatch, home):
    """The whole point of the layout: read_source must find the retry's winner."""
    _fake_solver(["41", "42"], tmp_path, monkeypatch)
    row = _row()
    inputs = tmp_path / "in"
    inputs.mkdir()

    assert read_source(row, "test") is None
    gen_refs.reference(row, "test", "m", lambda r: inputs, 5, attempts=2)
    assert read_source(row, "test") is not None, "promotion did not reach read_source"
    # and the promoted text is the verifying attempt's, not attempt 1's
    assert "42" in read_source(row, "test")


def test_a_timeout_attempt_is_recorded_and_retried(tmp_path, monkeypatch, home):
    _fake_solver(["x"], tmp_path, monkeypatch, statuses=["timeout"])
    row = _row()
    inputs = tmp_path / "in"
    inputs.mkdir()

    got = gen_refs.reference(row, "test", "m", lambda r: inputs, 5, attempts=1)
    assert got["reward"] == 0.0 and got["keep"] is False
    task = home / "solutions" / "test" / "t1"
    assert (task / "attempt_1" / "result.json").exists()
    assert json.loads((task / "attempt_1" / "result.json").read_text())["agent_status"] == "timeout"


def test_the_sweep_skips_tasks_that_already_have_a_reference(tmp_path, monkeypatch, home):
    """Only unverified tasks are re-rolled. Running the budget against all 250 would spend 250
    attempts to learn the 181 answers we already have."""
    rows = [_row("verified"), _row("failed")]
    task = home / "solutions" / "test" / "verified"
    task.mkdir(parents=True)
    (task / "result.json").write_text(json.dumps({"reward": 1.0, "agent_status": "exit 0"}))
    (task / "solution.py").write_text("print(42)\n")

    inputs = tmp_path / "in"
    inputs.mkdir()
    seen = _fake_solver(["42"], tmp_path, monkeypatch)
    out = gen_refs.run_sweep(rows, "test", "m", lambda r: inputs, 5, 1, workers=2)
    attempted = [r["task_id"] for r in out if r.get("attempts_made")]
    assert attempted == ["failed"], out
    assert read_source(rows[0], "test") is not None


def test_a_rate_limited_attempt_does_not_spend_the_budget(tmp_path, monkeypatch, home):
    """429 is the free model's quota, not a verdict on the task.

    The solver runs in a subprocess, so a throttle surfaces as a failed agent run with the
    refusal in its stderr. Counting that against the N attempts would spend the budget on quota
    and then report the task as unresolvable when the model never got to answer it -- so the
    same attempt index is retried, and only real answers consume the budget.
    """
    ran = {"n": 0}
    real_sleep = time.sleep

    def fake_once(*a, **kw):
        ran["n"] += 1
        if ran["n"] == 1:
            return {"reward": 0.0, "prediction": "", "agent_status": "exit 1",
                    "stderr": "HTTP 429: rate limit exceeded"}
        return {"reward": 1.0, "prediction": "42", "agent_status": "exit 0"}

    monkeypatch.setattr(gen_refs, "once", fake_once)
    monkeypatch.setattr(gen_refs.time, "sleep", lambda s: real_sleep(0))
    inputs = tmp_path / "in"
    inputs.mkdir()

    got = gen_refs.reference(_row(), "test", "m", lambda r: inputs, 5, attempts=1)

    assert ran["n"] == 2, "the throttled run was not retried"
    assert got["attempts_throttled"] == 1
    assert got["attempts_made"] == 1, "the throttle was charged to the attempt budget"
    assert got["reward"] == 1.0


def test_a_persistently_throttled_task_gives_up_rather_than_spinning(tmp_path, monkeypatch,
                                                                     home):
    """Bounded, so one task stuck on quota cannot pin a worker for the length of the sweep."""
    monkeypatch.setattr(gen_refs, "once", lambda *a, **kw: {
        "reward": 0.0, "prediction": "", "agent_status": "exit 1",
        "stderr": "429 Too Many Requests"})
    monkeypatch.setattr(gen_refs.time, "sleep", lambda s: None)
    inputs = tmp_path / "in"
    inputs.mkdir()

    got = gen_refs.reference(_row(), "test", "m", lambda r: inputs, 5, attempts=3,
                             max_throttles=2)
    assert got["attempts_made"] == 0, "no real attempt was ever made"
    assert got["keep"] is False


def test_throttle_detection_reads_the_refusal_where_it_actually_lands():
    assert gen_refs.throttled({"agent_status": "exit 1", "stderr": "HTTP 429: slow down"})
    assert gen_refs.throttled({"agent_status": "429"})
    assert gen_refs.throttled({"stderr": "Rate limit reached for model"})
    # a task the model actually failed on is not a throttle, and must be charged normally
    assert not gen_refs.throttled({"agent_status": "exit 0", "stderr": ""})
    assert not gen_refs.throttled({"agent_status": "exit 8", "stderr": "KeyError"})