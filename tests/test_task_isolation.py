"""One task raising must not take the sweep with it.

The ja3 transcript sweep (run tag `ja3`, split `jupyter-agent-v3`, L1) wrote 3,879 of 3,880
results and then died on task 3,880: `input_dir` raised `FileExistsError` for
`ja_0001_026_1026154.ipynb_qa_1`, the exception came back out of `f.result()` in `main`, and every
task behind it in the pool -- 2,257 of them -- was never attempted.

So the loop owes two things, and both are pinned here:

- a task that raises is recorded as a **harness failure** for that task -- the exception text
  included -- and the other tasks keep going;
- `RUN.json` closes out, with the counts it had, and no error.

The record has to be *on disk*, not just in the return value: a resumed sweep reuses
`result.json`, and a task whose failure left nothing behind is a task that is re-attempted
forever. It has to carry `agent_status != "exit 0"` too, which is what `summarize` keys on to keep
it out of every pass rate: a crash is not a model failure, and a harness failure booked as a 0.0
is a measurement that says nothing about the model.

Two layers are tested, because there are two places an exception can come from. `task_trials`
isolates a rung, so the rung is known and the failure is recorded against it; `main` isolates a
task, for an exception raised outside that per-rung loop (there is no rung to name then).

    uv run --with pytest pytest -q tests/test_task_isolation.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from smol_ladder import run_ladder as runner


def _rows(n: int) -> list[dict]:
    return [{"task_id": f"t{i}", "question": "q", "files": [], "answer": "1",
             "reward_mode": "numeric", "atol": 0.0, "rtol": 0.0} for i in range(n)]


def _ok(task_id: str, rung: str) -> dict:
    return {"task_id": task_id, "rung": rung, "sample": 0, "reward": 0.0,
            "agent_status": "exit 0", "prediction": ""}


def _main(monkey, tmp_path, tasks, rungs=("L1",), argv_extra=()):
    monkey.setattr(runner, "DATA", tmp_path)
    monkey.setattr(runner, "source_for", lambda split: (tasks, lambda r: tmp_path / "in"))
    monkey.setattr(sys, "argv", ["run_ladder", "--run-tag", "t1", "--split", "test",
                                 "--rungs", ",".join(rungs), "--workers", "4", *argv_extra])
    runner.main()
    return json.loads((tmp_path / "runs" / "t1" / "RUN.json").read_text())


def _launch(tmp_path, body, n=4, rungs=("L1",), **kwargs):
    """main() with `task_trials` itself stubbed, for what raises outside it."""
    monkey = pytest.MonkeyPatch()
    monkey.setattr(runner, "task_trials", body)
    try:
        return _main(monkey, tmp_path, _rows(n), rungs, **kwargs)
    finally:
        monkey.undo()


# ── a rung that raises, inside the real task_trials ───────────────────────────────────────────

def _raising_once(raise_on: str, seen: list[str] | None = None, only: str | None = None):
    """A stand-in for `once` that raises on one rung label and otherwise passes cleanly.

    `only` names the one task that raises, so a test can watch the others go through.
    """
    def once(row, prompt, work, venv, model, max_turns, retry_failed=False, inputs_of=None,
             rung_label="run", provenance=None, was_run=None, agent="tools",
             save_transcript=True):
        if seen is not None:
            seen.append(f"{row['task_id']}/{rung_label}")
        work = Path(work)
        if was_run is not None:
            was_run.append(True)
        work.mkdir(parents=True, exist_ok=True)
        if rung_label == raise_on and (only is None or row["task_id"] == only):
            raise FileExistsError(17, f"File exists: 'bubble_volume.csv' -> '{work}/bubble_volume.csv'")
        return _ok(row["task_id"], rung_label)
    return once


def _real_task_trials(tmp_path, monkeypatch, once, rungs):
    """main() with the real task_trials and a stubbed once, so the per-rung path is the one
    under test. `read_source` answers yes, because L2+ are gated on a reference and a stub that
    said no would have every rung above L1 skipped before any of this was reached."""
    monkeypatch.setattr(runner, "prompt_for", lambda row, split, rung: f"prompt:{rung}")
    monkeypatch.setattr(runner, "hint_source", lambda row, split, rung: "ast")
    monkeypatch.setattr(runner, "read_source", lambda row, split: "REF")
    monkeypatch.setattr(runner, "once", once)


def test_one_task_raising_does_not_stop_the_others(tmp_path, monkeypatch):
    seen: list[str] = []
    _real_task_trials(tmp_path, monkeypatch, _raising_once("L1", seen, only="t1"), ("L1",))

    run = _main(monkeypatch, tmp_path, _rows(4))

    assert sorted(seen) == ["t0/L1", "t1/L1", "t2/L1", "t3/L1"], \
        f"the sweep stopped early: only {sorted(seen)} were attempted"
    assert run["tasks"] == 4


def test_the_raised_task_is_booked_as_a_harness_failure_with_its_text(tmp_path, monkeypatch):
    _real_task_trials(tmp_path, monkeypatch, _raising_once("L1", only="t1"), ("L1",))

    run = _main(monkeypatch, tmp_path, _rows(4))

    failures = run["harness_failures"]
    assert len(failures) == 1, failures
    failure = failures[0]
    assert failure["task_id"] == "t1"
    assert failure["rung"] == "L1", "the rung is known, so the record has to name it"
    assert failure["agent_status"] != "exit 0", "a harness failure that reads as a clean run"
    assert "FileExistsError" in failure["error"], failure["error"]
    assert "bubble_volume.csv" in failure["error"], failure["error"]


def test_a_failure_is_written_to_the_tasks_own_result_file(tmp_path, monkeypatch):
    """So a resumed sweep reuses it instead of re-attempting the task forever, and so the tree
    explains itself to anyone reading it without RUN.json."""
    _real_task_trials(tmp_path, monkeypatch, _raising_once("L1", only="t1"), ("L1",))

    _main(monkeypatch, tmp_path, _rows(4))

    result = json.loads((tmp_path / "runs" / "t1" / "test" / "t1" / "L1" / "result.json").read_text())
    assert result["task_id"] == "t1"
    assert result["rung"] == "L1"
    assert result["reward"] == 0.0, "a crash has no prediction to grade"
    assert "FileExistsError" in result["error"]
    assert result["agent_status"] != "exit 0"


def test_a_harness_failure_is_counted_apart_from_trials_and_passes(tmp_path, monkeypatch):
    """Neither of the other two counters: it was attempted, and it is not a pass. Counting it as
    a trial puts it in the denominator of a pass rate; counting it as a pass is absurd."""
    _real_task_trials(tmp_path, monkeypatch, _raising_once("L1", only="t1"), ("L1",))

    run = _main(monkeypatch, tmp_path, _rows(4))

    assert run["trials"] == 3, run["trials"]
    assert run["passes"] == 0
    assert run["tasks"] == 4
    assert len(run["harness_failures"]) == 1


def test_run_json_closes_out_without_an_error(tmp_path, monkeypatch):
    """The contract the dead sweep broke: a sweep whose task raised is still a run, and its
    record has an end time and no `error` key."""
    _real_task_trials(tmp_path, monkeypatch, _raising_once("L1", only="t1"), ("L1",))

    run = _main(monkeypatch, tmp_path, _rows(4))

    assert run["end_time"], "the run closed without an end time"
    assert "error" not in run, run.get("error")


def test_a_rung_that_raises_does_not_abandon_the_tasks_later_rungs(tmp_path, monkeypatch):
    """`task_trials` runs every rung of a task, so an exception from L2 must not cost L3. A fix
    that only caught around the whole rung loop would keep the isolation and lose the rung."""
    seen: list[str] = []
    _real_task_trials(tmp_path, monkeypatch, _raising_once("L2", seen, only="t1"), ("L1", "L2", "L3"))

    run = _main(monkeypatch, tmp_path, _rows(2), rungs=("L1", "L2", "L3"))

    assert "t1/L3" in seen, f"the task's later rungs were abandoned: {seen}"
    failures = run["harness_failures"]
    assert len(failures) == 1 and failures[0]["rung"] == "L2", failures


def test_the_later_rungs_of_that_task_are_still_run_and_counted(tmp_path, monkeypatch):
    _real_task_trials(tmp_path, monkeypatch, _raising_once("L2", only="t1"), ("L1", "L2", "L3"))

    run = _main(monkeypatch, tmp_path, _rows(2), rungs=("L1", "L2", "L3"))

    # t0 L1+L2+L3, t1 L1+L3 -- six trials, one of them a harness failure.
    assert run["trials"] == 5, run["trials"]
    assert len(run["harness_failures"]) == 1


# ── resumability ─────────────────────────────────────────────────────────────────────────────

def test_the_real_once_reuses_a_recorded_failure_without_touching_the_inputs(tmp_path,
                                                                            monkeypatch):
    """Resumability is the point of writing the record. `once()` returns a cached `result.json`
    before it ever calls `inputs_of`, so a recorded failure costs nothing on the next launch
    unless `--retry-failed` asks for another go -- and the launcher that re-raises is the one that
    raised because its cache entry is broken, which is exactly what should be retried once."""
    monkeypatch.setenv("SMOL_LADDER_SCRATCH", str(tmp_path / "scratch"))
    work = tmp_path / "runs" / "t1" / "test" / "t1" / "L1"
    work.mkdir(parents=True)
    (work / "result.json").write_text(json.dumps(
        {"task_id": "t1", "rung": "L1", "reward": 0.0, "agent_status": "error: FileExistsError",
         "error": "FileExistsError: File exists"}))
    row = _rows(1)[0]

    def exploding(row, *a, **k):
        raise FileExistsError(17, "File exists")

    ran: list[bool] = []
    reused = runner.once(row, "p", work, Path(sys.prefix), "m", 2, retry_failed=False,
                         inputs_of=exploding, rung_label="L1", was_run=ran)
    assert reused["error"] == "FileExistsError: File exists"
    assert ran == [], "the cached result was not reused: the inputs were fetched anyway"

    # and with --retry-failed it does go back out to the inputs, which raises here
    with pytest.raises(FileExistsError):
        runner.once(row, "p", work, Path(sys.prefix), "m", 2, retry_failed=True,
                    inputs_of=exploding, rung_label="L1")


def test_a_resumed_sweep_retries_a_failure_when_asked(tmp_path, monkeypatch):
    """`--retry-failed` exists so a crashed trial gets another go. The recorded failure has to be
    distinguishable from a real result for that flag to mean anything: `reuse_recorded` is the
    predicate, and it must be false for a non-"exit 0" record only when the caller asked."""
    work = tmp_path / "runs" / "t1" / "test" / "t1" / "L1"
    work.mkdir(parents=True)
    (work / "result.json").write_text(json.dumps(
        {"task_id": "t1", "rung": "L1", "reward": 0.0, "agent_status": "error",
         "error": "FileExistsError: File exists"}))
    clean = tmp_path / "clean"
    clean.mkdir()
    (clean / "result.json").write_text(json.dumps(
        {"task_id": "t0", "rung": "L1", "reward": 1.0, "agent_status": "exit 0"}))

    assert runner.reuse_recorded(work, retry_failed=False) is True
    assert runner.reuse_recorded(work, retry_failed=True) is False
    assert runner.reuse_recorded(clean, retry_failed=True) is True


# ── a task raising outside the rung loop ──────────────────────────────────────────────────────

def test_an_exception_from_outside_the_rung_loop_is_still_isolated(tmp_path):
    """`task_trials` can raise before it reaches a rung -- resolving a prompt reads a cached hint,
    which is a file. The task backstop in main catches that; there is no rung to name, and
    pretending otherwise would put `L1` on a record that never got there."""

    def body(row, *a, **k):
        if row["task_id"] == "t1":
            raise ValueError("hint file is not valid JSON")
        return [_ok(row["task_id"], "L1")]

    run = _launch(tmp_path, body)

    assert len(run["harness_failures"]) == 1
    assert run["harness_failures"][0]["task_id"] == "t1"
    assert run["harness_failures"][0]["error"] == "ValueError: hint file is not valid JSON"
    assert run["tasks"] == 4
    assert "error" not in run


def test_a_dead_launch_keyboard_interrupt_still_closes_the_record(tmp_path):
    """A task raising is isolated; an interrupt is not a task, and it must still leave the record
    closed with an error. Pinned here so the isolation cannot be implemented by swallowing
    `BaseException`."""
    def body(row, *a, **k):
        if row["task_id"] == "t1":
            raise KeyboardInterrupt
        return [_ok(row["task_id"], "L1")]

    monkey = pytest.MonkeyPatch()
    monkey.setattr(runner, "DATA", tmp_path)
    monkey.setattr(runner, "source_for", lambda split: (_rows(4), lambda r: tmp_path / "in"))
    monkey.setattr(runner, "task_trials", body)
    monkey.setattr(sys, "argv", ["run_ladder", "--run-tag", "t1", "--split", "test",
                                 "--rungs", "L1"])
    try:
        with pytest.raises(KeyboardInterrupt):
            runner.main()
    finally:
        monkey.undo()

    run = json.loads((tmp_path / "runs" / "t1" / "RUN.json").read_text())
    assert run["end_time"]
    assert "KeyboardInterrupt" in run["error"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))