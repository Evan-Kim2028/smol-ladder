"""The verifier's copy of the task's tables must not outlive the grading pass.

The offline pass copies a task's input tables into `<trial>/verify/input` and left them there.
At ~30 MB per trial that is what a 1,600-trial v2 run turned into 24 GB of disk for no result:
`verify/solution.py` is the program and `result.json` beside it is the measurement, and the
tables are reconstructible from the shared cache by path. So the copy is removed once grading
finishes, and reclaim.py can clear the ones an older run left behind.

The tests below are split deliberately. The first group is about a trial we can run (the fake
runner's grading pass is the real bubblewrap, so the prediction still comes from solution.py).
The second is about an existing tree, which is only files on disk.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from test_runner import grading_pass_argv


def _trial(tmp_path, monkeypatch, *, code="print(42)\n", inputs=None):
    """A graded trial: the model loop is stubbed, the offline pass is real."""
    import smol_ladder.run_ladder as runner

    real_run = runner._run_jailed

    def fake_run(cmd, cwd, env, timeout):
        if cmd == grading_pass_argv(Path(cwd), Path(cwd)):
            return real_run(cmd, cwd, env, timeout)
        (Path(cwd) / "solution.py").write_text(code)
        return real_run(["bash", "-c", "true"], cwd, env, timeout)

    monkeypatch.setattr(runner, "_run_jailed", fake_run)
    monkeypatch.setenv("SMOL_LADDER_SCRATCH", str(tmp_path / "scratch"))
    if inputs is None:
        inputs = tmp_path / "in"
        inputs.mkdir()
        (inputs / "t.csv").write_text("a\n1\n")
    work = tmp_path / "trial" / "L1"
    row = {"task_id": "t1", "question": "Q?", "files": ["t.csv"], "answer": "42",
           "reward_mode": "numeric", "atol": 0.0, "rtol": 0.0}
    result = runner.once(row, "Q?", work, Path(sys.prefix), "m", 2,
                         inputs_of=lambda r: inputs, rung_label="L1")
    return result, work, inputs


def test_the_verifiers_copy_of_the_tables_is_gone_after_grading(tmp_path, monkeypatch):
    """The disk bug. A passing trial kept a full copy of its input tables beside the result.

    This is what makes a sweep's disk use a function of the tables rather than of the results,
    and it is why the v2 run reached tens of gigabytes of `verify/input` for trials whose
    programs and results are a few kilobytes each.
    """
    result, work, _ = _trial(tmp_path, monkeypatch)
    assert result["reward"] == 1.0, result.get("stderr", "")[:300]

    assert not (work / "verify" / "input").exists(), \
        "the verifier's copy of the tables survived the trial"
    # What is left in verify/ is the program, which is a result.
    assert (work / "verify" / "solution.py").exists()
    assert (work / "verify" / "solution.py").read_text() == "print(42)\n"


def test_grading_still_sees_the_tables_while_it_runs(tmp_path, monkeypatch):
    """The removal must not happen before the pass that needs the tables.

    A trial that removes verify/input on entry would grade every solution 0.0 with a clean exit
    code, which is the worst possible failure: it looks like a model that cannot read a CSV.
    So the copy is built, the pass runs against it, and only then is it dropped.
    """
    seen: list[str] = []
    import smol_ladder.run_ladder as runner

    real_run = runner._run_jailed

    def fake_run(cmd, cwd, env, timeout):
        if cmd == grading_pass_argv(Path(cwd), Path(cwd)):
            # The tables are present at the moment the pass is invoked.
            seen.append(str(Path(cwd, "input")))
            assert Path(cwd, "input").is_dir(), "grading started without the tables"
            return real_run(cmd, cwd, env, timeout)
        (Path(cwd) / "solution.py").write_text("print(42)\n")
        return real_run(["bash", "-c", "true"], cwd, env, timeout)

    monkeypatch.setattr(runner, "_run_jailed", fake_run)
    monkeypatch.setenv("SMOL_LADDER_SCRATCH", str(tmp_path / "scratch"))
    inputs = tmp_path / "in"
    inputs.mkdir()
    (inputs / "t.csv").write_text("a\n1\n")
    work = tmp_path / "trial" / "L1"
    result = runner.once({"task_id": "t1", "question": "Q?", "files": ["t.csv"], "answer": "42",
                          "reward_mode": "numeric", "atol": 0.0, "rtol": 0.0},
                         "Q?", work, Path(sys.prefix), "m", 2, inputs_of=lambda r: inputs)

    assert seen, "the grading pass never ran"
    assert result["reward"] == 1.0, (result["prediction"], result.get("stderr", "")[:300])


def test_a_failed_trial_is_cleaned_up_too(tmp_path, monkeypatch):
    """A harness failure is still a trial whose copy was made.

    Leaving the tables behind for exactly the trials that failed is the worst of both: no result
    to show for the space, and the failures are the ones a regrade is most likely to revisit.
    """
    import smol_ladder.run_ladder as runner

    real_run = runner._run_jailed

    def fake_run(cmd, cwd, env, timeout):
        if cmd == grading_pass_argv(Path(cwd), Path(cwd)):
            raise subprocess.TimeoutExpired(cmd, timeout)
        (Path(cwd) / "solution.py").write_text("print(42)\n")
        return real_run(["bash", "-c", "true"], cwd, env, timeout)

    monkeypatch.setattr(runner, "_run_jailed", fake_run)
    monkeypatch.setenv("SMOL_LADDER_SCRATCH", str(tmp_path / "scratch"))
    inputs = tmp_path / "in"
    inputs.mkdir()
    (inputs / "t.csv").write_text("a\n1\n")
    work = tmp_path / "trial" / "L1"
    result = runner.once({"task_id": "t1", "question": "Q?", "files": ["t.csv"], "answer": "42",
                          "reward_mode": "numeric", "atol": 0.0, "rtol": 0.0},
                         "Q?", work, Path(sys.prefix), "m", 2, inputs_of=lambda r: inputs)

    assert result["verify_status"] == "verify timeout", result
    assert not (work / "verify" / "input").exists()


def test_a_resumed_trial_does_not_need_the_copy_again(tmp_path, monkeypatch):
    """Resumability has to survive the removal, in both directions.

    A cached result is returned before anything is fetched or copied, so a resumed sweep neither
    re-downloads nor re-copies -- and the tree it resumes over no longer has the tables in it,
    which is precisely the case this pins. If resuming ever did need verify/input, this fails on
    the second call with the directory already gone.
    """
    import smol_ladder.run_ladder as runner

    calls: list[str] = []
    result, work, inputs = _trial(tmp_path, monkeypatch)
    assert result["reward"] == 1.0
    assert not (work / "verify" / "input").exists()

    real_run = runner._run_jailed

    def counting_run(cmd, cwd, env, timeout):
        calls.append(str(cwd))
        return real_run(cmd, cwd, env, timeout)

    monkeypatch.setattr(runner, "_run_jailed", counting_run)

    # task_trials is what persists the result, so a resumable trial is one whose result.json is
    # on disk -- writing it here is what the sweep does, not a shortcut around once().
    (work / "result.json").write_text(json.dumps(result, indent=1))
    again = runner.once({"task_id": "t1", "question": "Q?", "files": ["t.csv"], "answer": "42",
                         "reward_mode": "numeric", "atol": 0.0, "rtol": 0.0},
                        "Q?", work, Path(sys.prefix), "m", 2,
                        inputs_of=lambda r: pytest.fail("a cached result re-fetched inputs"))
    assert again["reward"] == 1.0
    assert calls == [], "a resumed trial re-ran something"

    # And a regrade -- retrying the failed half -- rebuilds the copy from the shared cache, so
    # the removal is recoverable rather than destructive. Only the fallback symlink is used if
    # the copy cannot be made, which is why the pass can still run.
    def fake_run(cmd, cwd, env, timeout):
        if cmd == grading_pass_argv(Path(cwd), Path(cwd)):
            assert Path(cwd, "input").is_dir() or Path(cwd, "input").is_symlink()
            return real_run(cmd, cwd, env, timeout)
        (Path(cwd) / "solution.py").write_text("print(42)\n")
        return real_run(["bash", "-c", "true"], cwd, env, timeout)

    monkeypatch.setattr(runner, "_run_jailed", fake_run)
    retried = runner.once({"task_id": "t1", "question": "Q?", "files": ["t.csv"], "answer": "42",
                           "reward_mode": "numeric", "atol": 0.0, "rtol": 0.0},
                          "Q?", work, Path(sys.prefix), "m", 2, inputs_of=lambda r: inputs,
                          retry_failed=True)
    assert retried["reward"] == 1.0, retried.get("stderr", "")[:300]
    assert not (work / "verify" / "input").exists()


def test_the_trial_keeps_its_own_input_symlink(tmp_path, monkeypatch):
    """The trial's own `input` symlink is provenance and costs nothing; the copy is neither.

    One points at the shared cache and is the record of which tables the trial was run against,
    which is what makes a re-run of a rung checkable by hand. The other is a copy of those same
    tables, made only so the sealed pass could bind-mount a self-contained directory.
    """
    result, work, inputs = _trial(tmp_path, monkeypatch)
    assert result["reward"] == 1.0
    link = work / "input"
    assert link.is_symlink()
    assert link.resolve() == inputs.resolve()
    assert inputs.exists(), "the shared cache was disturbed"


# ── an existing tree: reclaiming what an older run left ────────────────────────


def _plant(root: Path, task: str, rung: str, *, result=True, tables=True):
    """One trial directory as an older run left it: a verify/ with the copy and the program."""
    trial = root / task / rung
    verify = trial / "verify"
    (verify / "input").mkdir(parents=True)
    (verify / "input" / "t.csv").write_text("a,b\n1,2\n" * 10)
    (verify / "solution.py").write_text("print(2)\n")
    if result:
        (trial / "result.json").write_text(json.dumps({"reward": 1.0, "prediction": "2"}))
    return trial


def test_reclaim_removes_the_copies_an_older_run_left(tmp_path):
    """The tree already on disk is 24 GB of these, and no trial is re-run to clear them."""
    from smol_ladder.reclaim import verify_copies

    root = tmp_path / "runs" / "test"
    _plant(root, "t1", "L1")
    _plant(root, "t1", "L2")
    _plant(root, "t2", "L1")

    found, total = verify_copies(root)
    assert len(found) == 3, found
    assert total > 0

    from smol_ladder.reclaim import reclaim_verify_copies

    removed, freed = reclaim_verify_copies(root)
    assert removed == 3
    assert freed == total
    assert not list(root.rglob("verify/input")), "a verify copy survived"
    # and every result is still there, because none of them is a copy
    assert (root / "t1" / "L1" / "verify" / "solution.py").exists()
    assert (root / "t2" / "L1" / "result.json").exists()


def test_reclaim_leaves_a_copy_alone_when_the_trial_never_produced_a_result(tmp_path):
    """No result.json means the trial was killed mid-grading, and its copy may be the only
    evidence of what it was doing. The instruction is explicit about this guard, and it is the
    difference between reclaiming disk and destroying an in-flight run."""
    from smol_ladder.reclaim import reclaim_verify_copies, verify_copies

    root = tmp_path / "runs" / "test"
    kept = _plant(root, "done", "L1", result=True)
    partial = _plant(root, "inflight", "L1", result=False)

    found, _ = verify_copies(root)
    assert [p.parent.parent.parent.name for p in found] == ["done"], \
        "a trial with no result was treated as reclaimable"

    removed, _ = reclaim_verify_copies(root)
    assert removed == 1
    assert (kept / "verify" / "solution.py").exists()
    assert (partial / "verify" / "input").exists(), "an in-flight trial's copy was deleted"


def test_reclaim_dry_run_reports_without_removing(tmp_path, capsys):
    """The default has to be safe to run over the shared tree, and it prints the same numbers."""
    from smol_ladder.reclaim import main

    root = tmp_path / "runs" / "test"
    _plant(root, "t1", "L1")

    argv = ["reclaim", "--split", "test", "--dry-run"]
    import smol_ladder.reclaim as reclaim
    import smol_ladder.tasks as tasks

    original_data, original_argv = reclaim.DATA, sys.argv
    reclaim.DATA = tmp_path
    sys.argv = argv
    try:
        main()
    finally:
        reclaim.DATA, sys.argv = original_data, original_argv

    out = capsys.readouterr().out
    assert "verify/input" in out, out
    assert (root / "t1" / "L1" / "verify" / "input").exists(), "the dry run deleted something"
