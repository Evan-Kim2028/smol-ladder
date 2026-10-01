"""K samples per (task, rung), no-climb mode, and provenance on every result.

The runner stored ONE result per (task, rung) and stopped climbing at the first pass. Four L1
reruns disagreed on 18.9% of the 244 test tasks, so one sample cannot identify a task's first
passing rung and climb-stop makes the per-rung histogram selection-biased. These tests pin the
sampling layout, --no-climb, and the provenance that stops results from two ladder versions being
pooled by accident.

No model calls: the solver runs in a subprocess, so the model loop is stubbed the same way
test_once_runs_end_to_end stubs it, and the offline grading pass runs for real.

    uv run --with pytest pytest -q tests/test_samples.py
"""

import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from smol_ladder import run_ladder as runner

REPO = Path(runner.__file__).resolve().parent.parent


def grading_pass_argv(inputs: Path, verify: Path) -> list[str]:
    """The exact argv once() runs for a trial's offline grading pass (see test_runner.py)."""
    return ["nice", "-n", "15", "bwrap", "--ro-bind", "/", "/", "--dev", "/dev",
            "--proc", "/proc", "--unshare-net", "--unshare-pid", "--tmpfs", "/tmp",
            "--bind", str(verify), "/tmp/work",
            "--chdir", "/tmp/work", "--die-with-parent",
            "--setenv", "OMP_NUM_THREADS", "1", "--setenv", "OPENBLAS_NUM_THREADS", "1",
            sys.executable, "solution.py"]


def fake_once(ran: list, rewards: list[float] | None = None):
    """A stand-in for once() that records what it was handed and returns a scripted reward.

    It reproduces once()'s two contracts that the layout depends on -- a clean cached result is
    reused, and a fresh trial writes prompt.txt beside its result -- because a stub that skips
    either one leaves a tree shape the real runner never produces, and then the tests below are
    testing the stub rather than the layout.
    """
    rewards = list(rewards or [])

    def once(row, prompt, work, venv, model, max_turns, retry_failed=False, inputs_of=None,
             rung_label="run", provenance=None, was_run=None):
        if was_run is not None:
            was_run.clear()
        work = Path(work)
        cached = work / "result.json"
        if cached.exists():
            prior = json.loads(cached.read_text())
            if prior.get("agent_status") == "exit 0" or not retry_failed:
                return prior
        if was_run is not None:
            was_run.append(True)
        ran.append({"work": work, "rung_label": rung_label, "provenance": provenance,
                    "prompt": prompt, "model": model})
        (work / "prompt.txt").write_text(prompt)
        reward = rewards.pop(0) if rewards else 0.0
        return {"task_id": row["task_id"], "reward": reward, "agent_status": "exit 0",
                "prediction": str(reward),
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()}
    return once


def prepped(monkeypatch, ran, rewards=None, reference="REF"):
    monkeypatch.setattr(runner, "once", fake_once(ran, rewards))
    monkeypatch.setattr(runner, "prompt_for", lambda row, split, rung: f"prompt:{rung}")
    monkeypatch.setattr(runner, "read_source", lambda row, split: reference)
    return {"task_id": "t1", "question": "q", "files": [], "answer": "1"}


def results_on_disk(runs_root: Path) -> list[str]:
    return sorted(str(p.relative_to(runs_root)) for p in runs_root.rglob("result.json"))


# --- the samples layout --------------------------------------------------------------------

def test_k_samples_land_in_suffixed_directories(tmp_path, monkeypatch):
    """--samples 3 writes the bare rung dir (sample 0) plus s1 and s2 beside it."""
    ran = []
    row = prepped(monkeypatch, ran)
    root = tmp_path / "runs"

    out = runner.task_trials(row, "test", ["L1"], tmp_path, "m", 5, samples=3, runs_root=root)

    assert [str(Path(w["work"]).relative_to(root / "t1")) for w in ran] == ["L1", "L1/s1", "L1/s2"]
    assert [r["sample"] for r in out] == [0, 1, 2]
    assert results_on_disk(root) == ["t1/L1/result.json", "t1/L1/s1/result.json",
                                     "t1/L1/s2/result.json"]


def test_an_existing_unsuffixed_result_is_read_in_place_as_sample_zero(tmp_path, monkeypatch):
    """Existing results stay where they are and are reused; nothing is moved or rewritten."""
    prior = {"task_id": "t1", "reward": 1.0, "agent_status": "exit 0", "prediction": "1"}
    root = tmp_path / "runs"
    bare = root / "t1" / "L1"
    bare.mkdir(parents=True)
    (bare / "result.json").write_text(json.dumps(prior))
    stamp = (bare / "result.json").stat().st_mtime_ns
    ran = []
    row = prepped(monkeypatch, ran)

    out = runner.task_trials(row, "test", ["L1"], tmp_path, "m", 5, samples=3, runs_root=root)

    # sample 0 was reused, so it was never handed to once(): no second trial on it.
    assert [str(Path(w["work"]).relative_to(root / "t1")) for w in ran] == ["L1/s1", "L1/s2"]
    assert out[0]["prediction"] == "1"
    assert json.loads((bare / "result.json").read_text()) == prior
    assert (bare / "result.json").stat().st_mtime_ns == stamp, "the legacy result was rewritten"
    # and no migration directory appeared
    assert not (root / "t1" / "L1" / "s0").exists()


def test_a_rerun_of_the_same_sample_count_does_not_re_run_anything(tmp_path, monkeypatch):
    """Resumability has to survive the sample axis, or every sweep is paid for twice."""
    ran = []
    row = prepped(monkeypatch, ran)
    root = tmp_path / "runs"

    for _ in range(2):
        runner.task_trials(row, "test", ["L1"], tmp_path, "m", 5, samples=3, runs_root=root)

    assert [str(Path(w["work"]).relative_to(root / "t1")) for w in ran] == \
        ["L1", "L1/s1", "L1/s2"], "the second pass re-ran trials that were already on disk"


def test_rerunning_with_more_samples_only_adds_the_missing_ones(tmp_path, monkeypatch):
    ran = []
    row = prepped(monkeypatch, ran)
    root = tmp_path / "runs"
    runner.task_trials(row, "test", ["L1"], tmp_path, "m", 5, samples=2, runs_root=root)
    runner.task_trials(row, "test", ["L1"], tmp_path, "m", 5, samples=4, runs_root=root)

    assert [str(Path(w["work"]).relative_to(root / "t1")) for w in ran] == \
        ["L1", "L1/s1", "L1/s2", "L1/s3"]
    assert results_on_disk(root) == ["t1/L1/result.json", "t1/L1/s1/result.json",
                                     "t1/L1/s2/result.json", "t1/L1/s3/result.json"]


# --- --no-climb ----------------------------------------------------------------------------

def test_climbing_stops_after_the_rung_that_passed_but_finishes_its_samples(tmp_path, monkeypatch):
    ran = []
    row = prepped(monkeypatch, ran, rewards=[1.0, 1.0, 0.0])
    root = tmp_path / "runs"

    out = runner.task_trials(row, "test", ["L1", "L2", "L3"], tmp_path, "m", 5,
                             samples=3, runs_root=root)

    assert [str(Path(w["work"]).relative_to(root / "t1")) for w in ran] == \
        ["L1", "L1/s1", "L1/s2"]
    assert [r["rung"] for r in out] == ["L1"] * 3


def test_no_climb_runs_every_requested_rung_on_every_task(tmp_path, monkeypatch):
    """The A7 recommendation: an unconditional denominator makes P(pass at k, fail at k+1)
    measurable on every adjacent pair instead of only on tasks that happened to climb."""
    ran = []
    row = prepped(monkeypatch, ran, rewards=[1.0, 0.0, 1.0, 1.0, 0.0])
    root = tmp_path / "runs"

    out = runner.task_trials(row, "test", ["L1", "L2", "L3"], tmp_path, "m", 5,
                             samples=2, climb=False, runs_root=root)

    assert [str(Path(w["work"]).relative_to(root / "t1")) for w in ran] == \
        ["L1", "L1/s1", "L2", "L2/s1", "L3", "L3/s1"]
    assert [r["rung"] for r in out] == ["L1", "L1", "L2", "L2", "L3", "L3"]


def test_no_climb_still_skips_the_rungs_that_need_a_reference_it_does_not_have(tmp_path, monkeypatch):
    """--no-climb is about the ladder, not about the reference gate. L2-L4 stay gated."""
    ran = []
    row = prepped(monkeypatch, ran, reference=None)
    root = tmp_path / "runs"

    out = runner.task_trials(row, "test", ["L1", "L1_schema", "L2"], tmp_path, "m", 5,
                             samples=2, climb=False, runs_root=root)

    ran_rungs = [w["provenance"]["rung"] for w in ran]
    assert sorted(set(ran_rungs)) == ["L1", "L1+schema"]
    skipped = [r for r in out if r.get("skipped")]
    assert len(skipped) == 2 and {r["rung"] for r in skipped} == {"L2"}
    assert skipped[0]["skipped"] == "no verified reference"


# --- per-sample scratch --------------------------------------------------------------------

def test_each_sample_of_one_rung_gets_its_own_scratch(tmp_path, monkeypatch):
    """Two samples of the same (task, rung) run concurrently and must not share $HOME.

    once() rmtree's the scratch on entry, so a shared directory means one trial deletes the
    other's files mid-run -- the same leak test_each_trial_gets_its_own_home guards across tasks,
    now across samples."""
    real_run = runner._run_jailed
    homes: list[str] = []

    def fake_run(cmd, cwd, env, timeout):
        if cmd == grading_pass_argv(Path(cwd), Path(cwd)):
            return real_run(cmd, cwd, env, timeout)
        homes.append(env["HOME"])
        (Path(cwd) / "solution.py").write_text("print(42)\n")
        return real_run(["bash", "-c", "echo ok"], cwd, env, timeout)

    monkeypatch.setattr(runner, "_run_jailed", fake_run)
    monkeypatch.setattr(runner, "prompt_for", lambda row, split, rung: "p")
    monkeypatch.setattr(runner, "read_source", lambda row, split: "ref")
    monkeypatch.setenv("SMOL_LADDER_SCRATCH", str(tmp_path / "scratch"))
    inputs = tmp_path / "in"
    inputs.mkdir()
    (inputs / "t.csv").write_text("a\n1\n")
    row = {"task_id": "t1", "question": "Q?", "files": ["t.csv"], "answer": "42",
           "reward_mode": "numeric", "atol": 0.0, "rtol": 0.0}

    runner.task_trials(row, "test", ["L1"], Path(sys.prefix), "m", 2, samples=3,
                       inputs_of=lambda r: inputs, runs_root=tmp_path / "runs")

    assert len(homes) == 3
    assert len(set(homes)) == 3, f"two samples shared a HOME: {homes}"
    assert sorted(Path(h).relative_to(tmp_path / "scratch" / "trials" / "t1").as_posix()
                  for h in homes) == ["L1", "L1s1", "L1s2"]


# --- provenance ----------------------------------------------------------------------------

def run_one_real_trial(tmp_path, monkeypatch, work: Path, prompt: str = "Q?",
                       provenance=None, inputs_of=None):
    """once() for real, with only the model loop stubbed. Returns the result dict."""
    real_run = runner._run_jailed

    def fake_run(cmd, cwd, env, timeout):
        if cmd == grading_pass_argv(Path(cwd), Path(cwd)):
            return real_run(cmd, cwd, env, timeout)
        (Path(cwd) / "solution.py").write_text("print(42)\n")
        return real_run(["bash", "-c", "echo ok"], cwd, env, timeout)

    monkeypatch.setattr(runner, "_run_jailed", fake_run)
    monkeypatch.setenv("SMOL_LADDER_SCRATCH", str(tmp_path / "scratch"))
    inputs = inputs_of or (lambda _r: _mkinputs(tmp_path))
    row = {"task_id": "t1", "question": "Q?", "files": ["t.csv"], "answer": "42",
           "reward_mode": "numeric", "atol": 0.0, "rtol": 0.0}
    return runner.once(row, prompt, work, Path(sys.prefix), "model-x", 2,
                       inputs_of=inputs, rung_label="L1s1", provenance=provenance)


def _mkinputs(tmp_path: Path) -> Path:
    inputs = tmp_path / "in"
    inputs.mkdir(exist_ok=True)
    (inputs / "t.csv").write_text("a\n1\n")
    return inputs


def test_a_result_records_the_prompt_hash_rung_sample_model_commit_and_time(tmp_path, monkeypatch):
    work = tmp_path / "trial" / "L1" / "s1"
    result = run_one_real_trial(tmp_path, monkeypatch, work, prompt="Q? exactly",
                                provenance={"rung": "L1", "sample": 1})

    assert result["prompt_sha256"] == hashlib.sha256(b"Q? exactly").hexdigest()
    assert result["rung"] == "L1"
    assert result["sample"] == 1
    assert result["model"] == "model-x"
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True,
                          text=True, check=True).stdout.strip()
    assert result["git_commit"] == head
    assert datetime.fromisoformat(result["timestamp"]).tzinfo is not None


def test_the_prompt_is_saved_beside_the_result(tmp_path, monkeypatch):
    """prompt.txt is what makes a prompt hash checkable instead of a bare assertion."""
    work = tmp_path / "trial" / "L1" / "s1"
    prompt = "Q? with a\nnewline and unicode: é"
    result = run_one_real_trial(tmp_path, monkeypatch, work, prompt=prompt)
    saved = (work / "prompt.txt").read_text()
    assert saved == prompt
    assert hashlib.sha256(saved.encode()).hexdigest() == result["prompt_sha256"]


def test_the_saved_prompt_matches_the_hash_in_the_written_result(tmp_path, monkeypatch):
    """End to end through the runner, so the two files on disk agree with each other."""
    ran = []
    row = prepped(monkeypatch, ran, rewards=[1.0])
    root = tmp_path / "runs"

    runner.task_trials(row, "test", ["L1"], tmp_path, "m", 5, samples=2, runs_root=root)

    for relative in ["t1/L1", "t1/L1/s1"]:
        saved = (root / relative / "prompt.txt").read_text()
        stored = json.loads((root / relative / "result.json").read_text())
        assert saved == "prompt:L1"
        assert stored["prompt_sha256"] == hashlib.sha256(saved.encode()).hexdigest()
        assert stored["rung"] == "L1" and stored["sample"] in (0, 1)


def test_a_cached_result_without_provenance_is_not_rewritten(tmp_path, monkeypatch):
    """Old results predate provenance. Reusing them must not stamp today's commit on them."""
    work = tmp_path / "trial" / "L1"
    work.mkdir(parents=True)
    prior = {"task_id": "t1", "reward": 1.0, "agent_status": "exit 0", "prediction": "1"}
    (work / "result.json").write_text(json.dumps(prior))

    result = runner.once({"task_id": "t1", "question": "q", "files": [], "answer": "1"}, "Q?",
                         work, Path(sys.prefix), "m", 2,
                         inputs_of=lambda _r: _mkinputs(tmp_path), provenance={"rung": "L1"})
    assert result == prior
    assert json.loads((work / "result.json").read_text()) == prior


def test_the_git_commit_is_read_once_per_process(tmp_path, monkeypatch):
    """Per-trial `git rev-parse` over a few hundred tasks is a subprocess storm; memoise it."""
    calls = []
    real = subprocess.run

    def counting(cmd, *a, **kw):
        if cmd[:2] == ["git", "rev-parse"]:
            calls.append(cmd)
        return real(cmd, *a, **kw)

    monkeypatch.setattr(runner.subprocess, "run", counting)
    runner.git_provenance.cache_clear()
    for i in range(5):
        runner.git_provenance()
    runner.git_provenance.cache_clear()
    assert len(calls) == 1, calls


if __name__ == "__main__":
    raise SystemExit(__import__("pytest").main([__file__, "-q"]))