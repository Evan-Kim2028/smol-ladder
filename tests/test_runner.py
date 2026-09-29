import subprocess
import sys
import time
from pathlib import Path

from smol_ladder.run_ladder import _run_jailed


def test_a_hanging_command_actually_terminates(tmp_path):
    """The regression that mattered: a trial whose grandchildren hold the pipes open.

    subprocess.run's timeout kills only the direct child, so the read blocked long past the
    deadline and six workers sat for 14 minutes. This must come back at roughly the timeout.
    """
    script = "import subprocess,sys,time\n" \
             "subprocess.Popen(['sleep', '300'])\n" \
             "time.sleep(300)\n"
    start = time.time()
    try:
        _run_jailed([sys.executable, "-c", script], tmp_path,
                    {"PATH": f"{Path(sys.executable).parent}:/usr/bin:/bin"}, 5)
    except subprocess.TimeoutExpired:
        pass
    elapsed = time.time() - start
    assert elapsed < 45, f"took {elapsed:.0f}s for a 5s timeout"


def test_a_normal_command_returns_its_output(tmp_path):
    proc = _run_jailed([sys.executable, "-c", "print('hello')"], tmp_path,
                       {"PATH": f"{Path(sys.executable).parent}:/usr/bin:/bin"}, 60)
    assert proc.returncode == 0
    assert b"hello" in proc.stdout


def test_a_failing_command_reports_its_exit_code(tmp_path):
    proc = _run_jailed([sys.executable, "-c", "raise SystemExit(3)"], tmp_path,
                       {"PATH": f"{Path(sys.executable).parent}:/usr/bin:/bin"}, 60)
    assert proc.returncode == 3


def test_killing_a_group_leaves_no_zombies(tmp_path):
    """The bug this catches: killpg on an already-dead leader raised ProcessLookupError and the
    early return skipped the reap. Thirty workers x one uncollected child each left the runner
    with 25 zombies and 63 threads, at which point it stopped scheduling work entirely."""
    import subprocess as sp
    from smol_ladder.run_ladder import _kill_group

    before = _zombie_count()
    for _ in range(5):
        proc = sp.Popen([sys.executable, "-c", "import time; time.sleep(300)"],
                        stdout=sp.PIPE, stderr=sp.PIPE, start_new_session=True)
        time.sleep(0.2)
        _kill_group(proc)
        proc.stdout.close()
        proc.stderr.close()
    # Give the reaper a moment, then confirm the kernel has collected them.
    time.sleep(0.5)
    assert _zombie_count() <= before + 1, "killed children were not reaped"


def _zombie_count() -> int:
    import subprocess as sp
    out = sp.run(["ps", "-eo", "stat="], capture_output=True, text=True).stdout
    return sum(1 for line in out.splitlines() if line.strip().startswith("Z"))


def test_agent_command_with_grandchildren_holding_the_pipes_returns():
    """The hang that actually stalled the run: a model command that spawns workers.

    The parent python exits but its children keep the captured pipes open, so a plain
    subprocess.run waits on processes the model never asked about. One trial sat 16 minutes
    on joblib workers before this was fixed.
    """
    from smol_ladder.or_agent import run_command
    cmd = ("python3 -c \"import multiprocessing as m, time\n"
           "ps = [m.Process(target=time.sleep, args=(300,)) for _ in range(3)]\n"
           "[p.start() for p in ps]\"")
    start = time.time()
    out = run_command(cmd, timeout=5)
    elapsed = time.time() - start
    assert elapsed < 60, f"took {elapsed:.0f}s for a 5s timeout"
    assert "timed out" in out


def test_agent_command_returns_ordinary_output():
    from smol_ladder.or_agent import run_command
    assert "hello" in run_command("echo hello")
    # non-UTF-8 output must not raise
    assert run_command("printf '\\xff\\xfe'") is not None


def test_agent_command_reports_a_timeout_instead_of_raising():
    from smol_ladder.or_agent import run_command
    out = run_command("sleep 300", timeout=3)
    assert "timed out" in out


def test_a_backgrounded_command_does_not_defeat_the_timeout(tmp_path):
    """`nohup ... &` is what an agent reaches for to keep a long job running.

    The detached grandchild holds the write end of the captured pipe, so communicate() blocks
    on a descriptor nothing will close. A trial sat 31 minutes against a 20-minute cap
    because of exactly this. The deadline has to win.
    """
    script = "nohup sleep 300 >/dev/null 2>&1 & echo started"
    start = time.time()
    try:
        _run_jailed(["bash", "-c", script], tmp_path,
                    {"PATH": "/usr/bin:/bin"}, 5)
    except subprocess.TimeoutExpired:
        pass
    elapsed = time.time() - start
    assert elapsed < 45, f"took {elapsed:.0f}s for a 5s timeout"


def test_the_agent_shell_survives_a_backgrounded_command():
    from smol_ladder.or_agent import run_command
    start = time.time()
    out = run_command("nohup sleep 300 >/dev/null 2>&1 & echo started", timeout=5)
    elapsed = time.time() - start
    assert elapsed < 45, f"run_command took {elapsed:.0f}s for a 5s timeout"
    assert isinstance(out, str)


def test_the_control_runs_without_a_reference(monkeypatch, tmp_path):
    """The L1+schema control is built from the tables alone.

    Gating it on a verified reference would discard the L1-vs-control comparison on exactly
    the tasks whose reference we failed to build, which is most of the interesting failures.
    """
    import smol_ladder.run_ladder as runner

    ran: list[str] = []
    monkeypatch.setattr(runner, "read_source", lambda row, split: None)

    def fake_once(row, prompt, work, venv, model, max_turns, retry_failed, inputs_of):
        ran.append(work.name)
        return {"reward": 0.0, "agent_status": "exit 0", "prediction": ""}

    monkeypatch.setattr(runner, "once", fake_once)
    monkeypatch.setattr(runner, "prompt_for", lambda row, split, rung: "p")
    row = {"task_id": "t1", "question": "q", "files": [], "answer": "1"}

    out = runner.task_trials(row, "test", ["L1_schema", "L2"], tmp_path, "m", 5,
                             runs_root=tmp_path / "runs")

    assert ran == ["L1_schema"], f"control should run without a reference, ran {ran}"
    # and the rung that genuinely needs one is reported as skipped, not silently passed off
    assert out[1]["skipped"] == "no verified reference"
