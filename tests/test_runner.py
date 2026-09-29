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
