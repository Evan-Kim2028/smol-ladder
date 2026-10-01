import json
import subprocess
import sys
import time
from pathlib import Path

from smol_ladder.run_ladder import _run_jailed


def grading_pass_argv(inputs: Path, verify: Path) -> list[str]:
    """The exact argv once() runs for a trial's offline grading pass.

    Built rather than pattern-matched, because a test that guesses at the command list silently
    stops exercising the thing it claims to: here that meant skipping the pass, reading stdout
    that was never written, and passing an assertion about the wrong prediction on the agent's
    output instead.
    """
    return ["nice", "-n", "15", "bwrap", "--ro-bind", "/", "/", "--dev", "/dev",
            "--proc", "/proc", "--unshare-net", "--unshare-pid", "--tmpfs", "/tmp",
            "--bind", str(verify), "/tmp/work",
            "--chdir", "/tmp/work", "--die-with-parent",
            "--setenv", "OMP_NUM_THREADS", "1", "--setenv", "OPENBLAS_NUM_THREADS", "1",
            sys.executable, "solution.py"]


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


def test_once_runs_end_to_end(tmp_path, monkeypatch):
    """Exercise once() itself, not a stand-in.

    The scratch change introduced a NameError in once() that 67 tests missed, because every
    test either stubbed once() out or never called it. The solver runs in a *subprocess*, so it
    cannot be monkeypatched from here; instead the first jailed call is short-circuited with a
    stub that stands in for the model loop and writes solution.py into the cwd it was handed,
    which is the contract the real solver honours.
    """
    import smol_ladder.run_ladder as runner

    real_run = runner._run_jailed
    scratch_marker = tmp_path / "scratch"
    seen_cwd: list[str] = []

    def fake_run(cmd, cwd, env, timeout):
        # Anything that is not the grading pass is the model loop, and only that is stubbed:
        # the offline pass runs the real bubblewrap so the prediction comes from solution.py.
        if cmd == grading_pass_argv(Path(cwd), Path(cwd)):
            return real_run(cmd, cwd, env, timeout)
        seen_cwd.append(str(cwd))
        (Path(cwd) / "solution.py").write_text("print(42)\n")
        return real_run(["bash", "-c", "echo ok"], cwd, env, timeout)

    monkeypatch.setattr(runner, "_run_jailed", fake_run)
    monkeypatch.setenv("SMOL_LADDER_SCRATCH", str(scratch_marker))

    inputs = tmp_path / "in"
    inputs.mkdir()
    (inputs / "t.csv").write_text("a\n1\n")
    work = tmp_path / "trial" / "L1"
    row = {"task_id": "t1", "question": "Q?", "files": ["t.csv"], "answer": "42",
           "reward_mode": "numeric", "atol": 0.0, "rtol": 0.0}
    result = runner.once(row, "Q?", work, Path(sys.prefix), "m", 2,
                         inputs_of=lambda r: inputs, rung_label="L1")
    assert result["agent_status"] == "exit 0", result.get("stderr", "")[:300]
    assert (work / "solution.py").exists()
    assert (work / "solution.py").read_text() == "print(42)\n"

    # the agent worked in scratch and left nothing behind there
    trial_scratch = scratch_marker / "trials" / "t1" / "L1"
    assert seen_cwd[0] == str(trial_scratch), f"agent ran in {seen_cwd[0]}"
    assert not trial_scratch.exists(), "per-trial scratch survived the trial"
    # The empty trials/<task> parents stay: a few directories cost nothing, and pruning them
    # per trial is a race with every other trial using the same root. What must not survive is
    # anything the agent wrote.
    assert not [p for p in scratch_marker.rglob("*") if p.is_file()], \
        f"scratch kept the agent's files: {list(scratch_marker.rglob('*'))}"

    # nothing the agent wrote may reach the trial directory, which is a result we keep
    assert not list(work.glob("*.whl"))
    assert not list(work.glob("tmp"))
    # input is a symlink to the shared table cache. The verifier's copy is real and expected:
    # the offline grading pass cannot bind-mount into /tmp/work, so it gets its own copy.
    link = work / "input"
    assert link.is_symlink()
    assert link.resolve() == inputs.resolve()
    # nothing beyond the result, the prompt, the solution, and the verifier's own working copy
    allowed = {"input", "solution.py", "verify", "result.json", "turns.json", "prompt.txt"}
    assert {p.name for p in work.iterdir()} <= allowed, sorted(p.name for p in work.iterdir())

    # the prediction is solution.py's own output when re-run offline, not the agent's stdout
    assert result["prediction"] == "42", result["prediction"]
    assert result["reward"] == 1.0


def test_once_grades_the_offline_run_and_not_the_agents_stdout(tmp_path, monkeypatch):
    """The failure that made the end-to-end test fail, pinned on its own.

    once() runs solution.py in a second, sealed bubblewrap -- that is the "like the Harbor
    verifier" part of its docstring -- and grades what *that* printed. If the prediction were
    taken from the agent's stdout instead, any trial whose script printed nothing would still be
    scored on whatever the agent's last shell command echoed, and the two would be
    indistinguishable in the results.
    """
    import smol_ladder.run_ladder as runner

    real_run = runner._run_jailed

    def fake_run(cmd, cwd, env, timeout):
        if cmd == grading_pass_argv(Path(cwd), Path(cwd)):
            return real_run(cmd, cwd, env, timeout)
        # A script that produces the right answer only offline, and an agent whose last words
        # are the wrong answer. Reading the agent's stdout would record 0 here.
        (Path(cwd) / "solution.py").write_text("print(6 * 7)\n")
        return real_run(["bash", "-c", "echo 'I could not solve this'"], cwd, env, timeout)

    monkeypatch.setattr(runner, "_run_jailed", fake_run)
    monkeypatch.setenv("SMOL_LADDER_SCRATCH", str(tmp_path / "scratch"))

    inputs = tmp_path / "in"
    inputs.mkdir()
    (inputs / "t.csv").write_text("a\n1\n")
    work = tmp_path / "trial" / "L1"
    row = {"task_id": "t1", "question": "Q?", "files": ["t.csv"], "answer": "42",
           "reward_mode": "numeric", "atol": 0.0, "rtol": 0.0}
    result = runner.once(row, "Q?", work, Path(sys.prefix), "m", 2,
                         inputs_of=lambda r: inputs)

    assert result["prediction"] == "42", result["prediction"]
    assert result["reward"] == 1.0
    assert "could not solve" not in json.dumps(result)


def test_each_trial_gets_its_own_home(tmp_path, monkeypatch):
    """A shared $HOME for the whole sweep is the leak once()'s comment warns about.

    Concurrent agents can read each other's pip cache and ~/.cache, and nothing ever cleans it
    up. HOME is also the directory once() wipes, so pointing it at a shared root would have the
    runner deleting another trial's scratch out from under it.
    """
    import smol_ladder.run_ladder as runner

    homes: list[str] = []
    real_run = runner._run_jailed

    def fake_run(cmd, cwd, env, timeout):
        if cmd == grading_pass_argv(Path(cwd), Path(cwd)):  # HOME there is the verify dir
            return real_run(cmd, cwd, env, timeout)
        homes.append(env["HOME"])
        (Path(cwd) / "solution.py").write_text("print(1)\n")
        return real_run(["bash", "-c", "true"], cwd, env, timeout)

    monkeypatch.setattr(runner, "_run_jailed", fake_run)
    monkeypatch.setenv("SMOL_LADDER_SCRATCH", str(tmp_path / "scratch"))

    inputs = tmp_path / "in"
    inputs.mkdir()
    (inputs / "t.csv").write_text("a\n1\n")
    base = {"files": ["t.csv"], "answer": "1", "reward_mode": "numeric", "atol": 0.0, "rtol": 0.0}
    for i, (task_id, rung) in enumerate([("t1", "L1"), ("t2", "L1"), ("t1", "L2")]):
        work = tmp_path / "trial" / task_id / rung
        runner.once({"task_id": task_id, "question": "Q?", **base}, "Q?", work,
                    Path(sys.prefix), "m", 2, inputs_of=lambda r: inputs, rung_label=rung)

    assert len(set(homes)) == 3, f"two trials shared a HOME: {homes}"
    assert all(h.endswith(("t1/L1", "t2/L1", "t1/L2")) for h in homes), homes


def test_jail_chdirs_into_scratch_not_the_trial_dir(tmp_path):
    """The jail's --chdir must be the scratch dir, or pip download -d . writes to the results."""
    from smol_ladder.run_ladder import jail
    work = tmp_path / "trial"
    work.mkdir()
    inputs = tmp_path / "in"
    inputs.mkdir()
    (inputs / "t.csv").write_text("a\n1\n")
    scratch = tmp_path / "scratch"
    args = jail(work, inputs, Path(sys.prefix), scratch)
    chdir = args[args.index("--chdir") + 1]
    assert chdir == str(scratch), f"jail chdirs to {chdir}, not scratch"
    assert str(work) not in args, "the trial directory is still bound writable into the jail"
    # and the agent still sees its tables as ./input
    assert (scratch / "input").is_symlink()


def test_a_program_can_read_its_input_directory(tmp_path):
    """The regression: a --tmpfs /tmp plus a nested bind into /tmp/work is order-sensitive.

    bwrap applies the tmpfs after the --bind that created /tmp/work, so the destination is
    gone and the mount fails with "Unable to mount source on destination". The solution then
    never runs, stdout is empty, and the trial grades 0.0 while looking like an ordinary
    failure. The fix is a self-contained work dir: copy the tables in, bind once.
    """
    work = tmp_path / "verify"
    (work / "input").mkdir(parents=True)
    (work / "input" / "t.csv").write_text("a,b\n1,2\n3,4\n")
    (work / "solution.py").write_text(
        "import pandas as pd\n"
        "df = pd.read_csv('input/t.csv')\n"
        "print(int(df['a'].sum()))\n")
    env = {"PATH": f"{Path(sys.executable).parent}:/usr/bin:/bin", "HOME": str(work)}
    proc = _run_jailed(
        ["nice", "-n", "15", "bwrap", "--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc",
         "--unshare-net", "--unshare-pid", "--tmpfs", "/tmp", "--bind", str(work), "/tmp/work",
         "--chdir", "/tmp/work", "--die-with-parent", sys.executable, "solution.py"],
        work, env, 120)
    assert proc.returncode == 0, proc.stderr.decode()
    assert b"4" in proc.stdout, "solution did not run: " + proc.stdout.decode()[:200]


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


def test_a_verification_pass_that_times_out_is_a_harness_failure_not_a_zero(tmp_path, monkeypatch):
    """The bug the smoke run turned up: a solution that takes longer than the offline deadline.

    The sealed grading pass has a 180s cap, and a TimeoutExpired was caught and turned into out=""
    -- which is then graded exactly like a program that printed nothing. The result kept the
    solver's own "exit 0", so the trial looked clean everywhere: reward 0.0, scored, and counted
    in the pass rate. Two of the six smoke tasks were this, both of which print the right answer
    when given time. A harness deadline is not evidence the model cannot solve the task.
    """
    import smol_ladder.run_ladder as runner

    real_run = runner._run_jailed

    def fake_run(cmd, cwd, env, timeout):
        # The solver finishes and writes a solution; only the offline grading pass overruns.
        if cmd == grading_pass_argv(Path(cwd), Path(cwd)):
            raise subprocess.TimeoutExpired(cmd, timeout)
        (Path(cwd) / "solution.py").write_text("print(42)\n")
        return real_run(["bash", "-c", "true"], cwd, env, timeout)

    monkeypatch.setattr(runner, "_run_jailed", fake_run)
    monkeypatch.setenv("SMOL_LADDER_SCRATCH", str(tmp_path / "scratch"))

    inputs = tmp_path / "in"
    inputs.mkdir()
    (inputs / "t.csv").write_text("a\n1\n")
    row = {"task_id": "t1", "question": "Q?", "files": ["t.csv"], "answer": "42",
           "reward_mode": "numeric", "atol": 0.0, "rtol": 0.0}
    result = runner.once(row, "Q?", tmp_path / "trial" / "L1", Path(sys.prefix), "m", 2,
                         inputs_of=lambda r: inputs)

    assert result["agent_status"] == "exit 0", "the solver did finish; only the grading pass did not"
    assert result["verify_status"] == "verify timeout"
    assert result["reward"] == 0.0
    # and the summary must keep it out of the pass rate entirely
    from smol_ladder.summarize import _finished, pass_fraction
    assert not _finished(result), "a timed-out grading pass is a harness failure"
    assert pass_fraction([result]) is None, "and it contributes no pass fraction"


def test_a_solution_that_crashes_offline_is_also_a_harness_failure(tmp_path, monkeypatch):
    """Same failure, other shape: the program raises on re-run, so again there is no prediction.

    Without the marker this is indistinguishable from a solution that printed the wrong value, and
    the two are very different facts -- one is the model's arithmetic, the other is a program that
    does not run.
    """
    import smol_ladder.run_ladder as runner

    real_run = runner._run_jailed

    def fake_run(cmd, cwd, env, timeout):
        if cmd == grading_pass_argv(Path(cwd), Path(cwd)):
            (Path(cwd) / "solution.py").write_text("raise SystemExit(1)\n")
            return real_run(["bash", "-c", "exit 1"], cwd, env, timeout)
        (Path(cwd) / "solution.py").write_text("print(42)\n")
        return real_run(["bash", "-c", "true"], cwd, env, timeout)

    monkeypatch.setattr(runner, "_run_jailed", fake_run)
    monkeypatch.setenv("SMOL_LADDER_SCRATCH", str(tmp_path / "scratch"))

    inputs = tmp_path / "in"
    inputs.mkdir()
    (inputs / "t.csv").write_text("a\n1\n")
    row = {"task_id": "t1", "question": "Q?", "files": ["t.csv"], "answer": "42",
           "reward_mode": "numeric", "atol": 0.0, "rtol": 0.0}
    result = runner.once(row, "Q?", tmp_path / "trial" / "L1", Path(sys.prefix), "m", 2,
                         inputs_of=lambda r: inputs)

    assert result["verify_status"] == "verify exit 1", result
    from smol_ladder.summarize import _finished
    assert not _finished(result)


def test_a_clean_offline_run_records_no_verify_status(tmp_path, monkeypatch):
    """The marker is only written when something went wrong, so a good trial's record is unchanged
    and old results -- which have no such key -- keep reading as clean."""
    import smol_ladder.run_ladder as runner

    real_run = runner._run_jailed

    def fake_run(cmd, cwd, env, timeout):
        if cmd == grading_pass_argv(Path(cwd), Path(cwd)):
            return real_run(cmd, cwd, env, timeout)
        (Path(cwd) / "solution.py").write_text("print(42)\n")
        return real_run(["bash", "-c", "true"], cwd, env, timeout)

    monkeypatch.setattr(runner, "_run_jailed", fake_run)
    monkeypatch.setenv("SMOL_LADDER_SCRATCH", str(tmp_path / "scratch"))

    inputs = tmp_path / "in"
    inputs.mkdir()
    (inputs / "t.csv").write_text("a\n1\n")
    row = {"task_id": "t1", "question": "Q?", "files": ["t.csv"], "answer": "42",
           "reward_mode": "numeric", "atol": 0.0, "rtol": 0.0}
    result = runner.once(row, "Q?", tmp_path / "trial" / "L1", Path(sys.prefix), "m", 2,
                         inputs_of=lambda r: inputs)

    assert result["reward"] == 1.0
    assert "verify_status" not in result, result


def test_the_control_runs_without_a_reference(monkeypatch, tmp_path):
    """The L1+schema control is built from the tables alone.

    Gating it on a verified reference would discard the L1-vs-control comparison on exactly
    the tasks whose reference we failed to build, which is most of the interesting failures.
    """
    import smol_ladder.run_ladder as runner

    ran: list[str] = []
    monkeypatch.setattr(runner, "read_source", lambda row, split: None)

    def fake_once(row, prompt, work, venv, model, max_turns, retry_failed, inputs_of,
                  rung_label="run", provenance=None, was_run=None, agent="tools"):
        ran.append(work.name)
        if was_run is not None:
            was_run.clear()
            was_run.append(True)
        return {"reward": 0.0, "agent_status": "exit 0", "prediction": ""}

    monkeypatch.setattr(runner, "once", fake_once)
    monkeypatch.setattr(runner, "prompt_for", lambda row, split, rung: "p")
    row = {"task_id": "t1", "question": "q", "files": [], "answer": "1"}

    out = runner.task_trials(row, "test", ["L1_schema", "L2"], tmp_path, "m", 5,
                             runs_root=tmp_path / "runs")

    assert ran == ["L1_schema"], f"control should run without a reference, ran {ran}"
    # and the rung that genuinely needs one is reported as skipped, not silently passed off
    assert out[1]["skipped"] == "no verified reference"
