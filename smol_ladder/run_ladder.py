"""Run the ladder: every rung's prompt through the solver, graded against gold.

The solver runs in our process and executes the agent's shell commands in a bubblewrap jail
holding only the task's tables, so a trial costs no container and no per-trial install.

Pass-rate curve by rung is the headline; "lowest rung that passes" is derived from it.

One sample per (task, rung) cannot support that derivation. Four independent L1 runs on the same
244 test tasks disagreed on 18.9% of them, so a task recorded as an L1 pass at k=1 may have been
about to be an L1 failure, and climbing stops the very first time one happens. `--samples K` runs
K independent trials per (task, rung) under `<task>/<rung>/s<k>/`; the existing unsuffixed layout
is sample 0 and is read in place, never moved. `--no-climb` runs every requested rung on every
task, which is what gives each rung the same denominator.

    uv run python -m smol_ladder.run_ladder --split test --rungs L1 --workers 20
    uv run python -m smol_ladder.run_ladder --split test --workers 20
    uv run python -m smol_ladder.run_ladder --split test --samples 3 --no-climb --workers 20
    uv run python -m smol_ladder.run_ladder --run-tag v2 --split test --no-climb --workers 32

Results of two different ladder versions must not share a directory, so `--run-tag TAG` gives a
sweep its own tree: data/runs/TAG/<split>/, summarised to data/runs/TAG/summary_<split>.json, with
the run's own provenance and counts in data/runs/TAG/RUN.json. No tag is exactly the old tree,
because the existing results live there and nothing is ever moved.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from smol_ladder.grade import grade
from smol_ladder.ladder import hint_source, prompt_for, read_source
from smol_ladder.tasks import DATA, input_dir, load_split
from smol_ladder.upstream import looks_like_a_command

JAIL_RO = ["/usr", "/bin", "/lib", "/lib64", "/etc", "/opt"]
# A trial is one model loop of at most 40 turns, each shell command capped at 150s. 20 minutes
# is generous for that; the old 45-minute cap let one stuck trial hold a worker for three
# quarters of an hour, and with sixteen workers the sweep crawled.
AGENT_TIMEOUT = 1200
# The offline grading pass re-runs the agent's own solution.py under a sealed jail. It is a
# deadline, not a budget: a program that outruns it has told us nothing, so the trial is recorded as
# a harness failure rather than scored on the empty output. 180s is generous for reading a task's
# tables with one BLAS thread.
VERIFY_TIMEOUT = 180


@functools.lru_cache(maxsize=1)
def git_provenance() -> dict:
    """The commit of the code doing the running, and whether it can even be asked.

    Recorded on every result so a number can be traced to the code that produced it. A dirty tree
    is flagged rather than hidden: `git rev-parse HEAD` reports the commit, not the edits on top
    of it, so a results tree written from a modified checkout is otherwise indistinguishable from
    a clean one. Memoised because a `git` subprocess per trial over a few hundred tasks is a
    storm, and a sweep's code cannot change under itself.
    """
    try:
        proc = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent,
                              capture_output=True, text=True, timeout=30)
        commit = proc.stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=Path(__file__).parent,
                                    capture_output=True, text=True,
                                    timeout=30).stdout.strip())
    except (OSError, subprocess.SubprocessError):
        commit, dirty = "", False
    return {"git_commit": commit or "unknown", "git_dirty": dirty}


def prompt_sha256(prompt: str) -> str:
    """A hash of the exact prompt text, so two ladder versions' results cannot be pooled blind.

    The text is saved next to the result as prompt.txt, which is what makes the hash checkable
    instead of a claim. Anything that changes the prompt -- a reworded header, a schema dump that
    got wider -- changes this, and the summariser then refuses to average across the two.
    """
    return hashlib.sha256(prompt.encode()).hexdigest()


def source_for(split: str):
    """Task rows and their input directories, for any of the three sources.

    SmolDataEnvs rows come from the Hub, jupyter-agent rows from the local extract with their
    tables from Kaggle, and synthetic rows from a specification we executed ourselves. All
    three return the same row shape, so everything downstream is source-agnostic.
    """
    if split == "jupyter-agent":
        from smol_ladder.jtasks import input_dir as ja_input_dir
        from smol_ladder.jtasks import load_rows

        return load_rows(), ja_input_dir
    if split == "synthetic":
        from smol_ladder.jtasks import load_synthetic, synthetic_input_dir

        return load_synthetic(), synthetic_input_dir
    return load_split(split), input_dir


def runs_dir(split: str, tag: str | None = None, data: Path | None = None) -> Path:
    """Where a split's trials live. `data/runs/<split>`, or `data/runs/<tag>/<split>` under a tag.

    The tag exists because two ladder versions used to land in one tree, and the prompt-hash check
    only refuses to pool them at summary time -- after the second version has been written over the
    first, and after a resumed sweep has reused the first version's cached result.json as its own.
    The separation has to be in the path.

    The tag is one path segment and is checked as one. Unchecked, `--run-tag ../v2` is a way to
    write into another run's tree, and `--run-tag ../../..` is a way out of data/runs entirely.

    `data` is the tree's root, passed rather than read from the module so a caller with its own root
    (a test, or a summariser pointed somewhere else) gets the same layout and the same validation.
    """
    root = (data or DATA) / "runs"
    if not tag:
        return root / split
    if tag in {".", ".."} or "/" in tag or "\\" in tag or tag.startswith("-"):
        raise ValueError(f"invalid --run-tag {tag!r}: a tag is a single path segment "
                         f"(letters, digits, dot, dash, underscore)")
    return root / tag / split


def jail(work: Path, inputs: Path, venv: Path, scratch: Path | None = None) -> list[str]:
    """The task's tables, the toolchain, and a writable scratch directory.

    The writable directory is NOT the trial directory. $HOME pointed there, and the agents also
    run `pip download <pkg> -d .`, so a single trial collected 2.9 GB of cuda wheels next to
    its own solution.py and the results tree grew to tens of gigabytes. The agent works in
    scratch; once() copies solution.py back, so the only thing kept in the trial directory is
    the result.

    The task's tables and the toolchain are bound read-only. Nothing else: not the repo, not the
    HF cache that holds the gold answers, not a sibling task's solution. Each trial gets its own
    $HOME, and that is the other half of the isolation -- the repo's real $HOME is a tmpfs, so
    the HF cache and every sibling solution are gone.

    The interpreter is bound by its *resolved* path too: .venv/bin/python3 is a symlink into
    uv's managed CPython, and bwrap execs the literal path, so the run fails without it.

    PIDs are unshared so a runaway python cannot outlive its trial and eat cores. The network
    is NOT unshared: the model is remote, and separating that from the agent's own tools would
    need a proxy, not a namespace. The offline grading pass below is the one that must be
    sealed, and it is.
    """
    package = Path(__file__).resolve().parent
    # uv keeps its managed CPython under $HOME, which the tmpfs below hides. Bind the whole
    # interpreter prefix and its lib directory back, or the stdlib (encodings, and the
    # site-packages holding pandas) is not importable.
    interpreter = Path(sys.executable).resolve()
    # inputs is a directory of symlinks into the Kaggle cache, so bind its target too:
    # bwrap follows neither symlinks nor paths through them on the host side.
    ro = {venv, inputs, package, interpreter.parent, interpreter.parent.parent,
          Path(sys.prefix), Path(sys.base_prefix), Path(sys.base_prefix) / "lib"}
    for link in sorted(inputs.glob("*")):
        if link.is_symlink():
            ro.add(link.resolve().parent)
    writable = scratch or work
    writable.mkdir(parents=True, exist_ok=True)
    # The agent must see its tables as ./input inside the writable dir.
    link = writable / "input"
    if not link.exists():
        try:
            link.symlink_to(inputs.resolve())
        except OSError:
            pass
    # Order matters: the catch-all read-only bind of / must come first, or it shadows the
    # /dev and /proc mounts below and CPython cannot read urandom to seed its hash randomiser.
    args = ["bwrap", "--ro-bind", "/", "/", "--tmpfs", str(Path.home())]
    for d in sorted(ro, key=str):
        if d.exists():
            args += ["--ro-bind", str(d), str(d)]
    args += ["--dev", "/dev", "--proc", "/proc", "--unshare-pid", "--tmpfs", "/tmp",
             "--bind", str(writable), str(writable),
             "--chdir", str(writable), "--die-with-parent"]
    return args


def _run_jailed(cmd: list[str], cwd: Path, env: dict, timeout: int) -> subprocess.CompletedProcess:
    """Run a jailed command with a real deadline.

    subprocess.run's own timeout is not enough here: on expiry it kills only the direct child,
    but bwrap's grandchildren still hold the captured pipes open, so the read blocks and the
    worker never returns. Six trials sat that way for 14 minutes. Popen with a manual wait
    lets us kill the whole process group and drain what was written.
    """
    proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, start_new_session=True)
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        # Kill the group, then close our ends of the pipes before draining. An agent that
        # backgrounds a job (`nohup ... &`) leaves a grandchild holding the write end, so
        # communicate() blocks on a descriptor nothing will ever close. The 20-minute cap
        # was not firing for 31 because of exactly this.
        _kill_group(proc)
        for stream in (proc.stdout, proc.stderr):
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    pass
        raise
    finally:
        # Reap. bwrap exits via --die-with-parent, so it can outlive Popen's own wait by a
        # moment; without this the runner accumulated 25 zombies and 63 threads and stopped
        # scheduling new work entirely.
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _kill_group(proc)
    return subprocess.CompletedProcess(cmd, proc.returncode, out, err)


def _kill_group(proc: subprocess.Popen) -> None:
    """Kill the whole group, then always reap.

    The reap is the part that matters and the part that is easy to skip: killpg raises
    ProcessLookupError when the leader is already gone, and returning there leaves its exit
    status uncollected. Thirty workers x one uncollected child each is how the runner ended up
    with 25 zombies and 63 threads, at which point it stopped scheduling anything.
    """
    import signal
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(os.getpgid(proc.pid), sig)
        except (ProcessLookupError, PermissionError):
            break
        try:
            proc.wait(timeout=5)
            return
        except subprocess.TimeoutExpired:
            continue
    # Either the group is already gone or SIGKILL did not land: collect the status anyway.
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass


def once(row: dict, prompt: str, work: Path, venv: Path, model: str, max_turns: int,
         retry_failed: bool = False, inputs_of=input_dir, rung_label: str = "run",
         provenance: dict | None = None, was_run: list | None = None,
         agent: str = "tools") -> dict:
    """One attempt at one rung: run the solver in the jail, then grade its solution offline.

    Resumable: a result.json from a clean run is reused. A crashed trial is only retried when
    asked, so a rerun does not quietly re-roll a genuinely failed task.

    A cached result from before this carried no provenance, so it is returned untouched: stamping
    today's git commit onto a result some older code produced would be a lie, and summarize.py
    treats a missing prompt hash as "unknown ladder version" rather than as agreement.

    `was_run` is a one-element list the caller may pass to learn whether this trial actually ran
    or was reused. It is a list rather than a return flag because the returned dict is the result
    record, and a key like `_fresh` in it would be read by the summariser and written into the
    summary JSON as if it were part of the measurement.

    `agent` picks the protocol, and it is not cosmetic:

    - "tools"  ours. run_shell + write_solution, a persistent ./solution.py, re-run offline.
    - "program" upstream's GRPO/eval protocol. One turn, no tools, one fenced program.
    - "bash"   upstream's SFT protocol. One `bash` tool, submit by writing answer.txt.

    Both upstream protocols run in the same jail and are graded by the same offline pass and the
    same grader, so a local 2B model's number and our solver's number are comparable on the
    grading side and differ only where the protocol differs. That difference is the measurement.

    A fresh result records the protocol in its "agent" key, beside the provenance below, so a
    summary over samples can tell a rung's pass rate apart by which contract was actually run.
    """
    if was_run is not None:
        was_run.clear()
    cached = work / "result.json"
    if cached.exists():
        prior = json.loads(cached.read_text())
        if prior.get("agent_status") == "exit 0" or not retry_failed:
            return prior
    if was_run is not None:
        was_run.append(True)
    work.mkdir(parents=True, exist_ok=True)
    # The prompt, verbatim, next to the result. The hash alone cannot be checked against anything;
    # this can, and it is the only thing that distinguishes two ladder versions on disk.
    (work / "prompt.txt").write_text(prompt)
    inputs = inputs_of(row)  # fetched here, not inside the jail: kagglehub needs $HOME
    # The trial directory keeps a pointer to the tables, and only that: it is provenance on
    # disk (a later re-run of a rung, a human reading a failure) and costs one symlink.
    inp = work / "input"
    if not inp.exists():
        inp.symlink_to(inputs.resolve())
    # One per-trial directory holds everything the agent may write, and once() copies back the
    # single artifact worth keeping. The trial directory itself is a result we keep, and
    # $HOME pointed there, so one `pip install xgboost` left 660 MB of wheels beside
    # solution.py. Redirecting HOME alone was not enough: the agents also run
    # `pip download <package> -d .`, which writes into the *current* directory, so the cwd has
    # to move too.
    trial_scratch = Path(os.environ.get("SMOL_LADDER_SCRATCH", "/var/tmp/smol-ladder/scratch"))
    trial_scratch = trial_scratch / "trials" / row["task_id"] / rung_label
    # Cleared on entry so a retry never inherits the previous attempt's files, and again on the
    # way out: a sweep over hundreds of tasks would otherwise leave every wheel it ever collected.
    shutil.rmtree(trial_scratch, ignore_errors=True)
    trial_scratch.mkdir(parents=True, exist_ok=True)
    env = {
        "PATH": f"{Path(sys.executable).parent}:/usr/local/bin:/usr/bin:/bin",
        # Per trial, not a shared root: with one $HOME for the whole sweep, concurrent agents
        # could read each other's pip cache and ~/.cache, and nothing ever cleaned it up.
        "HOME": str(trial_scratch),
        "LANG": "C.UTF-8",
        # Inside the jail /tmp is its own tmpfs, but the host needs a real directory: a missing
        # TMPDIR makes tools fall back to /tmp, which the bwrap invocation below does not bind.
        "TMPDIR": str(trial_scratch / "tmp"),
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        "PIP_NO_INPUT": "1",
        "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1",
        # joblib and multiprocessing ignore the BLAS caps and will happily take every core
        # (n_jobs=-1). One run had joblib holding 21 of 32 for 16 minutes, which starved the
        # other fifteen workers. Cap the process pool and default n_jobs to that same ceiling.
        "LOKY_MAX_CPU_COUNT": "2", "JOBLIB_START_METHOD": "loky",
        "MKL_DYNAMIC": "FALSE", "NUMEXPR_MAX_THREADS": "1",
        # The endpoint, handed in whole so the solver inside the jail resolves the same server
        # this process would. Every one of these is allowlisted rather than inherited: passing
        # os.environ through gave the agent ten API keys it could read and exfiltrate over the
        # jail's open network. The base URL is not a credential, so it goes in either way, and
        # it is the one value a local server cannot do without.
        "SMOL_LADDER_BASE_URL": os.environ.get("SMOL_LADDER_BASE_URL", ""),
        "SMOL_LADDER_API_KEY_ENV": os.environ.get("SMOL_LADDER_API_KEY_ENV",
                                                  "OPENROUTER_API_KEY"),
        "SMOL_LADDER_CHAT_TEMPLATE_KWARGS": os.environ.get("SMOL_LADDER_CHAT_TEMPLATE_KWARGS",
                                                           ""),
        "OPENROUTER_API_KEY": os.environ.get("OPENROUTER_API_KEY", ""),
    }
    # A key held under a non-default name has to travel under its own name too, since that is
    # what SMOL_LADDER_API_KEY_ENV now points at.
    key_env = env["SMOL_LADDER_API_KEY_ENV"]
    if key_env != "OPENROUTER_API_KEY":
        env[key_env] = os.environ.get(key_env, "")
    script = _agent_script(agent, model, max_turns)
    t0 = time.time()
    try:
        proc = _run_jailed(
            jail(work, inputs, venv, trial_scratch)
            + [sys.executable, "-c", script, prompt],
            trial_scratch, env, timeout=AGENT_TIMEOUT)
        agent_status = f"exit {proc.returncode}"
        stderr = proc.stderr.decode("utf-8", "replace")[-2000:] \
            if isinstance(proc.stderr, bytes) else (proc.stderr or "")[-2000:]
    except subprocess.TimeoutExpired:
        agent_status, stderr = "timeout", ""
    except Exception as e:  # noqa: BLE001 - one bad trial must not kill the sweep
        agent_status, stderr = f"error: {type(e).__name__}", str(e)[-2000:]
    # The agent wrote solution.py into scratch; that is the one artifact we keep. Its stdout
    # is deliberately not the prediction: the agent's last command prints whatever it pleased,
    # so the number graded is whatever solution.py itself printed when re-run offline below.
    produced = trial_scratch / "solution.py"
    if produced.exists():
        shutil.copy(produced, work / "solution.py")
    # The bash protocol's artifact is a submitted answer rather than a program, so there is
    # nothing to re-run offline: the value the model chose IS the prediction. Kept on disk
    # because it is the only evidence of what the run produced.
    submitted = trial_scratch / "answer.txt"
    if submitted.exists():
        shutil.copy(submitted, work / "answer.txt")
    shutil.rmtree(trial_scratch, ignore_errors=True)
    result = {"task_id": row["task_id"], "model": model, "agent_status": agent_status,
              "agent": agent, "base_url": env["SMOL_LADDER_BASE_URL"],
              "agent_seconds": round(time.time() - t0, 1),
              "prediction": "", "reward": 0.0}
    result.update(git_provenance())
    result["prompt_sha256"] = prompt_sha256(prompt)
    result["timestamp"] = datetime.now(timezone.utc).isoformat()
    # rung and sample default to what the caller passed anyway: gen_refs drives once() directly
    # with a rung label and no sample axis, and a reference has rung "reference", sample 0.
    result["rung"] = (provenance or {}).get("rung", rung_label)
    result["sample"] = (provenance or {}).get("sample", 0)
    # Which hand built this rung's text. L2-L4 can come from a validated model hint or from the
    # AST fallback, per task, and a result that did not say which cannot be told apart from one
    # that did; gen_refs calls once() without a split, so it simply goes unstamped.
    if "hint_source" in (provenance or {}):
        result["hint_source"] = provenance["hint_source"]
    if agent == "bash":
        if (work / "answer.txt").exists():
            raw = (work / "answer.txt").read_text().strip()
            if looks_like_a_command(raw):
                # Upstream's own guard: `echo -n 2.14 > answer.txt` prints as the answer is a
                # redirect that never ran, and grading it as the value is how 42% of the old
                # reward's partial credit once went to strings that merely contained it.
                result["prediction"] = raw
                result["reward"] = 0.0
                result["stderr"] = "answer is a command, not a value"
            else:
                result["prediction"] = raw
                result["reward"] = grade(row, raw)
        if stderr:
            result["stderr"] = stderr
        return result
    solution = work / "solution.py"
    if solution.exists():
        # Grade the agent's own last printed line from a clean run, like the Harbor verifier.
        verify = work / "verify"
        verify.mkdir(exist_ok=True)
        (verify / "solution.py").write_text(solution.read_text())
        # The input is copied in rather than bind-mounted at /tmp/work/input. A --tmpfs /tmp
        # plus a nested --ro-bind into /tmp/work is order-sensitive: the tmpfs erases the
        # /tmp/work the previous --bind created, and bwrap then fails with "Unable to mount
        # source on destination", so the solution never runs and every such trial grades 0.0.
        # A plain self-contained work dir has no such ordering to get wrong.
        tables = verify / "input"
        if not tables.exists():
            try:
                shutil.copytree(inputs, tables, symlinks=True)
            except Exception:
                if tables.is_symlink() or tables.exists():
                    tables.unlink()
                tables.symlink_to(inputs.resolve())
        # Upstream runs a program from inside the table directory (`cd /home/user/input &&
        # python3 /tmp/solve.py`), so a bare `pd.read_csv('a.csv')` is the idiomatic spelling in
        # every program these models produced -- but our ladder prompts, and our own agent, are
        # written for `input/a.csv`. Rather than pick one and mis-measure the other, lay the
        # tables out under BOTH roots: keep the cwd one level up (so `input/...` resolves) and
        # link each table beside solution.py (so a bare `a.csv` resolves). Both idioms then run,
        # and the program is measured on its arithmetic rather than on our directory layout.
        #
        # The links are RELATIVE. The offline pass bind-mounts this directory at /tmp/work, and
        # bwrap does not follow a symlink whose target lies outside the mount -- an absolute
        # link back into the trial directory resolves to a path that does not exist inside the
        # jail, so every bare-filename program raised FileNotFoundError and scored 0.0.
        if tables.is_dir():
            for item in sorted(tables.iterdir()):
                if item.name.startswith("."):
                    continue
                link = verify / item.name
                if link.exists() or link.is_symlink():
                    continue
                try:
                    link.symlink_to(Path("input") / item.name)
                except OSError:
                    pass
        # A verification run that hangs is a harness failure, not a model failure. It used to be
        # caught, and out = "" was then graded exactly like a solution that printed nothing --
        # scoring 0.0 and leaving agent_status at the solver's own "exit 0". So a program that
        # simply takes longer than VERIFY_TIMEOUT to re-run was booked as a task the model got
        # wrong, and summarize, which trusts "exit 0", counted it in the pass rate. Two of the
        # smoke run's clean-exit trials were exactly this. A timeout now keeps the agent's own
        # status for the trial but marks the verification separately, so the trial leaves the
        # denominator instead of reading as a 0.
        verify_status = "exit 0"
        try:
            run = _run_jailed(
                ["nice", "-n", "15", "bwrap", "--ro-bind", "/", "/", "--dev", "/dev",
                 "--proc", "/proc", "--unshare-net", "--unshare-pid", "--tmpfs", "/tmp",
                 "--bind", str(verify), "/tmp/work",
                 "--chdir", "/tmp/work", "--die-with-parent",
                 "--setenv", "OMP_NUM_THREADS", "1", "--setenv", "OPENBLAS_NUM_THREADS", "1",
                 sys.executable, "solution.py"],
                verify, {"PATH": f"{Path(sys.executable).parent}:/usr/bin:/bin",
                         "HOME": str(verify), "LANG": "C.UTF-8",
                         "OMP_NUM_THREADS": "1", "MPLBACKEND": "Agg"}, VERIFY_TIMEOUT)
            out = run.stdout.decode("utf-8", "replace") if isinstance(run.stdout, bytes) \
                else (run.stdout or "")
            if run.returncode != 0:
                verify_status = f"verify exit {run.returncode}"
        except subprocess.TimeoutExpired:
            out, verify_status = "", "verify timeout"
        if verify_status != "exit 0":
            result["verify_status"] = verify_status
        lines = [l.strip() for l in out.splitlines() if l.strip()]
        result["prediction"] = lines[-1] if lines else ""
        result["reward"] = grade(row, result["prediction"])
    if stderr:
        result["stderr"] = stderr
    return result


def sample_dir(rung_dir: Path, k: int) -> Path:
    """Where sample k of a rung lives. Sample 0 is the rung directory itself.

    Keeping the first sample where every existing result already is means no result has to be
    moved: a tree of <task>/<rung>/result.json is read as-is as k=1, and --samples K adds
    <task>/<rung>/s1 .. s(K-1) beside it.
    """
    return rung_dir if k == 0 else rung_dir / f"s{k}"


def _agent_script(agent: str, model: str, max_turns: int) -> str:
    """The program that runs *inside* the jail, one per protocol.

    argv[1] is the user turn of the conversation, which for the upstream protocols is built by
    `upstream.program_prompt` / `upstream.bash_prompt` so the text matches what the model was
    trained on; the system turn comes from the same module and is not the rung prompt, because a
    rung's extra information has to arrive in the user turn to be a rung at all.

    All three write turns.json so once() can tell a clean finish from a crash. Built rather than
    switched at the call site because the difference between the three is invisible in a one-line
    %-format, and a test that guesses which body it is running silently stops exercising it.
    """
    head = ("import json,os,sys;"
            "sys.path.insert(0, %r);"
            "import smol_ladder.or_agent as A;"
            "import smol_ladder.upstream as U;"
            % str(Path(__file__).resolve().parent.parent))
    if agent == "tools":
        return head + (
            "log=A.solve_loop(sys.argv[1], lambda c: A.run_command(c),"
            "lambda c: open('solution.py','w').write(c), %r, %d);"
            "open('turns.json','w').write(json.dumps(len(log)))" % (model, max_turns))
    if agent == "program":
        # Upstream's generation step exactly: one turn, no tools, 1024 new tokens. The extracted
        # program is written to solution.py so the offline grading pass below runs it sealed, the
        # same way it runs our agent's -- the prediction is what the program printed when re-run,
        # never the model's stdout.
        return head + (
            "M=[{'role':'system','content':U.PROGRAM_SYSTEM},"
            "{'role':'user','content':sys.argv[1]}];"
            "out=A.program_once(M, %r);"
            "open('solution.py','w').write(out['code']);"
            "open('turns.json','w').write('1')" % model)
    if agent == "bash":
        return head + (
            "M=[{'role':'system','content':U.BASH_SYSTEM},"
            "{'role':'user','content':sys.argv[1]}];"
            "log=A.bash_loop(M, lambda c: A.run_command(c),"
            "lambda: (open('answer.txt').read() if os.path.exists('answer.txt') else None),"
            "%r, %d);"
            "open('turns.json','w').write(json.dumps(len(log)))" % (model, max_turns))
    raise ValueError(f"unknown agent protocol {agent!r}")


def task_trials(row: dict, split: str, rungs: list[str], venv: Path, model: str,
                max_turns: int, retry_failed: bool = False, inputs_of=input_dir,
                runs_root: Path | None = None, samples: int = 1,
                climb: bool = True, agent: str = "tools") -> list[dict]:
    """Run the ladder for one task: `samples` trials per rung, optionally climbing.

    Rung names are given as on the command line. "L1_schema" is the filesystem-safe spelling of
    the "L1+schema" control, since "+" would need quoting in a comma-separated list.

    Climbing stops after a rung that passed on any of its samples. That is the old behaviour and
    it stays the default so a resumed sweep does not silently start running rungs it never
    intended to; `--no-climb` is what makes the per-rung denominators equal. A rung's samples are
    always all run before the climb decision is taken, so `--samples 4` is 4 real observations of
    the rung rather than 1 and 3 for whatever split the coin-flip landed on.
    """
    out = []
    root = runs_root or (DATA / "runs" / split)
    have_source = read_source(row, split) is not None
    for rung in rungs:
        prompt_rung = rung.replace("_schema", "+schema")
        # The control is built from the tables alone, so it needs no reference. Gating it on
        # one would throw away the L1-vs-L1+schema comparison on every task whose reference we
        # failed to build, which is most of the failures we care about.
        #
        # L2-L4 need one, and a task without one is skipped rather than run: prompt_for now
        # marks those rungs as adding nothing, so running them would re-measure L1 and spend a
        # trial to learn it again. The article's own table calls this case "Nothing new".
        needs_reference = prompt_rung in {"L2", "L3", "L4"} and not have_source
        if needs_reference:
            # One skipped record per sample, so the sample count of the summary still lines up
            # with the rung's budget: a rung with no reference was not attempted K times.
            out.extend({"task_id": row["task_id"], "rung": prompt_rung, "reward": 0.0,
                        "sample": k, "skipped": "no verified reference"} for k in range(samples))
            continue
        prompt = prompt_for(row, split, prompt_rung)
        # Resolved once beside the prompt it describes, so the rung's text and the record of where
        # that text came from cannot disagree: both read the same cached hint.
        src = hint_source(row, split, prompt_rung)
        rung_dir = root / row["task_id"] / rung.replace("+", "_")
        passed = False
        for k in range(samples):
            work = sample_dir(rung_dir, k)
            work.mkdir(parents=True, exist_ok=True)
            # The scratch label carries the sample so two samples of one rung, which run
            # concurrently, never share a $HOME -- once() rmtree's it on entry.
            label = rung if k == 0 else f"{rung}s{k}"
            ran: list = []
            r = once(row, prompt, work, venv, model, max_turns,
                     retry_failed, inputs_of, label,
                     {"rung": prompt_rung, "sample": k, "hint_source": src}, ran,
                     agent)
            # Write only what once() actually produced. A reused result is already on disk with
            # whatever provenance it was written with, and rewriting it here would stamp this
            # run's rung, sample index and git commit onto a trial some earlier code ran -- and
            # would modify results in the shared data tree just by resuming a sweep over them.
            if ran:
                r["rung"] = prompt_rung
                r["sample"] = k
                (work / "result.json").write_text(json.dumps(r, indent=1))
            out.append(r)
            passed |= r["reward"] >= 1.0
        if climb and passed:
            break
    return out


def _stamp() -> str:
    return datetime.now(timezone.utc).isoformat()


# The keys a launch owns: what code was running, what it was asked to do, and when it started.
# The top-level copy of each describes the RUN and the first launch wins it, so a relaunch cannot
# restamp the provenance of the trials already on disk; `launches` keeps one entry per launch.
_LAUNCH_KEYS = ("git_commit", "git_dirty", "command_line", "start_time", "model", "agent",
                "rungs", "samples", "climb", "workers", "max_turns", "limit", "split",
                "run_tag", "tasks_planned")
# What a launch reports when it ends, so these are not part of the run's lasting shape.
_COUNT_KEYS = ("tasks", "trials", "skipped", "trials_scored", "passes")


def _read_run_record(path: Path) -> dict:
    """The record as it stands, or an empty one. A corrupt file is not worth crashing a sweep over:
    the results themselves are on disk and the sweep is resumable, so the cost of losing the record
    is far lower than the cost of refusing to run."""
    if path.exists():
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_run_record(path: Path, record: dict) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=1))
    return record


def open_run_record(path: Path, header: dict) -> int:
    """Register this launch in RUN.json and return its index. Called before the first trial.

    A resumed sweep is a second launch of the same run, and the first launch's code and start time
    are the ones that produced the results already on disk -- so the top-level keys keep the first
    launch's values, and this launch is appended to `launches` with its own provenance. Written
    before the sweep rather than after, because a sweep that dies an hour in has to still say which
    code was running.
    """
    existing = _read_run_record(path)
    launches = list(existing.get("launches") or [])
    this = dict(header)
    if launches:
        # Carry the run's shape forward so a launch entry reads on its own, but never an earlier
        # launch's provenance: this is a record of what ran now.
        this = {**{k: v for k, v in header.items() if k not in _LAUNCH_KEYS},
                **{k: v for k, v in launches[0].items()
                   if k not in _COUNT_KEYS and k not in {"start_time", "command_line",
                                                         "git_commit", "git_dirty", "error",
                                                         "end_time"}},
                **{k: v for k, v in header.items() if k in _LAUNCH_KEYS}}
    launches.append(this)
    record = {k: v for k, v in existing.items() if k != "launches"}
    for key in _LAUNCH_KEYS:
        record.setdefault(key, header.get(key))
    record = {k: v for k, v in record.items() if v is not None}
    record["launches"] = launches
    record.pop("end_time", None)
    record.pop("error", None)
    _save_run_record(path, record)
    return len(launches) - 1


def close_run_record(path: Path, index: int, counts: dict, error: str | None = None) -> dict:
    """Fill in one launch's counts and end time. Always called, including on the way out of a crash.

    `end_time` is set even for a failed launch: a sweep that stopped without one cannot say when it
    stopped, and that is the first question anyone asks of a partial tree.
    """
    record = _read_run_record(path)
    launches = list(record.get("launches") or [])
    if 0 <= index < len(launches):
        launches[index].update(counts)
        if error is not None:
            launches[index]["error"] = error
    record["launches"] = launches
    record.update(counts)
    record["end_time"] = _stamp()
    if error is not None:
        record["error"] = error
    else:
        record.pop("error", None)
    return _save_run_record(path, record)


def reference_state(rows: list[dict], split: str, rungs: list[str]) -> dict:
    """Which tasks could run a rung that needs a reference, as of this launch.

    L2-L4 are gated on a verified reference, and references arrive while a sweep runs -- a retry
    sweep is building them concurrently with this one. So the set of tasks that could attempt L2 is
    not a property of the run, it is a property of when the run started, and reading it back off
    disk at summary time silently moves the denominator: a task that had no reference when L2 was
    skipped would later count as one that was never scored, and the rung's pass rate would be read
    against the wrong population.

    So the ids are recorded here, at launch, and the summary's "not climbable" bucket is pinned to
    this list rather than to whatever read_source says today.
    """
    needed = any(r.replace("_schema", "+schema") in {"L2", "L3", "L4"} for r in rungs)
    if not needed:
        return {}
    with_ref = sorted(row["task_id"] for row in rows if read_source(row, split) is not None)
    return {"reference_task_ids_at_launch": with_ref,
            "reference_tasks_at_launch": len(with_ref),
            "tasks_at_launch": len(rows)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test",
                    choices=["test", "eval", "train", "jupyter-agent", "synthetic"])
    ap.add_argument("--run-tag", default=None,
                    help="keep this run's results to themselves: trials go to "
                         "data/runs/<tag>/<split>/ and the summary to "
                         "data/runs/<tag>/summary_<split>.json. Without a tag the legacy "
                         "data/runs/<split>/ tree is used and nothing moves. Use it whenever the "
                         "ladder text has changed: two ladder versions in one tree are pooled or "
                         "worse.")
    ap.add_argument("--rungs", default="L1", help="comma-separated, e.g. L1,L1+schema,L2,L3,L4")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--model", default="stealth/space-bunny-alpha")
    ap.add_argument("--agent", default="tools", choices=["tools", "program", "bash"],
                    help="which protocol to run. 'program' is upstream's GRPO/eval_pass1 "
                         "protocol (one turn, no tools); 'bash' is the SmolDataEnvs-sft bash "
                         "agent; 'tools' is ours. A released 2B model must be run under its own "
                         "protocol or the number is about the protocol, not the model.")
    ap.add_argument("--max-turns", type=int, default=40)
    ap.add_argument("--retry-failed", action="store_true",
                    help="re-run trials whose agent crashed; a clean pass is never re-rolled")
    ap.add_argument("--samples", type=int, default=1,
                    help="independent trials per (task, rung); sample 0 is the existing "
                         "<task>/<rung>/result.json, sample k lands in <task>/<rung>/s<k>/")
    ap.add_argument("--no-climb", dest="climb", action="store_false",
                    help="run every requested rung on every task, so each rung's pass rate has "
                         "the same denominator. Rungs that need a reference are still skipped "
                         "for tasks without one.")
    ap.set_defaults(climb=True)
    args = ap.parse_args()
    if args.samples < 1:
        ap.error("--samples must be at least 1")

    rows, inputs_of = source_for(args.split)
    rows = rows[: args.limit]
    rungs = args.rungs.split(",")
    venv = Path(sys.prefix)
    try:
        root = runs_dir(args.split, args.run_tag)
    except ValueError as e:
        ap.error(str(e))
    root.mkdir(parents=True, exist_ok=True)
    # RUN.json is written before the first trial, not after: a sweep that dies an hour in has to
    # still say which code was running and what it was asked to do. It is only written for a
    # tagged run -- the legacy tree is shared by every run that ever used it, so one launch's
    # provenance stamped there would be a lie about who owns the directory.
    run_record = (root.parent / "RUN.json") if args.run_tag else None
    header = {
        "run_tag": args.run_tag, "split": args.split, "model": args.model, "agent": args.agent,
        "rungs": rungs, "samples": args.samples, "climb": bool(args.climb),
        "workers": args.workers, "max_turns": args.max_turns, "limit": args.limit,
        "tasks_planned": len(rows),
        "command_line": [sys.executable, "-m", "smol_ladder.run_ladder", *sys.argv[1:]],
        "start_time": _stamp(), **git_provenance(),
        **reference_state(rows, args.split, rungs),
    }
    if run_record is not None:
        launch = open_run_record(run_record, header)
    print(f"{len(rows)} tasks x {len(rungs)} rungs x {args.samples} samples"
          f"{'' if args.climb else ', no climb'}; code {git_provenance()['git_commit'][:8]}"
          f"; results under {root}")
    done = 0
    counts = {"tasks": 0, "trials": 0, "skipped": 0, "trials_scored": 0, "passes": 0}
    try:
        with ThreadPoolExecutor(args.workers) as pool:
            futures = [pool.submit(task_trials, row, args.split, rungs, venv, args.model,
                                   args.max_turns, args.retry_failed, inputs_of,
                                   root, args.samples, args.climb, args.agent) for row in rows]
            for f in as_completed(futures):
                done += 1
                for r in f.result():
                    if r.get("skipped"):
                        # Not attempted, so not a trial: a rung the task could not run is in
                        # neither the numerator nor the denominator of anything measured here.
                        counts["skipped"] += 1
                        continue
                    counts["trials"] += 1
                    counts["trials_scored"] += r.get("agent_status") == "exit 0"
                    counts["passes"] += r.get("reward", 0.0) >= 1.0
                    print(f"[{done}/{len(rows)}] {r['task_id']} {r['rung']} "
                          f"s{r.get('sample', 0)} "
                          f"reward={r['reward']} pred={r.get('prediction','')[:40]!r} "
                          f"{r.get('agent_status','')}", flush=True)
    except BaseException as e:  # noqa: BLE001 - the record is what has to survive the crash
        if run_record is not None:
            close_run_record(run_record, launch, {**counts, "tasks": done},
                             error=f"{type(e).__name__}: {e}")
        raise
    if run_record is not None:
        close_run_record(run_record, launch, {**counts, "tasks": done})
    print(f"done: {done} tasks x {len(rungs)} rungs x {args.samples} samples")


if __name__ == "__main__":
    main()
