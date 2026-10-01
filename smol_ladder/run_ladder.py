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
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from smol_ladder.grade import grade
from smol_ladder.ladder import prompt_for, read_source
from smol_ladder.tasks import DATA, input_dir, load_split

JAIL_RO = ["/usr", "/bin", "/lib", "/lib64", "/etc", "/opt"]
# A trial is one model loop of at most 40 turns, each shell command capped at 150s. 20 minutes
# is generous for that; the old 45-minute cap let one stuck trial hold a worker for three
# quarters of an hour, and with sixteen workers the sweep crawled.
AGENT_TIMEOUT = 1200


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
         provenance: dict | None = None, was_run: list | None = None) -> dict:
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
        "OPENROUTER_API_KEY": os.environ.get("OPENROUTER_API_KEY", ""),
    }
    script = (
        "import json,sys;"
        "sys.path.insert(0, %r);"
        "import smol_ladder.or_agent as A;"
        "log=A.solve_loop(sys.argv[1], lambda c: A.run_command(c),"
        "lambda c: open('solution.py','w').write(c), %r, %d);"
        "open('turns.json','w').write(json.dumps(len(log)))"
        % (str(Path(__file__).resolve().parent.parent), model, max_turns)
    )
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
    shutil.rmtree(trial_scratch, ignore_errors=True)
    result = {"task_id": row["task_id"], "model": model, "agent_status": agent_status,
              "agent_seconds": round(time.time() - t0, 1), "prediction": "", "reward": 0.0}
    result.update(git_provenance())
    result["prompt_sha256"] = prompt_sha256(prompt)
    result["timestamp"] = datetime.now(timezone.utc).isoformat()
    # rung and sample default to what the caller passed anyway: gen_refs drives once() directly
    # with a rung label and no sample axis, and a reference has rung "reference", sample 0.
    result["rung"] = (provenance or {}).get("rung", rung_label)
    result["sample"] = (provenance or {}).get("sample", 0)
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
        # A verification run that hangs is a failed trial, not a crashed harness: the agent
        # wrote a program that never terminates offline. Catch it or the whole run dies.
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
                         "OMP_NUM_THREADS": "1", "MPLBACKEND": "Agg"}, 180)
            out = run.stdout.decode("utf-8", "replace") if isinstance(run.stdout, bytes) \
                else (run.stdout or "")
        except subprocess.TimeoutExpired:
            out = ""
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


def task_trials(row: dict, split: str, rungs: list[str], venv: Path, model: str,
                max_turns: int, retry_failed: bool = False, inputs_of=input_dir,
                runs_root: Path | None = None, samples: int = 1,
                climb: bool = True) -> list[dict]:
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
                     retry_failed, inputs_of, label, {"rung": prompt_rung, "sample": k}, ran)
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test",
                    choices=["test", "eval", "train", "jupyter-agent", "synthetic"])
    ap.add_argument("--rungs", default="L1", help="comma-separated, e.g. L1,L1+schema,L2,L3,L4")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--model", default="stealth/space-bunny-alpha")
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
    (DATA / "runs" / args.split).mkdir(parents=True, exist_ok=True)
    print(f"{len(rows)} tasks x {len(rungs)} rungs x {args.samples} samples"
          f"{'' if args.climb else ', no climb'}; code {git_provenance()['git_commit'][:8]}")
    done = 0
    with ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(task_trials, row, args.split, rungs, venv, args.model,
                               args.max_turns, args.retry_failed, inputs_of,
                               None, args.samples, args.climb) for row in rows]
        for f in as_completed(futures):
            done += 1
            for r in f.result():
                if r.get("skipped"):
                    continue
                print(f"[{done}/{len(rows)}] {r['task_id']} {r['rung']} "
                      f"s{r.get('sample', 0)} "
                      f"reward={r['reward']} pred={r.get('prediction','')[:40]!r} "
                      f"{r.get('agent_status','')}", flush=True)
    print(f"done: {done} tasks x {len(rungs)} rungs x {args.samples} samples")


if __name__ == "__main__":
    main()
