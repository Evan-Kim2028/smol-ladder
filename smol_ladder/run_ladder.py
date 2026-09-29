"""Run the ladder: every rung's prompt through the solver, graded against gold.

The solver runs in our process and executes the agent's shell commands in a bubblewrap jail
holding only the task's tables, so a trial costs no container and no per-trial install.

Pass-rate curve by rung is the headline; "lowest rung that passes" is derived from it. Climbing
stops at the first pass, so a task that passes at L1 costs one trial, not four.

    uv run python -m smol_ladder.run_ladder --split test --rungs L1 --workers 20
    uv run python -m smol_ladder.run_ladder --split test --workers 20
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from smol_ladder.grade import grade
from smol_ladder.ladder import prompt_for, read_source
from smol_ladder.tasks import DATA, input_dir, load_split

JAIL_RO = ["/usr", "/bin", "/lib", "/lib64", "/etc", "/opt"]


def jail(work: Path, inputs: Path, venv: Path) -> list[str]:
    """The task's tables, the toolchain, and the solver package. Nothing else.

    $HOME is a tmpfs, so the HF cache (which holds the gold answers), the repo's data/ dir and
    every sibling task's solution are gone. Only the solver package is bound back in, because
    the model loop has to be importable, and it contains no answers.

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
    ro = {venv, inputs, package, interpreter.parent, interpreter.parent.parent,
          Path(sys.prefix), Path(sys.base_prefix), Path(sys.base_prefix) / "lib"}
    # Order matters: the catch-all read-only bind of / must come first, or it shadows the
    # /dev and /proc mounts below and CPython cannot read urandom to seed its hash randomiser.
    args = ["bwrap", "--ro-bind", "/", "/", "--tmpfs", str(Path.home())]
    for d in sorted(ro, key=str):
        if d.exists():
            args += ["--ro-bind", str(d), str(d)]
    args += ["--dev", "/dev", "--proc", "/proc", "--unshare-pid", "--tmpfs", "/tmp",
             "--bind", str(work), str(work),
             "--chdir", str(work), "--die-with-parent"]
    return args


def once(row: dict, prompt: str, work: Path, venv: Path, model: str, max_turns: int,
         retry_failed: bool = False) -> dict:
    """One attempt at one rung: run the solver in the jail, then grade its solution offline.

    Resumable: a result.json from a clean run is reused. A crashed trial is only retried when
    asked, so a rerun does not quietly re-roll a genuinely failed task.
    """
    cached = work / "result.json"
    if cached.exists():
        prior = json.loads(cached.read_text())
        if prior.get("agent_status") == "exit 0" or not retry_failed:
            return prior
    work.mkdir(parents=True, exist_ok=True)
    inp = work / "input"
    if not inp.exists():
        inp.symlink_to(input_dir(row).resolve())
    env = {
        "PATH": f"{Path(sys.executable).parent}:/usr/local/bin:/usr/bin:/bin",
        "HOME": str(work),
        "LANG": "C.UTF-8",
        "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1",
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
        proc = subprocess.run(
            jail(work, input_dir(row), venv) + [sys.executable, "-c", script, prompt],
            cwd=work, env=env, capture_output=True, timeout=max_turns * 60 + 300)
        agent_status = f"exit {proc.returncode}"
        stderr = proc.stderr.decode("utf-8", "replace")[-2000:] \
            if isinstance(proc.stderr, bytes) else (proc.stderr or "")[-2000:]
    except subprocess.TimeoutExpired:
        agent_status, stderr = "timeout", ""
    result = {"task_id": row["task_id"], "model": model, "agent_status": agent_status,
              "agent_seconds": round(time.time() - t0, 1), "prediction": "", "reward": 0.0}
    solution = work / "solution.py"
    if solution.exists():
        # Grade the agent's own last printed line from a clean run, like the Harbor verifier.
        verify = work / "verify"
        verify.mkdir(exist_ok=True)
        (verify / "solution.py").write_text(solution.read_text())
        # A verification run that hangs is a failed trial, not a crashed harness: the agent
        # wrote a program that never terminates offline. Catch it or the whole run dies.
        try:
            run = subprocess.run(
                ["nice", "-n", "15", "bwrap", "--ro-bind", "/", "/", "--dev", "/dev",
                 "--proc", "/proc", "--unshare-net", "--unshare-pid", "--tmpfs", "/tmp",
                 "--bind", str(verify), "/tmp/work",
                 "--ro-bind", str((work / "input").resolve()), "/tmp/work/input",
                 "--chdir", "/tmp/work", "--die-with-parent",
                 "--setenv", "OMP_NUM_THREADS", "1", "--setenv", "OPENBLAS_NUM_THREADS", "1",
                 sys.executable, "solution.py"],
                capture_output=True, timeout=180)
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


def task_trials(row: dict, split: str, rungs: list[str], venv: Path, model: str,
                max_turns: int, retry_failed: bool = False) -> list[dict]:
    """Climb the ladder for one task: stop at the first rung that passes.

    Rung names are given as on the command line. "L1_schema" is the filesystem-safe spelling of
    the "L1+schema" control, since "+" would need quoting in a comma-separated list.
    """
    out = []
    have_source = read_source(row, split) is not None
    for rung in rungs:
        prompt_rung = rung.replace("_schema", "+schema")
        if prompt_rung != "L1" and not have_source:
            out.append({"task_id": row["task_id"], "rung": prompt_rung, "reward": 0.0,
                        "skipped": "no verified reference"})
            continue
        work = DATA / "runs" / split / row["task_id"] / rung.replace("+", "_")
        r = once(row, prompt_for(row, split, prompt_rung), work, venv, model, max_turns,
                 retry_failed)
        r["rung"] = prompt_rung
        (work / "result.json").write_text(json.dumps(r, indent=1))
        out.append(r)
        if r["reward"] >= 1.0:
            break
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["test", "eval", "train"])
    ap.add_argument("--rungs", default="L1", help="comma-separated, e.g. L1,L1+schema,L2,L3,L4")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--model", default="stealth/space-bunny-alpha")
    ap.add_argument("--max-turns", type=int, default=40)
    ap.add_argument("--retry-failed", action="store_true",
                    help="re-run trials whose agent crashed; a clean pass is never re-rolled")
    args = ap.parse_args()

    rows = load_split(args.split)[: args.limit]
    rungs = args.rungs.split(",")
    venv = Path(sys.prefix)
    (DATA / "runs" / args.split).mkdir(parents=True, exist_ok=True)
    done = 0
    with ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(task_trials, row, args.split, rungs, venv, args.model,
                               args.max_turns, args.retry_failed) for row in rows]
        for f in as_completed(futures):
            done += 1
            for r in f.result():
                print(f"[{done}/{len(rows)}] {r['task_id']} {r['rung']} "
                      f"reward={r['reward']} pred={r.get('prediction','')[:40]!r} "
                      f"{r.get('agent_status','')}", flush=True)
    print(f"done: {done} tasks x {len(rungs)} rungs")


if __name__ == "__main__":
    main()
