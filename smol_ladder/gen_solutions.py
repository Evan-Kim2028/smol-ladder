"""Generate reference solutions for held-out tasks with a headless `cmd` agent.

Each task gets data/solutions/<split>/<task_id>/ holding the agent's transcript,
its solution.py, and result.json. solution.py is re-run offline in bubblewrap and
graded against the gold answer, so a kept solution is one we reproduced, not one
the agent claimed. Tasks with a result.json are skipped: the run is resumable.

These solutions feed ladder hints only. Never train on them: they come from
the held-out splits.

    uv run python -m smol_ladder.gen_solutions --split test --limit 5
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from smol_ladder.grade import grade, last_line
from smol_ladder.sandbox import run_script
from smol_ladder.tasks import DATA, input_dir, load_split

MODEL = "stealth/space-bunny-alpha"
# The agent's `python3` is this venv, which has pandas and friends.
AGENT_ENV = {**os.environ, "PATH": f"{Path(sys.executable).parent}:{os.environ['PATH']}"}

PROMPT = """You are solving a data-analysis question. The input tables are in ./input (read-only).

Question: {question}

Files:
{files}

Explore the data with Python as much as you need. Then write ./solution.py: a self-contained
script that reads only from ./input, computes the answer, and prints the final answer as its
LAST line of output. The final answer is just the value: a number (no commas or units),
a short label, yes/no, or a comma-separated list. Run `python3 solution.py` to check it works.
Do not look the answer up online or in any dataset; compute it from the files."""

# Signs the agent went looking for the gold answer instead of computing it.
API_DOWN = "Unable to connect to the API"
API_RETRIES = 5

LEAK_RE = re.compile(r"SmolDataEnvs|FineEnvs|huggingface\.co/datasets|hf_hub_download", re.I)


def jail(work: Path, inputs: Path) -> list[str]:
    """Hide $HOME from the agent. It sees the toolchain and venv (read-only), its task's
    tables (read-only), cmd's state dir, and its own task folder. Nothing else: not this
    repo, not other tasks' solutions, not the HF cache that holds the gold answers.
    Network stays on because the model is remote."""
    home = Path.home()
    tmp = work / ".tmp"
    tmp.mkdir(exist_ok=True)
    ro = [home / ".nvm", home / ".local", Path(sys.prefix), inputs]
    args = ["bwrap", "--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc",
            "--tmpfs", str(home), "--bind", str(tmp), "/tmp"]
    for d in ro:
        args += ["--ro-bind", str(d), str(d)]
    args += ["--bind", str(home / ".commandcode"), str(home / ".commandcode"),
             "--bind", str(work), str(work), "--chdir", str(work), "--die-with-parent"]
    return args


def solve(row: dict, split: str, model: str, timeout: int) -> dict:
    work = DATA / "solutions" / split / row["task_id"]
    result_path = work / "result.json"
    if result_path.exists():
        return json.loads(result_path.read_text())
    if work.exists():
        shutil.rmtree(work)  # half-finished attempt from a crash
    work.mkdir(parents=True)
    inputs = input_dir(row)
    (work / "input").symlink_to(inputs)

    prompt = PROMPT.format(
        question=row["question"], files="\n".join(f"- {f}" for f in row["files"])
    )
    t0 = time.time()
    for attempt in range(API_RETRIES):
        try:
            p = subprocess.run(
                jail(work, inputs) + ["cmd", "-p", prompt, "-m", model, "--yolo", "-t", "--skip-onboarding",
                 "--no-session", "--max-turns", "40", "--output-format", "json"],
                cwd=work, env=AGENT_ENV, capture_output=True, text=True, timeout=timeout,
            )
            transcript, agent_status = p.stdout, f"exit {p.returncode}"
        except subprocess.TimeoutExpired as e:
            transcript = e.stdout.decode(errors="replace") if isinstance(e.stdout, bytes) else e.stdout or ""
            agent_status = "timeout"
        if API_DOWN not in transcript:
            break
        # The endpoint refused us; that says nothing about the task. Back off and retry.
        time.sleep(30 * 2**attempt)
    else:
        raise RuntimeError(f"{row['task_id']}: API unreachable after {API_RETRIES} attempts")
    (work / "transcript.jsonl").write_text(transcript)

    result = {
        "task_id": row["task_id"], "split": split, "model": model,
        "difficulty_tier": row["difficulty_tier"], "agent_status": agent_status,
        "agent_seconds": round(time.time() - t0, 1),
        "suspected_lookup": bool(LEAK_RE.search(transcript)),
        "prediction": "", "reward": 0.0, "verify": "no solution.py",
    }
    solution = work / "solution.py"
    if solution.exists():
        verify = work / "verify"
        verify.mkdir(exist_ok=True)
        shutil.copy(solution, verify / "solution.py")
        run = run_script(verify / "solution.py", work / "input")
        result["prediction"] = last_line(run.stdout)
        result["reward"] = grade(row, result["prediction"])
        result["verify"] = "timeout" if run.timed_out else f"exit {run.returncode}"
        (work / "verify_stdout.txt").write_text(run.stdout + "\n--- stderr ---\n" + run.stderr)
    result_path.write_text(json.dumps(result, indent=1))
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["test", "eval", "train"])
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--timeout", type=int, default=900, help="seconds per agent run")
    args = ap.parse_args()

    rows = load_split(args.split)[: args.limit]
    done = passed = 0
    with ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(solve, row, args.split, args.model, args.timeout) for row in rows]
        for f in as_completed(futures):
            done += 1
            try:
                r = f.result()
            except RuntimeError as e:  # no result.json written; the next run retries it
                print(f"[{done}/{len(rows)}] SKIPPED {e}", flush=True)
                continue
            passed += r["reward"] >= 1.0
            flag = " LOOKUP?" if r["suspected_lookup"] else ""
            print(f"[{done}/{len(rows)}] {r['task_id']} reward={r['reward']} "
                  f"pred={r['prediction'][:40]!r} {r['verify']}{flag}  pass={passed}", flush=True)


if __name__ == "__main__":
    main()
