"""Generate reference solutions for a split, with the OpenRouter solver.

L2-L4 are built from a solution that reproduced the gold answer, so a split with no references
has no rungs to measure. This is the same loop run_ladder uses, with one difference: a trial
that passes is *kept* as the reference rather than merely scored.

It supersedes the original gen_solutions.py, which drove Command Code inside the jail. Same
guarantees: the tables and the toolchain are bound read-only, $HOME is a tmpfs so the HF cache
and every sibling task's solution are gone, and the kept solution is re-run offline and graded,
so what we keep is what we reproduced.

Retrying. An attempt is kept or dropped on one number, the reward of the offline grading pass.
`agent_status` says only that the agent process finished: a solution that runs cleanly and
prints the wrong answer is exactly the failure we most want a second opinion on, and treating a
clean exit as final left 69 test and 49 eval tasks permanently stuck -- a rerun printed 118
cached lines and produced nothing. --attempts N therefore gives every task without a verified
reference up to N fresh tries, and each one lands in its own directory:

    data/solutions/<split>/<task_id>/
        result.json, solution.py      the legacy slot gen_solutions.py wrote, read as attempt 0,
                                      and where a verified attempt is promoted
        attempt_<i>/                  one directory per fresh attempt, never overwritten
            result.json               attempt index, timestamp, git commit, model, and for a
                                      failure the prediction next to the gold it missed

The attempt directories are evidence rather than scratch: when three tries disagree, which
answer each of them gave is the evidence for whether the gold is wrong, and overwriting it to
save a few kilobytes would throw away the only record we have. So promotion *copies* the
winner up to the task directory, where ladder.read_source looks for it, and leaves the attempt
where it was.

    uv run python -m smol_ladder.gen_refs --split jupyter-agent --workers 30
    uv run python -m smol_ladder.gen_refs --split test --attempts 3 --workers 8
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

from smol_ladder.grade import grade, last_line
from smol_ladder.run_ladder import jail, once, source_for
from smol_ladder.tasks import DATA

ROOT = Path(__file__).resolve().parent.parent


class RateLimited(Exception):
    """OpenRouter refused the call for quota reasons, so the task was never really attempted."""


def git_commit() -> str:
    """The commit a reference was produced from, so a result can be traced to the code that
    made it. Empty rather than fatal when git is unavailable: a missing commit is not a reason
    to lose the reference."""
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:  # noqa: BLE001 - provenance is best-effort, the reference is not
        return ""


def attempt_dir(task: Path, index: int) -> Path:
    """Where attempt `index` runs. Zero is the flat legacy slot, so index N is `attempt_N`."""
    return task if index == 0 else task / f"attempt_{index}"


def attempt_indices(task: Path) -> list[int]:
    """Every attempt already on disk, legacy slot included.

    The flat directory counts as attempt 0 whether or not it holds a result: gen_solutions.py
    wrote one, and a half-finished attempt left by a crash still occupies the number, so
    counting it is what keeps a retry from landing on top of it.
    """
    if not task.exists():
        return []
    found = []
    if (task / "result.json").exists():
        found.append(0)
    found += [int(p.name.split("_")[1]) for p in task.glob("attempt_*")
              if p.name.split("_")[1].isdigit() and p.is_dir()]
    return sorted(set(found))


def verified(task: Path) -> dict | None:
    """The result of a reference we already have, or None.

    reward >= 1.0 is the whole test. agent_status is deliberately not consulted: "exit 0" is
    the status of a task the model got wrong as often as one it got right, and gating on it is
    what made the cached failures unreachable.
    """
    result = task / "result.json"
    if not result.exists():
        return None
    try:
        prior = json.loads(result.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if prior.get("reward", 0.0) < 1.0:
        return None
    if not (task / "solution.py").exists():
        # A verified reward with no script to show for it is not a reference: read_source would
        # find the result and no solution, and L2-L4 would silently measure nothing.
        return None
    return {**prior, "keep": True, "attempts_made": 0}


def throttled(result: dict) -> bool:
    """Whether the attempt died on a rate limit rather than on the task.

    The solver runs in a subprocess, so a 429 surfaces as a failed agent run with the refusal in
    its stderr. Counting that against the attempt budget would spend the budget on quota and
    then report the task as unresolvable when the model never got to answer it.
    """
    status = str(result.get("agent_status", ""))
    stderr = str(result.get("stderr", "")).lower()
    if "429" in status:
        return True
    return any(marker in stderr for marker in
               ("429", "rate limit", "too many requests", "quota"))


def record(work: Path, result: dict, row: dict, index: int, model: str, split: str,
           started_at: float) -> dict:
    """Write one attempt's result.json, with everything needed to read it later."""
    result = dict(result)
    result.update({"task_id": row["task_id"], "split": split, "attempt": index, "model": model,
                   "started_at": round(started_at, 1), "git_commit": git_commit(),
                   "gold": row.get("answer", ""), "keep": result.get("reward", 0.0) >= 1.0,
                   "reward": result.get("reward", 0.0)})
    work.mkdir(parents=True, exist_ok=True)
    (work / "result.json").write_text(json.dumps(result, indent=1))
    return result


def promote(task: Path, work: Path, index: int) -> None:
    """Copy a verifying attempt up to the slot read_source reads.

    read_source looks for solutions/<split>/<task_id>/result.json next to solution.py, so that
    is where a reference has to be. The attempt directory stays: it is the record of the run
    that earned it.

    The transcript is promoted with it, and this is the whole reason the reference slot matters
    twice over. `verified()` reads this directory, so a reference that reached the slot without
    its conversation would be counted as a reference with no trajectory beside it -- the exact
    state this file's path is built to avoid. An exporter that walks
    `data/solutions/<split>/<task_id>/` for training data would then find the program, the answer
    and no way it was reached, and fall back to inventing the exploration turns.
    """
    result = json.loads((work / "result.json").read_text())
    result["reference_attempt"] = index
    task.mkdir(parents=True, exist_ok=True)
    (task / "result.json").write_text(json.dumps(result, indent=1))
    for name in ("solution.py", "transcript.json"):
        if (work / name).exists():
            (task / name).write_text((work / name).read_text(errors="replace"))


def attempt(row: dict, split: str, model: str, inputs_of, max_turns: int, work: Path,
            save_transcript: bool = True) -> dict:
    """One attempt at a reference solution, run and graded (never raises).

    `save_transcript` defaults on and is passed through explicitly rather than left to once()'s
    own default: gen_refs is the path that builds references, so a reference is the artifact an
    SFT exporter reads, and this call site is the one place that fact has to be visible. It is
    still the default in once() too, because the ladder sweep wants it as well -- but a caller
    that reads as if it does not ask for a transcript is exactly the bug this module had.
    """
    prompt = PROMPT.format(
        question=row["question"], files="\n".join(f"- {f}" for f in row["files"]))
    # once() runs the solver in the jail, then re-runs the written solution offline and grades
    # it. reward == 1.0 therefore means the solution reproduced the gold answer, which is the
    # only case we keep: a reference is something we verified, not something the agent claimed.
    return once(row, prompt, work, Path(sys.prefix), model, max_turns,
                retry_failed=True, inputs_of=inputs_of, save_transcript=save_transcript)


def reference(row: dict, split: str, model: str, inputs_of, max_turns: int, attempts: int = 1,
              max_throttles: int = 3, save_transcript: bool = True) -> dict:
    """Attempt a reference solution for one task, keeping every attempt.

    Returns the result dict (never raises). A task that already has a verified reference is
    returned untouched -- it is never re-rolled, whatever the budget -- and a task that runs out
    of attempts returns its last one, so the caller sees why it has no reference.
    """
    task = DATA / "solutions" / split / row["task_id"]
    prior = verified(task)
    if prior is not None:
        return prior

    indices = attempt_indices(task)
    index = (max(indices) + 1) if indices else 0
    if index == 0:  # a fresh task has no flat slot yet; attempt 0 is the flat one
        index = 1
    made = throttles = 0
    result = {"task_id": row["task_id"], "reward": 0.0, "keep": False, "prediction": "",
              "agent_status": "throttled"}
    while True:
        work = attempt_dir(task, index)
        started = time.time()
        result = attempt(row, split, model, inputs_of, max_turns, work, save_transcript)
        result["reference_seconds"] = round(time.time() - started, 1)
        if throttled(result):
            # Quota, not an answer. Wait it out and run the same attempt index again rather
            # than spending one of the N tries on a call the model never got to make. Once the
            # backoff budget is spent the task is reported as throttled and left for a later
            # sweep: a 429 is not evidence that the task is unsolvable, and booking it as one
            # would hide it from the next run behind a recorded failure.
            if throttles < max_throttles:
                throttles += 1
                time.sleep(min(300, 20 * 2**throttles))
                continue
            result = {"task_id": row["task_id"], "reward": 0.0, "keep": False, "prediction": "",
                      "agent_status": "throttled"}
            break
        result = record(work, result, row, index, model, split, started)
        made += 1
        if result["keep"]:
            promote(task, work, index)
            break
        index += 1
        if made >= attempts:
            break
    result = dict(result)
    result["attempts_made"] = made
    result["attempts_throttled"] = throttles
    return result


def run_sweep(rows: list[dict], split: str, model: str, inputs_of, max_turns: int,
              attempts: int, workers: int = 20, save_transcript: bool = True) -> list[dict]:
    """Retry every task that lacks a verified reference, and skip the ones that have one.

    Skipping is the point: running the budget over all 250 test tasks would spend 250 attempts
    to rediscover the 181 answers already on disk, and the model is free but the wall clock is
    not.
    """
    todo = []
    out = []
    for row in rows:
        prior = verified(DATA / "solutions" / split / row["task_id"])
        if prior is not None:
            out.append({**prior, "task_id": row["task_id"], "attempts_made": 0})
        else:
            todo.append(row)
    if not todo:
        return out
    (DATA / "solutions" / split).mkdir(parents=True, exist_ok=True)
    kept = 0
    with ThreadPoolExecutor(workers) as pool:
        futures = [pool.submit(reference, row, split, model, inputs_of, max_turns, attempts,
                               max_throttles=3, save_transcript=save_transcript)
                   for row in todo]
        for done, f in enumerate(as_completed(futures), 1):
            try:
                r = f.result()
            except Exception as e:  # noqa: BLE001 - one bad task must not stop the sweep
                print(f"[{done}/{len(todo)}] ERROR {type(e).__name__}: {str(e)[:120]}",
                      flush=True)
                continue
            kept += bool(r.get("keep"))
            if done % 10 == 0 or not r["keep"]:
                print(f"[{done}/{len(todo)}] {r['task_id']} reward={r['reward']} "
                      f"pred={r.get('prediction','')[:32]!r} "
                      f"tries={r.get('attempts_made')} kept={kept}", flush=True)
            out.append(r)
    print(f"done: {len(todo)} retried, {kept} verified references "
          f"({kept/max(len(todo),1):.0%})", flush=True)
    return out


PROMPT = """You are solving a data-analysis question. The input tables are in ./input (read-only).

Question: {question}

Files:
{files}

Explore the data with Python as much as you need. Then write ./solution.py: a self-contained
script that reads only from ./input, computes the answer, and prints the final answer as its
LAST line of output. The final answer is just the value: a number (no commas or units),
a short label, yes/no, or a comma-separated list. Run `python3 solution.py` to check it works.
Do not look the answer up online or in any dataset; compute it from the files."""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="jupyter-agent")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--model", default="stealth/space-bunny-alpha")
    ap.add_argument("--max-turns", type=int, default=40)
    ap.add_argument("--attempts", type=int, default=1,
                    help="fresh attempts per task without a verified reference; each one is "
                         "kept in its own directory and none is ever overwritten")
    ap.add_argument("--no-transcript", dest="transcript", action="store_false",
                    help="do not keep each attempt's assistant/tool conversation. On by default: "
                         "this is the path that builds references, and a verified reference whose "
                         "trajectory was thrown away is a program and an answer (train/traces.py).")
    ap.set_defaults(transcript=True)
    args = ap.parse_args()

    rows, inputs_of = source_for(args.split)
    rows = rows[: args.limit]
    (DATA / "solutions" / args.split).mkdir(parents=True, exist_ok=True)
    skipped = sum(1 for row in rows if verified(DATA / "solutions" / args.split / row["task_id"]))
    print(f"{len(rows)} tasks, {len(rows) - skipped} without a verified reference, "
          f"up to {args.attempts} attempt(s) each", flush=True)
    out = run_sweep(rows, args.split, args.model, inputs_of, args.max_turns, args.attempts,
                    args.workers, save_transcript=args.transcript)
    kept = sum(1 for r in out if r.get("keep"))
    print(f"done: {len(rows)} tasks, {kept} verified references")


if __name__ == "__main__":
    main()