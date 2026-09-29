"""Generate reference solutions for a split, with the OpenRouter solver.

L2-L4 are built from a solution that reproduced the gold answer, so a split with no references
has no rungs to measure. This is the same loop run_ladder uses, with one difference: a trial
that passes is *kept* as the reference rather than merely scored.

It supersedes the original gen_solutions.py, which drove Command Code inside the jail. Same
guarantees: the tables and the toolchain are bound read-only, $HOME is a tmpfs so the HF cache
and every sibling task's solution are gone, and the kept solution is re-run offline and graded,
so what we keep is what we reproduced.

    uv run python -m smol_ladder.gen_refs --split jupyter-agent --workers 30
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from smol_ladder.grade import grade, last_line
from smol_ladder.run_ladder import jail, once, source_for
from smol_ladder.tasks import DATA


def reference(row: dict, split: str, model: str, inputs_of, max_turns: int) -> dict:
    """One attempt at a reference solution. Returns the result dict (never raises)."""
    work = DATA / "solutions" / split / row["task_id"]
    cached = work / "result.json"
    if cached.exists():
        prior = json.loads(cached.read_text())
        if prior.get("agent_status") == "exit 0":
            return prior

    prompt = PROMPT.format(
        question=row["question"], files="\n".join(f"- {f}" for f in row["files"]))
    t0 = time.time()
    # once() runs the solver in the jail, then re-runs the written solution offline and grades
    # it. reward == 1.0 therefore means the solution reproduced the gold answer, which is the
    # only case we keep: a reference is something we verified, not something the agent claimed.
    result = once(row, prompt, work, Path(sys.prefix), model, max_turns,
                  retry_failed=False, inputs_of=inputs_of)
    result["split"] = split
    result["reference_seconds"] = round(time.time() - t0, 1)
    result["keep"] = result["reward"] >= 1.0
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_text(json.dumps(result, indent=1))
    return result


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
    args = ap.parse_args()

    rows, inputs_of = source_for(args.split)
    rows = rows[: args.limit]
    (DATA / "solutions" / args.split).mkdir(parents=True, exist_ok=True)
    done = kept = 0
    with ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(reference, row, args.split, args.model, inputs_of, args.max_turns)
                   for row in rows]
        for f in as_completed(futures):
            done += 1
            try:
                r = f.result()
            except Exception as e:  # noqa: BLE001 - one bad task must not stop the sweep
                print(f"[{done}/{len(rows)}] ERROR {type(e).__name__}: {str(e)[:120]}", flush=True)
                continue
            kept += r["keep"]
            if done % 10 == 0 or not r["keep"]:
                print(f"[{done}/{len(rows)}] reward={r['reward']} "
                      f"pred={r.get('prediction','')[:32]!r} kept={kept}", flush=True)
    print(f"done: {done} tasks, {kept} verified references")


if __name__ == "__main__":
    main()
