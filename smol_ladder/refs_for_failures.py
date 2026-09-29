"""Reference solutions for the tasks that failed L1.

A rung is only worth measuring on a task the model cannot already solve, so the references
that matter for the ladder are the ones for failures. This collects those task ids from a
split's L1 results and hands them to gen_refs, skipping the ones already solved.

    uv run python -m smol_ladder.refs_for_failures --split jupyter-agent --workers 30
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from smol_ladder.gen_refs import PROMPT, reference
from smol_ladder.run_ladder import source_for
from smol_ladder.tasks import DATA


def failed_task_ids(split: str) -> list[str]:
    """Task ids whose L1 trial did not pass. The id is the grandparent: <task>/<rung>/result."""
    out = []
    for path in (DATA / "runs" / split).glob("*/*/result.json"):
        if path.parent.name != "L1":
            continue
        if json.loads(path.read_text())["reward"] < 1.0:
            out.append(path.parent.parent.name)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="jupyter-agent")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--model", default="stealth/space-bunny-alpha")
    ap.add_argument("--max-turns", type=int, default=40)
    ap.add_argument("--all", action="store_true",
                    help="build references for every task, not only the L1 failures")
    args = ap.parse_args()

    rows, inputs_of = source_for(args.split)
    by_id = {r["task_id"]: r for r in rows}
    if args.all:
        targets = list(by_id)
    else:
        targets = [t for t in failed_task_ids(args.split) if t in by_id]
    targets = targets[: args.limit] if args.limit else targets
    print(f"building references for {len(targets)} tasks with {args.workers} workers", flush=True)

    (DATA / "solutions" / args.split).mkdir(parents=True, exist_ok=True)
    done = kept = 0
    start = time.time()
    with ThreadPoolExecutor(args.workers) as pool:
        futures = {pool.submit(reference, by_id[t], args.split, args.model, inputs_of,
                               args.max_turns): t for t in targets}
        for f in as_completed(futures):
            done += 1
            try:
                r = f.result()
            except Exception as e:  # noqa: BLE001 - one bad task must not stop the sweep
                print(f"[{done}/{len(targets)}] ERROR {type(e).__name__}: {str(e)[:100]}",
                      flush=True)
                continue
            kept += bool(r.get("keep"))
            if done % 10 == 0:
                rate = done / max(time.time() - start, 1)
                print(f"[{done}/{len(targets)}] kept={kept} ({kept/done:.0%}) "
                      f"{rate:.1f}/s", flush=True)
    print(f"done: {done} tasks, {kept} verified references "
          f"({kept/max(done,1):.0%}) in {time.time()-start:.0f}s")


if __name__ == "__main__":
    main()
