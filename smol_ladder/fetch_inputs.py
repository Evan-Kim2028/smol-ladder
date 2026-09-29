"""Pre-fetch every jupyter-agent task's tables, in parallel.

A trial fetches its own tables before entering the jail, so without this the run serialises
on Kaggle: sixteen workers all waiting on downloads, and a single 42 MB dataset stalls the
whole batch. Fetching up front makes the trial loop do only what it is for.

Idempotent: input_dir skips a task whose directory already exists, and each distinct Kaggle
dataset is downloaded once.

    uv run python -m smol_ladder.fetch_inputs --split jupyter-agent --workers 12
"""

from __future__ import annotations

import argparse
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

_lock = threading.Lock()
_done = Counter()
_failed: list[tuple[str, str]] = []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="jupyter-agent")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    from smol_ladder.jtasks import input_dir, load_rows
    from smol_ladder.run_ladder import source_for

    rows, inputs_of = source_for(args.split)
    rows = rows[: args.limit]
    print(f"fetching tables for {len(rows)} tasks with {args.workers} workers", flush=True)
    start = time.time()

    def one(row: dict) -> str | None:
        try:
            inputs_of(row)
            with _lock:
                _done["ok"] += 1
                n = _done["ok"]
            if n % 25 == 0:
                print(f"  {n}/{len(rows)} ({time.time()-start:.0f}s)", flush=True)
            return None
        except Exception as e:  # noqa: BLE001 - a missing dataset must not stop the sweep
            with _lock:
                _done["failed"] += 1
                _failed.append((row["task_id"], f"{type(e).__name__}: {e}"[:120]))
            return row["task_id"]

    with ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(one, row) for row in rows]
        for f in as_completed(futures):
            f.result()

    print(f"\nok={_done['ok']} failed={_done['failed']} in {time.time()-start:.0f}s")
    for task_id, err in _failed[:20]:
        print(f"  FAILED {task_id}: {err}")
    if _failed:
        print(f"  ... and {len(_failed)-20} more" if len(_failed) > 20 else "")


if __name__ == "__main__":
    main()
