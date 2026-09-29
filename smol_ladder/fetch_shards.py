"""Download the jupyter-agent shards once, locally, so no run re-fetches them.

Every extraction pass used to call hf_hub_download per shard, and a trial's table fetch went
to Kaggle. Neither is a rate limit problem on the Hub side — the 429s we hit were Kaggle's —
but re-reading 103 shards for every sweep is slow and fragile, so the parquet files are pulled
once into data/jl_shards/ and every later pass reads from disk.

Narrow columns only, and never the ~72 GB original_notebook blob: we need the question, the
answer, the file list and the executor type, not the notebook that produced them.

    uv run python -m smol_ladder.fetch_shards --workers 4
"""

from __future__ import annotations

import argparse
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from smol_ladder.tasks import DATA

DATASET = "jupyter-agent/jupyter-agent-dataset"
SHARDS = 103
DEST = DATA / "jl_shards"
# The eight columns the task builder reads. original_notebook is 72 GB and never needed.
COLUMNS = ["id", "question", "answer", "executor_type", "files_used", "packages_used",
           "kaggle_dataset_name", "edu_score"]

_lock = threading.Lock()
_done = 0
_failed: list[tuple[str, str]] = []


def shard_path(index: int) -> Path:
    return DEST / f"non_thinking-{index:05d}-of-{SHARDS:05d}.parquet"


def fetch(index: int, retries: int = 4) -> str | None:
    """Download one shard, resuming by skipping any already on disk."""
    global _done
    target = shard_path(index)
    if target.exists() and target.stat().st_size > 0:
        with _lock:
            _done += 1
            n = _done
        if n % 10 == 0:
            print(f"  {n}/{SHARDS} (cached)", flush=True)
        return None
    from huggingface_hub import hf_hub_download

    last = ""
    for attempt in range(retries):
        try:
            path = hf_hub_download(
                DATASET, f"data/{target.name}", repo_type="dataset")
            # Copy out of the blob store so later runs do not depend on the Hub cache, and so
            # the file survives a cache prune.
            tmp = target.with_suffix(".partial")
            tmp.write_bytes(Path(path).read_bytes())
            tmp.rename(target)
            with _lock:
                _done += 1
                n = _done
                size = target.stat().st_size
            if n % 5 == 0:
                print(f"  {n}/{SHARDS} ({size/1e6:.0f} MB each)", flush=True)
            return None
        except Exception as e:  # noqa: BLE001 - a flaky shard must not stop the sweep
            last = f"{type(e).__name__}: {str(e)[:100]}"
            time.sleep(5 * 2 ** attempt)
    with _lock:
        _failed.append((target.name, last))
    return target.name


def read_shard(index: int):
    """Rows from a shard, read from the local copy."""
    import pyarrow.parquet as pq
    return pq.ParquetFile(shard_path(index)).read(columns=COLUMNS).to_pylist()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    DEST.mkdir(parents=True, exist_ok=True)
    have = sum(shard_path(i).exists() for i in range(SHARDS))
    print(f"{have}/{SHARDS} shards already local; downloading the rest with {args.workers} workers",
          flush=True)
    start = time.time()
    with ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(fetch, i) for i in range(SHARDS)]
        for f in as_completed(futures):
            f.result()

    present = sum(shard_path(i).exists() for i in range(SHARDS))
    total = sum(shard_path(i).stat().st_size for i in range(SHARDS) if shard_path(i).exists())
    print(f"\n{present}/{SHARDS} shards, {total/1e9:.1f} GB, in {time.time()-start:.0f}s")
    for name, err in _failed[:10]:
        print(f"  FAILED {name}: {err}")


if __name__ == "__main__":
    main()
