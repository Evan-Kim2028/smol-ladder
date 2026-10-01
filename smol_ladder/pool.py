"""Picking which jupyter-agent pool a sweep runs on.

`data/jtasks.jsonl` is the v1 extract: 2,000 tasks, and the population every existing
jupyter-agent run was measured against. It stays exactly as it is -- other checkouts have live
sweeps reading it, and a pool is an input to a measurement rather than an output of one. So the
corrected pool is reached by naming it, not by overwriting the old file.

**The split name.** `--split jupyter-agent-v3` reads `data/jtasks_v3.jsonl` and keeps the
*ladder-grade* subset: not nondeterministic, not ambiguous, and it names at least one input file.
That is 4,217 of the 7,518 v3 tasks. The filter is `jtasks_v2.is_ladder_grade` imported, not
copied, so a reviewer who changes the definition changes this selection with it.

Naming the split is also what keeps the two sweeps apart. `run_ladder` derives its tree from the
split name, so v3 trials land in `data/runs/<tag>/jupyter-agent-v3/` and never touch
`data/runs/jupyter-agent/`; `gen_refs` and `ladder.read_source` derive their directory the same
way, so a reference verified under v3's rules is never served to a rung built from v1's gold.

**The cached-input skip.** 7,018 of the 7,518 v3 tasks (93.4%) reuse datasets already in the
Kaggle cache. The other 500 are skipped rather than fetched: they span 144 uncached datasets, and
Kaggle was returning 403s on the day the pool was built, so a sweep that tried would burn its
wall clock on downloads and book the failures against the model. The ids are written to
`data/skipped_<split>.json`, because "500 tasks were skipped" is only auditable if the 500 are
named.

    uv run python -m smol_ladder.run_ladder --split jupyter-agent-v3 --run-tag ja3 --rungs L1
    uv run python -m smol_ladder.pool --split jupyter-agent-v3          # counts, nothing written
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from smol_ladder.jtasks import input_dir
from smol_ladder.jtasks_v2 import is_ladder_grade
from smol_ladder.tasks import DATA

#: The filter, by reference. `jtasks_v2.is_ladder_grade` is the one definition of "a task whose
#: answer a careful reader can derive"; re-implementing it here would be a second definition that
#: could disagree with the first on a boundary case and quietly change a population.
LADDER_GRADE = is_ladder_grade

#: The corrected pool's split name. It is a *new* split rather than a new default so that
#: `--split jupyter-agent` keeps meaning the v1 population every published number was measured
#: over, and so the two sweeps get separate results trees by construction.
V3_SPLIT = "jupyter-agent-v3"

#: Split name -> pool file. The v1 entry is what `--split jupyter-agent` has always meant.
POOLS = {
    "jupyter-agent": "jtasks.jsonl",
    V3_SPLIT: "jtasks_v3.jsonl",
    "jupyter-agent-v2": "jtasks_v2.jsonl",
}


def pool_path(split: str, data: Path | None = None) -> Path:
    """Which file a split name reads. The pool is opened, never written."""
    try:
        name = POOLS[split]
    except KeyError:
        raise ValueError(f"unknown jupyter-agent pool {split!r}; "
                         f"known: {', '.join(sorted(POOLS))}") from None
    return (data or DATA) / name


def load_pool(split: str, data: Path | None = None) -> list[dict]:
    """The raw rows of a pool, whatever it contains."""
    path = pool_path(split, data)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; see smol_ladder.jtasks_v2 for how to build it")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def source_for(split: str, data: Path | None = None):
    """Task rows and their input directories, for the runnable jupyter-agent splits.

    Same signature as `run_ladder.source_for`, so a split name is all that decides which pool and
    which filter run. The v1 split keeps its old meaning exactly -- unfiltered v1 rows -- because
    every published jupyter-agent number is a measurement over that population.

    v1 is read through `pool_path` rather than `jtasks.load_rows`, which resolves its own `DATA`
    at call time. Both name the same file, but one of them can be pointed at a fixture and the
    other cannot, and a selector whose v1 branch cannot be tested is a selector whose v1 branch
    is not.
    """
    if split == "jupyter-agent":
        return load_pool(split, data), input_dir
    return ladder_grade(load_pool(split, data)), input_dir


def ladder_grade(rows: list[dict]) -> list[dict]:
    """The rows whose answer a careful reader can derive, in pool order.

    Order is preserved rather than sorted: a sweep's row order is its task order, and a resumed
    sweep must see the same tasks in the same order to resume the same work.
    """
    return [row for row in rows if LADDER_GRADE(row)]


def inputs_are_cached(row: dict, inputs_of=input_dir) -> bool:
    """Whether fetching this task's tables would hit the network.

    The cache has two levels and the difference matters here. `kagglehub` keeps the dataset
    *archives* under `kaggle/datasets/<owner>/<name>/`, and `input_dir` additionally builds a
    per-task directory of symlinks under `kaggle/tasks/<task_id>/` the first time a task runs.
    Only 1,623 of the 4,217 ladder-grade tasks have that per-task directory, but 3,880 have the
    archive they would be built from -- so testing for the per-task directory skips 2,257 tasks
    whose tables are sitting on disk and would cost nothing to read.

    So the question is "would `dataset_download` hit the network?", and the archive directory is
    what answers it. `input_dir(..., fetch=False)` is still consulted, but only as a second
    opinion: it is the per-task directory, which is a stricter test and would report "not cached"
    for a task whose archive is present. Either signal means no download is needed.

    The row's own `inputs_cached` flag is a third signal, and the loosest: it is a snapshot of
    the cache when the pool was built, and `/var/tmp` is not durable. It is honoured only when it
    says yes and something else corroborates it, so a stale `true` cannot make a task look cached
    when its archive has since been evicted.
    """
    try:
        if inputs_of(row, fetch=False).is_dir():
            return True
    except TypeError:
        try:
            if inputs_of(row).is_dir():
                return True
        except Exception:  # noqa: BLE001 - an unresolvable path is not a cached one
            pass
    except Exception:  # noqa: BLE001 - an unresolvable path is not a cached one
        pass
    if row.get("inputs_cached") is not True:
        return False
    dataset = row.get("kaggle_dataset_name")
    if not dataset:
        return False
    import os

    cache = Path(os.environ.get("SMOL_LADDER_CACHE", "/var/tmp/smol-ladder")) / "kaggle"
    # The full `owner/name`, because that is the directory kagglehub creates. Matching on the
    # bare name instead would call a mirror of a cached dataset "cached" and then read the wrong
    # owner's files -- the same over-broad key that let 121 tasks through the overlap firewall.
    return (cache / "datasets" / dataset).is_dir()


def without_cached_inputs(rows: list[dict], inputs_of=input_dir,
                          enabled: bool = True) -> tuple[list[dict], list[str]]:
    """`(runnable, skipped_ids)` -- the tasks to attempt and the ids deliberately not attempted.

    `enabled=False` skips nothing, and is what every split other than a Kaggle-backed pool passes.
    The check exists because the jupyter-agent pool spans 144 datasets the endpoint would have to
    fetch; SmolDataEnvs rows come from a single HuggingFace bucket and synthetic rows point at a
    table that is already local, so applying the same test there would skip the whole corpus for
    no reason and report a population of zero.
    """
    if not enabled:
        return list(rows), []
    runnable, skipped = [], []
    for row in rows:
        if inputs_are_cached(row, inputs_of):
            runnable.append(row)
        else:
            skipped.append(row["task_id"])
    return runnable, skipped


def record_skipped(split: str, skipped: list[str], path: Path | None = None,
                   planned: int | None = None) -> Path:
    """Write the skipped ids where the run record can point at them.

    Every launch writes its own file, named for the split, and never truncates a previous sweep's
    list: two sweeps over the same pool with different cached-input populations are both facts.
    """
    out = Path(path or DATA / f"skipped_{split}.json")
    record = {"split": split, "tasks_planned": planned, "skipped_for_inputs": len(skipped),
              "skipped_task_ids": skipped}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=1))
    return out


def counts(split: str, data: Path | None = None) -> dict:
    """What a sweep over this pool would actually do. Writes nothing."""
    rows = load_pool(split, data)
    selected = ladder_grade(rows)
    runnable, skipped = without_cached_inputs(selected)
    return {
        "split": split,
        "pool_tasks": len(rows),
        "ladder_grade": len(selected),
        "excluded_by_filter": len(rows) - len(selected),
        "inputs_cached": len(runnable),
        "skipped_for_inputs": len(skipped),
        "tasks_planned": len(runnable),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default=V3_SPLIT, choices=sorted(POOLS))
    ap.add_argument("--record-skipped", action="store_true",
                    help="also write data/skipped_<split>.json, so the skipped ids are on disk")
    args = ap.parse_args()
    summary = counts(args.split)
    print(json.dumps(summary, indent=1))
    if args.record_skipped:
        _, skipped = without_cached_inputs(ladder_grade(load_pool(args.split)))
        print(f"wrote {record_skipped(args.split, skipped, planned=summary['tasks_planned'])}")


if __name__ == "__main__":
    main()
