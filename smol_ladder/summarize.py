"""Summarise ladder runs: a partition of tasks by first passing rung, plus per-rung counts.

Three tables, and they are kept apart on purpose.

1. `first_passing_rung` — one bucket per task, over the ladder rungs only. It is a partition:
   every task lands in exactly one bucket and the buckets sum to the task count. The L1+schema
   control is excluded from it. The control is not a rung; it adds no information, only
   cheaper reading, so booking a rescue there would double-count a task that the ladder also
   books. It is reported on its own, in (2).
2. `control` — the L1+schema trials: attempted and rescued, split by whether the task has a
   reference, because the control runs on L1 failures whether or not they can be climbed.
3. `rungs` — attempted, passed, and harness failures (`agent_status != "exit 0"`) per rung,
   over the tasks actually run at that rung.

The partition also separates three states that used to collapse into one: a task that passed at
no rung, a task with no reference and therefore nothing above L1 to try, and a task no rung was
ever run on at all. "Tried and failed" and "never tried" are different findings.

    uv run python -m smol_ladder.summarize --split test
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path

from smol_ladder.ladder import read_source
from smol_ladder.run_ladder import source_for
from smol_ladder.tasks import DATA

RUNGS = ["L1", "L2", "L3", "L4"]
CLIMBABLE = RUNGS[1:]
CONTROL = "L1+schema"
ALL = RUNGS + [CONTROL]
# The control's directory has no "+" in it; everything else is the rung name verbatim.
DIRS = {rung: rung.replace("+", "_") for rung in ALL}

# Every bucket `first_passing_rung` can hold, so the key set is fixed and a reader never has to
# guess whether a zero bucket is missing or empty.
BUCKETS = RUNGS + ["never", "not climbable (no reference)", "not attempted"]


def collect(split: str) -> dict[str, dict[str, dict]]:
    """task_id -> rung -> result, read off disk."""
    out: dict[str, dict[str, dict]] = {}
    by_dir = {v: k for k, v in DIRS.items()}
    for path in (DATA / "runs" / split).glob("*/*/result.json"):
        rung = by_dir.get(path.parent.name, path.parent.name)
        out.setdefault(path.parent.parent.name, {})[rung] = json.loads(path.read_text())
    return out


def _passed(result: dict | None) -> bool:
    return bool(result) and result.get("reward", 0.0) >= 1.0


def _finished(result: dict | None) -> bool:
    """Did the harness get a clean run out of the trial? Anything else is not a model failure."""
    return bool(result) and result.get("agent_status") == "exit 0"


def first_passing_rung(rungs: dict[str, dict], has_reference: bool) -> str:
    """The one bucket this task belongs to, out of BUCKETS.

    The control cannot appear here. A task that the control rescued is booked by the rung that
    passed it, if any, and by `never` if no rung did — which is the honest reading, because the
    control result says nothing about the ladder rungs.
    """
    if "L1" not in rungs:
        return "not attempted"
    if _passed(rungs["L1"]):
        return "L1"
    if not has_reference:
        # Nothing above L1 was ever built for this task, so it cannot have passed one. It is
        # not a ladder failure; it is outside the ladder.
        return "not climbable (no reference)"
    for rung in CLIMBABLE:
        if _passed(rungs.get(rung)):
            return rung
    return "never"


def partition(runs: dict[str, dict[str, dict]],
             has_reference: Callable[[str], bool]) -> dict[str, int]:
    """task -> exactly one bucket. Sums to len(runs) by construction, one task per iteration."""
    hist = dict.fromkeys(BUCKETS, 0)
    for task, rungs in runs.items():
        hist[first_passing_rung(rungs, has_reference(task))] += 1
    return hist


def control_block(runs: dict[str, dict[str, dict]],
                  has_reference: Callable[[str], bool]) -> dict:
    """The L1+schema trials, on the tasks they were actually run on, split by reference status.

    Split by reference because the control is gated on nothing: it runs on every L1 failure, so
    most of its rescues are on tasks the ladder never climbs, and a single blended rate hides
    that.
    """
    block: dict = {"attempted": 0, "rescued": 0,
                   "with reference": {"attempted": 0, "rescued": 0},
                   "without reference": {"attempted": 0, "rescued": 0}}
    harness = 0
    for task, rungs in runs.items():
        result = rungs.get(CONTROL)
        if result is None:
            continue
        group = "with reference" if has_reference(task) else "without reference"
        block["attempted"] += 1
        block[group]["attempted"] += 1
        harness += not _finished(result)
        if _passed(result):
            block["rescued"] += 1
            block[group]["rescued"] += 1
    block["harness_failures"] = harness
    return block


def rung_counts(runs: dict[str, dict[str, dict]]) -> dict[str, dict]:
    """Attempted, passed and harness failures per rung, over the tasks run at that rung."""
    out: dict[str, dict] = {}
    for rung in ALL:
        seen = [r[rung] for r in runs.values() if rung in r]
        out[rung] = {
            "attempted": len(seen),
            "passed": sum(_passed(r) for r in seen),
            "harness_failures": sum(not _finished(r) for r in seen),
        }
    return out


def summarise(split: str, runs: dict[str, dict[str, dict]],
              has_reference: Callable[[str], bool]) -> dict:
    """The whole report, as a dict. `summarize` asserts nothing; this asserts the partition."""
    hist = partition(runs, has_reference)
    assert sum(hist.values()) == len(runs), (hist, len(runs))
    return {
        "split": split,
        "tasks": len(runs),
        "first_passing_rung": hist,
        "control": control_block(runs, has_reference),
        "rungs": rung_counts(runs),
    }


def _rows_for(split: str) -> tuple[list[dict], Callable[[str], bool]]:
    rows, _ = source_for(split)
    by_id = {r["task_id"]: r for r in rows}
    return rows, lambda task: task in by_id and read_source(by_id[task], split) is not None


def _print(report: dict) -> None:
    print(f"split={report['split']}  tasks={report['tasks']}")
    print()
    print("first passing rung (mutually exclusive, control excluded, sums to tasks)")
    for key in BUCKETS:
        value = report["first_passing_rung"][key]
        if value:
            print(f"  {key:<30} {value:>4}")
    print(f"  {'sum':<30} {sum(report['first_passing_rung'].values()):>4}")

    control = report["control"]
    print()
    print("L1+schema control (not a rung: adds no information, only cheaper reading)")
    for group in ("with reference", "without reference"):
        block = control[group]
        rate = block["rescued"] / block["attempted"] if block["attempted"] else float("nan")
        print(f"  {group:<20} {block['rescued']:>3}/{block['attempted']:<3} = {rate:>5.1%}")
    rate = control["rescued"] / control["attempted"] if control["attempted"] else float("nan")
    print(f"  {'all':<20} {control['rescued']:>3}/{control['attempted']:<3} = {rate:>5.1%}"
          f"   (harness failures {control['harness_failures']})")

    print()
    print("per rung, over the tasks actually run at that rung")
    for rung in ALL:
        block = report["rungs"][rung]
        if not block["attempted"]:
            print(f"  {rung:<10} not run")
            continue
        print(f"  {rung:<10} {block['passed']:>3}/{block['attempted']:<3} = "
              f"{block['passed']/block['attempted']:>5.1%}"
              f"   (harness failures {block['harness_failures']})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test",
                    choices=["test", "eval", "train", "jupyter-agent", "synthetic"])
    ap.add_argument("--out", type=Path, help="where to write the JSON report")
    args = ap.parse_args()

    rows, has_reference = _rows_for(args.split)
    runs = collect(args.split)
    n_ref = sum(has_reference(row["task_id"]) for row in rows)
    total = len(rows)
    print(f"split={args.split}  tasks in the source={total}  "
          f"with verified reference={n_ref} ({n_ref/total:.0%})")

    report = summarise(args.split, runs, has_reference)
    report["source_tasks"] = total
    report["with_reference"] = n_ref
    _print(report)

    dest = args.out or DATA / "runs" / f"summary_{args.split}.json"
    dest.write_text(json.dumps(report, indent=1))
    print(f"\nwrote {dest}")
    missing = {r["task_id"] for r in rows} - set(runs)
    if missing:
        print(f"{len(missing)} source tasks have no trial on disk, "
              f"counted as 'not attempted', e.g. {sorted(missing)[:3]}")


if __name__ == "__main__":
    main()
