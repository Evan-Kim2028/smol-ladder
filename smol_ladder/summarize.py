"""Summarise ladder runs: pass rate by rung, and the lowest rung each task passes.

The headline is the pass-rate curve. "Lowest passing rung" is a lossy summary of the same runs
and is reported second, with the no-reference tasks kept in their own row so they are never
silently counted as "never passes".

    uv run python -m smol_ladder.summarize --split test
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from smol_ladder.ladder import read_source
from smol_ladder.run_ladder import source_for
from smol_ladder.tasks import DATA

RUNGS = ["L1", "L1+schema", "L2", "L3", "L4"]
# The control's directory has no "+" in it; everything else is the rung name verbatim.
DIRS = {"L1": "L1", "L1+schema": "L1_schema", "L2": "L2", "L3": "L3", "L4": "L4"}


def collect(split: str) -> dict[str, dict[str, dict]]:
    """task_id -> rung -> result, read off disk."""
    out: dict[str, dict[str, dict]] = {}
    by_dir = {v: k for k, v in DIRS.items()}
    for path in (DATA / "runs" / split).glob("*/*/result.json"):
        rung = by_dir.get(path.parent.name, path.parent.name)
        out.setdefault(path.parent.parent.name, {})[rung] = json.loads(path.read_text())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test",
                    choices=["test", "eval", "train", "jupyter-agent", "synthetic"])
    args = ap.parse_args()

    rows, _ = source_for(args.split)
    by_id = {r["task_id"]: r for r in rows}
    runs = collect(args.split)
    total = len(by_id)
    n_ref = sum(read_source(r, args.split) is not None for r in by_id.values())

    print(f"split={args.split}  tasks={total}  with verified reference={n_ref} "
          f"({n_ref/total:.0%})")
    print()
    print("pass rate by rung, over the tasks actually run at that rung")
    for rung in RUNGS:
        seen = [t for t, r in runs.items() if rung in r]
        if not seen:
            print(f"  {rung:<10} not run")
            continue
        ok = sum(runs[t][rung]["reward"] >= 1.0 for t in seen)
        crashed = sum(runs[t][rung]["agent_status"] != "exit 0" for t in seen)
        print(f"  {rung:<10} {ok:>3}/{len(seen):<3} = {ok/len(seen):>5.1%}"
              f"   (crashed {crashed})")

    print()
    print("lowest rung that passed (tasks that failed L1 and were climbed)")
    hist: dict[str, int] = {"L1": 0, "L1+schema": 0, "L2": 0, "L3": 0, "L4": 0, "never": 0,
                            "no reference": 0}
    climbed = 0
    for task, rungs in runs.items():
        if "L1" not in rungs:
            continue
        if rungs["L1"]["reward"] >= 1.0:
            hist["L1"] += 1
            continue
        if read_source(by_id[task], args.split) is None:
            hist["no reference"] += 1
            continue
        climbed += 1
        # The control is not a rung: a pass there is a skill (exploration) result, so it is
        # recorded on its own and never counts as the "information" rescues below.
        if rungs.get("L1+schema", {}).get("reward", 0.0) >= 1.0:
            hist["L1+schema"] += 1
        first = next((r for r in ["L2", "L3", "L4"]
                      if rungs.get(r, {}).get("reward", 0.0) >= 1.0), None)
        hist[first or "never"] += 1
    for key in ["L1", "L1+schema", "L2", "L3", "L4", "never", "no reference"]:
        if hist[key]:
            print(f"  {key:<12} {hist[key]}")

    out = DATA / "runs" / f"summary_{args.split}.json"
    out.write_text(json.dumps({"tasks": total, "with_reference": n_ref, "rung_counts": hist},
                              indent=1))
    print(f"\nwrote {out}")
    print(f"climbed {climbed} tasks; {hist['L2']+hist['L3']+hist['L4']} were rescued by a hint")


if __name__ == "__main__":
    main()
