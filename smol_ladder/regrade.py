"""Re-grade stored predictions offline: what the run scored, against what it would score if the
`ANSWER: ` label were gone.

Why this exists. SmolDataEnvs' grader strips nothing -- its own `_normalize` lowercases and
collapses whitespace, and its Harbor verifier pipes `/workdir/answer.txt` into the grader
verbatim -- so a prediction that grades 0 upstream grades 0 here too. Our pass rate is therefore
already the benchmark's, and loosening `grade()` to strip a prefix would quietly measure a grader
of our own instead. The right layer for that defect is the prompt, which is where the fix went.

That leaves the question of how much the stored runs are worth. This tool answers it without a
model and without touching `data/runs`: for every stored `result.json` it re-scores the recorded
prediction twice, strict and with one `ANSWER:`-style prefix removed, and prints the per-rung
counts for both plus every task that flips. Both numbers are always shown, because the normalised
one is a measurement and not a score: nothing here writes back to the results tree.

    uv run python -m smol_ladder.regrade --split test
    uv run python -m smol_ladder.regrade --split jupyter-agent --flips
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from smol_ladder.grade import grade
from smol_ladder.run_ladder import source_for
from smol_ladder.tasks import DATA

# Anchored, and the one label regrade removes: case-insensitive, optional inner spaces, and a
# marker rather than a word, so a gold answer that legitimately reads "the answer: 42" is not in
# the blast radius. Nothing else is stripped -- no token, no hyphen, no punctuation.
REREDUCED_PREFIX = re.compile(r"^[ \t]*answer[ \t]*:[ \t]*", re.IGNORECASE)


def strip_prefix(prediction: str) -> str:
    """The prediction with one leading `ANSWER:` marker removed. Identity otherwise."""
    return REREDUCED_PREFIX.sub("", prediction or "").strip()


def regrade_prediction(row: dict, prediction: str) -> tuple[float, float]:
    """(strict, prefix-stripped) reward for one stored prediction. No model, no writes.

    The second number is what the trial would have scored had the agent printed the value alone,
    so it is the strict number for every prediction that carried no label and can only rise.
    """
    if not prediction:
        return 0.0, 0.0
    return grade(row, prediction), grade(row, strip_prefix(prediction))


def collect(split: str) -> dict[str, dict[str, dict]]:
    """task_id -> rung dir -> stored result, read off disk and left exactly as found."""
    out: dict[str, dict[str, dict]] = {}
    for path in (DATA / "runs" / split).glob("*/*/result.json"):
        out.setdefault(path.parent.parent.name, {})[path.parent.name] = json.loads(path.read_text())
    return out


def _blank() -> dict:
    return {"attempted": 0, "graded": 0, "passed_strict": 0, "passed_prefix_stripped": 0,
            "harness_failures": 0, "no_prediction": 0, "prefixed": 0}


def report(runs: dict[str, dict[str, dict]], rows: dict[str, dict]) -> dict:
    """Per-rung counts for both readings of every stored trial, plus the tasks that flip.

    Three states stay apart, because one blended row is what made the hand-read necessary: a
    trial that produced a line we graded, a trial whose agent never finished, and a trial that
    finished with nothing to grade. Only the first has a surface form to be wrong about.
    """
    rungs: dict[str, dict] = {}
    flips: list[dict] = []
    for task, by_rung in runs.items():
        row = rows.get(task)
        for rung, result in by_rung.items():
            block = rungs.setdefault(rung, _blank())
            block["attempted"] += 1
            prediction = result.get("prediction", "")
            if result.get("agent_status") != "exit 0":
                block["harness_failures"] += 1
                continue
            if not prediction:
                block["no_prediction"] += 1
                continue
            if row is None:
                continue
            block["graded"] += 1
            strict, normalised = regrade_prediction(row, prediction)
            prefixed = strip_prefix(prediction) != prediction.strip()
            block["prefixed"] += prefixed
            block["passed_strict"] += strict >= 1.0
            block["passed_prefix_stripped"] += normalised >= 1.0
            if normalised >= 1.0 > strict:
                flips.append({"task_id": task, "rung": rung, "reward_mode": row["reward_mode"],
                              "gold": row["answer"], "prediction": prediction})
    return {"rungs": rungs, "flips": sorted(flips, key=lambda f: (f["rung"], f["task_id"]))}


def _print(split: str, got: dict, show_flips: bool) -> None:
    print(f"split={split}   strict = SmolDataEnvs' grader as shipped; "
          f"normalised = one leading `ANSWER:` label removed")
    print()
    header = f"  {'rung':<10} {'graded':>7} {'strict':>8} {'normalised':>11} {'flips':>6}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for rung, block in sorted(got["rungs"].items()):
        if not block["graded"]:
            continue
        gained = block["passed_prefix_stripped"] - block["passed_strict"]
        print(f"  {rung:<10} {block['graded']:>7} {block['passed_strict']:>8} "
              f"{block['passed_prefix_stripped']:>11} {gained:>6}")
    print()
    total = {k: sum(b[k] for b in got["rungs"].values())
             for k in ("graded", "passed_strict", "passed_prefix_stripped", "prefixed",
                       "harness_failures", "no_prediction")}
    print(f"  {'total':<10} {total['graded']:>7} {total['passed_strict']:>8} "
          f"{total['passed_prefix_stripped']:>11} "
          f"{total['passed_prefix_stripped'] - total['passed_strict']:>6}")
    print(f"  {total['prefixed']} graded predictions carried an `ANSWER:` label; "
          f"{total['harness_failures']} trials never finished; "
          f"{total['no_prediction']} finished with nothing to grade")
    flips = got["flips"]
    print(f"\n  {len(flips)} tasks flip 0.0 -> 1.0 on the label alone")
    for flip in flips[: len(flips) if show_flips else 10]:
        print(f"    {flip['task_id']} {flip['rung']:<10} gold={flip['gold']!r:<34} "
              f"pred={flip['prediction']!r}")
    if len(flips) > 10 and not show_flips:
        print(f"    ... {len(flips) - 10} more; pass --flips to list them all")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test",
                    choices=["test", "eval", "train", "jupyter-agent", "synthetic"])
    ap.add_argument("--flips", action="store_true", help="list every flipping task, not the first 10")
    args = ap.parse_args()

    source, _ = source_for(args.split)
    rows = {r["task_id"]: r for r in source}
    _print(args.split, report(collect(args.split), rows), args.flips)


if __name__ == "__main__":
    main()