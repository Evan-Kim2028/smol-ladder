"""Re-grade stored trials against a *corrected* gold, without touching the original rewards.

Named `regrade_gold` because the correction here is the gold, and nothing else. A parallel
branch's `regrade` answers a different question — what the same stored predictions are worth if
the `ANSWER:` label were stripped — and the two must be able to coexist in one tree.

Why a tool at all. A gold fix invalidates the rewards computed against the old one, and the
ladder's shape depends on those rewards: climbing stops at the first pass, so a task that passed
under a wrong gold never saw the rungs above it, and a task that failed under a wrong gold was
sent up rungs it should never have reached. Re-grading the predictions that *were* stored
therefore answers two questions with two different answers:

- What the stored predictions are worth now, at the rung they were run at. That is a regrade.
- What the pass-rate curve would look like if every rung had been run against the corrected
  gold. That is not recoverable, because the missing predictions were never produced. Tasks
  that would need a rung rerun are named explicitly instead.

The gold a stored trial is graded against is whichever version that task's id carries, and both
are recorded per row as `gold_source`. This is not bookkeeping: the corrected corpus drops ids
the shipped-file gate refused, and every stored trial under a dropped id still has a prediction
and a recorded reward. Grading those against nothing would silently remove half the results tree
from the report — 136 of 275 synthetic trials — and the before/after table would then compare
two differently-populated samples rather than two golds.

    uv run python -m smol_ladder.regrade_gold --split synthetic
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from smol_ladder.grade import grade
from smol_ladder.jtasks import load_synthetic
from smol_ladder.tasks import DATA

# The filesystem spells the control without a "+"; everything else is the rung name verbatim.
DIR_TO_RUNG = {"L1_schema": "L1+schema"}
RUNG_ORDER = ["L1", "L1+schema", "L2", "L3", "L4"]
# The ladder proper. The control is not in it: L1+schema adds no information, so it is never a
# rung a task climbs to, and it runs on every L1 failure whether or not the ladder can climb the
# task at all. Ordering matters below and nowhere else.
CLIMBABLE = ["L2", "L3", "L4"]
BACKUP = DATA / "synthetic.jsonl.bak-pre-goldfix"


def current_gold() -> dict[str, dict]:
    """The corrected corpus, keyed by task id."""
    return {row["task_id"]: row for row in load_synthetic()}


def superseded_gold(backup: Path = BACKUP) -> dict[str, dict]:
    """The pre-fix corpus, keyed by task id, for ids the corrected one no longer carries.

    Its golds are known wrong — that is why they were replaced — but a trial graded against the
    old gold is still the only measurement that exists for a task the gate removed, and dropping
    it would make the regrade a comparison over a task set that changed at the same time as the
    gold did.
    """
    if not backup.exists():
        return {}
    return {row["task_id"]: row for row in
            (json.loads(line) for line in backup.read_text().splitlines() if line.strip())}


def collect(split: str) -> dict[str, dict[str, dict]]:
    """task_id -> rung -> result, read off the results tree without modifying it."""
    out: dict[str, dict[str, dict]] = {}
    for path in sorted((DATA / "runs" / split).glob("*/*/result.json")):
        rung = DIR_TO_RUNG.get(path.parent.name, path.parent.name)
        out.setdefault(path.parent.parent.name, {})[rung] = json.loads(path.read_text())
    return out


def resolve(task: str, current: dict[str, dict], old: dict[str, dict]):
    """(gold row, which version it came from) for a stored trial, or (None, None)."""
    if task in current:
        return current[task], "corrected"
    if task in old:
        return old[task], "superseded"
    return None, None


def regrade(split: str, current: dict[str, dict], old: dict[str, dict],
            out: Path) -> dict:
    """Re-grade every stored trial against its task's current gold.

    `result.json` is read and never written: the original `reward` is the record of what the run
    scored under the gold that was in force then, and overwriting it would destroy the only copy
    of that.
    """
    runs = collect(split)
    # Pre-seeded so a zero is present rather than absent: a consumer summing the per-gold counts
    # to reconcile against `trials` must not have to know which counter keys were never touched.
    per_rung: dict[str, Counter] = {
        rung: Counter({"trials": 0, "old_harness": 0,
                       "old_pass_corrected": 0, "new_pass_corrected": 0,
                       "old_pass_superseded": 0, "new_pass_superseded": 0,
                       "trials_corrected": 0, "trials_superseded": 0})
        for rung in RUNG_ORDER
    }
    flips: list[dict] = []
    ungraded: list[str] = []
    with out.open("w") as fh:
        for task, rungs in sorted(runs.items()):
            row, source = resolve(task, current, old)
            if row is None:
                ungraded.append(task)
                continue
            for rung in RUNG_ORDER:
                stored = rungs.get(rung)
                if stored is None:
                    continue
                reward = grade(row, stored.get("prediction", "") or "")
                old_reward = float(stored.get("reward", 0.0) or 0.0)
                counts = per_rung[rung]
                counts["trials"] += 1
                counts[f"trials_{source}"] += 1
                counts[f"old_pass_{source}"] += old_reward >= 1.0
                counts[f"new_pass_{source}"] += reward >= 1.0
                counts["old_harness"] += stored.get("agent_status") != "exit 0"
                if (reward >= 1.0) != (old_reward >= 1.0):
                    flips.append({"task_id": task, "rung": rung, "reward_old": old_reward,
                                  "reward": reward, "gold_source": source,
                                  "prediction": stored.get("prediction", "")})
                fh.write(json.dumps({
                    "task_id": task, "rung": rung, "reward_old": old_reward, "reward": reward,
                    "agent_status": stored.get("agent_status"),
                    "prediction": stored.get("prediction", ""),
                    "gold": row["answer"], "gold_source": source,
                }) + "\n")
    return {"split": split, "out": str(out), "ungraded": sorted(ungraded),
            "rungs": {r: dict(c) for r, c in per_rung.items()}, "flips": flips}


def rerun_table(current: dict[str, dict], old: dict[str, dict],
                runs: dict[str, dict[str, dict]]) -> dict:
    """Which stored trials are now misleading, and which rung each task still owes a trial on.

    A rung owes a rerun for one of two reasons. The trial exists and its reward moved, so the
    number it contributed to the curve is wrong. Or the trial does not exist at all, because the
    climb stopped on a reward that the corrected gold reverses: a task that passes at L1 under
    the corrected gold but did not under the old one never saw L2, so its place on the ladder is
    unknown rather than known, and no amount of regrading can recover it.
    """
    stale: list[tuple[str, str]] = []
    owed: list[tuple[str, str]] = []
    for task, rungs in sorted(runs.items()):
        row, _ = resolve(task, current, old)
        if row is None:
            continue
        for rung in RUNG_ORDER:
            if rung not in rungs:
                continue
            moved = grade(row, rungs[rung].get("prediction", "") or "") >= 1.0
            if moved != (float(rungs[rung].get("reward", 0.0) or 0.0) >= 1.0):
                stale.append((task, rung))
        # The climb starts at L2 and only if L1 did not pass, so a task that still clears L1 owes
        # nothing at all -- there is no gap, the climb stopped where it always did.
        l1 = rungs.get("L1")
        if l1 is not None and grade(row, l1.get("prediction", "") or "") >= 1.0:
            continue
        # Re-simulate the climb against the corrected gold. The climb is bottom-up and stops at
        # the first pass, so a rung is owed exactly when the corrected gold reaches it and no
        # stored trial there would pass: the old run left it out because a rung below passed, and
        # that rung no longer passes. The control is not a rung and is never part of the climb --
        # it runs on every L1 failure whether or not the ladder can climb the task -- so it is
        # excluded here, and its own flips are already in `stale_trials`.
        climbed = _climb_gap(row, rungs)
        owed.extend((task, rung) for rung in climbed)
    return {"stale_trials": sorted(set(stale)), "owed_trials": sorted(set(owed))}


def _climb_gap(row: dict, rungs: dict[str, dict]) -> list[str]:
    """The rungs a fresh climb against `row` would have to run, and so never ran before.

    The climb starts at L2 and stops at the first rung that passes. A rung that exists and passes
    is where a fresh climb stops too, so nothing above it is owed. Every missing rung from there
    to L4 is owed, because the old run left it out precisely when a rung below passed and that
    rung no longer does.
    """
    owed: list[str] = []
    for rung in CLIMBABLE:
        stored = rungs.get(rung)
        if stored is None:
            owed.append(rung)
            continue
        if grade(row, stored.get("prediction", "") or "") >= 1.0:
            break
    return owed


def table(report: dict) -> str:
    """The per-rung before/after, split by which gold each row was graded against."""
    lines = [f"{'rung':<10}{'trials':>8}{'pass old':>10}{'pass new':>10}{'flipped':>9}   split"]
    for rung in RUNG_ORDER:
        counts = report["rungs"][rung]
        if not counts.get("trials"):
            lines.append(f"{rung:<10}  (no trials)")
            continue
        flipped = sum(1 for f in report["flips"] if f["rung"] == rung)
        old = sum(v for k, v in counts.items() if k.startswith("old_pass_"))
        new = sum(v for k, v in counts.items() if k.startswith("new_pass_"))
        split = ", ".join(f"{n} {counts[f'trials_{n}']}"
                          for n in ("corrected", "superseded") if counts.get(f"trials_{n}"))
        lines.append(f"{rung:<10}{counts['trials']:>8}{old:>10}{new:>10}{flipped:>9}   {split}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="synthetic")
    ap.add_argument("--tag", default="post-goldfix")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--backup", type=Path, default=BACKUP,
                    help="the pre-fix corpus, used for ids the corrected one dropped")
    args = ap.parse_args()

    current, old = current_gold(), superseded_gold(args.backup)
    runs = collect(args.split)
    out = args.out or DATA / "runs" / f"regrade_gold_{args.tag}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"{args.split}: {len(runs)} tasks on disk, "
          f"{sum(len(r) for r in runs.values())} stored trials")
    print(f"  {len(current)} corrected golds, {len(old)} superseded, "
          f"{len(set(current) & set(old))} ids in both")
    print(f"  tasks covered by a gold: "
          f"{len(set(runs) & (set(current) | set(old)))}/{len(runs)}; "
          f"ungraded: {len(set(runs) - set(current) - set(old))}")

    report = regrade(args.split, current, old, out)
    print(f"\nwrote {report['out']}")
    print()
    print(table(report))

    report.update(rerun_table(current, old, runs))
    print()
    print(f"stored trials whose reward moved: {len(report['stale_trials'])}")
    print(f"rungs that never ran but that the corrected gold now makes matter: "
          f"{len(report['owed_trials'])}")
    by_rung = Counter(rung for _, rung in report["owed_trials"])
    for rung in RUNG_ORDER:
        if by_rung.get(rung):
            print(f"  {rung:<10} {by_rung[rung]} tasks")
    for task, rung in report["owed_trials"][:40]:
        print(f"  {task} {rung}")
    if len(report["owed_trials"]) > 40:
        print(f"  ... and {len(report['owed_trials']) - 40} more")


if __name__ == "__main__":
    main()