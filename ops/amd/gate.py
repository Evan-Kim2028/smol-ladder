"""The gate: does the whole stack work on a model somebody else already trained?

Before any training is paid for, the released adapter (arm `R`) is merged, served beside the base
and run through the harness at L1 on a fixed 60-task subset, one sample each, both at once. The
first paid session produced nothing usable because the stack was wrong in ways that every number
still looked plausible under (adapters evaluated as the base, a prompt that contradicted the
training format, runs that hung and kept "arriving" as harness errors). A released model with a
known recipe is the cheapest probe for all of them together.

Pure functions over result files, so the decision is tested without a droplet.

GO needs all of:

  (a) harness failures within a small tolerance, per model (a missing trial counts as a failure:
      a sweep that died has no files, not bad ones);
  (b) the merge report passed, the adapter produced a parsed tool call, and its temperature-0
      output on a fixed training prompt differs from the base's;
  (c) the released adapter's pass rate is at least the base's plus a margin.

(a) and (b) are hard. (c) is a judgement about whether the evaluation protocol reproduces the
released model's advantage, so when it fails the two rates, a paired comparison and the stop-reason
histograms are printed and `--accept-gate` is required to go on.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

# The gate's tasks: a fixed seeded sample stratified by difficulty tier (committed; regenerate with
# `python -m ops.amd.gate subset`). The first 60 task ids of the split are 25% easy against the split's
# 13%, which made the gate easier than the evaluation it vouches for.
SUBSET_IDS = "ops/amd/gate_subset.txt"       # one id per line: what `run_ladder --task-ids` reads
SUBSET_JSON = "ops/amd/gate_subset.json"     # the same ids with the seed and the tier allocation
SUBSET_SEED = 42


def stratified_subset(rows: list[dict], n: int, seed: int = SUBSET_SEED) -> tuple[list[str], dict]:
    """`n` task ids, drawn at random within each difficulty tier in proportion to the tier's share
    of `rows` (largest remainder, ties by tier name), in split order. Deterministic: the draw per
    tier uses its own `random.Random(f"{seed}:{tier}")` over the tier's sorted ids, so it does not
    depend on row order or on the other tiers."""
    import random
    if n > len(rows):
        raise ValueError(f"asked for {n} tasks from a split of {len(rows)}")
    by_tier: dict[str, list[str]] = {}
    for r in rows:
        by_tier.setdefault(str(r["difficulty_tier"]), []).append(r["task_id"])
    total = len(rows)
    quota = {t: n * len(ids) / total for t, ids in by_tier.items()}
    alloc = {t: int(q) for t, q in quota.items()}
    for t in sorted(by_tier, key=lambda t: (-(quota[t] - alloc[t]), t))[: n - sum(alloc.values())]:
        alloc[t] += 1
    chosen: set[str] = set()
    for tier, ids in by_tier.items():
        chosen |= set(random.Random(f"{seed}:{tier}").sample(sorted(ids), alloc[tier]))
    order = sorted(r["task_id"] for r in rows)
    ids = [t for t in order if t in chosen]
    meta = {"n": n, "seed": seed, "split_counts": {t: len(v) for t, v in sorted(by_tier.items())},
            "allocation": dict(sorted(alloc.items())), "ids": ids,
            "derivation": "ops.amd.gate.stratified_subset(rows of the SmolDataEnvs test split, n, seed): "
                          "per difficulty_tier, random.Random(f'{seed}:{tier}').sample(sorted ids, k), "
                          "k by largest remainder of n x the tier's share; ids in sorted order"}
    return ids, meta


STOP_REASONS = ("answer_submitted", "model_stopped", "max_turns", "context_exhausted")
ANSWER_STOPS = ("answer_submitted", "model_stopped")


def _main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="regenerate the committed gate subset from the dataset")
    ap.add_argument("cmd", choices=["subset"])
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=SUBSET_SEED)
    args = ap.parse_args(argv)
    from smol_ladder.tasks import load_split
    root = Path(__file__).resolve().parents[2]
    ids, meta = stratified_subset(load_split("test"), args.n, args.seed)
    (root / SUBSET_IDS).write_text("\n".join(ids) + "\n")
    (root / SUBSET_JSON).write_text(json.dumps(meta, indent=1) + "\n")
    print(meta["allocation"], "of", meta["split_counts"])
    return 0


def is_clean(result: dict) -> bool:
    """A trial the harness ran to the end. Anything else is a harness failure, never a model one."""
    return result.get("agent_status") == "exit 0"


def passed(result: dict) -> bool:
    return is_clean(result) and float(result.get("reward") or 0.0) >= 1.0


def read_results(root: Path, tag: str, split: str, rung: str = "L1",
                 task_ids: set[str] | None = None) -> dict[str, dict]:
    """task id -> result.json of sample 0 under data/runs/<tag>/<split>/<task>/<rung>/."""
    base = Path(root) / tag / split
    out: dict[str, dict] = {}
    if not base.exists():
        return out
    for path in base.glob(f"*/{rung}/result.json"):
        task = path.parent.parent.name
        if task_ids is not None and task not in task_ids:
            continue
        try:
            out[task] = json.loads(path.read_text())
        except (OSError, ValueError):
            out[task] = {"agent_status": "unreadable result.json"}
    return out


def task_ids_present(root: Path, tags: list[str], split: str, rung: str = "L1") -> list[str]:
    ids: set[str] = set()
    for tag in tags:
        ids |= set(read_results(root, tag, split, rung))
    return sorted(ids)


@dataclass
class ModelStats:
    arm: str
    expected: int                         # the tasks the gate set holds
    results: dict[str, dict] = field(default_factory=dict)

    @property
    def clean(self) -> int:
        return sum(is_clean(r) for r in self.results.values())

    @property
    def failures(self) -> int:
        """Expected trials that are not clean results: crashed, timed out, or never written."""
        return self.expected - self.clean

    @property
    def passes(self) -> int:
        return sum(passed(r) for r in self.results.values())

    def stop_histogram(self) -> dict[str, int]:
        hist: dict[str, int] = {}
        for r in self.results.values():
            if is_clean(r):
                key = r.get("stop_reason") or "unknown"
                hist[key] = hist.get(key, 0) + 1
        return hist

    def ended_with_answer(self) -> int:
        return sum(1 for r in self.results.values()
                   if is_clean(r) and r.get("stop_reason") in ANSWER_STOPS
                   and str(r.get("prediction") or "").strip())


def stats_for(arm: str, results: dict[str, dict], task_ids: list[str]) -> ModelStats:
    return ModelStats(arm, len(task_ids), {t: results[t] for t in task_ids if t in results})


def paired(base: ModelStats, other: ModelStats) -> dict:
    """The two models on the tasks both ran cleanly: who passed what, and an exact two-sided sign
    test on the discordant pairs (is the difference more than a coin flip?)."""
    both_clean = [t for t in base.results
                  if t in other.results and is_clean(base.results[t]) and is_clean(other.results[t])]
    b = o = both = neither = 0
    for t in both_clean:
        pb, po = passed(base.results[t]), passed(other.results[t])
        both += pb and po
        neither += not pb and not po
        b += pb and not po
        o += po and not pb
    n = b + o
    p = 1.0 if n == 0 else min(1.0, 2.0 * sum(math.comb(n, k) for k in range(min(b, o) + 1)) / 2 ** n)
    return {"tasks": len(both_clean), "both": both, "only_base": b, "only_other": o,
            "neither": neither, "p_value": p,
            "base_rate": (both + b) / len(both_clean) if both_clean else 0.0,
            "other_rate": (both + o) / len(both_clean) if both_clean else 0.0}


@dataclass
class Decision:
    go: bool
    needs_accept: bool                    # (c) failed, (a) and (b) held: only a human can go on
    hard_failures: list[str]
    accepted: bool = False
    lines: list[str] = field(default_factory=list)
    trials_per_min: float = 0.0


def histogram_line(label: str, s: ModelStats) -> str:
    hist, n = s.stop_histogram(), max(s.clean, 1)
    shown = {k: hist.get(k, 0) for k in STOP_REASONS}
    other = sum(v for k, v in hist.items() if k not in STOP_REASONS)
    parts = [f"{k} {shown[k]} ({shown[k] / n:.0%})" for k in STOP_REASONS]
    if other:
        parts.append(f"other {other} ({other / n:.0%})")
    return (f"  {label:<6} stop reasons: " + ", ".join(parts)
            + f"; ended with an answer {s.ended_with_answer()} ({s.ended_with_answer() / n:.0%})")


def decide(base: ModelStats, adapter: ModelStats, *, adapter_name: str, differs: bool | None,
           tool_calls_ok: bool | None, merge_ok: bool | None, margin: float, max_failures: int,
           accepted: bool = False) -> Decision:
    """The gate's verdict and the report that explains it (`Decision.lines`)."""
    hard: list[str] = []
    lines = [f"## gate: base vs {adapter_name} on {base.expected} tasks (L1, one sample each, "
             "--agent bash, the harness's default --bash-stop model)"]
    lines.append(f"  {'model':<6} {'trials':>6} {'clean':>6} {'failed':>6} {'pass':>5}")
    for label, s in (("base", base), (adapter_name, adapter)):
        lines.append(f"  {label:<6} {s.expected:>6} {s.clean:>6} {s.failures:>6} {s.passes:>5}")
    for label, s in (("base", base), (adapter_name, adapter)):
        lines.append(histogram_line(label, s))
        if s.failures > max_failures:
            hard.append(f"{label}: {s.failures} harness failures of {s.expected} trials "
                        f"(tolerance {max_failures}): the harness or the server is not healthy")
    if base.expected == 0:
        hard.append("no trials at all: the gate evaluation produced no results")
    for what, val in (("the merge report passed", merge_ok),
                      (f"{adapter_name} produced a parsed bash tool call", tool_calls_ok),
                      (f"{adapter_name}'s temperature-0 output differs from the base's", differs)):
        if val is not True:
            hard.append(f"{what}: {'FAILED' if val is False else 'not measured'}")
    pr = paired(base, adapter)
    lines.append(f"  paired on {pr['tasks']} tasks clean in both: both pass {pr['both']}, only base "
                 f"{pr['only_base']}, only {adapter_name} {pr['only_other']}, neither {pr['neither']}; "
                 f"exact sign test p={pr['p_value']:.3f}")
    lines.append(f"  pass rate: base {pr['base_rate']:.3f}, {adapter_name} {pr['other_rate']:.3f} "
                 f"(difference {pr['other_rate'] - pr['base_rate']:+.3f}; required {margin:+.3f})")
    if adapter.clean and adapter.stop_histogram().get("max_turns", 0) / adapter.clean >= 0.5:
        lines.append(f"  WARNING: {adapter_name} runs out of turns in at least half its trials; a "
                     "healthy SFT model mostly ends with a submitted answer")
    beats = pr["other_rate"] - pr["base_rate"] >= margin - 1e-9 and pr["tasks"] > 0
    if hard:
        lines.append("### GATE: NO-GO")
        lines += [f"  - {h}" for h in hard]
        return Decision(False, False, hard, False, lines)
    if beats:
        lines.append("### GATE: GO")
        return Decision(True, False, [], False, lines)
    if accepted:
        lines.append(f"### GATE: GO, ACCEPTED by --accept-gate although {adapter_name} did not beat "
                     f"the base by {margin:.3f}")
        return Decision(True, False, [], True, lines)
    lines.append(f"### GATE: STOP. {adapter_name} did not beat the base by {margin:.3f}. Everything "
                 "else passed, so the stack works; whether the protocol reproduces the released "
                 "model's advantage is a judgement. Read the paired comparison and the stop "
                 "reasons, then `driver.py gate-decide --accept-gate` (free) to continue, or "
                 "destroy.")
    return Decision(False, True, [], False, lines)


if __name__ == "__main__":
    raise SystemExit(_main())
