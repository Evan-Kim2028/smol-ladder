"""Paired comparison of run tags: the reading rules of docs/SFT_RESULTS.md, applied.

    uv run python -m smol_ladder.report_runs --runs base=amd2-base,A=amd2-a,B=amd3-b4 --rung L1
    uv run python -m smol_ladder.report_runs --runs base=amd3-base,A=amd3-a --rung L3 --reference base
    uv run python -m smol_ladder.report_runs --runs base=amd3s-base,A=amd3s-a --rung L1 --json

Every number is computed on the COMMON task set: the tasks that every named run has a scored
trial for at that rung (harness errors and skips excluded identically). A model's score is the
mean over tasks of its per-task pass rate (one attempt: 0 or 1; k attempts: the fraction). The
interval is a 95% bootstrap over tasks; the comparison against the reference model is paired per
task (gained / lost, a sign test, and a bootstrap interval on the mean difference). Beside
accuracy: how episodes ended, the share that wrote any answer, and the share that ended in a
repeat loop, because those are less noisy than accuracy and say whether a gap is behavioural.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

from smol_ladder.tasks import DATA, load_split

RUNS = DATA / "runs"
LOOP_TAIL = 4           # an episode "looped" when its last 4 commands are identical


def rung_dir(rung: str) -> str:
    return rung.replace("+", "_")


def commands_of(transcript_path: Path) -> list[str]:
    try:
        t = json.loads(transcript_path.read_text())
    except (OSError, ValueError):
        return []
    out = []
    for m in (t if isinstance(t, list) else t.get("messages") or []):
        if m.get("role") == "assistant":
            for c in m.get("tool_calls") or []:
                a = c["function"]["arguments"]
                a = json.loads(a) if isinstance(a, str) else a
                out.append(str(a.get("command")))
    return out


def load_run(tag: str, rung: str, split: str = "test") -> dict[str, dict]:
    """task_id -> {'passes': [0/1 per attempt], 'stops': [...], 'answered': [...], 'looped': [...]}"""
    out: dict[str, dict] = {}
    for result in (RUNS / tag / split).glob(f"*/{rung_dir(rung)}/**/result.json"):
        r = json.loads(result.read_text())
        if r.get("skipped") or r.get("stop_reason") == "error" or r.get("rung") != rung:
            continue
        d = out.setdefault(r["task_id"], {"passes": [], "stops": [], "answered": [], "looped": []})
        d["passes"].append(1 if (r.get("reward") or 0) >= 1 else 0)
        d["stops"].append(r.get("stop_reason") or "none")
        d["answered"].append(bool((r.get("prediction") or "").strip()))
        cmds = commands_of(result.parent / "transcript.json")
        d["looped"].append(len(cmds) >= LOOP_TAIL and len(set(cmds[-LOOP_TAIL:])) == 1)
    return out


def mean(xs) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def bootstrap_ci(values: list[float], n: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = random.Random(seed)
    k = len(values)
    if k == 0:
        return (float("nan"), float("nan"))
    means = sorted(sum(values[rng.randrange(k)] for _ in range(k)) / k for _ in range(n))
    return (means[int(0.025 * n)], means[int(0.975 * n) - 1])


def sign_test(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def report(runs: dict[str, str], rung: str, reference: str | None, split: str = "test",
           exclude: set[str] | None = None) -> dict:
    loaded = {name: load_run(tag, rung, split) for name, tag in runs.items()}
    common = set.intersection(*(set(d) for d in loaded.values()))
    if exclude:
        common -= exclude
    tiers = {r["task_id"]: r.get("difficulty_tier") for r in load_split(split)}
    tasks = sorted(common)
    out = {"rung": rung, "common_tasks": len(tasks), "models": {}}
    rates = {name: {t: mean(loaded[name][t]["passes"]) for t in tasks} for name in runs}
    for name in runs:
        d = loaded[name]
        per_task = [rates[name][t] for t in tasks]
        attempts = sum(len(d[t]["passes"]) for t in tasks)
        stops = Counter(s for t in tasks for s in d[t]["stops"])
        by_tier: dict[str, list[float]] = defaultdict(list)
        for t in tasks:
            by_tier[tiers.get(t) or "?"].append(rates[name][t])
        m = {"run": runs[name], "attempts": attempts, "attempts_per_task": round(attempts / max(len(tasks), 1), 2),
             "pass_rate": round(mean(per_task), 4), "ci95": [round(x, 4) for x in bootstrap_ci(per_task)],
             "wrote_any_answer": round(mean(a for t in tasks for a in d[t]["answered"]), 4),
             "ended_in_repeat_loop": round(mean(l for t in tasks for l in d[t]["looped"]), 4),
             "stop_reasons": {k: round(v / attempts, 4) for k, v in stops.most_common()},
             "by_tier": {k: round(mean(v), 4) for k, v in sorted(by_tier.items())}}
        if reference and name != reference:
            diffs = [rates[name][t] - rates[reference][t] for t in tasks]
            wins = sum(x > 0 for x in diffs)
            losses = sum(x < 0 for x in diffs)
            m["vs_" + reference] = {"gained": wins, "lost": losses, "sign_test_p": round(sign_test(wins, losses), 4),
                                    "mean_difference": round(mean(diffs), 4),
                                    "difference_ci95": [round(x, 4) for x in bootstrap_ci(diffs)]}
        out["models"][name] = m
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True, help="name=tag,name=tag,...")
    ap.add_argument("--rung", default="L1")
    ap.add_argument("--reference", default=None, help="model name the others are paired against")
    ap.add_argument("--split", default="test")
    ap.add_argument("--exclude-ids", type=Path, help="JSON file with an 'ids' list to leave out (the notebook overlap)")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    runs = dict(x.split("=", 1) for x in a.runs.split(","))
    exclude = set(json.loads(a.exclude_ids.read_text())["ids"]) if a.exclude_ids else None
    rep = report(runs, a.rung, a.reference or next(iter(runs)), a.split, exclude)
    if a.json:
        print(json.dumps(rep, indent=1))
        return
    print(f"{rep['rung']}: {rep['common_tasks']} common tasks")
    for name, m in rep["models"].items():
        line = (f"  {name:>6}  pass {100 * m['pass_rate']:5.1f}%  [{100 * m['ci95'][0]:.1f}, {100 * m['ci95'][1]:.1f}]"
                f"  answered {100 * m['wrote_any_answer']:.0f}%  looped {100 * m['ended_in_repeat_loop']:.0f}%"
                f"  attempts/task {m['attempts_per_task']}")
        for k, v in m.items():
            if k.startswith("vs_"):
                line += (f"  | {k}: +{v['gained']} -{v['lost']} p={v['sign_test_p']}"
                         f" diff {100 * v['mean_difference']:+.1f} [{100 * v['difference_ci95'][0]:+.1f}, {100 * v['difference_ci95'][1]:+.1f}]")
        print(line)
        print(f"          tiers {m['by_tier']}")


if __name__ == "__main__":
    main()
