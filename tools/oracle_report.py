"""Compare an oracle run with the trajectories it replayed: where did the sandbox disagree?

    uv run python -m tools.oracle_report data/runs/<tag>/train --data data/train/sft_upstream/train.jsonl ...

For each trial: the graded reward, the submitted answer against the recorded one, and the first
tool result that differs from the recorded one. A recorded run was verified upstream, so every
difference is the sandbox's -- a missing package, a path, a version, or output the recorded
environment printed differently.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from tools.oracle_server import question_of, read_rows

_ANSWER = re.compile(r"""echo\s+-n\s+(["'])(.*?)\1\s*>\s*/workdir/answer\.txt""", re.S)


def recorded_answer(row: dict) -> str | None:
    for m in reversed(row["messages"]):
        for call in m.get("tool_calls") or []:
            found = _ANSWER.search(call["function"]["arguments"].get("command", ""))
            if found:
                return found.group(2)
    return None


def tool_results(messages: list[dict]) -> list[str]:
    return [m["content"] for m in messages if m["role"] == "tool"]


def compare(trial: Path, rows_by_question: dict[str, dict]) -> dict:
    result = json.loads((trial / "result.json").read_text())
    transcript = json.loads((trial / "transcript.json").read_text())
    row = rows_by_question.get(question_of(transcript) or "")
    out = {"task_id": result["task_id"], "reward": result["reward"],
           "prediction": result.get("prediction", ""), "stop_reason": result.get("stop_reason"),
           "recorded_answer": recorded_answer(row) if row else None, "first_difference": None}
    if row is None:
        return out
    ours, theirs = tool_results(transcript), tool_results(row["messages"])
    for i, (a, b) in enumerate(zip(ours, theirs)):
        if a.strip() != b.strip():
            out["first_difference"] = {"tool_result": i, "ours": a[:600], "recorded": b[:600]}
            break
    out["tool_results"] = [len(ours), len(theirs)]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("tree", type=Path)
    ap.add_argument("--data", type=Path, nargs="+", required=True)
    ap.add_argument("--only-failures", action="store_true")
    args = ap.parse_args()
    by_q = {question_of(r["messages"]): r for r in read_rows(*args.data)}
    reports = [compare(p.parent, by_q) for p in sorted(args.tree.glob("*/L1*/result.json"))]
    passed = sum(r["reward"] >= 1.0 for r in reports)
    for r in reports:
        if args.only_failures and r["reward"] >= 1.0:
            continue
        print(json.dumps(r, indent=1))
    print(f"{passed}/{len(reports)} graded equal to gold")


if __name__ == "__main__":
    main()
