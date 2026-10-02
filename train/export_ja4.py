"""Arm B, native: `ja4_sft.jsonl`, the teacher's own trajectories in the shell harness.

`ja3_sft_v2` was a translation: the sweep ran the teacher in the tools loop (which demands a
`solution.py`) and the export re-expressed it as a bash conversation. Its adapter was far worse than
the base model (docs/SFT_RESULTS.md). The `ja4` sweep ran the same teacher on the same tasks through
`--agent bash` itself, so a row here is the conversation exactly as the harness recorded it: nothing
is invented or rewritten. A trial is kept when the grader passed it and the model ended the episode.

    uv run python -m train.export_ja4            # data/train/ja4_sft.{jsonl,index.jsonl,manifest.json}
    uv run python -m train.export_ja4 --dry-run
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from smol_ladder import ladder as L
from smol_ladder.tasks import DATA
from smol_ladder.upstream import BASH_SYSTEM, BASH_TOOL
from train import export_ja3 as X
from train.export_ja3_v2 import call_id
from train.replay import assert_replays

NAME = "ja4_sft"
RUN = Path("runs") / "ja4" / X.SPLIT


def row_of(task: dict, transcript: list[dict]) -> dict:
    """The recorded conversation as a training row: tool-call arguments as objects, ids in the
    exported shape, and every tool result named. Raises ValueError when it is not a whole episode."""
    if [m["role"] for m in transcript[:2]] != ["system", "user"]:
        raise ValueError("does not open with system and user")
    if transcript[0]["content"] != BASH_SYSTEM:
        raise ValueError("system prompt differs from the harness's")
    if transcript[1]["content"] != L.prompt_for(task, X.SPLIT, "L1", "bash"):
        raise ValueError("user turn differs from today's L1 prompt")
    last = transcript[-1]
    if last["role"] != "assistant" or last.get("tool_calls"):
        raise ValueError("does not end with the model's closing message")
    out, ids, n = [dict(m) for m in transcript[:2]], {}, 0
    for m in transcript[2:]:
        if m["role"] == "assistant":
            new = {"role": "assistant", "content": m.get("content") or ""}
            calls = []
            for c in m.get("tool_calls") or []:
                n += 1
                ids[c["id"]] = call_id(task["task_id"], n)
                args = c["function"]["arguments"]
                calls.append({"id": ids[c["id"]], "type": "function", "function": {
                    "name": c["function"]["name"],
                    "arguments": json.loads(args) if isinstance(args, str) else args}})
            if calls:
                new["tool_calls"] = calls
            out.append(new)
        elif m["role"] == "tool":
            out.append({"role": "tool", "tool_call_id": ids[m["tool_call_id"]],
                        "content": m["content"], "name": "bash"})
        else:
            raise ValueError(f"unexpected role {m['role']}")
    return {"messages": out, "tools": BASH_TOOL}


def commands(messages: list[dict]) -> list[str]:
    return [c["function"]["arguments"]["command"] for m in messages if m["role"] == "assistant"
            for c in m.get("tool_calls") or []]


def export(data: Path) -> tuple[list[dict], list[dict], dict]:
    catalogue, keys = X.pool_rows(X.SPLIT), X.heldout_keys_for()
    rows, index, refused, trials = [], [], Counter(), 0
    for result_path in sorted((data / RUN).glob("*/L1/result.json")):
        result = json.loads(result_path.read_text())
        trials += 1
        task = catalogue.get(result["task_id"])
        if (result.get("reward") or 0) < 1:
            refused["not graded correct"] += 1
            continue
        if result.get("stop_reason") != "model_stopped":
            refused[f"episode ended by {result.get('stop_reason')}"] += 1
            continue
        column = X.row_is_heldout(task, keys)
        if column:
            refused[f"heldout:{column}"] += 1
            continue
        try:
            row = row_of(task, json.loads((result_path.parent / "transcript.json").read_text()))
        except (OSError, ValueError, KeyError) as e:
            refused[str(e)[:60]] += 1
            continue
        if X.hints_absent(row["messages"], task):
            refused["hint text in the conversation"] += 1
            continue
        leaked = X.leaks_machine(json.dumps(row, ensure_ascii=False))
        if leaked:
            refused["names this machine: " + ", ".join(leaked)] += 1
            continue
        try:
            assert_replays(row["messages"], strict=True)
        except AssertionError:
            refused["does not replay through the harness"] += 1
            continue
        cmds = commands(row["messages"])
        rows.append(row)
        index.append({"task_id": result["task_id"], "commands": len(cmds),
                      "longest_command": max(map(len, cmds)) if cmds else 0})
    n = [e["commands"] for e in index]
    report = {"trials": trials, "rows": len(rows), "refused": dict(refused.most_common()),
              "commands_per_row": {"mean": round(statistics.mean(n), 2), "median": statistics.median(n)} if n else None,
              "longest_command_median": statistics.median(e["longest_command"] for e in index) if index else None}
    return rows, index, report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=DATA / "train")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    rows, index, report = export(DATA)
    print(json.dumps(report, indent=1))
    if args.dry_run:
        return
    (args.out / f"{NAME}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    (args.out / f"{NAME}.index.jsonl").write_text("".join(json.dumps(e) + "\n" for e in index))
    (args.out / f"{NAME}.manifest.json").write_text(json.dumps({
        "source": NAME, "run": str(RUN), "teacher": "stealth/space-bunny-alpha, --agent bash, 16 turns",
        "method": "the recorded conversation, verbatim; no translation", **report,
        "generated_at": datetime.now(timezone.utc).isoformat()}, indent=1))
    print(f"wrote {len(rows)} rows to {args.out / (NAME + '.jsonl')}")


if __name__ == "__main__":
    main()
