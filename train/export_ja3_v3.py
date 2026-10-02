"""Arm B, leaner: `ja3_sft_v3.jsonl`, a mechanical rewrite of `ja3_sft_v2.jsonl`.

Session 2's B adapter failed at evaluation by imitating v2's long multi-step style: 7 commands a
row against upstream's 4, of which 2.4 were the write-then-run of `solution.py` that our sweep
required for verification, and 0.3 were library version checks. Nothing here calls a model; every
command and every tool output that remains was recorded in the same trajectory.

    uv run python -m train.export_ja3_v3            # writes data/train/ja3_sft_v3.{jsonl,index.jsonl,manifest.json}
    uv run python -m train.export_ja3_v3 --dry-run  # counts only

What changes, in order:
  1. the no-op `cd . && ` prefix is removed from every command;
  2. a call that only prints library versions is dropped with its output;
  3. a `cd /app && ...` that failed because /app did not exist is dropped (the sweep's write tool
     had said "written /app/solution.py"; the teacher's next command is the same one without it);
  3b. the solution script is dropped (write, runs, rewrites) when an earlier command's output already
     ended in the submitted answer and the closing message does not talk about a script;
  4. otherwise each write of `solution.py` and the run that follows become ONE command, whose
     output is the run's (the write printed nothing);
  5. a row whose longest remaining command exceeds --max-command characters is left out.
v2 stays on disk untouched.
"""
from __future__ import annotations

import argparse
import copy
import json
import re
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from smol_ladder.ladder import DATA
from train.replay import assert_replays

NAME = "ja3_sft_v3"
SOURCE = "ja3_sft_v2"
MAX_COMMAND = 1000                  # characters; upstream's 90th percentile longest command is ~900
EMPTY = "(empty output, rc=0)"
_CD_NOOP = re.compile(r"^cd \. && ")
_WRITE = re.compile(r"^cat > /workdir/solution\.py << ?'EOF")
_RUN = re.compile(r"^[^\n]*\bpython3? (?:/workdir/)?solution\.py[^\n]*$")   # one line that runs the script
_CD_APP = re.compile(r"^cd /app\b")
_NO_APP = "cd: /app: No such file or directory"
_SUBMIT = re.compile(r"^echo -n (?P<q>['\"])(?P<value>.*)(?P=q) > /workdir/answer\.txt$", re.S)
_SCRIPT_TALK = re.compile(r"script|solution", re.I)


def command_of(call: dict) -> str:
    args = call["function"]["arguments"]
    return (json.loads(args) if isinstance(args, str) else args)["command"]


def set_command(call: dict, command: str) -> None:
    args = call["function"]["arguments"]
    if isinstance(args, str):
        call["function"]["arguments"] = json.dumps({**json.loads(args), "command": command})
    else:
        args["command"] = command


def is_version_check(command: str) -> bool:
    return ("__version__" in command and "/home/user/input" not in command
            and "solution.py" not in command and "<<" not in command)


def turns_of(messages: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    """(head, turns, tail): head is system+user, each turn an assistant message with its calls and
    their outputs, tail the closing assistant message(s) that call nothing."""
    head, turns, tail, outputs = [], [], [], {}
    for m in messages:
        if m["role"] == "tool":
            outputs[m["tool_call_id"]] = m
    for m in messages:
        if m["role"] in ("system", "user"):
            head.append(m)
        elif m["role"] == "assistant" and m.get("tool_calls"):
            turns.append({"message": m, "steps": [[c, outputs[c["id"]]] for c in m["tool_calls"]]})
        elif m["role"] == "assistant":
            tail.append(m)
    return head, turns, tail


def answered_inline(turns: list[dict], value: str) -> bool:
    """Did a command BEFORE the first solution write already print the answer as its last line?"""
    for turn in turns:
        for call, out in turn["steps"]:
            command = command_of(call)
            if _WRITE.match(command):
                return False
            lines = [x.strip() for x in out["content"].splitlines() if x.strip()]
            if "python" in command and lines and lines[-1] == value.strip():
                return True
    return False


def lean(messages: list[dict]) -> tuple[list[dict], Counter]:
    counts: Counter = Counter()
    head, turns, tail = turns_of(copy.deepcopy(messages))
    for turn in turns:
        for call, _ in turn["steps"]:
            command = command_of(call)
            if _CD_NOOP.match(command):
                set_command(call, _CD_NOOP.sub("", command, count=1))
                counts["cd_noop_removed"] += 1
    for turn in turns:
        kept = [s for s in turn["steps"] if not is_version_check(command_of(s[0]))]
        counts["version_check_dropped"] += len(turn["steps"]) - len(kept)
        turn["steps"] = kept
    # The sweep's write tool answered "written /app/solution.py", so the teacher tried `cd /app`,
    # which did not exist, and then repeated the command without it. The failed try goes.
    for turn in turns:
        kept = [s for s in turn["steps"]
                if not (_CD_APP.match(command_of(s[0])) and _NO_APP in s[1]["content"])]
        counts["failed_cd_app_dropped"] += len(turn["steps"]) - len(kept)
        turn["steps"] = kept
    if counts["failed_cd_app_dropped"]:
        for turn in turns:
            if "/app" in (turn["message"].get("content") or ""):
                turn["message"]["content"] = ""
                counts["prose_about_app_blanked"] += 1

    flat = [(turn, step) for turn in turns for step in turn["steps"]]
    submit = next((_SUBMIT.match(command_of(s[0])) for _, s in reversed(flat)
                   if _SUBMIT.match(command_of(s[0]))), None)
    closing = " ".join(m.get("content") or "" for m in tail)
    drop_script = bool(submit and not _SCRIPT_TALK.search(closing)
                       and answered_inline(turns, submit["value"]))
    if drop_script:
        for turn in turns:
            kept = [s for s in turn["steps"] if "solution.py" not in command_of(s[0])]
            if len(kept) != len(turn["steps"]) and _SCRIPT_TALK.search(turn["message"].get("content") or ""):
                turn["message"]["content"] = ""       # "Now writing the solution script." has no script left
            counts["solution_step_dropped"] += len(turn["steps"]) - len(kept)
            turn["steps"] = kept
        counts["rows_solution_dropped"] += 1
    else:
        i = 0
        while i + 1 < len(flat):
            (turn, step), (next_turn, next_step) = flat[i], flat[i + 1]
            write, run = command_of(step[0]), command_of(next_step[0])
            if _WRITE.match(write) and step[1]["content"] == EMPTY and _RUN.match(run):
                set_command(step[0], f"{write}\n{run.strip()}")
                step[1]["content"] = next_step[1]["content"]
                next_turn["steps"].remove(next_step)
                del flat[i + 1]
                counts["write_and_run_folded"] += 1
            i += 1

    out = list(head)
    carried = ""                    # prose of a turn that lost all its calls moves to the next one
    for turn in turns:
        message = turn["message"]
        text = message.get("content") or ""
        if not turn["steps"]:
            carried = carried or text
            counts["empty_turn_removed"] += 1
            continue
        if carried and not text:
            message["content"] = carried
        carried = ""
        message["tool_calls"] = [c for c, _ in turn["steps"]]
        out.append(message)
        out.extend(o for _, o in turn["steps"])
    out.extend(tail)
    return out, counts


def commands(messages: list[dict]) -> list[str]:
    return [command_of(c) for m in messages if m["role"] == "assistant" for c in m.get("tool_calls") or []]


def export(source: Path, max_command: int) -> tuple[list[dict], list[dict], dict]:
    rows, index, totals, left_out = [], [], Counter(), Counter()
    source_index = [json.loads(x) for x in source.with_name(f"{SOURCE}.index.jsonl").read_text().splitlines()]
    before, after = [], []
    for n, line in enumerate(source.read_text().splitlines()):
        row = json.loads(line)
        messages, counts = lean(row["messages"])
        cmds = commands(messages)
        before.append(len(commands(row["messages"])))
        if not cmds or not _SUBMIT.match(cmds[-1]):
            left_out["no_submission_as_last_command"] += 1
            continue
        if max(map(len, cmds)) > max_command:
            left_out["command_too_long"] += 1
            continue
        try:
            assert_replays(messages, strict=True)
        except AssertionError:
            left_out["does_not_replay"] += 1
            continue
        totals.update(counts)
        after.append(len(cmds))
        rows.append({"messages": messages, "tools": row["tools"]})
        index.append({"task_id": source_index[n]["task_id"], "v2_row": n, "commands": len(cmds),
                      "longest_command": max(map(len, cmds)), "counts": dict(counts)})
    report = {"source_rows": len(before), "rows": len(rows), "left_out": dict(left_out),
              "changes": dict(totals), "max_command": max_command,
              "commands_per_row": {"v2_median": statistics.median(before),
                                   "v2_mean": round(statistics.mean(before), 2),
                                   "v3_median": statistics.median(after) if after else None,
                                   "v3_mean": round(statistics.mean(after), 2) if after else None},
              "longest_command_median": statistics.median(e["longest_command"] for e in index) if index else None}
    return rows, index, report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=DATA / "train")
    ap.add_argument("--max-command", type=int, default=MAX_COMMAND)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    rows, index, report = export(args.out / f"{SOURCE}.jsonl", args.max_command)
    print(json.dumps(report, indent=1))
    if args.dry_run:
        return
    (args.out / f"{NAME}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    (args.out / f"{NAME}.index.jsonl").write_text("".join(json.dumps(e) + "\n" for e in index))
    (args.out / f"{NAME}.manifest.json").write_text(json.dumps({
        "source": NAME, "derived_from": f"{SOURCE}.jsonl (left in place)",
        "method": "mechanical rewrite, no model output added; see train/export_ja3_v3.py",
        **report, "generated_at": datetime.now(timezone.utc).isoformat()}, indent=1))
    print(f"wrote {len(rows)} rows to {args.out / (NAME + '.jsonl')}")


if __name__ == "__main__":
    main()
