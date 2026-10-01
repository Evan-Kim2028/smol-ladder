"""Our own traces, translated into upstream's bash-agent format.

## What we actually have, measured on 2026-10-01

**The sweep runner does not save transcripts.** `run_ladder.once()` writes `turns.json` -- and
that file contains `len(log)`, a single integer, not the log. It copies exactly two artifacts out
of the per-trial scratch (`solution.py`, or `answer.txt` for the bash protocol) and then deletes
the scratch. A `data/runs/**` trial therefore holds a result, a turn *count*, the final program,
and the prompt; the assistant/tool conversation that got there is gone.

**`gen_solutions.py` does save transcripts** -- `data/solutions/<split>/<task_id>/transcript.jsonl`
-- but it has only ever been run on the held-out splits:

| source | verified tasks | full transcript |
|---|---|---|
| `data/solutions/test` | 211 | 211 |
| `data/solutions/eval` | 116 | 116 |
| `data/solutions/jupyter-agent` | 479 | **0** |
| `data/runs/test` | 217 | **0** |
| `data/runs/synthetic` | 211 | **0** |
| `data/runs/jupyter-agent.bak-pre-rerun-20261001` | 290 | **0** |

So: **327 full multi-turn transcripts exist, and every one is from `test` or `eval`** -- the two
splits the firewall refuses. There is no trainable transcript on this machine at all.

## What that means for training

`collect_traces` therefore has two paths and only one of them can return anything today:

- `transcript_path`: parse `transcript.jsonl` into real multi-turn trajectories. Implemented and
  tested, and it is the path that matters the moment anyone runs `gen_solutions --split train`.
  Returns nothing today, and the firewall -- not this module -- is why.
- `solution_path` (the fallback, and the only one in use): build a **degenerate** trajectory from
  the question and the verified `solution.py`: write the program, submit the answer, stop. It
  teaches the *contract* (inspect-with-bash -> program -> submit -> stop) and nothing about
  exploration. A training set of these is a weaker but honest dataset, not a broken one; it is
  emphatically **not** a substitute for real traces and must not be reported as one.

`smol_ladder/run_ladder.py` now saves transcripts by default (`--no-transcript` opts out), so the
next sweep produces the real thing. See `docs/TRAINING.md`.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from smol_ladder.tasks import DATA

from train.format import (answer_file_command, bash_row, closing_turn, heldout_keys,
                          is_heldout, shell_turn, submission_turn)

# Where verified trials live, and which of them we are allowed to train on. `solutions/test` and
# `solutions/eval` are listed for completeness and contribute nothing: they are held out, and
# `gen_solutions.py` says so itself ("Never train on them: they come from the held-out splits").
SOURCES = (
    ("solutions/jupyter-agent", "jupyter-agent"),
    ("runs/jupyter-agent.bak-pre-rerun-20261001", "jupyter-agent"),
    ("runs/synthetic", "synthetic"),
    ("runs/test", "smoldataenvs"),
)

# Empty stdout is the truthful result of a write-only command, and upstream's own trajectories end
# their submission with exactly this string.
EMPTY_OUTPUT = "(empty output, rc=0)"


def find_verified(root: Path) -> dict[str, Path]:
    """`task_id -> directory` for every trial that passed offline *and* left a program behind.

    `reward >= 1.0` is our own grader's verdict from the sealed offline pass, not the agent's
    claim, so a trial in here is one we re-ran and reproduced. When one task has several, the
    first by sorted path wins, so the choice is stable across runs.
    """
    found: dict[str, Path] = {}
    for dirpath, _dirs, names in os.walk(root):
        if "result.json" not in names:
            continue
        directory = Path(dirpath)
        if not (directory / "solution.py").exists():
            continue
        try:
            result = json.loads((directory / "result.json").read_text())
        except (OSError, ValueError):
            continue
        if result.get("reward", 0.0) < 1.0:
            continue
        task_id = result.get("task_id")
        if task_id:
            found.setdefault(task_id, directory)
    return found


def parse_transcript(path: Path) -> list[dict]:
    """`transcript.jsonl` (cmd's event stream) -> bash-format turns.

    The event stream is verbose and streaming: text arrives as deltas, a tool call appears inside
    a `message_end` content block, and its output arrives as `tool_update` partials keyed by
    `toolCallId`. Only the settled events are read -- a `message_end` and the *last* `tool_update`
    per call -- because a transcript assembled from deltas would contain every intermediate token
    state of a streamed answer.

    `shell_command` is renamed to `bash`: it is the same tool (a fresh `bash -c` per call, combined
    output), and the protocol a model is being taught to speak should not depend on which runner
    produced the trace.
    """
    turns: dict[str, dict] = {}
    order: list[str] = []
    for line in path.read_text(errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except ValueError:
            continue
        event = record.get("event") or {}
        kind = event.get("type")
        if kind == "message_end":
            text_parts, calls = [], []
            for block in event.get("content") or []:
                if block.get("type") == "text" and block.get("text"):
                    text_parts.append(block["text"])
                elif block.get("type") == "tool_use":
                    call_id = block.get("id")
                    command = (block.get("input") or {}).get("command", "")
                    if call_id:
                        calls.append((call_id, command))
            for call_id, command in calls:
                if call_id not in turns:
                    order.append(call_id)
                turns[call_id] = shell_turn(command, "", call_id)
            if text_parts and not calls:
                # A closing message: attach to the last turn, or make one if the assistant spoke
                # without ever calling a tool.
                if order:
                    turns[order[-1]]["content"] = "\n\n".join(text_parts)
                else:
                    order.append("__text__")
                    turns["__text__"] = {"content": "\n\n".join(text_parts)}
        elif kind == "tool_update":
            call_id = event.get("toolCallId")
            output = "".join(part.get("text", "")
                             for part in (event.get("partial") or []) if part.get("type") == "text")
            if call_id in turns and output:
                turns[call_id].setdefault("results", [])
                for result in turns[call_id]["results"]:
                    result["content"] = output
    return [turns[key] for key in order if key in turns]


def task_rows(source: str) -> dict[str, dict]:
    """`task_id -> row` for a source's tasks, for the question and file list."""
    if source == "jupyter-agent":
        from smol_ladder.jtasks import load_rows

        return {row["task_id"]: row for row in load_rows()}
    if source == "synthetic":
        from smol_ladder.jtasks import load_synthetic

        return {row["task_id"]: row for row in load_synthetic()}
    from smol_ladder.tasks import load_split

    rows: dict[str, dict] = {}
    for split in ("train", "test", "eval"):
        for row in load_split(split):
            rows.setdefault(row["task_id"], row)
    return rows


def write_program_command(code: str) -> str:
    """A single bash call that writes solution.py, as upstream's traces write code.

    A quoted heredoc, so nothing in the program -- `$`, backticks, the `EOF` of its own accord --
    is expanded by the shell. The command is never executed here, but the model is being taught to
    produce it, and an unquoted heredoc in the training set is a bug the model would faithfully
    reproduce on a table full of dollar signs.
    """
    return "cat > solution.py << 'SMOL_EOF'\n" + code.rstrip("\n") + "\nSMOL_EOF"


def fallback_turns(code: str, prediction: str) -> list[dict]:
    """The degenerate trajectory: write the program, submit the answer, stop.

    Deliberately short and with no exploration in it. Two facts are known and both are used: the
    program in `solution.py`, which reproduced the gold answer in the sealed offline pass, and the
    value it printed (`result.json["prediction"]`, the last line of that same run). What is *not*
    known is what the program printed before that last line, or what the agent saw while computing
    it, so this trajectory invents neither -- it writes the real program in one call, with the
    truthful empty result, then submits, then stops.

    That makes it a *contract* trajectory: it teaches write-program-then-submit-then-stop and the
    shape of a submission, and it teaches nothing about exploration. A set of these is a smaller,
    honest dataset. It is not a stand-in for real traces and must never be reported as one.
    """
    return [
        {"content": "I'll write a program that reads the input table and computes the answer."},
        shell_turn(write_program_command(code), EMPTY_OUTPUT, "call_write"),
        submission_turn(answer_file_command(prediction)) | {
            "results": [{"tool_call_id": "call_submit", "content": EMPTY_OUTPUT}]},
        closing_turn("The answer is written to /workdir/answer.txt."),
    ]


def collect_traces(data: Path | None = None, limit: int | None = None,
                   keys: dict[str, set[str]] | None = None) -> tuple[list[dict], dict]:
    """Our verified trials as upstream-format rows, firewall applied.

    Returns `(rows, dropped_by_column)`. `dropped` is reported rather than logged because the
    number that matters -- how many held-out tasks were refused -- is the one a training run has to
    be able to state.
    """
    data = data or DATA
    if keys is None:
        from smol_ladder.tasks import load_split

        keys = heldout_keys({"test": load_split("test"), "eval": load_split("eval")})
    dropped: dict[str, int] = {}
    rows: list[dict] = []
    for relative, source in SOURCES:
        root = data / relative
        if not root.exists():
            continue
        catalogue = task_rows(source)
        for task_id, directory in sorted(find_verified(root).items()):
            row = catalogue.get(task_id)
            if row is None:
                continue
            column = is_heldout(row, keys)
            if column:
                dropped[column] = dropped.get(column, 0) + 1
                continue
            result = json.loads((directory / "result.json").read_text())
            prediction = result.get("prediction") or ""
            if not prediction:
                # No prediction means the sealed run printed nothing, so there is no value to
                # submit and the trajectory would teach a model to submit an empty answer.
                continue
            transcript = directory / "transcript.jsonl"
            if transcript.exists():
                turns = parse_transcript(transcript)
            else:
                turns = fallback_turns(
                    (directory / "solution.py").read_text(errors="replace"), prediction)
            rows.append(bash_row(row["question"], row.get("files") or [], turns))
            rows[-1]["task_id"] = task_id
            if limit and len(rows) >= limit:
                return rows, dropped
    return rows, dropped