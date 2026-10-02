"""The one training target format, and the held-out firewall.

**The format decision.** Every trajectory we export -- upstream's, ours, and the rung variants --
is written in upstream's `FineEnvs/SmolDataEnvs-sft` format: `messages` + `tools`, one tool named
`bash`, the answer submitted by `echo -n "<value>" > /workdir/answer.txt`, and the non-thinking
chat template. Two reasons, and the second is the one that actually binds:

1. Upstream's released 2B model was trained in it, and the released numbers (~0.28 -> ~0.40) are
   only comparable to runs in it. A different target format would make "did our training beat
   theirs" unanswerable, because the arms would differ in format as well as in data. That is the
   confound `docs/PLAN.md` already names as the risk that decides whether A vs B means anything.
2. TRL consumes `messages` + `tools` with no preprocessing at all (upstream says so verbatim:
   "there is no preprocessing step here and none is hiding in a helper: load, train, push"). So
   the format that is easiest to train is also the format that is comparable.

The cost, stated plainly: our traces are *ours* -- a different solver, `run_shell` +
`write_solution`, `./input` paths, a persistent `solution.py` -- so re-expressing them as bash
trajectories is a **translation**, not a passthrough. `traces.py` says exactly how, and drops the
translation it cannot do faithfully rather than emitting a subtly wrong one. Anything the
translation cannot preserve is a fact about the translation, and a training set full of quietly
inconsistent trajectories is worse than a smaller honest one.

**The firewall.** `FineEnvs/SmolDataEnvs-sft` ships 4,677 rows and the audit found test-split
questions inside it, so a naive `load_dataset(...)` trains on the held-out set and every later
number is meaningless. `is_heldout()` is the check, and it is enforced in code at export time
rather than left to a report nobody reads. See `heldout_keys()` for what it matches on.
"""

from __future__ import annotations

import json
from pathlib import Path

from smol_ladder.upstream import BASH_TOOL, bash_prompt

HELD_OUT_SPLITS = ("test", "eval")

# What one exported row looks like on disk. Kept as a dict of plain JSON types so a row can be
# written to JSONL and read back by TRL without a custom loading script.
SCHEMA = ("messages", "tools")

# The firewall's default keys. `bucket_prefix` is deliberately absent and deliberately available:
# see `heldout_keys` for the measurement that puts it behind a flag rather than in front of it.
DEFAULT_HELDOUT_COLUMNS = "task_id,question"


def heldout_keys(rows_by_split: dict[str, list[dict]]) -> dict[str, set[str]]:
    """Every key that identifies a held-out task, from the *task* dataset's own columns.

    Two keys by default, and a third that is available but off by default:

    - `task_id` catches an exact task appearing in the training split.
    - `question` catches the same *question text* asked over a *different* table, which is the case
      the other keys miss. Measured on `SmolDataEnvs-sft`: 4 such rows, e.g. "How many samples
      belong to each species in the dataset?" appears in the SFT set and in `test`. Four rows is
      not a rounding error when the metric is pass@1 on 250 tasks -- one leaked question is 0.4
      points.

    **`bucket_prefix` is available but is NOT the default, because it costs 65% of the dataset.**
    The coarse check is the right idea and the wrong threshold: 115 of the 170 held-out tables
    *also* appear in SmolDataEnvs' own `train` split. Upstream therefore treats a table as shared
    training material, not as held out, and a table-level firewall refuses 3,019 of 4,677 SFT
    rows -- including every row SmolDataEnvs itself would have trained on. Enabling it
    (`--heldout-columns bucket_prefix,...`) is defensible for the ladder, where knowing a table's
    shape is the signal being measured; it is not defensible for an arm-A replication, where it
    changes the dataset and the arm stops being arm A. 55 tables appear *only* in held-out
    splits, and no SFT row sits on any of them.

    So the default is `task_id,question`: every training row that is a held-out *task* or a
    held-out *question*, and nothing else. The owner can widen or narrow this from the CLI; see
    `docs/TRAINING.md`, which lists the decision as theirs.
    """
    keys: dict[str, set[str]] = {"task_id": set(), "bucket_prefix": set(), "question": set()}
    for split in HELD_OUT_SPLITS:
        for row in rows_by_split.get(split) or []:
            for column in keys:
                value = row.get(column)
                if column == "question" and value:
                    value = str(value).strip().lower()
                if value:
                    keys[column].add(str(value))
    return keys


def normalise_for_firewall(row: dict) -> dict:
    """A row's firewall keys, with `question` folded the same way `heldout_keys` folds it.

    Without this the question key would compare raw text against lowercased, stripped text and
    silently never match -- a firewall that is always on and never fires looks exactly like a
    clean dataset.
    """
    out = dict(row)
    if row.get("question"):
        out["question"] = str(row["question"]).strip().lower()
    return out


def is_heldout(row: dict, keys: dict[str, set[str]]) -> str | None:
    """The held-out key this row trips, or None. The column name is the reason.

    Returning the column rather than a bool is what makes the export report legible: "dropped 41
    rows, 12 by task_id and 29 by bucket_prefix" tells you which firewall did the work, and a bare
    count does not.
    """
    row = normalise_for_firewall(row)
    for column, values in keys.items():
        if column not in keys:
            continue
        value = row.get(column)
        if value and str(value) in values:
            return column
    return None


# ── the bash-agent row ────────────────────────────────────────────────────────


def bash_row(question: str, files: list[str], turns: list[dict], answer_format: str = "",
             source: str = "", task_id: str = "", rung: str = "") -> dict:
    """One SmolDataEnvs-sft-shaped row: `messages` + `tools`, and nothing else.

    `turns` is the transcript as assistant/tool pairs *after* the opening user turn, in upstream's
    own wire shape: assistant messages carry `tool_calls` with `function.arguments` as a **dict**
    (upstream's parquet stores it that way, not as a JSON string), and tool messages carry both
    `tool_call_id` and `name`.

    Upstream's rows are `messages` + `tools` only -- `task_id`, `difficulty_tier` and friends exist
    in the parquet but TRL never reads them, and carrying our own bookkeeping into the training
    text is how provenance ends up in the prompt. Provenance therefore goes in a sidecar index
    file (see `export_sft.py`), keyed by position, which keeps the training rows byte-identical in
    shape to upstream's.
    """
    messages = bash_prompt(question, files, answer_format)  # the rows' own template, one builder
    for turn in turns:
        assistant = {"role": "assistant", "content": turn.get("content") or ""}
        if turn.get("tool_calls"):
            assistant["tool_calls"] = turn["tool_calls"]
        elif not assistant["content"]:
            # An assistant turn with neither text nor a call is not a turn upstream's format can
            # express, and TRL's template would render an empty block. Dropping it keeps the
            # assistant/tool pairing contiguous, which is what apply_chat_template assumes.
            continue
        messages.append(assistant)
        for result in turn.get("results") or []:
            messages.append({"role": "tool", "tool_call_id": result["tool_call_id"],
                             "content": result.get("content") or "", "name": "bash"})
    return {"messages": messages, "tools": BASH_TOOL}


def submission_turn(command: str) -> dict:
    """The `echo -n "<value>" > /workdir/answer.txt` call, in upstream's wire shape.

    Upstream's published rows end with exactly this: a bash call writing the answer, its empty
    tool result, and one closing assistant sentence. Reproducing the shape byte-for-byte is the
    point of the translation -- the model has to learn to stop here, and the only evidence of
    "how to stop" in the training set is that every row stops here.
    """
    return {
        "tool_calls": [{"id": "call_submit", "type": "function", "function": {
            "name": "bash", "arguments": {"command": command}}}],
    }


def closing_turn(text: str) -> dict:
    """The one-sentence assistant message after submission. No tool call: the turn is over."""
    return {"content": text}


def shell_turn(command: str, output: str, call_id: str) -> dict:
    """One inspect/compute bash call and its result."""
    return {
        "tool_calls": [{"id": call_id, "type": "function", "function": {
            "name": "bash", "arguments": {"command": command}}}],
        "results": [{"tool_call_id": call_id, "content": output}],
    }


def answer_file_command(answer: str) -> str:
    """The submission command for a known-correct answer.

    `shlex.quote` rather than the naive f-string upstream's agents happened to emit: an answer
    containing a quote or a `$` would otherwise produce a command that does not reconstruct the
    answer, and the trajectory would teach the model a submission that silently corrupts its own
    result. Upstream's own guard (`upstream.looks_like_a_command`) exists precisely because this
    string is graded as text when it leaks out.
    """
    import shlex

    return f"echo -n {shlex.quote(answer)} > /workdir/answer.txt"


def write_jsonl(rows: list[dict], path: Path) -> None:
    """One JSON object per line, which is what `datasets.load_dataset("json", ...)` reads."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows