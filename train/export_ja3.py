"""Arm B: the ja3 transcript sweep as an SFT dataset, in upstream's `SmolDataEnvs-sft` format.

`data/runs/ja3/jupyter-agent-v3` is the first tree in this project that holds real multi-turn
trajectories at training quality (2,106 verified passes out of 3,880 L1 trials, model
`stealth/space-bunny-alpha`, each trial's whole conversation saved by `run_ladder.once`). The
target format is upstream's `messages` + `tools` with one tool named `bash` -- the decision and its
two reasons are in `docs/TRAINING.md` §2 and `train/format.py`, and this module does not reopen it.

    uv run python -m train.export_ja3 --out data/train
    uv run python -m train.export_ja3 --out data/train --dry-run

**A translation, and what it costs.** The sweep ran `run_shell` + `write_solution` against
`./input`, with a persistent `./solution.py` whose last printed line was graded offline and sealed.
Upstream's protocol is one `bash` tool against `/home/user/input`, submitting by writing
`/workdir/answer.txt`. Both are the same contract -- inspect, compute, submit, stop -- with
different spellings, so the trajectory is re-expressed rather than passed through:

- every command, `write_solution`'s `code` and every tool result are scrubbed and renamed;
- the submission is *appended*, as `echo -n "<graded prediction>" > /workdir/answer.txt`.

Appending is the only part that is not a renaming, and it has to be said plainly: the model's own
text almost never prints the graded value in the graded form (it prints `Answer: 141`, or prose,
or nothing at all), while the transcript's *last* assistant message routinely does. So the
trajectory is cut at the model's final answer and the submission turn the sweep's own sealed
`result.json["prediction"]` -- the value our grader gave 1.0 -- is appended after it. What the
model is taught is therefore "compute it, state it, submit it"; what it is not taught is the
sweep's own re-formatting of the value. That is the whole of the invention, and it is one turn.

## The four rules, each enforced here and each with a test

1. **The held-out firewall.** No task whose *table* or *question* belongs to SmolDataEnvs
   `test`/`eval`. The v3 pool is built that way already (`jtasks_v2.sde_overlap` on
   `dataset_key`, the bare Kaggle name, so a mirror under another owner cannot slip through), and
   the check is repeated at export time from the *live* split load rather than read off the pool's
   own tag -- a tag is a claim, the split is the fact. Expected drops: **0**. A non-zero drop is
   reported, and the export refuses to finish silently around it.
2. **No machine-specific strings.** `/home/<user>/...`, `/var/tmp/smol-ladder/...`, this
   worktree, the Kaggle cache layout, the local username, hostnames, and anything shaped like an
   API key or token. All of it becomes `./input/<file>` or a neutral path. The export *fails*
   rather than writing a row that still contains `/home/`, `/var/tmp/` or the username.
3. **The trajectory ends with the answer that was graded.** A transcript that is truncated,
   unparseable, or whose final assistant message does not state the graded prediction is dropped
   and counted, never repaired by invention.
4. **No hints, no gold.** These are L1 prompts only, so no rung's hint text can be in them, and
   the gold answer is checked against the system and user turns with `ladder.leaks`.

## What is *not* here

`solutions/jupyter-agent` (the v1 verified references) has no conversation -- 0 of 479 -- so it can
only yield the contract rows, and those are written to their own file
(`ja3_fallback.jsonl`) and labelled as such. They are a smaller, honest dataset that teaches the
contract and nothing about exploration, and a manifest that mixes them in with the traces without
saying so is worse than no manifest.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import socket
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from smol_ladder.jtasks import dataset_key, smoldataenvs_datasets, HELDOUT_SPLITS
from smol_ladder.tasks import DATA

from train.format import (answer_file_command, bash_row, heldout_keys, is_heldout,
                          shell_turn, write_jsonl)

#: Where the transcript sweep lives, and the pool its rows come from. The split name and the pool
#: are one thing, so a trajectory cannot be paired with a different task's question.
RUN_RELATIVE = "runs/ja3/jupyter-agent-v3"
SPLIT = "jupyter-agent-v3"
#: The v1 verified references. No conversations, so only the contract rows can come from them.
SOLUTIONS_RELATIVE = "solutions/jupyter-agent"
V1_SPLIT = "jupyter-agent"

#: Tokens per trajectory at which a row stops fitting a training context on its own. The number
#: the export reports is the measurement; this is only the threshold behind "share over 8,192".
LONG_CONTEXT = 8192

# ── rule 2: the scrub ─────────────────────────────────────────────────────────────────────────

#: The local user, resolved rather than hardcoded: a scrubber that only knew one person's name
#: would leave every other checkout's transcripts dirty. Only words of four letters or more, so
#: the shortest real usernames are covered without eating the English.
LOCAL_USER = os.environ.get("USER") or os.environ.get("LOGNAME") or ""
_USERNAME = re.compile(rf"\b{re.escape(LOCAL_USER)}\b") if len(LOCAL_USER) >= 4 else None
#: Anything that looks like this machine. `/home/<user>`, `/Users/<user>`, any `/var/tmp` path
#: (not only this project's -- other sessions' scratch shows up in these transcripts: `/var/tmp/
#: synprobe2/...` from a synthetic-probe run was found in 7 of the first 320 verified trials), this
#: checkout's worktrees, and the Kaggle cache layout the jail binds in.
_MACHINE = re.compile(
    r"/(?:home|Users)/(?!user\b)[^\s\"':,)\]]+"
    r"|/var/tmp/[^\s\"':,)\]]*"
    r"|/tmp/[^\s\"':,)\]]*smol[^\s\"':,)\]]*"
    r"|\b[a-z0-9_.-]+-wt-[a-z0-9_-]+"          # worktree directories
    r"|/mnt/[^\s\"':,)\]]+"
    r"|/data/[^\s\"':,)\]]+", re.I)
#: The Kaggle cache, spelled out so `kaggle/datasets/<owner>/<name>/versions/<n>/<file>` becomes a
#: neutral input file rather than a path into somebody's download cache.
_KAGGLE = re.compile(r"(?:\./)?(?:input/)?kaggle/datasets/[^\s\"':,)\]]+")
#: Secrets. OpenRouter keys are `sk-or-v1-...`, HuggingFace `hf_...`; the generic alternative is
#: what the sweep itself used to keep one from reaching the training text.
_SECRET = re.compile(
    r"\b(?:sk-or-v1-|sk-|hf_|ghp_|gho_|xoxb-|AKIA)[A-Za-z0-9_\-]{8,}"
    r"|\b[A-Za-z0-9_\-]*(?:api[_-]?key|token|secret)[A-Za-z0-9_\-]*\s*[=:]\s*[\"']?[A-Za-z0-9_\-]{12,}"
    r"|\bBearer\s+[A-Za-z0-9._\-]{12,}", re.I)
#: A machine's own name. Matched as a whole word and only when it is not a table's column, which
#: is why it is applied after the path rules.
_HOSTNAME = re.compile(rf"\b{re.escape(socket.gethostname())}\b")
#: A home directory that is not the sandbox's own. `/home/user` is where the upstream protocol
#: puts the tables (`/home/user/input`), so it is the intended path and is neither scrubbed nor
#: flagged; every other `/home/<name>` and `/Users/<name>` still is.
_HOME_PATH = re.compile(r"/(?:home|Users)/(?!user\b)")

#: Replacements, applied in order. The path rules collapse to `./input/<file>` for anything whose
#: last component is a real table name, and to a neutral `./input` for anything else -- an `ls -la`
#: of the input directory is the informative part of that output, not where the cache lives.
#: The username goes last, so a path has already been replaced by the time a bare `ls -la` owner
#: column is reached; that one cannot be neutralised to a path, so it becomes "user".
_SCRUBBERS = (_KAGGLE, _MACHINE, _SECRET, _HOSTNAME, _USERNAME)


def _neutral(text: str) -> str:
    """Replace every machine-specific string, in one pass over a list of rules."""
    for rule in _SCRUBBERS:
        if rule is None:
            continue
        text = rule.sub(lambda m: _replacement(m.group()), text)
    return text


def _replacement(match: str) -> str:
    """`/.../input/t.csv` -> `./input/t.csv`; anything else machine-specific -> `./input`."""
    tail = match.rsplit("/", 1)[-1]
    if tail and re.search(r"\.(csv|tsv|parquet|json|jsonl|xlsx|xls|sqlite|db|txt|tif|png|jpg|"
                          r"jpeg|avro|feather|pickle|pkl|h5|parq)$", tail, re.I):
        return f"./input/{tail}"
    if match == LOCAL_USER or match == socket.gethostname():
        # A bare word that names this machine: the `evan evan` owner columns of an `ls -la`. Not a
        # path, so not `./input` -- `user` keeps the column's shape without naming anybody.
        return "user"
    return "./input"


def scrub(text: str) -> str:
    """One string, scrubbed. Used for every message, every command and every tool result."""
    return _neutral(text or "")


def leaks_machine(text: str) -> list[str]:
    """What survived the scrub, by rule. An empty list is the export's precondition for writing."""
    found = []
    if _HOME_PATH.search(text):
        found.append("home path")
    if "/var/tmp/" in text or "/tmp/smol" in text:
        found.append("scratch or cache path")
    if _USERNAME is not None and _USERNAME.search(text):
        found.append("local username")
    if "kaggle/datasets" in text:
        found.append("kaggle cache layout")
    if _SECRET.search(text):
        found.append("api key or token")
    if _HOSTNAME.search(text):
        found.append("hostname")
    return found


def scrub_row(row: dict) -> tuple[dict, list[str]]:
    """Scrub every string in a `messages` + `tools` row, and report what survived."""
    text = json.dumps(row, ensure_ascii=False)
    clean = scrub(text)
    try:
        out = json.loads(clean)
    except ValueError:
        # A secret or a path inside a string that the scrub shortened into invalid JSON. Losing the
        # row is the honest outcome; the alternative is writing something that does not parse.
        return {}, ["row is no longer valid JSON after scrubbing"]
    return out, leaks_machine(clean)


# ── rule 1: the firewall ─────────────────────────────────────────────────────────────────────

def heldout_keys_for(splits=HELDOUT_SPLITS) -> dict[str, set[str]]:
    """Every held-out identity key, built from the splits themselves.

    `question` from `smol_ladder.tasks.load_split`, and `bucket_prefix` from the *Kaggle* table
    identity -- `jtasks.dataset_key`'s bare name -- because that is the key `jtasks_v2` fires the
    v3 pool on, and re-deriving it here from a different normaliser is how a firewall ends up
    checking something other than what the pool was built with.
    """
    from smol_ladder.tasks import load_split

    keys = heldout_keys({split: load_split(split) for split in splits})
    keys["kaggle_table"] = {dataset_key(s)
                            for s in smoldataenvs_datasets(tuple(splits))}
    return keys


def firewall_key(row: dict) -> dict:
    """The keys `heldout_keys_for` compares a jupyter-agent row on."""
    return {"question": row.get("question"), "bucket_prefix": dataset_key(
        row.get("kaggle_dataset_name") or "")}


def row_is_heldout(row: dict, keys: dict[str, set[str]]) -> str | None:
    """The key this task trips, or None. Same shape as `train.format.is_heldout` so the report
    reads the same way as every other export's."""
    if dataset_key(row.get("kaggle_dataset_name") or "") in keys.get("kaggle_table", set()):
        return "kaggle_table"
    merged = {**row, **firewall_key(row)}
    return is_heldout(merged, {k: v for k, v in keys.items() if k != "kaggle_table"})


# ── rule 3 and 4: the trajectory itself ──────────────────────────────────────────────────────

#: The L1 prompt's own headings, and any sentence of ours that could ride along in a tool result.
#: Checked against the system and user turns so "no hints" is a fact about the row rather than a
#: promise about which rung was run.
RUNG_MARKERS = ("A verified reference solution to this question is below",
                "Notes on the intended computation", "Schema of the input tables",
                "Columns used:", "Filters applied:", "Files read:")

_STARTS = ("the answer is", "answer:", "the final answer is", "answer is", "result:",
           "final answer:", "so the answer", "therefore the answer")
_VALUE = re.compile(r"(?<![A-Za-z0-9._+-])(\d[\d,]*(?:\.\d+)?|[A-Za-z][\w \-/&'.]{0,48})")


def same_value(left: str, right: str) -> bool:
    """Whether two surface forms are the same value.

    Through the dataset's own grader where possible, because that is the comparison the reward
    makes: "141" and "141.0" are the same answer, and "141" and "41" are not. Falls back to
    `ladder.normalise` for label answers, where the grader needs a reward_mode the row does not
    necessarily carry.
    """
    from smol_ladder.grade import grade
    from smol_ladder.ladder import normalise

    a, b = str(left).strip().strip("."), str(right).strip().strip(".")
    if not a or not b:
        return False
    if normalise(a) == normalise(b):
        return True
    try:
        return grade({"answer": a, "reward_mode": "numeric", "atol": 1e-4, "rtol": 1e-4}, b) >= 1.0
    except Exception:  # noqa: BLE001 - a label answer has no numeric grader
        return False


def final_answer_text(messages: list[dict], prediction: str) -> tuple[int, bool]:
    """Where the conversation's own final answer is, and whether it states the graded prediction.

    Returns `(index, ok)`. `index` is the last assistant message, which is where the trajectory is
    cut. `ok` is false when that message is missing, is a tool call with no text, or states
    something other than what the sealed pass graded -- a transcript that ends anywhere else is
    truncated or malformed, and is dropped rather than closed with a turn we invented.
    """
    last = None
    for index, message in enumerate(messages):
        if message.get("role") == "assistant":
            last = index
    if last is None:
        return -1, False
    text = str(messages[last].get("content") or "").strip()
    if not text:
        return last, False
    for pattern in _VALUE.finditer(text):
        if same_value(pattern.group(), prediction):
            return last, True
    return last, False


def turn_arguments(tool_call: dict) -> dict:
    """A tool call's arguments as a dict, whether the sweep stored them as a JSON string or a dict.

    `or_agent` logs what the endpoint returned, which is a string; upstream's parquet stores a dict.
    Anything else is `{}` rather than a guess.
    """
    arguments = tool_call.get("function", {}).get("arguments")
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except ValueError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def to_bash_turns(messages: list[dict], stop: int, prediction: str) -> list[dict]:
    """The conversation from `stop` onward, as upstream-format turns, plus the submission.

    Two renamings and nothing else: `run_shell` and `write_solution` both become `bash` (the
    model's own command text is kept verbatim inside the arguments), and every call id is
    rewritten so the tool results attach to the calls that produced them. `cd . &&` is dropped
    because it is the scratch the sweep ran in, not something the model chose to say.
    """
    turns: list[dict] = []
    # old call id -> the new one. The ids are rewritten because upstream's rows use its own
    # `call_*` names, and a tool result that matched its old id would otherwise attach to nothing.
    renamed: dict[str, str] = {}
    for index, message in enumerate(messages):
        role = message.get("role")
        if index > stop:
            break                       # the final answer is the last thing that is kept
        if role == "assistant":
            text = scrub(message.get("content") or "")
            calls = []
            for call in message.get("tool_calls") or []:
                # Scrub the argument *values*, not the JSON text around them: a rule that matches
                # across the escaped quotes of a re-serialised string can produce text that no
                # longer parses, and a command is only usable if it can be read back.
                arguments = {k: scrub(v if isinstance(v, str) else json.dumps(v))
                             for k, v in turn_arguments(call).items()}
                command = arguments.get("command") or ""
                command = re.sub(r"^(?:cd\s+\.\s*&&\s*|cd\s+\./\s*&&\s*)", "", command)
                call_id = f"call_{len(turns)}_{len(calls)}"
                calls.append({"id": call_id, "type": "function",
                              "function": {"name": "bash",
                                           "arguments": {"command": command}}})
                if call.get("id") is not None:
                    renamed[str(call["id"])] = call_id
            turn = {"content": text}
            if calls:
                turn["tool_calls"] = calls
                turn["results"] = [{"tool_call_id": call["id"], "content": "", "name": "bash"}
                                   for call in calls]
            elif not text:
                continue
            turns.append(turn)
        elif role == "tool":
            call_id = renamed.get(str(message.get("tool_call_id") or ""))
            if call_id is None:
                continue
            for turn in turns:
                for result in turn.get("results") or []:
                    if result["tool_call_id"] == call_id:
                        result["content"] = scrub(message.get("content") or "")
    turns.append({"content": "",
                  "tool_calls": [{"id": "call_submit", "type": "function",
                                  "function": {"name": "bash",
                                               "arguments": {
                                                   "command": answer_file_command(prediction)}}}],
                  "results": [{"tool_call_id": "call_submit",
                               "content": "(empty output, rc=0)", "name": "bash"}]})
    return turns


# ── the collection ───────────────────────────────────────────────────────────────────────────

def pool_rows(split: str) -> dict[str, dict]:
    """`task_id -> row` for a split's pool."""
    from smol_ladder import pool

    return {row["task_id"]: row for row in pool.load_pool(split)}


def verified_trials(root: Path, catalogue: dict[str, dict]) -> list[tuple[str, Path, dict]]:
    """`(task_id, directory, result)` for every trial that passed and left a conversation.

    `reward >= 1.0` is our own grader's verdict from the sealed offline pass; `agent_status ==
    "exit 0"` and a clean `verify_status` are what make that verdict mean something. A trial whose
    own solution did not re-run sealed has no prediction to grade and is not training data.
    """
    out = []
    for directory in sorted(root.glob("*/*")):
        result_path = directory / "result.json"
        if not result_path.exists():
            continue
        try:
            result = json.loads(result_path.read_text())
        except (OSError, ValueError):
            continue
        if result.get("reward", 0.0) < 1.0:
            continue
        if result.get("agent_status") != "exit 0":
            continue
        if result.get("verify_status", "exit 0") != "exit 0":
            continue
        task_id = result.get("task_id")
        if task_id and task_id in catalogue:
            out.append((task_id, directory, result))
    return out


def hints_absent(messages: list[dict], row: dict) -> list[str]:
    """Why these prompts may not be shipped, if they may not be.

    Two separate rules and they fail differently. A rung's marker text in a prompt is a hint
    whatever rung was nominally run. The gold answer is checked with `ladder.leaks`, which is the
    check every rung in this project is held to -- so a row is refused on exactly the standard a
    hint would be.

    That standard has a documented floor: `ladder.leaks` skips an answer under four normalised
    characters, because "3" occurs in every column dump and "yes"/"no" in prose. The floor is
    right for a *hint*, which is generated text and only needs to be free of real leaks. Here the
    text being checked is the task's own question, which every row of a dataset must contain, and
    which therefore cannot be held to a stricter rule than the ladder holds its hints to -- doing
    so would silently drop short-answer tasks from the training set and make arm B's population
    depend on the answer's length. So the floor is inherited, not widened, and it is *named*: the
    refusal reason says which check fired.
    """
    from smol_ladder.ladder import MIN_LEAK_LEN, leaks, normalise

    text = "\n".join(str(m.get("content") or "") for m in messages
                     if m.get("role") in {"system", "user"})
    reasons = [f"hint marker: {marker[:40]!r}" for marker in RUNG_MARKERS if marker in text]
    hits = leaks(row, text)
    if hits:
        reasons.append(f"gold answer in the prompt ({', '.join(hits)})")
    if len(normalise(row.get("answer", ""))) < MIN_LEAK_LEN:
        # Recorded rather than acted on, so the manifest can say how many rows were checked by a
        # rule that cannot fire. A reader can then decide whether that is a fact they can live
        # with, rather than discovering it from a training run.
        reasons.append("")
    return [reason for reason in reasons if reason]


def export_traces(data: Path | None = None, limit: int | None = None) -> tuple[list[dict], dict]:
    """The verified ja3 conversations as upstream-format rows. Returns `(rows, report)`."""
    data = data or DATA
    root = data / RUN_RELATIVE
    catalogue = pool_rows(SPLIT)
    keys = heldout_keys_for()
    rows: list[dict] = []
    dropped: Counter = Counter()
    families: Counter = Counter()
    short_answer = 0
    turns: list[int] = []
    task_ids: list[str] = []          # row i of the output belongs to task_ids[i]
    seen: set[str] = set()
    for task_id, directory, result in verified_trials(root, catalogue):
        if task_id in seen:
            continue
        seen.add(task_id)
        row = catalogue[task_id]
        column = row_is_heldout(row, keys)
        if column:
            dropped[f"heldout:{column}"] += 1
            continue
        prediction = str(result.get("prediction") or "").strip()
        conversation = directory / "transcript.json"
        if not prediction or not conversation.exists():
            dropped["no prediction or no transcript"] += 1
            continue
        try:
            messages = json.loads(conversation.read_text())
        except (OSError, ValueError):
            dropped["transcript does not parse"] += 1
            continue
        if not isinstance(messages, list) or len(messages) < 3:
            dropped["transcript truncated or malformed"] += 1
            continue
        stop, ok = final_answer_text(messages, prediction)
        if not ok:
            dropped["transcript does not end with the graded answer"] += 1
            continue
        reasons = hints_absent(messages, row)
        if reasons:
            dropped["; ".join(reasons)] += 1
            continue
        from smol_ladder.ladder import MIN_LEAK_LEN, normalise
        # Counted, not acted on: `ladder.leaks` cannot fire on an answer under four normalised
        # characters, and that floor is the right one for a hint but not for a question. The
        # number goes in the manifest so the fact is stated rather than discovered.
        short_answer += len(normalise(str(row.get("answer", "")))) < MIN_LEAK_LEN
        turns.append(sum(1 for m in messages[: stop + 1] if m.get("role") == "assistant"))
        families[row.get("op_family") or "other"] += 1
        exported = bash_row(row["question"], row.get("files") or [],
                            to_bash_turns(messages, stop, prediction))
        clean, leaked = scrub_row(exported)
        if leaked:
            dropped["scrub could not remove: " + ", ".join(leaked)] += 1
            continue
        rows.append(clean)
        task_ids.append(task_id)
        if limit and len(rows) >= limit:
            break
    return rows, {"dropped": dict(dropped), "task_ids": task_ids, "op_family": dict(families.most_common()),
                  "turns": turns, "verified_trials": len(seen),
                  "gold_answer_below_the_leak_floor": short_answer}


def export_fallbacks(data: Path | None = None, limit: int | None = None) -> tuple[list[dict], dict]:
    """The v1 verified references that are also in v3, as single-turn contract rows.

    `solutions/jupyter-agent` has 479 verified references and **zero** conversations, so all that
    can come out of it is the contract trajectory -- write the verified program, submit the graded
    answer, stop. Held-out *tables* are excluded here as they are in the trace path: the 383 tasks
    whose table is in a SmolDataEnvs held-out split are dropped by the same firewall.
    """
    from train.traces import fallback_turns, find_verified

    data = data or DATA
    catalogue = pool_rows(SPLIT)
    keys = heldout_keys_for()
    rows: list[dict] = []
    dropped: Counter = Counter()
    for task_id, directory in sorted(find_verified(data / SOLUTIONS_RELATIVE).items()):
        row = catalogue.get(task_id)
        if row is None:
            dropped["not in the v3 pool"] += 1
            continue
        column = row_is_heldout(row, keys)
        if column:
            dropped[f"heldout:{column}"] += 1
            continue
        result = json.loads((directory / "result.json").read_text())
        prediction = str(result.get("prediction") or "").strip()
        code = directory / "solution.py"
        if not prediction or not code.exists():
            dropped["no prediction or no program"] += 1
            continue
        exported = bash_row(row["question"], row.get("files") or [],
                            fallback_turns(code.read_text(errors="replace"), prediction))
        clean, leaked = scrub_row(exported)
        if leaked:
            dropped["scrub could not remove: " + ", ".join(leaked)] += 1
            continue
        rows.append(clean)
        if limit and len(rows) >= limit:
            break
    return rows, {"dropped": dict(dropped),
                  "solutions_checked": len(find_verified(data / SOLUTIONS_RELATIVE))}


# ── token statistics ─────────────────────────────────────────────────────────────────────────

def token_stats(rows: list[dict], model: str | None = None) -> dict:
    """Turns and tokens per trajectory, by the base model's tokenizer when one is available.

    The messages are rendered through the model's own chat template, because that is the text a
    training run actually tokenizes -- counting the JSON would report a different number from the
    one a `--max-length` has to be set against. Falls back to `chars / 4`, which is the usual
    approximation for English prose and code, and says so in the record.
    """
    from train.format import SCHEMA

    chats = []
    for row in rows:
        payload = [row[key] for key in SCHEMA if key in row]
        chats.append(json.dumps(payload, ensure_ascii=False))
    turns = []
    for row in rows:
        turns.append(sum(1 for m in row.get("messages") or []
                         if m.get("role") == "assistant"))
    tokens = None
    tokenizer = None
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model or "Qwen/Qwen3.5-0.8B",
                                                  local_files_only=True)
        texts = []
        for row in rows:
            messages = row.get("messages") or []
            tools = row.get("tools")
            if tools:
                texts.append(tokenizer.apply_chat_template(
                    messages, tools=tools, tokenize=False, enable_thinking=False))
            else:
                texts.append(tokenizer.apply_chat_template(
                    messages, tokenize=False, enable_thinking=False))
        tokens = [len(tokenizer.encode(t, add_special_tokens=False)) for t in texts]
    except Exception:  # noqa: BLE001 - no cached tokenizer, or no transformers here
        tokens = [max(1, len(c) // 4) for c in chats]
    if not tokens:
        return {"rows": 0}
    ordered = sorted(tokens)
    ordered_turns = sorted(turns) if turns else [0]

    def at(values: list[int], fraction: float) -> int:
        index = min(len(values) - 1, max(0, round(fraction * (len(values) - 1))))
        return values[index]

    return {"rows": len(rows),
            "tokens": {"min": ordered[0], "median": at(ordered, 0.5), "p90": at(ordered, 0.9),
                       "max": ordered[-1],
                       "share_over_8192": round(sum(1 for t in tokens if t > LONG_CONTEXT)
                                                / len(tokens), 4)},
            "turns": {"min": ordered_turns[0], "median": at(ordered_turns, 0.5),
                      "p90": at(ordered_turns, 0.9), "max": ordered_turns[-1]},
            "method": (f"{tokenizer.name_or_path} chat template, enable_thinking=False"
                       if tokenizer is not None else "chars/4 estimate")}


# ── the manifest ─────────────────────────────────────────────────────────────────────────────

def manifest(source: str, rows: list[dict], stats: dict, extra: dict, out: Path) -> Path:
    """The record a training run is quoted from, written beside the data and never into it.

    Every number a reader would otherwise have to take on trust is here: the paths, the counts, the
    firewall's drops *by key*, the scrub's rules, how many rows are real traces and how many are
    the contract fallback, and how the token statistics were measured.
    """
    record = {"source": source, "format": "FineEnvs/SmolDataEnvs-sft (messages + tools, one "
               "tool named bash)", "row": "messages + tools, nothing else",
              "rows": len(rows), "files": extra.get("files"),
              "firewall": {"keys": extra.get("firewall_keys"),
                           "dropped": stats.get("dropped", {}),
                           "expected_drops": 0},
              "scrub": {"rules": ["local absolute paths (/home/<user>, /Users/<user>, /var/tmp/"
                                  "smol-ladder, worktrees) -> ./input/<file>",
                                  "kaggle cache layout -> ./input/<file>",
                                  "local username, hostname -> removed",
                                  "api key / token patterns -> removed"],
                        "fails_the_export_if": ["/home/", "/var/tmp/", "the local username",
                                                "a hostname", "an api key or token"]},
              "token_stats": {k: v for k, v in stats.items() if k != "rows"},
              "generated_at": datetime.now(timezone.utc).isoformat(),
              "smol_ladder_commit": _commit(),
              **{k: v for k, v in extra.items() if k != "files"},
              }
    path = out / f"{source}.manifest.json"
    path.write_text(json.dumps(record, indent=1))
    return path


def _commit() -> str:
    try:
        return run_ladder_commit()
    except Exception:  # noqa: BLE001 - provenance is nice, not load-bearing
        return "unknown"


def run_ladder_commit() -> str:
    import subprocess

    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                          timeout=30).stdout.strip() or "unknown"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=DATA / "train")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--tokenizer", default=None,
                    help="the base model's tokenizer; the locally cached default is used")
    args = ap.parse_args()

    rows, report = export_traces(limit=args.limit)
    report.pop("task_ids", None)       # v2's join key; not part of v1's manifest
    stats = token_stats(rows, args.tokenizer)
    fallback_rows, fallback_report = export_fallbacks(limit=args.limit)
    fallback_stats = token_stats(fallback_rows, args.tokenizer)

    print(json.dumps({"traces": {**{k: v for k, v in report.items() if k != "turns"},
                                "turns": len(report.get("turns") or [])},
                      "fallbacks": fallback_report}, indent=1))
    print(json.dumps({"traces": stats, "fallbacks": fallback_stats}, indent=1))
    if args.dry_run:
        return
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(rows, out / "ja3_sft.jsonl")
    manifest("ja3_sft", rows, stats,
             {**report, "files": {"ja3_sft.jsonl": str(out / "ja3_sft.jsonl")},
              "firewall_keys": {k: len(v) for k, v in heldout_keys_for().items()},
              "op_family": report["op_family"]}, out)
    write_jsonl(fallback_rows, out / "ja3_fallback.jsonl")
    manifest("ja3_fallback", fallback_rows, fallback_stats,
             {**fallback_report, "files": {"ja3_fallback.jsonl": str(out / "ja3_fallback.jsonl")},
              "note": "single-turn CONTRACT rows from data/solutions/jupyter-agent (the v1 "
                      "verified references). No conversations exist for those trials, so there is "
                      "no exploration in them. They are NOT traces.",
              "firewall_keys": {k: len(v) for k, v in heldout_keys_for().items()}}, out)
    print(f"wrote {len(rows)} trace rows and {len(fallback_rows)} fallback rows to {out}")


if __name__ == "__main__":
    main()