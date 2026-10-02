"""Arm B, version 2: the ja3 transcripts re-expressed in the conversation the harness builds.

    uv run --extra train python -m train.export_ja3_v2 --out data/train
    uv run python -m train.export_ja3_v2 --dry-run --no-tokens

`ja3_sft.jsonl` (v1) kept the sweep's own conventions -- a `./input` system prompt, a hard-wrapped
user template, `./input` paths, `""` and `--- stderr ---` tool results, 20,000-character outputs and
a final appended submission turn. The evaluation harness (`--agent bash`) builds upstream's
conversation byte for byte, so a model trained on v1 is evaluated in a format it was not trained in
and A-versus-B measures format, not data. v2 writes arm B in the harness's format and then PROVES it:
every row is replayed through `or_agent.bash_loop` (`train/replay.py`), and a row the harness would
not rebuild exactly is refused.

## One code path with the harness

- the system turn is `upstream.BASH_SYSTEM`, the user turn is `ladder.prompt_for(row, split, "L1",
  "bash")` -- the function `run_ladder` calls for the same task -- so the file list (`row["files"]`)
  and the answer-format line (`upstream.answer_format_of`) are derived exactly as in evaluation;
- each tool result is `or_agent.format_tool_result` applied to the recorded output.

**The answer-format line.** Upstream's per-task line ("Express the value as a percentage ...",
"Answer as a comma-separated list ...") exists only for SmolDataEnvs tasks whose `reward_mode` is
`list`, `list_csv` or `flexible` (500 of 5,000 train tasks); every `numeric`, `exact_short` and
`exact_bool` task (4,442) has none, and the line is the task's own instruction text, not a function
of the mode. The ja3 tasks are `numeric` (5,660), `exact_short` (1,658) and `exact_bool` (200), so
all of them map to the empty line, which `answer_format_of` returns for a row with no `instruction`.

## What the translation does to a recorded trajectory

The sweep's tools loop (`or_agent.solve_loop`) offered `run_shell` and `write_solution`, in a
scratch directory holding `./input` (a symlink) and `solution.py`, and recorded results as
`stdout + "\\n--- stderr ---\\n" + stderr` (tail-cut at 20,000 characters), without exit codes.

1. `run_shell(command)` -> `bash(command)`. Text kept verbatim except paths: `./input/x`, `input/x`,
   a bare `./input`, `'input'` as a call argument and `cd|ls|... input` become `/home/user/input`;
   the task's own scratch directory becomes `/workdir`. `cd . &&` (the model's own text) is kept.
2. `write_solution(code)` -> `bash("cat > /workdir/solution.py << 'EOF' ... EOF")`, the model's code
   verbatim (paths mapped), result `(empty output, rc=0)` in place of `written /app/solution.py`.
3. Tool results: the stderr divider is removed (stderr is appended after stdout: the recording
   did not keep the interleaving), an empty or whitespace-only result becomes `(empty output,
   rc=0)` -- the exit status was NOT recorded, and 1,943 of the 1,952 empty results in the upstream
   rows are rc=0, so rc=0 is the convention the data supports -- and the harness's own cut applies
   (first 8,000 characters + marker).
4. The model's last message (text only, and it must state the graded value) becomes the closing
   assistant turn. Before it comes ONE submission turn: `echo -n "<graded prediction>" >
   /workdir/answer.txt` (upstream's most frequent submission idiom: 2,433 of 4,672 final
   submissions against 1,824 for `printf %s`), answered `(empty output, rc=0)` as the harness
   answers it.

Everything in 1-3 that is not a path mapping, and the whole of 4, is counted in the manifest as
`invented`. The closing message is the model's own text; no sentence is written by this module.

## Refusals

A row is refused, never repaired, when the recording environment shows through in a way the
translation cannot make true in the new one: a tool other than the two above, a recorded timeout
(150 s deadline, not the harness's 180), an output cut at 20,000 characters (its head is lost),
a machine path / username / hostname / symlinked-input listing in a command or output, a final
message that does not end the conversation, a submitted value that appears nowhere before the
submission, a multi-line prediction -- plus v1's own firewall and hint checks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import shutil
import subprocess
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from smol_ladder import ladder as L
from smol_ladder import pool
from smol_ladder.or_agent import EMPTY_OUTPUT, truncate_output
from smol_ladder.tasks import DATA
from smol_ladder.upstream import BASH_SYSTEM, BASH_TOOL, INPUT_DIR, WORKDIR
from train import export_ja3 as X
from train.format import write_jsonl
from train.replay import assert_replays

NAME = "ja3_sft_v2"
STDERR_DIVIDER = "\n--- stderr ---\n"
RECORDED_TAIL = 20_000          # `or_agent.run_command`'s `limit`: the recording kept the LAST 20,000
ANSWER_PATH = f"{WORKDIR}/answer.txt"
SOLUTION_PATH = f"{WORKDIR}/solution.py"
RECORDED_TOOLS = ("run_shell", "write_solution")

#: The kinds of text this module writes that the model did not produce. Counted per row.
INVENTED = (
    "submission_command",        # echo -n "<prediction>" > /workdir/answer.txt, one per row
    "submission_result",        # (empty output, rc=0) for it, as the harness answers `echo -n`
    "write_solution_wrapper",   # `cat > /workdir/solution.py << 'EOF'` framing around the model's code
    "write_solution_result",    # (empty output, rc=0) in place of "written /app/solution.py"
    "empty_result_rc0",         # (empty output, rc=0): the recording kept no exit status
    "stderr_appended",          # stderr placed after stdout; the recording lost the interleaving
    "input_listing",            # the `ls -la ./input` symlink line -> the directory listing
    "cwd_input_entry_dropped",  # the `input` entry of a bare `ls` in the recording's cwd, removed
)
#: Edits to text the model DID produce. Counted too, because they are still the translator's.
REWRITTEN = (
    "input_path",               # ./input, input/ ... -> /home/user/input (commands, code, outputs)
    "workdir_path",             # the trial's scratch directory -> /workdir
    "output_cut_8000",          # 8,001-19,999 recorded characters -> head 8,000 + the marker
    "python_library_path",      # the recording venv's frames -> /usr/local/lib/python3.N (the jail's)
)

# ── paths ─────────────────────────────────────────────────────────────────────────────────────

#: `./input/x`, `input/x`: unambiguous wherever they appear (command, code, prose or output).
_INPUT_SLASH = re.compile(r"(?<![\w/.\-])(?:\./)?input(?=/)")
#: A bare `./input` (`ls -la ./input && ...`, `'./input'`).
_INPUT_DOT = re.compile(r"(?<![\w/.\-])\./input(?![\w/.\-])")
#: `listdir('input')`, `Path("input")`, `INPUT = 'input'`; not `df['input']` or `x == 'input'`.
_INPUT_ARG = re.compile(r"(?P<pre>(?<![=!<>])[(=]\s*)(?P<q>['\"])input(?P=q)")
#: `cd input`, `ls -la input`, `find input -name ...`: a shell word, so commands only.
#: `os.path.join(here, "input", "x.csv")`: a program that finds its tables relative to its own
#: directory. The recording's scratch directory held an `input` link; /workdir does not, so the
#: program would not run there and its recorded output would be false.
_INPUT_BESIDE_SCRIPT = re.compile(r"(?:,|/)\s*['\"](?:\./)?input['\"]")
_INPUT_WORD = re.compile(r"(?P<verb>\b(?:cd|ls|du|tree|find|stat|file)\b(?:\s+-\S+)*\s+)input"
                         r"(?=\s|$|[;&|)])", re.M)


#: Where the recording interpreter's libraries lived, and where `run_ladder.jail_bash` shows the same
#: ones (`/usr/local/lib/python3.N/...`: the tools venv's site-packages and the uv CPython's stdlib
#: are bound there). A traceback frame is the one place a library path reaches the model.
_VENV_PACKAGES = re.compile(r"/home/[\w.\-]+/Documents/[\w.\-]+/\.venv/lib/(python3\.\d+)/site-packages")
_UV_STDLIB = re.compile(r"/home/[\w.\-]+/\.local/share/uv/python/cpython-[\w.\-]+/lib/(python3\.\d+)")


def world_text(text: str, task_id: str, command: bool) -> tuple[str, Counter]:
    """One recorded string, spelled as the harness's sandbox has it. Returns `(text, counts)`."""
    n: Counter = Counter()
    scratch = re.compile(rf"/var/tmp/smol-ladder/scratch/trials/{re.escape(task_id)}/L1(?:/s\d+)?")
    text, hits = scratch.subn(WORKDIR, text)
    n["workdir_path"] += hits
    text, hits = re.subn(rf"/var/tmp/smol-ladder/kaggle/tasks/{re.escape(task_id)}(?![\w.\-])",
                         INPUT_DIR, text)
    n["input_path"] += hits
    for rule, target in ((_VENV_PACKAGES, r"/usr/local/lib/\1/site-packages"),
                         (_UV_STDLIB, r"/usr/local/lib/\1")):
        text, hits = rule.subn(target, text)
        n["python_library_path"] += hits
    rules = [_INPUT_SLASH, _INPUT_DOT] + ([_INPUT_ARG, _INPUT_WORD] if command else [])
    for rule in rules:
        if rule is _INPUT_ARG:
            text, hits = rule.subn(lambda m: f"{m['pre']}{m['q']}{INPUT_DIR}{m['q']}", text)
        elif rule is _INPUT_WORD:
            text, hits = rule.subn(lambda m: m["verb"] + INPUT_DIR, text)
        else:
            text, hits = rule.subn(INPUT_DIR, text)
        n["input_path"] += hits
    return text, n


#: What would make a translated string false in the harness's sandbox: the recording machine, or
#: the recording directory's own layout (the symlink `ls -la` shows; a listing of the scratch dir).
_SYMLINKED_INPUT = re.compile(r"\binput -> ")
_SCRATCH_LISTING = re.compile(r"(?m)^input$")
#: A bare `ls` in the recording's working directory listed `input` (the symlink) beside the model's
#: own files; the sandbox's /workdir has no such entry. Dropping that one line is what `ls` would
#: print there. Only for a bare `ls` (no operands), where the listing is exactly the directory.
_BARE_LS = re.compile(r"(?<![\w/.\-])ls(?=\s*(?:$|[;&|)]))", re.M)
_INPUT_ENTRY = re.compile(r"(?m)^input(?:\n|\Z)")


def environment_leaks(text: str, listing: bool) -> list[str]:
    found = X.leaks_machine(text)
    if _SYMLINKED_INPUT.search(text):
        found.append("symlinked input listing")
    if listing and _SCRATCH_LISTING.search(text):
        found.append("scratch directory listing")
    return found


#: `ls -la ./input` on the recording machine printed the symlink itself, `lrwxrwxrwx 1 <user> <user>
#: 64 Oct  1 17:17 ./input -> /var/tmp/smol-ladder/kaggle/tasks/<task>`: in 1,936 of the 2,110
#: verified trials, because it is the first command nearly every transcript runs. That line names the
#: machine and the owner, and the harness's sandbox has a real directory there, so it prints a
#: listing instead. It is replaced by the listing of the task's actual input directory, in the format
#: of upstream's own `ls -la /home/user/input` results (mode, owner and date copied from them).
_SYMLINK_LINE = re.compile(r"(?m)^lrwxrwxrwx +\d+ +\S+ +\S+ +\d+ +\w{3} +\d+ +[\d:]{4,5} +"
                           r"(?:\./)?input -> /var/tmp/smol-ladder/kaggle/tasks/\S+$")
_LS_LA_INPUT = re.compile(rf"\bls -la {re.escape(INPUT_DIR)}(?![\w/.\-])")   # as already spelled
LISTING_DATE = "Aug 30 23:00"


def input_listing(files: list[tuple[str, int]]) -> str:
    """`ls -la /home/user/input` for a directory holding `files` (name, bytes), upstream's layout."""
    entries = [("d", "user", "user", 4096, "."), ("d", "user", "user", 4096, "..")]
    entries += [("-", "root", "root", size, name) for name, size in sorted(files)]
    width = max(len(str(e[3])) for e in entries)
    total = sum(-(-size // 4096) * 4 for _, size in files)
    lines = [f"total {total}"]
    for kind, owner, group, size, name in entries:
        mode = "drwxr-xr-x" if kind == "d" else "-rw-r--r--"
        links = 2 if name == "." else 3 if name == ".." else 1
        lines.append(f"{mode} {links} {owner} {group} {size:>{width}} {LISTING_DATE} {name}")
    return "\n".join(lines)


# ── the translation ───────────────────────────────────────────────────────────────────────────

#: Paths the harness's sandbox has and the recording machine did not. A command or output that
#: already names them ran (or failed) in a world where they did not exist -- `cd /home/user` that
#: silently failed, a FileNotFoundError for `/home/user/input/x.csv` -- and would not do the same
#: in the sandbox, where they do.
_SANDBOX_ONLY = re.compile(r"/home/user|/workdir")


class Refused(Exception):
    """A row the translation will not write, with the reason the manifest counts it under."""

    detail: str = ""


def call_id(task_id: str, n: int) -> str:
    """`chatcmpl-tool-<16 hex>`, the shape upstream's ids mostly have, stable per row and position."""
    return "chatcmpl-tool-" + hashlib.sha1(f"{task_id}:{n}".encode()).hexdigest()[:16]


def heredoc(path: str, code: str) -> str:
    delimiter, n = "EOF", 0
    lines = set(code.split("\n"))
    while delimiter in lines:
        n += 1
        delimiter = f"EOF_{n}"
    return f"cat > {path} << '{delimiter}'\n{code.rstrip(chr(10))}\n{delimiter}"


def submission_command(value: str) -> tuple[str, str]:
    """`echo -n "<value>" > /workdir/answer.txt` and which quoting it needed.

    Double quotes, as the prompt's own example and the commonest rows have them. A value that a
    double-quoted string would change (`"`, `$`, a backtick, `\\`, `!`) is single-quoted instead,
    which is the other form upstream's rows use. A newline or an `echo` flag cannot be written
    faithfully by `echo -n` at all, and the row is refused.
    """
    if "\n" in value or "\r" in value:
        raise Refused("prediction is multi-line")
    if re.fullmatch(r"-[neE]+", value):
        raise Refused("prediction is an echo flag")
    if any(c in value for c in '"$`\\!'):
        return f"echo -n {shlex.quote(value)} > {ANSWER_PATH}", "single"
    return f'echo -n "{value}" > {ANSWER_PATH}', "double"


def merged_result(recorded: str) -> tuple[str, bool]:
    """A recorded `run_command` string as one combined stream, and whether stderr was folded in."""
    if recorded.startswith("[timed out after"):
        raise Refused("a command timed out at recording time (150 s deadline, not the harness's 180)")
    if len(recorded) == RECORDED_TAIL:
        raise Refused("a tool output was tail-cut at recording time (20,000 characters; the head "
                      "the harness keeps is lost)")
    if STDERR_DIVIDER in recorded:
        return recorded.replace(STDERR_DIVIDER, "", 1), True
    return recorded, False


def translate(messages: list[dict], task_id: str, prediction: str,
              files=lambda: None) -> tuple[list[dict], Counter]:
    """The recorded conversation after the user turn -> bash-format messages, plus counts.

    `files()` returns the input directory's `[(name, bytes)]` (or None), for the one result that
    has to be re-made rather than re-spelled (`input_listing`). Raises `Refused`. `messages[-1]`
    must be the model's text-only final answer.
    """
    n: Counter = Counter()
    last = messages[-1]
    if last.get("role") != "assistant" or last.get("tool_calls") or not (last.get("content") or "").strip():
        raise Refused("transcript does not end with a text-only final answer")
    out: list[dict] = []
    pending: list[tuple[str, str, str]] = []     # (new id, tool, command) awaiting a tool message
    ids = 0
    visible: list[str] = []                      # what the model has said and what it has been shown

    def check(text: str, where: str, listing: bool) -> None:
        found = environment_leaks(text, listing)
        if found:
            refusal = Refused(f"recording environment in {where}: {found[0]}")
            refusal.detail = text[:300]
            raise refusal

    for message in messages[2:-1]:
        role = message.get("role")
        if role == "assistant":
            if pending:
                raise Refused("a tool call has no result")
            content, c = world_text(message.get("content") or "", task_id, False)
            n.update(c)
            check(content, "an assistant message", False)
            visible.append(content)
            turn: dict = {"role": "assistant", "content": content}
            calls = []
            for call in message.get("tool_calls") or []:
                name = (call.get("function") or {}).get("name")
                if name not in RECORDED_TOOLS:
                    raise Refused(f"unsupported tool call: {name}")
                try:
                    args = json.loads(call["function"].get("arguments") or "{}")
                except ValueError:
                    raise Refused("tool call arguments are not JSON") from None
                key = "command" if name == "run_shell" else "code"
                value = args.get(key) if isinstance(args, dict) else None
                if not isinstance(value, str) or not value.strip():
                    raise Refused("tool call without a command or code")
                if _SANDBOX_ONLY.search(value):
                    raise Refused("a command names /home/user or /workdir, which the "
                                  "recording machine did not have")
                value, c = world_text(value, task_id, True)
                n.update(c)
                if _INPUT_BESIDE_SCRIPT.search(value) or (
                        "__file__" in value
                        and re.search(r"['\"/]input\b", value.replace(INPUT_DIR, ""))):
                    raise Refused("a command finds `input` relative to the script's own directory")
                check(value, "a command", False)
                if name == "write_solution":
                    value = heredoc(SOLUTION_PATH, value)
                    n["write_solution_wrapper"] += 1
                new = call_id(task_id, ids)
                ids += 1
                calls.append({"id": new, "type": "function",
                              "function": {"name": "bash", "arguments": {"command": value}}})
                pending.append((new, name, value))
            if calls:
                turn["tool_calls"] = calls
            elif not content.strip():
                continue                                 # nothing to say, nothing to call
            out.append(turn)
        elif role == "tool":
            if not pending:
                raise Refused("a tool result matches no call")
            new, name, command = pending.pop(0)
            recorded = message.get("content") or ""
            if name == "write_solution":
                if recorded != "written /app/solution.py":
                    raise Refused("unexpected write_solution result")
                text = EMPTY_OUTPUT.format(rc=0)
                n["write_solution_result"] += 1
            else:
                if _SANDBOX_ONLY.search(recorded):
                    raise Refused("a tool output names /home/user or /workdir, which the "
                                  "recording machine did not have")
                merged, folded = merged_result(recorded)
                n["stderr_appended"] += folded
                if _SYMLINK_LINE.search(merged) and _LS_LA_INPUT.search(command):
                    listed = files()
                    if listed is None:
                        raise Refused("the input directory is not on this machine (for ls -la)")
                    merged, hits = _SYMLINK_LINE.subn(lambda _m: input_listing(listed), merged)
                    n["input_listing"] += hits
                text, c = world_text(merged, task_id, False)
                n.update(c)
                if _BARE_LS.search(command) and _SCRATCH_LISTING.search(text):
                    text, hits = _INPUT_ENTRY.subn("", text)
                    n["cwd_input_entry_dropped"] += hits
                check(text, "a tool output", True)
                if not text.strip():
                    text = EMPTY_OUTPUT.format(rc=0)
                    n["empty_result_rc0"] += 1
                else:
                    text, cut = truncate_output(text)
                    n["output_cut_8000"] += cut
                visible.append(text)
            out.append({"role": "tool", "tool_call_id": new, "content": text, "name": "bash"})
        else:
            raise Refused(f"unexpected role {role!r}")
    if pending:
        raise Refused("a tool call has no result")

    closing, c = world_text(last["content"], task_id, False)
    n.update(c)
    check(closing, "the final message", False)
    token = re.compile(rf"(?<![\w.,\-]){re.escape(prediction)}(?![\w])")
    if not any(token.search(text) for text in visible):
        raise Refused("the submitted value appears nowhere before the submission")
    command, quoting = submission_command(prediction)
    submit_id = call_id(task_id, ids)
    out.append({"role": "assistant", "content": "", "tool_calls": [{
        "id": submit_id, "type": "function",
        "function": {"name": "bash", "arguments": {"command": command}}}]})
    out.append({"role": "tool", "tool_call_id": submit_id, "content": EMPTY_OUTPUT.format(rc=0),
                "name": "bash"})
    out.append({"role": "assistant", "content": closing})
    n["submission_command"] += 1
    n["submission_result"] += 1
    n["quoting_" + quoting] += 1
    return out, n


def check_submission(command: str, prediction: str) -> bool:
    """Run the submission command for real, into a temp file, and compare bytes with the value."""
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "answer.txt"
        subprocess.run(["bash", "-c", command.replace(ANSWER_PATH, str(target))], check=True,
                       capture_output=True, timeout=30)
        return target.read_bytes() == prediction.encode()


def input_files_of(row: dict):
    """A thunk for `translate`: the task's input directory as `[(name, bytes)]`, or None."""
    def files():
        try:
            directory = pool.input_dir(row, fetch=False)      # never a download
            return [(p.name, p.stat().st_size) for p in directory.iterdir()]
        except Exception:  # noqa: BLE001 - not on this machine: the caller refuses the row
            return None
    return files


def build_row(row: dict, messages: list[dict], prediction: str) -> tuple[dict, Counter]:
    """The v2 row for one task: the harness's own opening turns, then the translated trajectory."""
    if not row.get("files"):
        # `prompt_for` would fall back to listing the directory, and for this split that lookup can
        # reach for a download; a task with no file list is not worth one.
        raise Refused("the task lists no files (the harness would list the input directory)")
    user = L.prompt_for(row, X.SPLIT, "L1", "bash")
    tail, counts = translate(messages, row["task_id"], prediction, input_files_of(row))
    return {"messages": [{"role": "system", "content": BASH_SYSTEM},
                         {"role": "user", "content": user}] + tail, "tools": BASH_TOOL}, counts


# ── the export ────────────────────────────────────────────────────────────────────────────────

def export(data: Path | None = None, limit: int | None = None) -> tuple[list[dict], list[dict], dict]:
    """`(rows, index, report)`. `index[i]` describes `rows[i]`; the order is v1's."""
    data = data or DATA
    catalogue = X.pool_rows(X.SPLIT)
    keys = X.heldout_keys_for()
    rows: list[dict] = []
    index: list[dict] = []
    refused: Counter = Counter()
    refused_examples: dict[str, str] = {}
    invented: Counter = Counter()
    invented_rows: Counter = Counter()
    invented_example: dict[str, str] = {}
    families: Counter = Counter()
    submitted_quoting: Counter = Counter()
    files_from_directory = 0
    seen: set[str] = set()
    trials = X.verified_trials(data / X.RUN_RELATIVE, catalogue)

    def refuse(task_id: str, reason: str) -> None:
        refused[reason] += 1
        refused_examples.setdefault(reason, task_id)

    for task_id, directory, result in trials:
        if task_id in seen:
            continue
        seen.add(task_id)
        row = catalogue[task_id]
        column = X.row_is_heldout(row, keys)
        if column:
            refuse(task_id, f"heldout:{column}")
            continue
        prediction = str(result.get("prediction") or "").strip()
        conversation = directory / "transcript.json"
        if not prediction or not conversation.exists():
            refuse(task_id, "no prediction or no transcript")
            continue
        try:
            messages = json.loads(conversation.read_text())
        except (OSError, ValueError):
            refuse(task_id, "transcript does not parse")
            continue
        if not isinstance(messages, list) or len(messages) < 4:
            refuse(task_id, "transcript truncated or malformed")
            continue
        stop, ok = X.final_answer_text(messages, prediction)
        if not ok or stop != len(messages) - 1:
            refuse(task_id, "transcript does not end with the graded answer")
            continue
        try:
            built, counts = build_row(row, messages, prediction)
        except Refused as e:
            refuse(task_id, str(e))
            continue
        reasons = X.hints_absent(built["messages"], row)
        if reasons:
            refuse(task_id, "; ".join(reasons))
            continue
        if not check_submission(built["messages"][-3]["tool_calls"][0]["function"]["arguments"]
                                ["command"], prediction):
            refuse(task_id, "submission command does not reproduce the value")
            continue
        leaked = X.leaks_machine(json.dumps(built, ensure_ascii=False))
        if leaked:
            refuse(task_id, "scrub could not remove: " + ", ".join(leaked))
            continue
        try:
            assert_replays(built["messages"], strict=True)
        except AssertionError as e:
            refuse(task_id, "does not replay through the harness: " + str(e)[:80])
            continue
        files_from_directory += not row.get("files")
        families[row.get("op_family") or "other"] += 1
        for kind in (*INVENTED, *REWRITTEN):
            if counts[kind]:
                invented[kind] += counts[kind]
                invented_rows[kind] += 1
                invented_example.setdefault(kind, task_id)
        submitted_quoting[("single" if counts["quoting_single"] else "double")] += 1
        rows.append(built)
        index.append({"task_id": task_id, "op_family": row.get("op_family") or "other",
                      "assistant_turns": sum(m["role"] == "assistant" for m in built["messages"]),
                      "counts": {k: v for k, v in sorted(counts.items()) if v}})
        if limit and len(rows) >= limit:
            break
    report = {"verified_trials": len(seen), "refused": dict(refused.most_common()),
              "refused_example_task": refused_examples, "op_family": dict(families.most_common()),
              "invented": {k: {"count": invented[k], "rows": invented_rows[k],
                               "example_task": invented_example.get(k)}
                           for k in (*INVENTED, *REWRITTEN)},
              "submission_quoting": dict(submitted_quoting),
              "rows_whose_file_list_came_from_the_directory": files_from_directory}
    return rows, index, report


# ── v1 join, rendering and tokens ────────────────────────────────────────────────────────────

def v1_positions(index: list[dict], v1_path: Path) -> dict:
    """Where each v2 row sits in `ja3_sft.jsonl`, by re-deriving v1's order from the same trials.

    v1 and v2 walk `verified_trials` in the same order and v2 only ever drops rows, so the order is
    shared; this re-runs v1's export to recover its task ids (v1 rows carry none) and checks the
    count against the file on disk. Returns `{task_id: position}` and what was verified.
    """
    from train.format import read_jsonl

    on_disk = read_jsonl(v1_path) if v1_path.exists() else []
    rows, report = X.export_traces()
    ids = report["task_ids"]
    # The system and user turns are what v2 changes, so the join is checked on everything after
    # them: the model's turns and the tool results, which v1's exporter has not changed since.
    same = sum(1 for a, b in zip(rows, on_disk) if a["messages"][2:] == b["messages"][2:])
    trusted = len(ids) == len(on_disk) and same >= 0.99 * len(on_disk)
    return {"positions": {t: i for i, t in enumerate(ids)} if trusted else {},
            "v1_rows_on_disk": len(on_disk), "v1_rows_rederived": len(rows),
            "v1_rows_with_identical_turns_after_the_user_message": same,
            "v1_row_in_index_is_null_unless": "the rederived order matches the file on >= 99% of rows"}


def token_report(rows: list[dict], max_length: int = 8192, a_path: Path | None = None) -> dict:
    """Rendered-token statistics with the Qwen3.5-2B tokenizer, and the audit `train.render` runs.

    The rendering is `sft_lora.prepare` through the chat template with tools and
    `enable_thinking=False`, the text TRL trains on. `A+B` is arm A's train split plus these rows.
    """
    from train.render import (MODEL, assert_tool_calls_rendered, load_tokenizer, render_row,
                              tokenizer_renderer)
    from train.sft_lora import prepare
    from train.format import read_jsonl

    tokenizer = load_tokenizer(MODEL)
    template = tokenizer_renderer(tokenizer)

    def counts(source: list[dict]) -> list[int]:
        return [len(tokenizer(render_row(prepare([r], "bash")[0], template),
                              add_special_tokens=False)["input_ids"]) for r in source]

    b = counts(rows)
    audit = assert_tool_calls_rendered(rows, template, prepare(rows, "bash"), label=NAME)
    audit.pop("bad_rows", None)
    ordered = sorted(b)

    def at(q: float) -> int:
        return ordered[min(len(ordered) - 1, round(q * (len(ordered) - 1)))]

    out = {"tokenizer": MODEL, "template": "chat template with tools, enable_thinking=False",
           "max_length": max_length,
           "B": {"rows": len(b), "min": ordered[0], "median": at(.5), "p90": at(.9), "max": ordered[-1],
                 "share_over_max_length": round(sum(t > max_length for t in b) / len(b), 4),
                 "raw_tokens": sum(b), "trained_tokens": sum(min(t, max_length) for t in b)},
           "render_audit": audit}
    if a_path and a_path.exists():
        a = counts(read_jsonl(a_path))
        out["A"] = {"rows": len(a), "raw_tokens": sum(a),
                    "trained_tokens": sum(min(t, max_length) for t in a)}
        out["AB"] = {"rows": len(a) + len(b), "raw_tokens": sum(a) + sum(b),
                     "trained_tokens": out["A"]["trained_tokens"] + out["B"]["trained_tokens"]}
    return out


def upstream_idioms(a_dir: Path) -> dict | None:
    """What the submission and the ending look like in upstream's rows (train + val), measured.

    The choices this module makes -- `echo -n "<value>"`, a closing assistant message -- are the
    commonest ones in the 4,673 rows, and the manifest records the counts they were chosen by.
    """
    from train.format import read_jsonl

    files = [a_dir / "train.jsonl", a_dir / "val.jsonl"]
    if not all(f.exists() for f in files):
        return None
    final: Counter = Counter()
    closing = rows = 0
    for f in files:
        for row in read_jsonl(f):
            rows += 1
            messages = row["messages"]
            commands = [c["function"]["arguments"]["command"].strip() for m in messages
                        for c in m.get("tool_calls") or []
                        if "answer.txt" in c["function"]["arguments"]["command"]]
            if commands:
                last = commands[-1]
                final["echo -n" if last.startswith("echo -n") else
                      "printf %s" if last.startswith("printf %s") else
                      "python" if "python" in last else "other"] += 1
            closing += messages[-1]["role"] == "assistant" and not messages[-1].get("tool_calls")
    return {"rows": rows, "final_submission_idiom": dict(final.most_common()),
            "rows_ending_in_a_closing_assistant_message": closing,
            "chosen": "echo -n \"<value>\" > /workdir/answer.txt, answered (empty output, rc=0)"}


def turn_stats(index: list[dict]) -> dict:
    turns = sorted(e["assistant_turns"] for e in index)

    def at(q: float) -> int:
        return turns[min(len(turns) - 1, round(q * (len(turns) - 1)))]

    return {"min": turns[0], "median": at(.5), "p90": at(.9), "max": turns[-1],
            "share_over_16": round(sum(t > 16 for t in turns) / len(turns), 4),
            "note": "assistant messages, closing message included; upstream's rows run 3-12 tool "
                    "turns and or_agent.bash_loop's own ceiling is 16"}


def write_all(out: Path, rows: list[dict], index: list[dict], report: dict, replay: dict,
              tokens: dict | None, v1: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{NAME}.jsonl"
    for p in (path, out / f"{NAME}.index.jsonl", out / f"{NAME}.manifest.json"):
        if p.exists():                      # never overwrite a file in place: back it up first
            shutil.copy2(p, p.with_name(p.name + datetime.now().strftime(".bak-%Y%m%d%H%M%S")))
    positions = v1.pop("positions")
    for entry in index:
        entry["v1_row"] = positions.get(entry["task_id"])
    write_jsonl(rows, path)
    (out / f"{NAME}.index.jsonl").write_text("".join(json.dumps(e) + "\n" for e in index))
    record = {
        "source": NAME, "supersedes": "ja3_sft (v1, left in place; its adapters are invalid)",
        "format": "FineEnvs/SmolDataEnvs-sft conventions as the harness builds them "
                  "(messages + tools, one tool named bash)",
        "row": "messages + tools, nothing else; task ids and v1 positions are in the .index.jsonl",
        "rows": len(rows), "files": {f"{NAME}.jsonl": str(path)},
        "verified_trials": report["verified_trials"],
        "refused": report["refused"], "refused_total": sum(report["refused"].values()),
        "refused_example_task": report["refused_example_task"],
        "replay": replay, "token_stats": tokens, "op_family": report["op_family"],
        "invented_text": report["invented"], "submission_quoting": report["submission_quoting"],
        "user_turn": {"builder": "ladder.prompt_for(row, 'jupyter-agent-v3', 'L1', 'bash')",
                      "answer_format_line": {"numeric": "", "exact_short": "", "exact_bool": ""},
                      "rows_whose_file_list_came_from_the_directory":
                          report["rows_whose_file_list_came_from_the_directory"]},
        "firewall": {"keys": {k: len(v) for k, v in X.heldout_keys_for().items()},
                     "refused_by_firewall": {k: v for k, v in report["refused"].items()
                                             if k.startswith("heldout:")},
                     "expected_drops": 0},
        "scrub": {"fails_the_export_if": ["/home/<anything but user>", "/Users/", "/var/tmp/",
                                          "the local username", "a hostname",
                                          "an api key or token", "kaggle cache layout",
                                          "a symlinked-input or scratch-dir listing"],
                  "intended_path_not_flagged": "/home/user/input"},
        "turns": turn_stats(index),
        "upstream_idioms_measured": upstream_idioms(out / "sft_upstream"),
        "v1_join": v1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "smol_ladder_commit": X._commit(),
    }
    (out / f"{NAME}.manifest.json").write_text(json.dumps(record, indent=1))


def replay_all(rows: list[dict]) -> dict:
    """Replay every row again from the file's content (the exporter already replayed each once)."""
    failed = []
    for i, row in enumerate(rows):
        try:
            assert_replays(row["messages"], strict=True)
        except AssertionError:
            failed.append(i)
    return {"rows_checked": len(rows), "rows_replayed_byte_identically": len(rows) - len(failed),
            "rate": round((len(rows) - len(failed)) / len(rows), 6) if rows else None,
            "stop_policy": "model", "strict": "every message of the row must be rebuilt, in order"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=DATA / "train")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-tokens", action="store_true", help="skip the tokenizer pass")
    args = ap.parse_args()

    rows, index, report = export(limit=args.limit)
    replay = replay_all(rows)
    print(json.dumps({"rows": len(rows), "refused": report["refused"], "replay": replay,
                      "invented": report["invented"]}, indent=1))
    if args.dry_run:
        return
    tokens = None if args.no_tokens else token_report(
        rows, a_path=args.out / "sft_upstream" / "train.jsonl")
    v1 = v1_positions(index, args.out / "ja3_sft.jsonl")
    write_all(args.out, rows, index, report, replay, tokens, v1)
    print(f"wrote {len(rows)} rows to {args.out / (NAME + '.jsonl')}")
    if tokens:
        print(json.dumps({k: tokens[k] for k in ("B", "A", "AB", "render_audit") if k in tokens},
                         indent=1))


if __name__ == "__main__":
    main()
