"""`--agent bash` must put the model in the conversation it was trained on.

Two bugs made every adapter evaluation about something other than the adapter: the harness sent
the generic ladder prompt (tables in ./input, write ./solution.py) as the user turn while the
system turn said "write /workdir/answer.txt", and it formatted tool results, paths and the Python
environment differently from the SmolDataEnvs-sft rows. The tests here pin each of those to the
rows themselves:

- the user turn is the rows' template, byte for byte (`test_the_user_turn_*`);
- every tool result is formatted the way the rows' are (`test_*_result_*`);
- the replay test is the regression test for the whole class: it feeds recorded trajectories
  through the harness's own loop and asserts that the conversation sent at every turn equals the
  recorded conversation up to that turn;
- the sandbox tests (tests/test_bash_sandbox.py) run real commands in the real jail.

Everything here runs without a GPU. `-m slow` adds the full 4,673 rows when data/ is present.
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path

import pytest

import smol_ladder.or_agent as agent
from smol_ladder import ladder as L
from smol_ladder.or_agent import (ANSWER_RECEIPT, CommandResult, Endpoint, Episode, bash_loop,
                                  format_tool_result, run_bash, sandbox_env)
from smol_ladder.upstream import (BASH_SYSTEM, BASH_TOOL, BASH_USER, answer_format_of,
                                  bash_prompt)
from train.render import normalise_messages

FIXTURES = Path(__file__).parent / "fixtures"
ROWS = [json.loads(line) for line in (FIXTURES / "bash_replay_rows.jsonl").read_text().splitlines()]
DATA = Path(__file__).resolve().parent.parent / "data" / "train"


# ── the user turn ─────────────────────────────────────────────────────────────────────────────

def task_row(record: dict) -> dict:
    return {**record["task"], "answer": "0", "reward_mode": "numeric", "atol": 0.0, "rtol": 0.0}


@pytest.mark.parametrize("record", ROWS, ids=[r["task"]["task_id"] for r in ROWS])
def test_the_user_turn_is_the_training_template_filled_with_the_tasks_fields(record, monkeypatch):
    """The harness's own builder (`ladder.prompt_for(..., "bash")`), for the rows' own task."""
    monkeypatch.setattr(L, "input_files", lambda row, split: ["not-the-dir-listing.csv"])
    built = L.prompt_for(task_row(record), "train", "L1", "bash")
    assert built == record["messages"][1]["content"]


def test_bash_mode_lists_the_tasks_files_and_falls_back_to_the_directory(row):
    """The rows list the task's `files`, not the directory (which held more in 178 rows)."""
    row = {**row, "files": ["b.csv", "a.csv"]}
    listing = L.prompt_for(row, "test", "L1", "bash").split("no subfolders):\n")[1].split("\n\n")[0]
    assert listing == "- b.csv\n- a.csv"
    assert "- t.csv" in L.prompt_for({**row, "files": []}, "test", "L1", "bash")   # the directory
    assert "- t.csv" in L.prompt_for({**row, "files": ["x.csv"]}, "test", "L1")   # other modes


def test_an_empty_answer_format_leaves_one_blank_line_and_a_given_one_leaves_two():
    plain = bash_prompt("Q?", ["a.csv"], "")[1]["content"]
    assert "then compute.\n\nAnswer with a single clean value" in plain
    given = bash_prompt("Q?", ["a.csv"], "Express the value as a percentage (e.g. 95.5), not a "
                                         "fraction.")[1]["content"]
    assert ("then compute.\n\nExpress the value as a percentage (e.g. 95.5), not a fraction.\n\n"
            "Answer with a single clean value") in given


def test_the_template_is_not_hard_wrapped():
    """An earlier BASH_USER wrapped at ~100 columns and carried an extra blank line."""
    lines = BASH_USER.split("\n")
    assert max(len(line) for line in lines) > 140
    assert "\n\n\n" not in bash_prompt("Q?", ["a.csv"], "")[1]["content"]


def test_the_answer_format_line_is_read_from_the_tasks_instruction():
    instruction = ("x\n\nWork it out step by step — inspect the data first (head, shape, dtypes), "
                   "then compute.\n\nAnswer as: <name>, <value>.\n\nAnswer with a single clean value")
    assert answer_format_of({"instruction": instruction}) == "Answer as: <name>, <value>."
    assert answer_format_of({"instruction": "Question:\nq\n\nWork it out step by step — x\n\n"
                                            "Answer with a single clean value"}) == ""
    assert answer_format_of({}) == ""  # jupyter-agent and synthetic tasks have none


def test_bash_system_and_tools_are_the_rows():
    for record in ROWS:
        assert record["messages"][0]["content"] == BASH_SYSTEM
        assert record["tools"] == BASH_TOOL


# ── the rungs ─────────────────────────────────────────────────────────────────────────────────

def test_l1_is_a_prefix_of_every_higher_rung_in_bash_mode(row):
    l1 = L.prompt_for(row, "test", "L1", "bash")
    for rung in ("L1+schema", "L2", "L3", "L4"):
        assert L.prompt_for(row, "test", rung, "bash").startswith(l1), rung
    chain = [L.prompt_for(row, "test", r, "bash") for r in ("L2", "L3", "L4")]
    assert chain[1].startswith(chain[0].split("\n\nNotes on the intended computation")[0])
    assert len(set(chain)) == 3


def test_bash_l1_is_the_training_template_not_the_generic_ladder_prompt(row):
    l1 = L.prompt_for(row, "test", "L1", "bash")
    assert l1 == bash_prompt(row["question"], row["files"], "")[1]["content"]
    assert "solution.py" not in l1 and "./input" not in l1
    assert L.prompt_for(row, "test", "L1") == L.prompt_for(row, "test", "L1", "tools")
    assert "solution.py" in L.prompt_for(row, "test", "L1")  # the default mode is unchanged


def test_no_bash_rung_contradicts_the_answer_file_contract(row):
    for rung in ("L1", "L1+schema", "L2", "L3", "L4"):
        prompt = L.prompt_for(row, "test", rung, "bash")
        assert "solution.py" not in prompt and "./input" not in prompt, rung
        assert "/workdir/answer.txt" in prompt
    l4 = L.prompt_for(row, "test", "L4", "bash")
    assert "pd.read_csv('/home/user/input/t.csv')" in l4   # the reference, at the sandbox's path
    assert "Do not copy its output as the answer file" in l4


def test_the_other_modes_prompts_are_unchanged(row):
    for rung in ("L1", "L2", "L3", "L4"):
        assert L.prompt_for(row, "test", rung) == L.prompt_for(row, "test", rung, "program")
    assert "A verified reference solution to this question" in L.prompt_for(row, "test", "L4")


# ── the tool result ───────────────────────────────────────────────────────────────────────────

def test_empty_output_names_the_exit_status_and_nonempty_output_does_not():
    assert format_tool_result(CommandResult("", 0)) == ("(empty output, rc=0)", False)
    assert format_tool_result(CommandResult("", 2)) == ("(empty output, rc=2)", False)
    assert format_tool_result(CommandResult("  \n", 137)) == ("(empty output, rc=137)", False)
    # a failing command's text is just its text: no marker, no status
    assert format_tool_result(CommandResult("Traceback ...\nValueError: x\n", 1)) == (
        "Traceback ...\nValueError: x\n", False)


def test_the_timeout_result_is_the_fixed_line():
    assert format_tool_result(CommandResult("", -1, timed_out=True)) == (
        "[shell_exec] error: RuntimeError: Command timed out after 180 seconds", False)


def test_output_is_cut_to_8000_characters_plus_the_marker():
    text, cut = format_tool_result(CommandResult("x" * 8000, 0))
    assert not cut and len(text) == 8000
    text, cut = format_tool_result(CommandResult("y" * 20_000, 0))
    assert cut and text == "y" * 8000 + "\n... [truncated]" and len(text) == 8016


def test_the_conventions_hold_on_every_recorded_result_in_the_fixture():
    for record in ROWS:
        for m in record["messages"]:
            if m["role"] == "tool":
                result = recorded_to_raw(m["content"])
                assert format_tool_result(result)[0] == m["content"]


def test_run_bash_merges_the_streams_in_order_and_reports_the_exit_status(tmp_path):
    result = run_bash("echo out1; echo err1 >&2; echo out2; exit 3", cwd=str(tmp_path))
    assert result.output == "out1\nerr1\nout2\n" and result.rc == 3 and not result.timed_out
    assert "--- stderr ---" not in result.output
    assert run_bash("true").output == "" and run_bash("true").rc == 0


def test_a_command_killed_by_a_signal_reports_128_plus_n_like_the_rows(tmp_path):
    assert run_bash("kill -9 $$", cwd=str(tmp_path)).rc == 137


def test_a_command_that_outruns_the_deadline_is_killed_with_its_children(tmp_path):
    result = run_bash("sleep 30 & sleep 30", cwd=str(tmp_path), timeout=1)
    assert result.timed_out
    assert format_tool_result(result, timeout=1)[0].endswith("Command timed out after 1 seconds")


def test_the_models_commands_do_not_see_the_api_key(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENROUTER_API_KEY", "not-a-real-key-for-the-test")
    monkeypatch.setenv("SMOL_LADDER_BASE_URL", "http://127.0.0.1:1/v1")
    out = run_bash("env", cwd=str(tmp_path)).output
    assert "not-a-real-key" not in out and "SMOL_LADDER" not in out and "OPENROUTER" not in out
    assert set(sandbox_env({"PATH": "/bin", "SECRET": "x", "HOME": "/h"})) == {"PATH", "HOME"}


def test_the_printf_submission_gets_the_receipt_the_rows_carry(tmp_path, monkeypatch):
    """1,833 of the rows' submissions are `printf %s v > /workdir/answer.txt` and all of them are
    answered "Wrote N bytes ..."; `echo -n` is answered "(empty output, rc=0)"."""
    target = tmp_path / "answer.txt"
    monkeypatch.setattr(agent, "_PRINTF_SUBMIT", re.compile(r"\Aprintf %s (.*) > (\S+)\Z", re.S))
    out = run_bash(f"printf %s 'e4 with 12598' > {target}", cwd=str(tmp_path),
                   answer_path=str(target))
    assert out.output == ANSWER_RECEIPT.format(n=13) == "Wrote 13 bytes to /workdir/answer.txt"
    echo = run_bash(f"echo -n 7 > {target}", cwd=str(tmp_path), answer_path=str(target))
    assert format_tool_result(echo)[0] == "(empty output, rc=0)"


# ── the replay: the conversation the harness sends is the conversation that was recorded ──────

def recorded_to_raw(content: str) -> CommandResult:
    """Invert `format_tool_result`: the raw command result that would have been formatted so."""
    empty = re.fullmatch(r"\(empty output, rc=(\d+)\)", content)
    if empty:
        return CommandResult("", int(empty.group(1)))
    if content.startswith("[shell_exec] error: RuntimeError: Command timed out after"):
        return CommandResult("", -1, timed_out=True)
    if content.endswith(agent.TRUNCATION_MARKER) and len(content) == 8016:
        return CommandResult(content[:-16] + "z" * 500, 0)  # the tail the cut removed
    return CommandResult(content, 0)


def wire(turn: dict) -> dict:
    """A recorded assistant turn as a server returns it: `arguments` is a JSON string."""
    message = {"role": "assistant", "content": turn.get("content") or ""}
    if turn.get("tool_calls"):
        message["tool_calls"] = [
            {"id": c["id"], "type": "function",
             "function": {"name": c["function"]["name"],
                          "arguments": json.dumps(c["function"]["arguments"])}}
            for c in turn["tool_calls"]]
    return message


class RecordedRunEnded(Exception):
    pass


def replay(messages: list[dict], monkeypatch, stop: str = "model") -> tuple[list, list, list]:
    """Run `bash_loop` against the recorded trajectory. Returns (requests, final, commands).

    The model is scripted with the recorded assistant turns, the shell with the recorded results
    (inverted to raw command output, so the loop's own formatter has to reproduce them). Each
    request's conversation is captured at the moment it is sent.
    """
    turns = iter(m for m in messages if m["role"] == "assistant")
    results = iter(m["content"] for m in messages if m["role"] == "tool")
    requests: list[dict] = []
    commands: list[str] = []
    wrote: list[str] = []

    def fake_call_model(sent, model, tools, ep=None, max_tokens=None):
        requests.append({"messages": copy.deepcopy(sent), "tools": tools, "max_tokens": max_tokens})
        try:
            return {"choices": [{"message": wire(next(turns))}]}
        except StopIteration:  # a recorded run that ends on a tool result: the loop asks once more
            raise RecordedRunEnded from None

    def run_shell(command):
        commands.append(command)
        if "answer.txt" in command:  # stands in for the file the real loop reads back
            wrote.append(command)
        return recorded_to_raw(next(results))

    monkeypatch.setattr(agent, "call_model", fake_call_model)
    live = [copy.deepcopy(messages[0]), copy.deepcopy(messages[1])]
    try:
        bash_loop(live, run_shell, lambda: "submitted" if wrote else None, "m", 99,
                  Endpoint("http://127.0.0.1:1/v1"), Episode(None), stop=stop)
    except RecordedRunEnded:
        pass
    return requests, live, commands


def assert_replays(messages: list[dict], monkeypatch) -> int:
    requests, final, commands = replay(messages, monkeypatch)
    assistants = [i for i, m in enumerate(messages) if m["role"] == "assistant"]
    ends_on_tool = messages[-1]["role"] == "tool"
    last_tool = max((i for i, m in enumerate(messages) if m["role"] == "tool"), default=-1)
    # The loop stops at the first closing message after the submission. 60 of the 4,673 recorded
    # runs go on with a second one; those extra messages (and nothing else) may go unreplayed.
    unreplayed = assistants[len(requests):]
    assert all(i > last_tool and not messages[i].get("tool_calls") for i in unreplayed), \
        "the loop stopped before the recorded run's work did"
    assert len(requests) <= len(assistants) + ends_on_tool
    want = normalise_messages(messages)
    for request, index in zip(requests, assistants + [len(messages)] * ends_on_tool):
        assert normalise_messages(request["messages"]) == want[:index], f"conversation at {index}"
        assert request["tools"] == BASH_TOOL and request["max_tokens"] == 1024
    for m in final:
        m.pop("submitted", None)
    assert normalise_messages(final) == want[:len(final)]
    expected_commands = [c["function"]["arguments"]["command"] for m in messages
                         for c in m.get("tool_calls") or []]
    assert commands == expected_commands
    return len(requests)


@pytest.mark.parametrize("record", ROWS, ids=[r["task"]["task_id"] for r in ROWS])
def test_the_harness_sends_the_recorded_conversation_at_every_turn(record, monkeypatch):
    assert assert_replays(record["messages"], monkeypatch) >= 3


def test_the_fixture_covers_the_conventions_that_used_to_differ():
    tool = [m["content"] for r in ROWS for m in r["messages"] if m["role"] == "tool"]
    assert any(t == "(empty output, rc=0)" for t in tool)
    assert any(re.fullmatch(r"\(empty output, rc=[1-9]\d*\)", t) for t in tool)
    assert any("timed out after 180" in t for t in tool)
    assert any(t.startswith("Wrote ") for t in tool)
    assert any(len(m.get("tool_calls") or []) == 2 for r in ROWS for m in r["messages"])
    assert any(m["role"] == "assistant" and not m.get("tool_calls") and i < len(r["messages"]) - 1
               for r in ROWS for i, m in enumerate(r["messages"]))


def test_the_submit_policy_ends_at_the_first_submission_and_the_model_policy_does_not(monkeypatch):
    record = next(r for r in ROWS if any(
        c["function"]["arguments"]["command"].strip().startswith("printf %s")
        for m in r["messages"] for c in m.get("tool_calls") or []))
    full, _, _ = replay(record["messages"], monkeypatch, stop="model")
    short, _, _ = replay(record["messages"], monkeypatch, stop="submit")
    assert len(short) < len(full)
    with pytest.raises(ValueError, match="stop must be"):
        bash_loop([], lambda c: CommandResult("", 0), lambda: None, "m", 1,
                  Endpoint("http://127.0.0.1:1/v1"), Episode(None), stop="forever")


def read_sft_rows() -> list[dict]:
    rows = []
    for name in ("train.jsonl", "val.jsonl"):
        path = DATA / "sft_upstream" / name
        if not path.exists():
            pytest.skip(f"{path} is not on this machine")
        rows += [json.loads(line) for line in path.read_text().splitlines()]
    return rows


@pytest.mark.slow
def test_every_recorded_trajectory_replays(monkeypatch):
    """All 4,673 SmolDataEnvs-sft rows, when data/ has them."""
    rows = read_sft_rows()
    assert len(rows) == 4673
    diverge = []
    for row in rows:
        try:
            assert_replays(row["messages"], monkeypatch)
        except AssertionError:
            diverge.append(row["task_id"])
    # One recorded run submits, speaks, and then keeps working; the "model" policy ends the episode
    # at the closing message that follows a submission, so it cannot follow that one (0.02%).
    assert diverge == ["0138_141_138141265_qa_5"]


@pytest.mark.slow
def test_the_template_rebuilds_the_user_turn_of_every_recorded_row():
    """Group-by-skeleton, then rebuild. (1) From each row's own fields -- the question, the file
    list and the answer-format line cut out of its user turn -- all 4,673 must match byte for
    byte, and the rows share ONE skeleton. (2) From the SmolDataEnvs task rows joined by task_id,
    at least 90%: the rest differ in the task's own file list or question text (the rows were made
    from another revision of the dataset than the one cached), which no template can repair."""
    rows = read_sft_rows()
    parts = re.compile(r"Files \(in /home/user/input, no subfolders\):\n(.*?)\n\nInstalled.*?"
                       r"Question:\n(.*?)\n\nWork it out step by step[^\n]*\n\n(.*?)"
                       r"Answer with a single clean value", re.S)
    skeletons, exact = set(), 0
    for row in rows:
        user = row["messages"][1]["content"]
        found = parts.search(user)
        files = [f[2:] for f in found.group(1).split("\n")]
        pieces, at = [], 0
        for i in (1, 2, 3):  # cut the three variable spans out; what is left is the skeleton
            pieces.append(user[at:found.start(i)])
            at = found.end(i)
        skeletons.add("\x00".join(pieces + [user[at:]]))
        exact += bash_prompt(found.group(2), files, found.group(3).strip())[1]["content"] == user
    assert len(skeletons) == 1 and exact == len(rows) == 4673

    pytest.importorskip("pandas")
    from smol_ladder.run_ladder import source_for
    try:
        tasks, _ = source_for("train")
    except Exception as e:  # noqa: BLE001 - no cached dataset and no network
        pytest.skip(f"SmolDataEnvs train rows unavailable: {type(e).__name__}")
    by_id = {t["task_id"]: t for t in tasks}
    matched = sum(
        bash_prompt(by_id[r["task_id"]]["question"], by_id[r["task_id"]]["files"],
                    answer_format_of(by_id[r["task_id"]]))[1]["content"]
        == r["messages"][1]["content"] for r in rows)
    assert matched / len(rows) >= 0.90, f"{matched}/{len(rows)}"
