"""Replay a recorded bash trajectory through the harness's own loop.

If the model emitted exactly the row's assistant turns and the sandbox returned the row's tool
outputs, would `or_agent.bash_loop` build exactly the row's messages? That is the question
`tests/test_bash_fidelity.py` asks of the 4,673 upstream rows, and it is the acceptance test for
every dataset this project trains on: a row that cannot be replayed is a row the model was trained
on in a format the evaluation never builds.

The model is scripted with the recorded assistant turns, the shell with the recorded results
(inverted to the raw command output the loop's own `format_tool_result` has to turn back into the
recorded text). Each request's conversation is captured at the moment it is sent. Lives here, not
in the tests, so the exporter can refuse a row with the same code the tests use.
"""

from __future__ import annotations

import contextlib
import copy
import json
import re

import smol_ladder.or_agent as agent
from smol_ladder.or_agent import CommandResult, Endpoint, Episode, bash_loop
from smol_ladder.upstream import BASH_TOOL
from train.render import normalise_messages


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


@contextlib.contextmanager
def _patched(target, name, value):
    original = getattr(target, name)
    setattr(target, name, value)
    try:
        yield
    finally:
        setattr(target, name, original)


def replay(messages: list[dict], monkeypatch=None, stop: str = "model") -> tuple[list, list, list]:
    """Run `bash_loop` against the recorded trajectory. Returns (requests, final, commands).

    `monkeypatch` is accepted for the tests' call shape and unused: the one patch (`call_model`)
    is scoped here, so the exporter can call this without pytest.
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

    live = [copy.deepcopy(messages[0]), copy.deepcopy(messages[1])]
    with _patched(agent, "call_model", fake_call_model):
        try:
            bash_loop(live, run_shell, lambda: "submitted" if wrote else None, "m", 99,
                      Endpoint("http://127.0.0.1:1/v1"), Episode(None), stop=stop)
        except RecordedRunEnded:
            pass
    return requests, live, commands


def assert_replays(messages: list[dict], monkeypatch=None, strict: bool = False) -> int:
    """Raise AssertionError unless the harness would build `messages`. Returns the request count.

    The upstream rows are replayed with a documented allowance (`strict=False`): the loop stops at
    the first closing message after the submission, and 60 of the 4,673 recorded runs go on with a
    second one, so trailing text-only messages may go unreplayed. `strict=True` is the standard for
    rows we write ourselves: every message of the row must come back out of the loop, in order.
    """
    requests, final, commands = replay(messages, monkeypatch)
    assistants = [i for i, m in enumerate(messages) if m["role"] == "assistant"]
    ends_on_tool = messages[-1]["role"] == "tool"
    last_tool = max((i for i, m in enumerate(messages) if m["role"] == "tool"), default=-1)
    unreplayed = assistants[len(requests):]
    if strict:
        assert not unreplayed, "the loop stopped before the recorded run's last assistant message"
    else:
        assert all(i > last_tool and not messages[i].get("tool_calls") for i in unreplayed), \
            "the loop stopped before the recorded run's work did"
    assert len(requests) <= len(assistants) + ends_on_tool
    want = normalise_messages(messages)
    for request, index in zip(requests, assistants + [len(messages)] * ends_on_tool):
        assert normalise_messages(request["messages"]) == want[:index], f"conversation at {index}"
        assert request["tools"] == BASH_TOOL and request["max_tokens"] == 1024
    for m in final:
        m.pop("submitted", None)
    if strict:
        assert normalise_messages(final) == want, "the loop's final conversation is not the row"
    else:
        assert normalise_messages(final) == want[:len(final)]
    expected_commands = [c["function"]["arguments"]["command"] for m in messages
                         for c in m.get("tool_calls") or []]
    assert commands == expected_commands
    return len(requests)
