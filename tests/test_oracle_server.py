"""The oracle endpoint replays a recorded trajectory one assistant turn per request.

It is the harness's no-GPU stand-in for a model (tools/oracle_server.py), so its own behaviour is
pinned here: the turn is chosen by how many assistant messages the conversation already holds,
`arguments` goes out as a JSON string whatever form the dataset stores, and a request it cannot
match is answered rather than crashing the run.
"""

from __future__ import annotations

import json
import urllib.request

from smol_ladder.upstream import bash_prompt
from tools.oracle_server import Oracle, OracleServer

QUESTION = "How many rows does t.csv have?"


def trajectory(commands: list[str]) -> dict:
    messages = list(bash_prompt(QUESTION, ["t.csv"]))
    for i, command in enumerate(commands):
        messages.append({"role": "assistant", "content": "", "tool_calls": [
            {"id": f"c{i}", "type": "function",
             "function": {"name": "bash", "arguments": {"command": command}}}]})
        messages.append({"role": "tool", "tool_call_id": f"c{i}", "name": "bash", "content": ""})
    return {"messages": messages, "tools": []}


def test_the_oracle_picks_the_recorded_turn_by_how_many_the_conversation_holds():
    row = trajectory(["ls", "pwd"])
    oracle = Oracle([row])
    base = list(bash_prompt(QUESTION, ["t.csv"]))
    first = oracle.reply(base)
    assert first["tool_calls"][0]["function"]["arguments"] == json.dumps({"command": "ls"})
    second = oracle.reply(base + [row["messages"][2], row["messages"][3]])
    assert json.loads(second["tool_calls"][0]["function"]["arguments"]) == {"command": "pwd"}
    assert "tool_calls" not in oracle.reply(row["messages"])           # exhausted: a closing line
    assert oracle.reply([{"role": "user", "content": "other"}])["content"].startswith("No recorded")
    assert oracle.unmatched == 1


def test_the_server_answers_chat_completions_and_models_over_http():
    row = trajectory(["ls"])
    with OracleServer([row]) as server:
        body = json.dumps({"model": "m", "messages": list(bash_prompt(QUESTION, ["t.csv"]))}).encode()
        request = urllib.request.Request(f"{server.base_url}/chat/completions", data=body,
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request) as resp:
            message = json.load(resp)["choices"][0]["message"]
        with urllib.request.urlopen(f"{server.base_url}/models") as resp:
            models = json.load(resp)
        assert len(server.requests) == 1
    assert json.loads(message["tool_calls"][0]["function"]["arguments"]) == {"command": "ls"}
    assert models["data"][0]["max_model_len"] > 0
