"""The ladder against a local OpenAI-compatible server, in both of its agent modes.

Nothing here loads a model. A stub HTTP server stands in for vLLM / llama.cpp / ollama and
replies with scripted completions, so the whole request/response contract is exercised: which
URL is called, what the body says, which tool schema is offered, and what gets graded.

The stub is a ThreadingHTTPServer on a loopback port rather than a mock of `urlopen`, because
the endpoint being configurable is the whole point of the change: a base URL, an API-key env
name and a model id that a local server understands are different things from OpenRouter's, and
only a real socket can prove the request got there.
"""

import json
import os
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from smol_ladder.or_agent import SYSTEM, Endpoint, endpoint
from smol_ladder.upstream import (BASH_TOOL, PROGRAM_SYSTEM, extract_code, localise_paths,
                                  looks_like_a_command, program_prompt, bash_prompt)


class Stub:
    """An OpenAI-compatible chat/completions endpoint that replies from a script."""

    def __init__(self, replies, error_aware=False, models=None):
        self.models = models  # body of GET /v1/models, or None for a server without one
        self.replies = list(replies)
        self.error_aware = error_aware
        self.requests: list[dict] = []
        self.headers: list[dict] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_args):
                pass

            def do_GET(self):
                body = json.dumps(outer.models or {}).encode()
                self.send_response(200 if outer.models else 404)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                outer.requests.append(json.loads(self.rfile.read(length)))
                outer.headers.append({k.lower(): v for k, v in self.headers.items()})
                reply = outer.replies.pop(0) if outer.replies else {
                    "choices": [{"message": {"content": ""}}]}
                status = 200
                if outer.error_aware and isinstance(reply, tuple):
                    status, reply = reply
                body = json.dumps(reply).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.base_url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"

    def __enter__(self):
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *_exc):
        self.server.shutdown()
        self.server.server_close()


def flaky_stub(replies):
    """A stub whose replies may be an error: `(status, body)` replies an HTTP status and a body.

    The free OpenRouter endpoint fails in two shapes that both have to be retried rather than
    booked as a model failure: a 429, and an HTTP 200 whose body carries no "choices" (a routed
    provider's error or a rate limit surfaced as JSON). Neither says anything about the model.
    """
    return Stub(replies, error_aware=True)


def assistant(content="", calls=None):
    message = {"role": "assistant", "content": content}
    if calls:
        # arguments is a JSON *string* on the wire. Handing the loop a dict would let it pass
        # while the server it actually talks to sends a string, so the stub has to match.
        message["tool_calls"] = [
            {"id": f"call_{i}", "type": "function",
             "function": {"name": name, "arguments": json.dumps(args)}}
            for i, (name, args) in enumerate(calls)]
    return {"choices": [{"message": message}]}


# ── the upstream protocol, byte for byte ──────────────────────────────────────

def test_the_program_prompt_is_the_one_the_grpo_model_was_trained_on():
    """The prompt is the protocol. A paraphrase is a different evaluation."""
    prompt = program_prompt("How many rows?", ["a.csv"])
    assert prompt[0] == {"role": "system", "content": PROGRAM_SYSTEM}
    user = prompt[1]["content"]
    assert user.startswith("How many rows?\n\nThe files are in /home/user/input")
    assert "- a.csv" in user
    assert "```python block" in user
    assert "/workdir/answer.txt" not in user, "the program protocol has no answer file"


def test_a_task_with_no_file_names_still_gets_a_prompt():
    """Upstream's build_prompt has a fallback here, and 1/4 of tasks hit it."""
    user = program_prompt("Q?", [])[1]["content"]
    assert "No file names were provided by the dataset" in user
    assert "os.listdir" in user


def test_the_bash_prompt_is_the_one_the_sft_model_was_trained_on():
    prompt = bash_prompt("How many rows?", ["a.csv"])
    system, user = prompt[0]["content"], prompt[1]["content"]
    assert system.startswith("You are an autonomous data-analysis agent")
    assert "/workdir/answer.txt" in system
    assert "Do NOT end your turn without submitting." in system
    assert "Question:\nHow many rows?" in user
    assert user.count("/workdir/answer.txt") >= 2
    assert BASH_TOOL[0]["function"]["name"] == "bash"
    assert "non-stateful" in BASH_TOOL[0]["function"]["description"]


def test_the_last_fenced_block_wins():
    """Models draft in one block and answer in the next; upstream takes the last."""
    out = extract_code("thinking\n```python\nprint(1)\n```\nactually\n```python\nprint(42)\n```")
    assert out == "print(42)"


def test_unfenced_output_is_treated_as_code_rather_than_scored_zero():
    assert extract_code("  print(42)  ") == "print(42)"


def test_the_command_guard_matches_upstreams_rule():
    # a bare value that begins with '>' is an answer, not a redirect
    assert not looks_like_a_command(">50K")
    # a value that redirects one is a command that never ran
    assert looks_like_a_command("2.14 > answer.txt")
    assert looks_like_a_command("echo -n 2.14 > answer.txt")
    assert looks_like_a_command("42 | tee out.txt")
    assert not looks_like_a_command("42")


def test_paths_are_localised_because_our_jail_is_not_their_sandbox():
    code = "df = pd.read_csv('/home/user/input/Iris.csv')\nprint(len(df))"
    out = localise_paths(code)
    assert "input/Iris.csv" in out
    assert "/home/user" not in out
    assert "answer.txt" in localise_paths("echo -n 7 > /workdir/answer.txt")


# ── endpoint resolution ───────────────────────────────────────────────────────

def test_a_local_server_needs_no_api_key(monkeypatch):
    monkeypatch.delenv("SMOL_LADDER_BASE_URL", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("SMOL_LADDER_API_KEY_ENV", raising=False)
    ep = endpoint({"SMOL_LADDER_BASE_URL": "http://127.0.0.1:8000/v1"})
    assert ep.url == "http://127.0.0.1:8000/v1/chat/completions"
    assert ep.api_key == ""


def test_a_remote_endpoint_without_a_key_is_an_error_not_a_401(monkeypatch):
    monkeypatch.delenv("SMOL_LADDER_API_KEY_ENV", raising=False)
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        endpoint({"SMOL_LADDER_BASE_URL": "https://openrouter.ai/api/v1"})


def test_the_key_env_var_is_itself_configurable():
    ep = endpoint({"SMOL_LADDER_BASE_URL": "http://127.0.0.1:1/v1",
                   "SMOL_LADDER_API_KEY_ENV": "LOCAL_TOKEN", "LOCAL_TOKEN": "secret"})
    assert ep.headers()["Authorization"] == "Bearer secret"


def test_a_trailing_slash_does_not_double_up():
    ep = Endpoint(base_url="http://127.0.0.1:8000/v1/", api_key="")
    assert ep.url == "http://127.0.0.1:8000/v1/chat/completions"


def test_non_thinking_is_requested_by_default_but_can_be_turned_off():
    assert endpoint({"SMOL_LADDER_BASE_URL": "http://127.0.0.1:1/v1"}).chat_template_kwargs \
        == {"enable_thinking": False}
    assert endpoint({"SMOL_LADDER_BASE_URL": "http://127.0.0.1:1/v1",
                     "SMOL_LADDER_CHAT_TEMPLATE_KWARGS": ""}).chat_template_kwargs is None


# ── the tools loop, against a stub server ─────────────────────────────────────

def test_the_tools_loop_calls_the_configured_server_with_the_configured_model(monkeypatch,
                                                                              tmp_path):
    """The endpoint is the change: base URL, key env and model id all come from config."""
    with Stub([assistant(calls=[("run_shell", {"command": "ls input"})]),
               assistant(calls=[("write_solution", {"code": "print(42)"})]),
               assistant("done")]) as stub:
        monkeypatch.setenv("SMOL_LADDER_BASE_URL", stub.base_url)
        monkeypatch.setenv("SMOL_LADDER_API_KEY_ENV", "LOCAL_TOKEN")
        monkeypatch.setenv("LOCAL_TOKEN", "tok")
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        seen: list[str] = []
        from smol_ladder.or_agent import solve_loop
        log = solve_loop("Q?", lambda c: (seen.append(c), "ok")[1],
                         lambda c: None, "AdithyaSK/smoldataenvs-grpo-2b-v0", 4)
    # 7 messages: system, user, then three assistant turns with two tool results between them.
    assert len(log) == 7, log
    # The returned log is the whole conversation, so it opens with the system and user turns.
    assert [m["role"] for m in log[:2]] == ["system", "user"], log
    assert len([m for m in log if m["role"] == "assistant"]) == 3
    assert len([m for m in log if m["role"] == "tool"]) == 2
    assert seen == ["ls input"]
    body = stub.requests[0]
    assert body["model"] == "AdithyaSK/smoldataenvs-grpo-2b-v0"
    assert body["temperature"] == 0.0
    assert {t["function"]["name"] for t in body["tools"]} == {"run_shell", "write_solution"}
    assert stub.headers[0]["authorization"] == "Bearer tok"


# ── the upstream program mode, end to end through once() ──────────────────────

def test_once_in_program_mode_grades_the_extracted_program(tmp_path, monkeypatch):
    """The faithful path: one program in a fence, run offline, graded by our grader."""
    import smol_ladder.run_ladder as runner

    program = "```python\nimport pandas as pd\ndf = pd.read_csv('input/t.csv')\nprint(6*7)\n```"
    inputs = tmp_path / "in"
    inputs.mkdir()
    (inputs / "t.csv").write_text("a\n1\n")
    work = tmp_path / "trial" / "L1"
    row = {"task_id": "t1", "question": "Q?", "files": ["t.csv"], "answer": "42",
           "reward_mode": "numeric", "atol": 0.0, "rtol": 0.0}

    with Stub([assistant(program)]) as stub:
        monkeypatch.setenv("SMOL_LADDER_BASE_URL", stub.base_url)
        monkeypatch.setenv("SMOL_LADDER_API_KEY_ENV", "LOCAL_TOKEN")
        monkeypatch.setenv("LOCAL_TOKEN", "tok")
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        result = runner.once(row, runner_program_prompt(), work, Path(sys.prefix), "local-2b",
                             1, inputs_of=lambda r: inputs, rung_label="L1", agent="program")

    assert result["agent_status"] == "exit 0", result.get("stderr", "")[:400]
    assert (work / "solution.py").read_text().endswith("print(6*7)")
    assert result["prediction"] == "42", result["prediction"]
    assert result["reward"] == 1.0
    # one turn, no tools offered: this mode is not a tool loop at all
    assert "tools" not in stub.requests[0]


def test_a_result_records_which_protocol_and_which_endpoint_produced_it(tmp_path, monkeypatch):
    """Provenance has to name the protocol, not just the model.

    The same model under `program` and under `tools` is a different measurement, and the same
    model served locally versus through OpenRouter is a third. A summary that cannot tell those
    apart pools them, so both the mode and the base URL that actually served the request are
    written next to the prompt hash and the rung.
    """
    import smol_ladder.run_ladder as runner

    inputs = tmp_path / "in"
    inputs.mkdir()
    (inputs / "t.csv").write_text("a\n1\n")
    row = {"task_id": "t1", "question": "Q?", "files": ["t.csv"], "answer": "42",
           "reward_mode": "numeric", "atol": 0.0, "rtol": 0.0}

    with Stub([assistant("```python\nprint(42)\n```")]) as stub:
        monkeypatch.setenv("SMOL_LADDER_BASE_URL", stub.base_url)
        monkeypatch.setenv("SMOL_LADDER_API_KEY_ENV", "LOCAL_TOKEN")
        monkeypatch.setenv("LOCAL_TOKEN", "tok")
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        result = runner.once(row, runner_program_prompt(), tmp_path / "p", Path(sys.prefix),
                             "local-2b", 1, inputs_of=lambda r: inputs, rung_label="L1",
                             provenance={"rung": "L1", "sample": 2}, agent="program")

    assert result["agent"] == "program"
    assert result["base_url"] == stub.base_url
    assert result["model"] == "local-2b"
    # ... beside the ladder provenance, so one record identifies the whole measurement
    assert result["rung"] == "L1" and result["sample"] == 2
    assert result["prompt_sha256"]

    with Stub([assistant("```python\nprint(42)\n```")]) as stub2:
        monkeypatch.setenv("SMOL_LADDER_BASE_URL", stub2.base_url)
        monkeypatch.setenv("SMOL_LADDER_API_KEY_ENV", "LOCAL_TOKEN")
        monkeypatch.setenv("LOCAL_TOKEN", "tok")
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        same_model = runner.once(row, runner_program_prompt(), tmp_path / "b", Path(sys.prefix),
                                 "local-2b", 1, inputs_of=lambda r: inputs, rung_label="L1",
                                 agent="bash")

    # same model, different protocol and different port: the record must tell them apart
    assert same_model["agent"] == "bash"
    assert same_model["base_url"] == stub2.base_url != result["base_url"]


def test_once_in_program_mode_does_not_offer_tools_even_though_the_model_can_call_them(
        tmp_path, monkeypatch):
    inputs = tmp_path / "in"
    inputs.mkdir()
    (inputs / "t.csv").write_text("a\n1\n")
    with Stub([assistant("```python\nprint(1)\n```")]) as stub:
        monkeypatch.setenv("SMOL_LADDER_BASE_URL", stub.base_url)
        monkeypatch.delenv("SMOL_LADDER_API_KEY_ENV", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        import smol_ladder.run_ladder as runner
        runner.once({"task_id": "t1", "question": "Q?", "files": ["t.csv"], "answer": "1",
                     "reward_mode": "numeric", "atol": 0.0, "rtol": 0.0},
                    runner_program_prompt(), tmp_path / "w", Path(sys.prefix), "m", 1,
                    inputs_of=lambda r: inputs, agent="program")
    assert "tool_calls" not in json.dumps(stub.requests[0])


def runner_program_prompt() -> str:
    from smol_ladder.upstream import program_prompt
    return program_prompt("Q?", ["t.csv"])[1]["content"]


# ── the transcript, end to end through the real agent script ────────────────────
#
# These drive once() against the stub server, so the program executed inside the jail is
# _agent_script's own body and or_agent.solve_loop is the loop that runs. Nothing here stubs the
# transcript: the file is written by the loop and read back by once(), which is the only way to
# show that what lands on disk is the conversation and not a fixture shaped like one.


def _stub_env(monkeypatch, stub):
    monkeypatch.setenv("SMOL_LADDER_BASE_URL", stub.base_url)
    monkeypatch.delenv("SMOL_LADDER_API_KEY_ENV", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)


def _inputs_outside_tmp(tmp_path, name="t.csv", body="a\n1\n") -> Path:
    """A task's tables somewhere the trial jail can actually reach them.

    The jail puts a tmpfs over /tmp, and the input directory is bound in by its *resolved* path.
    So an input dir under /tmp resolves to a path that does not exist inside the sandbox, and
    `ls input/` fails there while succeeding on every real task -- whose tables live under
    /var/tmp/smol-ladder/kaggle. pytest's tmp_path is under /tmp, so a transcript test that
    asserted on real shell output would be measuring the fixture's location rather than the
    transcript. The base dir is this project's own scratch root, which is where the cache is.
    """
    root = Path(os.environ.get("SMOL_LADDER_CACHE", "/var/tmp/smol-ladder")) / "pytest-inputs"
    root.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(dir=root))
    (directory / name).write_text(body)
    return directory


def test_the_saved_transcript_is_the_whole_multi_turn_conversation(tmp_path, monkeypatch):
    """A passing trial must leave a trajectory, not a turn count.

    The claim this pins is the whole reason the transcript is saved at all: a verified trial plus
    its conversation is an SFT example, and a verified trial without one is a program and an
    answer. So the file has to carry *every* message in order -- the system turn, the user turn,
    each assistant message, each tool call, and each tool result -- and not merely the assistant
    turns the loop happened to log.

    It asserts the shape rather than the sizes: the point is what is present and in what order,
    and a transcript that grew a fourth role later still has to read as a conversation.
    """
    import smol_ladder.run_ladder as runner

    inputs = _inputs_outside_tmp(tmp_path)
    work = tmp_path / "trial" / "L1"
    row = {"task_id": "t1", "question": "Q?", "files": ["t.csv"], "answer": "42",
           "reward_mode": "numeric", "atol": 0.0, "rtol": 0.0}
    replies = [
        assistant("First I will look at the table.",
                  calls=[("run_shell", {"command": "ls input/"})]),
        assistant("It has one column, so the sum is the answer.",
                  calls=[("write_solution", {"code": "print(6*7)"})]),
        assistant("The solution is written."),
    ]
    with Stub(replies) as stub:
        _stub_env(monkeypatch, stub)
        result = runner.once(row, "Q? Compute 6*7 from the table.", work, Path(sys.prefix),
                             "local", 6, inputs_of=lambda r: inputs, rung_label="L1")

    assert result["reward"] == 1.0, (result.get("prediction"),
                                     result.get("stderr", "")[:300])

    turns = json.loads((work / "transcript.json").read_text())
    roles = [m["role"] for m in turns]
    # The opening two are what make it a conversation rather than a bare transcript of replies.
    assert roles[:2] == ["system", "user"], roles
    assert turns[0]["content"] == SYSTEM, "the system prompt the agent actually ran with"
    assert turns[1]["content"] == "Q? Compute 6*7 from the table.", turns[1]["content"]

    # Three model turns, and they alternate assistant/tool in the order they happened.
    assert len([r for r in roles if r == "assistant"]) == 3
    assert len([r for r in roles if r == "tool"]) == 2
    assert roles[2:] == ["assistant", "tool", "assistant", "tool", "assistant"], roles

    # Every tool call is present with its arguments, and every tool result with its output --
    # the exploration is the training signal, so dropping either half trains on nothing.
    calls = [c for m in turns for c in (m.get("tool_calls") or [])]
    assert [c["function"]["name"] for c in calls] == ["run_shell", "write_solution"]
    assert json.loads(calls[0]["function"]["arguments"]) == {"command": "ls input/"}
    assert json.loads(calls[1]["function"]["arguments"])["code"] == "print(6*7)"
    outputs = [m["content"] for m in turns if m["role"] == "tool"]
    # The shell really ran: `ls input` inside the jail, where ./input is the symlink to the
    # task's tables. A transcript that recorded the call but not what came back would train the
    # model to invent results, which is the one thing a trajectory must never do.
    assert "t.csv" in outputs[0], outputs
    assert outputs[1] == "written /app/solution.py", outputs

    # Every tool result points back at the call that produced it, in order. A result whose
    # tool_call_id does not line up with the assistant turn above it is a trajectory TRL's
    # template will silently mis-render.
    ids = [c["id"] for c in calls]
    assert [m["tool_call_id"] for m in turns if m["role"] == "tool"] == ids

    # The assistant text survives too, including the closing turn that carries no call.
    assert turns[2]["content"] == "First I will look at the table."
    assert turns[4]["content"] == "It has one column, so the sum is the answer."
    assert turns[-1]["content"] == "The solution is written."
    # The closing turn is the one with no tool_calls: the loop stops on it, so a transcript that
    # dropped it would end mid-thought and read as a truncated run.
    assert "tool_calls" not in turns[-1]

    # And the trajectory describes what actually happened: three model turns, one of them writing
    # the program that the sealed grading pass then re-ran and graded 1.0. The turn count is
    # counted off the transcript rather than read from turns.json, which the agent writes into
    # its scratch and never copies out -- a number that only exists during the trial cannot be
    # checked afterwards, which is how it stayed wrong for so long.
    assert (work / "solution.py").read_text() == "print(6*7)"
    assert len([m for m in turns if m["role"] == "assistant"]) == 3


def test_the_saved_transcript_carries_the_prompt_the_trial_was_sent(tmp_path, monkeypatch):
    """The user turn is the rung's text, verbatim.

    A rung prompt is the thing the ladder varies, so a transcript whose user turn is a paraphrase
    cannot be matched back to the rung it came from -- and pairing a trajectory with the wrong
    prompt is how a training set quietly teaches the model a prompt it will never be given.
    """
    import smol_ladder.run_ladder as runner

    inputs = _inputs_outside_tmp(tmp_path)
    work = tmp_path / "trial" / "L2"
    prompt = "L2 text: the answer uses the mean of col_a where flag == 1."
    replies = [assistant(calls=[("write_solution", {"code": "print(42)"})]), assistant("done")]
    with Stub(replies) as stub:
        _stub_env(monkeypatch, stub)
        runner.once({"task_id": "t1", "question": "Q?", "files": ["t.csv"], "answer": "42",
                     "reward_mode": "numeric", "atol": 0.0, "rtol": 0.0},
                    prompt, work, Path(sys.prefix), "m", 4, inputs_of=lambda r: inputs,
                    rung_label="L2")

    turns = json.loads((work / "transcript.json").read_text())
    assert turns[1] == {"role": "user", "content": prompt}
    # and it is the same text the trial saved beside the result, so the two cannot disagree
    assert (work / "prompt.txt").read_text() == prompt


def test_the_bash_protocol_transcript_is_also_a_whole_conversation(tmp_path, monkeypatch):
    """The bash protocol is the format the SFT arm trains in, so its transcript is the one that
    most needs to be complete -- including the submission call, which is the turn the format's
    whole shape is built around."""
    import smol_ladder.run_ladder as runner

    inputs = _inputs_outside_tmp(tmp_path)
    work = tmp_path / "trial" / "L1"
    replies = [
        assistant(calls=[("bash", {"command": "ls input"})]),
        assistant(calls=[("bash", {"command": 'echo -n "42" > /workdir/answer.txt'})]),
        assistant("The answer is 42."),
    ]
    with Stub(replies) as stub:
        _stub_env(monkeypatch, stub)
        result = runner.once({"task_id": "t1", "question": "Q?", "files": ["t.csv"],
                              "answer": "42", "reward_mode": "numeric", "atol": 0.0, "rtol": 0.0},
                             runner_bash_prompt(), work, Path(sys.prefix), "sft-2b", 8,
                             inputs_of=lambda r: inputs, rung_label="L1", agent="bash")

    assert result["reward"] == 1.0
    turns = json.loads((work / "transcript.json").read_text())
    # system, user, then the bash turn and its result, then the submitting turn and its result.
    # The loop stops *on* submission rather than letting the model speak again, so there is no
    # trailing assistant turn -- which is the contract upstream's own rows follow.
    assert [m["role"] for m in turns] == ["system", "user", "assistant", "tool",
                                          "assistant", "tool"], [m["role"] for m in turns]
    # The submission is a bash call in the transcript, which is exactly how upstream's own rows
    # end. It is in the transcript, not only in answer.txt.
    submission = [c for m in turns for c in (m.get("tool_calls") or [])
                  if "answer.txt" in json.dumps(c)]
    assert len(submission) == 1, submission
    # The submitted value is recorded on the turn that made it, so an exporter can build the
    # final row without re-running the trial -- and without reading answer.txt, which the bash
    # protocol's own harness calls a side channel.
    assert turns[-2]["submitted"] == "42", turns[-2]


# ── the upstream bash mode, end to end through once() ─────────────────────────

def test_once_in_bash_mode_grades_the_answer_file(tmp_path, monkeypatch):
    """The SFT protocol: the model shells around and submits by writing a file."""
    import smol_ladder.run_ladder as runner

    inputs = tmp_path / "in"
    inputs.mkdir()
    (inputs / "t.csv").write_text("a\n1\n")
    work = tmp_path / "trial" / "L1"
    row = {"task_id": "t1", "question": "Q?", "files": ["t.csv"], "answer": "42",
           "reward_mode": "numeric", "atol": 0.0, "rtol": 0.0}
    replies = [
        assistant(calls=[("bash", {"command": "ls input"})]),
        assistant(calls=[("bash", {"command": "python3 -c \"print(6*7)\""})]),
        assistant(calls=[("bash", {"command": 'echo -n "42" > /workdir/answer.txt'})]),
        assistant("The answer is 42."),
    ]
    with Stub(replies) as stub:
        monkeypatch.setenv("SMOL_LADDER_BASE_URL", stub.base_url)
        monkeypatch.delenv("SMOL_LADDER_API_KEY_ENV", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        result = runner.once(row, runner_bash_prompt(), work, Path(sys.prefix), "sft-2b", 8,
                             inputs_of=lambda r: inputs, rung_label="L1", agent="bash")

    assert result["agent_status"] == "exit 0", result.get("stderr", "")[:400]
    assert result["prediction"] == "42", result["prediction"]
    assert result["reward"] == 1.0
    # the answer file is the run's evidence, so it is kept next to the result
    assert (work / "answer.txt").read_text() == "42"
    # the bash tool schema is the one from SmolDataEnvs-sft, verbatim
    assert [t["function"]["name"] for t in stub.requests[0]["tools"]] == ["bash"]
    # and the sandbox saw a localised command, not an absolute /workdir path
    assert "answer.txt" in json.dumps(stub.requests[1:])
    assert not (work / "solution.py").exists()


def test_a_submitted_answer_ends_the_bash_loop(tmp_path, monkeypatch):
    """"then stop" is part of the protocol: three more turns after submitting are not a
    measurement of anything, and on a 2B model they are 3 turns of rambling."""
    import smol_ladder.run_ladder as runner

    inputs = tmp_path / "in"
    inputs.mkdir()
    (inputs / "t.csv").write_text("a\n1\n")
    replies = [
        assistant(calls=[("bash", {"command": 'echo -n "42" > /workdir/answer.txt'})]),
        assistant(calls=[("bash", {"command": "ls"})]),
    ]
    with Stub(replies) as stub:
        monkeypatch.setenv("SMOL_LADDER_BASE_URL", stub.base_url)
        monkeypatch.delenv("SMOL_LADDER_API_KEY_ENV", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        result = runner.once({"task_id": "t1", "question": "Q?", "files": ["t.csv"],
                              "answer": "42", "reward_mode": "numeric", "atol": 0.0, "rtol": 0.0},
                             runner_bash_prompt(), tmp_path / "w", Path(sys.prefix), "m", 8,
                             inputs_of=lambda r: inputs, agent="bash")
    assert result["reward"] == 1.0
    assert len(stub.requests) == 1, "the loop kept going after the answer was submitted"


def test_the_bash_mode_refuses_to_credit_an_answer_that_is_just_a_command(tmp_path, monkeypatch):
    """Upstream's reward hack, guard and all.

    A 2B model that has not understood the protocol often "submits" by running
    `python3 -c "print(2.14)" > answer.txt` *as the echo payload*, i.e. the answer file ends up
    holding the text of the command rather than the value it would print. Upstream guards exactly
    this: `echo -n "2.14" > answer.txt` used to grade as 2.14 because the string contains it.
    """
    import smol_ladder.run_ladder as runner

    inputs = tmp_path / "in"
    inputs.mkdir()
    (inputs / "t.csv").write_text("a\n1\n")
    # The model echoes the whole shell line into the file instead of running it.
    replies = [assistant(calls=[("bash", {"command":
                           "echo -n 'echo 2.14 > answer.txt' > answer.txt"})])]
    with Stub(replies) as stub:
        monkeypatch.setenv("SMOL_LADDER_BASE_URL", stub.base_url)
        monkeypatch.delenv("SMOL_LADDER_API_KEY_ENV", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        result = runner.once({"task_id": "t1", "question": "Q?", "files": ["t.csv"],
                              "answer": "2.14", "reward_mode": "numeric", "atol": 0.0, "rtol": 0.0},
                             runner_bash_prompt(), tmp_path / "w", Path(sys.prefix), "m", 8,
                             inputs_of=lambda r: inputs, agent="bash")
    assert result["reward"] == 0.0, result["prediction"]
    assert "command" in result.get("stderr", "")


def test_the_bash_mode_accepts_a_value_that_merely_contains_a_redirect(tmp_path, monkeypatch):
    """The guard is not so blunt that it rejects a real gold answer. Upstream's regex requires
    something *before* the redirect, because gold answers legitimately start with `>`:
    `>50K`, `> 2 Years`, `>40hrs` are all real answers in SmolDataEnvs."""
    from smol_ladder.upstream import looks_like_a_command
    assert not looks_like_a_command(">50K")
    assert not looks_like_a_command(">40hrs")


def test_a_local_model_that_produces_nothing_scores_zero_not_a_crash(tmp_path, monkeypatch):
    import smol_ladder.run_ladder as runner

    inputs = tmp_path / "in"
    inputs.mkdir()
    (inputs / "t.csv").write_text("a\n1\n")
    with Stub([assistant("I am not sure how to answer that.")]) as stub:
        monkeypatch.setenv("SMOL_LADDER_BASE_URL", stub.base_url)
        monkeypatch.delenv("SMOL_LADDER_API_KEY_ENV", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        result = runner.once({"task_id": "t1", "question": "Q?", "files": ["t.csv"],
                              "answer": "42", "reward_mode": "numeric", "atol": 0.0, "rtol": 0.0},
                             runner_program_prompt(), tmp_path / "w", Path(sys.prefix), "m", 1,
                             inputs_of=lambda r: inputs, agent="program")
    assert result["agent_status"] == "exit 0"
    assert result["reward"] == 0.0


def test_a_program_that_reads_a_bare_filename_still_finds_its_table(tmp_path, monkeypatch):
    """Upstream programs run with the tables as the working directory, so `pd.read_csv('a.csv')`
    is idiomatic for them. Ours run one level up, next to ./input. Both spellings must work or
    we are not measuring their skill, ours."""
    import smol_ladder.run_ladder as runner

    inputs = tmp_path / "in"
    inputs.mkdir()
    (inputs / "t.csv").write_text("a\n7\n")
    work = tmp_path / "trial" / "L1"
    program = "```python\nimport pandas as pd\nprint(int(pd.read_csv('t.csv')['a'].sum()))\n```"
    with Stub([assistant(program)]) as stub:
        monkeypatch.setenv("SMOL_LADDER_BASE_URL", stub.base_url)
        monkeypatch.delenv("SMOL_LADDER_API_KEY_ENV", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        result = runner.once({"task_id": "t1", "question": "Q?", "files": ["t.csv"],
                              "answer": "7", "reward_mode": "numeric", "atol": 0.0, "rtol": 0.0},
                             runner_program_prompt(), work, Path(sys.prefix), "m", 1,
                             inputs_of=lambda r: inputs, agent="program")
    assert result["reward"] == 1.0, (result["prediction"], result.get("stderr", "")[:300])


def runner_bash_prompt() -> str:
    from smol_ladder.upstream import bash_prompt
    return bash_prompt("Q?", ["t.csv"])[1]["content"]


# ── the endpoint's transient failures ─────────────────────────────────────────

def call_with_retries(monkeypatch, stub, model="m"):
    """call_model() against the stub, with the backoff collapsed to nothing.

    The backoff is not the thing under test and costs 5s+ per attempt if it is left real, so it is
    stubbed; the number of attempts, and what is retried, are.
    """
    from smol_ladder.or_agent import call_model, endpoint
    monkeypatch.setenv("SMOL_LADDER_BASE_URL", stub.base_url)
    monkeypatch.delenv("SMOL_LADDER_API_KEY_ENV", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr("smol_ladder.or_agent.time.sleep", lambda _s: None)
    return call_model([{"role": "user", "content": "Q?"}], model, None, endpoint())


def test_a_429_is_retried_rather_than_recorded_as_a_model_failure(monkeypatch):
    """The free endpoint rate-limits under a parallel sweep, and a 429 says nothing about the model.

    Booking it as a failure would put the sweep's own concurrency into the pass rate, and the trial
    would then be unrecoverable: once() reuses a cached result.json, so a 429 written to disk
    survives every rerun that is not passed --retry-failed.
    """
    with flaky_stub([(429, {"error": {"message": "rate limited"}}),
                     (429, {"error": {"message": "rate limited"}}),
                     assistant("42")]) as stub:
        reply = call_with_retries(monkeypatch, stub)

    assert reply["choices"][0]["message"]["content"] == "42"
    assert len(stub.requests) == 3, "the 429s were not retried"


def test_a_200_with_no_choices_is_retried(monkeypatch):
    """The same endpoint intermittently answers 200 with a body that carries no "choices" at all --
    a routed provider's error, or a rate limit surfaced as JSON. A 200 status is not a completion,
    and taking the empty body for one loses a whole trial to a transient."""
    with flaky_stub([{"id": "gen-1", "error": {"message": "No endpoints available"}},
                     {"id": "gen-2", "choices": []},
                     assistant("42")]) as stub:
        reply = call_with_retries(monkeypatch, stub)

    assert reply["choices"][0]["message"]["content"] == "42"
    assert len(stub.requests) == 3, "an empty-choices 200 was treated as a completion"


def test_a_transient_burst_still_leaves_a_working_completion(monkeypatch):
    """The realistic shape under load: a few failures of each kind, then a real answer. This is the
    sequence a parallel sweep has to survive without any trial being booked as a model failure."""
    replies = [(429, {"error": "slow down"}),
               {"error": "no provider"},
               (503, {"error": "unavailable"}),
               {"choices": []},
               assistant("7")]
    with flaky_stub(replies) as stub:
        reply = call_with_retries(monkeypatch, stub)

    assert reply["choices"][0]["message"]["content"] == "7"
    assert len(stub.requests) == 5


def test_an_endpoint_that_never_answers_fails_the_trial_rather_than_hanging(monkeypatch):
    """After the retries are spent the call has to raise. A solve_loop that simply returned would
    look like a clean exit-0 trial with an empty prediction -- a model failure that was the
    endpoint's, scored into the pass rate."""
    with flaky_stub([(429, {"error": "rate limited"})] * 8) as stub:
        with pytest.raises(RuntimeError, match="no completion"):
            call_with_retries(monkeypatch, stub)

    assert len(stub.requests) == 5, "the retry budget is not 5 attempts"


def test_the_retry_budget_is_spent_and_the_error_says_why(monkeypatch):
    """The message has to name the endpoint and the last failure: a bare "gave no completion" in a
    32-worker log is 32 identical lines and no diagnosis at all."""
    with flaky_stub([(429, {"error": {"message": "rate limited"}})] * 8) as stub:
        with pytest.raises(RuntimeError) as caught:
            call_with_retries(monkeypatch, stub)

    message = str(caught.value)
    assert "429" in message, message
    assert stub.base_url in message, message

# ── tool-output truncation and the context budget ─────────────────────────────

OVERFLOW = {"error": {"message": (
    "You passed 16385 input tokens and requested 0 output tokens. However, the model's context "
    "length is only 16384 tokens, resulting in a maximum input length of 16384 tokens.")}}


def _endpoint_for(monkeypatch, stub, ctx=None):
    from smol_ladder.or_agent import endpoint
    monkeypatch.setenv("SMOL_LADDER_BASE_URL", stub.base_url)
    monkeypatch.delenv("SMOL_LADDER_API_KEY_ENV", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr("smol_ladder.or_agent.time.sleep", lambda _s: None)
    if ctx:
        monkeypatch.setenv("SMOL_LADDER_MAX_MODEL_LEN", str(ctx))
    else:
        monkeypatch.delenv("SMOL_LADDER_MAX_MODEL_LEN", raising=False)
    return endpoint()


def _run_bash_loop(monkeypatch, replies, ctx=None, shell=lambda c: "ok", answer=lambda: None,
                   max_turns=16, models=None):
    from smol_ladder.or_agent import Episode, bash_loop, context_length
    with flaky_stub(replies) as stub:
        stub.models = models
        ep = _endpoint_for(monkeypatch, stub, ctx)
        episode = Episode(context_length(ep, "m"))
        messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "q"}]
        bash_loop(messages, shell, answer, "m", max_turns, ep, episode)
    return messages, episode, stub


def test_a_tool_output_is_cut_the_way_upstreams_rows_are():
    """Upstream's SFT rows: first 8,000 characters + "\\n... [truncated]" (8,016 in all), never
    longer. Measured on data/train/sft_upstream; see the constants' comment."""
    from smol_ladder.or_agent import TOOL_OUTPUT_MAX_CHARS, TRUNCATION_MARKER, truncate_output
    assert (TOOL_OUTPUT_MAX_CHARS, TRUNCATION_MARKER) == (8000, "\n... [truncated]")
    assert truncate_output("a" * 8000) == ("a" * 8000, False)
    out, cut = truncate_output("a" * 8001)
    assert cut and len(out) == 8016 and out.endswith("\n... [truncated]")


def test_a_huge_tool_output_is_truncated_before_it_joins_the_conversation(monkeypatch):
    replies = [assistant(calls=[("bash", {"command": "cat big.csv"})]), assistant("done")]
    messages, episode, _ = _run_bash_loop(monkeypatch, replies, shell=lambda c: "x" * 50_000)
    tool = [m for m in messages if m["role"] == "tool"][0]
    assert len(tool["content"]) == 8016 and tool["content"].endswith("[truncated]")
    assert episode.truncated_outputs == 1
    assert episode.stop_reason == "model_stopped"


def test_a_non_retryable_4xx_is_raised_at_once_not_retried_five_times(monkeypatch):
    from smol_ladder.or_agent import ClientError
    with flaky_stub([(401, {"error": "bad key"})] * 8) as stub:
        with pytest.raises(ClientError, match="401"):
            call_with_retries(monkeypatch, stub)
    assert len(stub.requests) == 1


def test_a_5xx_is_still_retried(monkeypatch):
    with flaky_stub([(500, {"error": "boom"}), (502, {"error": "boom"}), assistant("1")]) as stub:
        call_with_retries(monkeypatch, stub)
    assert len(stub.requests) == 3


def test_max_tokens_is_upstreams_cap_but_never_more_than_the_room_left(monkeypatch):
    messages, episode, stub = _run_bash_loop(monkeypatch, [assistant("hi")], ctx=16384)
    assert stub.requests[0]["max_tokens"] == 1024
    messages, episode, stub = _run_bash_loop(monkeypatch, [assistant("hi")], ctx=400)
    assert 0 < stub.requests[0]["max_tokens"] < 400


def test_no_room_for_a_completion_ends_the_episode_without_calling_the_server(monkeypatch):
    messages, episode, stub = _run_bash_loop(monkeypatch, [assistant("hi")], ctx=50)
    assert stub.requests == []
    assert episode.stop_reason == "context_exhausted"


def test_a_server_side_context_overflow_ends_the_episode_cleanly(monkeypatch):
    """The reported crash: the conversation passed 16,384 tokens. No retries, no exception."""
    messages, episode, stub = _run_bash_loop(
        monkeypatch, [assistant(calls=[("bash", {"command": "cat x"})]), (400, OVERFLOW)])
    assert len(stub.requests) == 2, "the 400 was retried"
    assert episode.stop_reason == "context_exhausted"
    assert episode.turns == 1


def test_an_overflow_with_a_little_room_is_retried_once_with_the_exact_max_tokens(monkeypatch):
    near = {"error": {"message": "You passed 16000 input tokens and requested 1024 output "
                      "tokens. However, the model's context length is only 16384 tokens"}}
    messages, episode, stub = _run_bash_loop(monkeypatch, [(400, near), assistant("done")])
    assert [r["max_tokens"] for r in stub.requests] == [1024, 16384 - 16000 - 8]
    assert episode.stop_reason == "model_stopped"


def test_the_context_length_comes_from_the_server_with_an_env_override(monkeypatch):
    from smol_ladder.or_agent import context_length
    models = {"data": [{"id": "m", "max_model_len": 16384}]}
    with Stub([], models=models) as stub:
        ep = _endpoint_for(monkeypatch, stub)
        assert context_length(ep, "m") == 16384
        monkeypatch.setenv("SMOL_LADDER_MAX_MODEL_LEN", "4096")
        assert context_length(ep, "m") == 4096
    with Stub([]) as stub:  # no /models: unknown, not an error
        monkeypatch.delenv("SMOL_LADDER_MAX_MODEL_LEN")
        assert context_length(_endpoint_for(monkeypatch, stub), "m") is None


def test_the_stop_reasons(monkeypatch):
    call = lambda: assistant(calls=[("bash", {"command": "ls"})])  # noqa: E731
    _, ep_, _ = _run_bash_loop(monkeypatch, [call()] * 3, max_turns=2)
    assert (ep_.stop_reason, ep_.turns) == ("max_turns", 2)
    _, ep_, _ = _run_bash_loop(monkeypatch, [call()], answer=lambda: "42")
    assert ep_.stop_reason == "answer_submitted"
    _, ep_, _ = _run_bash_loop(monkeypatch, [assistant("no tools")])
    assert ep_.stop_reason == "model_stopped"


def test_a_transport_failure_still_raises_and_is_marked_error(monkeypatch):
    from smol_ladder.or_agent import ClientError, Episode, bash_loop
    with flaky_stub([(401, {"error": "no"})]) as stub:
        ep = _endpoint_for(monkeypatch, stub)
        episode = Episode()
        with pytest.raises(ClientError):
            bash_loop([{"role": "user", "content": "q"}], lambda c: "", lambda: None, "m", 4,
                      ep, episode)
    assert episode.stop_reason == "error"


_ROW = {"task_id": "t1", "question": "Q?", "files": ["t.csv"], "answer": "42",
        "reward_mode": "numeric", "atol": 0.0, "rtol": 0.0}


def _once(tmp_path, monkeypatch, replies, agent, prompt=None):
    import smol_ladder.run_ladder as runner
    inputs = _inputs_outside_tmp(tmp_path)
    work = tmp_path / "trial" / "L1"
    with flaky_stub(replies) as stub:
        _stub_env(monkeypatch, stub)
        monkeypatch.delenv("SMOL_LADDER_MAX_MODEL_LEN", raising=False)
        result = runner.once(_ROW, prompt or runner_bash_prompt(), work, Path(sys.prefix), "m", 8,
                             inputs_of=lambda r: inputs, rung_label="L1", agent=agent)
    return result, work


def test_a_context_exhausted_bash_trial_is_a_finished_model_failure_with_a_transcript(
        tmp_path, monkeypatch):
    replies = [assistant(calls=[("bash", {"command": "ls input"})]), (400, OVERFLOW)]
    result, work = _once(tmp_path, monkeypatch, replies, "bash")
    assert result["agent_status"] == "exit 0", result.get("stderr", "")[:400]
    assert result["stop_reason"] == "context_exhausted"
    assert result["reward"] == 0.0 and result["prediction"] == ""
    assert result["turns_used"] == 1 and result["truncated_outputs"] == 0
    roles = [m["role"] for m in json.loads((work / "transcript.json").read_text())]
    assert roles == ["system", "user", "assistant", "tool"]


def test_a_submitted_answer_records_stop_reason_turns_and_prompt_tokens(tmp_path, monkeypatch):
    submit = assistant(calls=[("bash", {"command": 'echo -n "42" > /workdir/answer.txt'})])
    submit["usage"] = {"prompt_tokens": 777, "completion_tokens": 20}
    result, _ = _once(tmp_path, monkeypatch, [submit], "bash")
    assert result["reward"] == 1.0
    assert (result["stop_reason"], result["turns_used"], result["last_prompt_tokens"]) == (
        "answer_submitted", 1, 777)


def test_a_huge_output_in_a_real_trial_is_truncated_and_counted(tmp_path, monkeypatch):
    replies = [assistant(calls=[("bash", {"command": "python3 -c \"print('x'*30000)\""})]),
               assistant("done")]
    result, work = _once(tmp_path, monkeypatch, replies, "bash")
    assert result["truncated_outputs"] == 1 and result["stop_reason"] == "model_stopped"
    tool = [m for m in json.loads((work / "transcript.json").read_text())
            if m["role"] == "tool"][0]
    assert len(tool["content"]) == 8016


@pytest.mark.parametrize("agent", ["bash", "tools", "program"])
def test_a_crashed_trial_still_leaves_its_transcript_and_stays_a_harness_failure(
        tmp_path, monkeypatch, agent):
    result, work = _once(tmp_path, monkeypatch, [(401, {"error": "no"})] * 3, agent,
                         prompt="Q?" if agent != "bash" else None)
    assert result["agent_status"] == "exit 1"
    roles = [m["role"] for m in json.loads((work / "transcript.json").read_text())]
    assert roles == ["system", "user"]
    assert result["stop_reason"] == "error"


def test_the_tools_mode_ends_cleanly_when_the_context_is_exhausted(tmp_path, monkeypatch):
    replies = [assistant(calls=[("run_shell", {"command": "ls input"})]), (400, OVERFLOW)]
    result, work = _once(tmp_path, monkeypatch, replies, "tools", prompt="Q?")
    assert result["agent_status"] == "exit 0", result.get("stderr", "")[:400]
    assert result["stop_reason"] == "context_exhausted"
    assert (work / "transcript.json").exists()
