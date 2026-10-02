"""The bash protocol's sandbox: real commands, real bubblewrap jail, a scripted model.

The model's commands say /home/user/input and /workdir/answer.txt, and the training rows show what
those commands printed in the container they were recorded in. The jail is built to fit the
commands, so these tests run commands through `run_ladder.once()` (the same path a sweep takes) and
read back what the model would have seen: paths work as typed, nothing of the host leaks, the tables
are read-only, the answer lands where the grader reads it.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import pytest

import smol_ladder.run_ladder as runner
from smol_ladder.upstream import bash_prompt
from tools.oracle_server import OracleServer

pytestmark = pytest.mark.skipif(shutil.which("bwrap") is None, reason="needs bubblewrap")

QUESTION = "How many rows does t.csv have?"
ROW = {"task_id": "sandbox_1", "question": QUESTION, "files": ["t.csv"], "answer": "42",
       "reward_mode": "numeric", "atol": 0.0, "rtol": 0.0}


def trajectory(commands: list[str]) -> dict:
    """A recorded-looking row whose assistant turns run `commands` in order."""
    messages = list(bash_prompt(QUESTION, ["t.csv"]))
    for i, command in enumerate(commands):
        messages.append({"role": "assistant", "content": "", "tool_calls": [
            {"id": f"c{i}", "type": "function",
             "function": {"name": "bash", "arguments": {"command": command}}}]})
        messages.append({"role": "tool", "tool_call_id": f"c{i}", "name": "bash", "content": ""})
    return {"messages": messages, "tools": []}


def run_commands(tmp_path, monkeypatch, commands, inputs=None, **kw):
    """Run `commands` as the model's turns; return (result, [tool message contents])."""
    if inputs is None:
        inputs = tmp_path / "in"
        inputs.mkdir()
        (inputs / "t.csv").write_text("a\n1\n2\n")
    work = tmp_path / "trial"
    with OracleServer([trajectory(commands)]) as server:
        monkeypatch.setenv("SMOL_LADDER_BASE_URL", server.base_url)
        monkeypatch.delenv("SMOL_LADDER_API_KEY_ENV", raising=False)
        monkeypatch.setenv("OPENROUTER_API_KEY", "not-a-real-key-for-the-test")
        monkeypatch.setenv("SMOL_LADDER_SCRATCH", str(tmp_path / "scratch"))
        result = runner.once(ROW, bash_prompt(QUESTION, ["t.csv"])[1]["content"], work,
                             Path(sys.prefix), "oracle", len(commands) + 2,
                             inputs_of=lambda r: inputs, rung_label="L1", agent="bash", **kw)
    transcript = json.loads((work / "transcript.json").read_text())
    return result, [m["content"] for m in transcript if m["role"] == "tool"]


def test_the_models_paths_work_as_typed_and_the_answer_is_graded(tmp_path, monkeypatch):
    result, out = run_commands(tmp_path, monkeypatch, [
        "ls /home/user/input && wc -l /home/user/input/t.csv",
        "pwd; echo $HOME; ls /workdir",
        "python3 -c \"import pandas as pd; print(len(pd.read_csv('/home/user/input/t.csv')))\"",
        'echo -n "42" > /workdir/answer.txt',
    ])
    assert out[0] == "t.csv\n3 /home/user/input/t.csv\n"
    assert out[1] == "/workdir\n/home/user\n"
    assert out[2] == "2\n"
    assert out[3] == "(empty output, rc=0)"
    assert result["agent_status"] == "exit 0" and result["prediction"] == "42"
    assert result["reward"] == 1.0 and result["stop_reason"] == "answer_submitted"


def test_python_output_and_its_traceback_arrive_in_the_order_they_were_written(tmp_path,
                                                                                 monkeypatch):
    """The rows show print output BEFORE the traceback that followed it. Over a pipe Python buffers
    stdout until exit, which would put the traceback first."""
    _, out = run_commands(tmp_path, monkeypatch, [
        "python3 -c \"print('before'); import nonexistent_module_x\""])
    assert out[0].startswith("before\nTraceback"), out[0]


def test_the_printf_submission_gets_the_rows_receipt_in_the_real_sandbox(tmp_path, monkeypatch):
    result, out = run_commands(tmp_path, monkeypatch, ["printf %s 42 > /workdir/answer.txt"])
    assert out == ["Wrote 2 bytes to /workdir/answer.txt"]
    assert result["reward"] == 1.0


def test_nothing_of_the_host_is_visible_in_what_the_model_reads_back(tmp_path, monkeypatch):
    user = os.environ.get("USER") or Path.home().name
    repo = str(Path(__file__).resolve().parent.parent)
    _, out = run_commands(tmp_path, monkeypatch, [
        "ls -la /home/user/input; ls /home; whoami; id -un",
        "python3 -c 'import nonexistent_module_x'",
        "python3 -c 'import pandas, sys; print(pandas.__file__, sys.executable)'",
        "env | grep -i -e openrouter -e smol_ladder -e key; echo done",
        "ls /; ls /home/user -a",
    ])
    everything = "\n".join(out)
    for secret in (str(Path.home()), repo, "not-a-real-key", str(tmp_path)):
        assert secret not in everything, secret
    if len(user) > 3:
        # the account is called `user` in there, as in the rows; the host account name is not
        assert not any(f" {user} " in line for line in out[0].splitlines())
    assert "user user" in out[0]
    assert out[0].rstrip().endswith("user\nuser")
    assert "ModuleNotFoundError: No module named 'nonexistent_module_x'" in out[1]
    assert out[2].startswith("/usr/local/lib/python3.") and out[2].strip().endswith(
        "/usr/local/bin/python3"), out[2]
    assert out[3] == "done\n"
    assert "home" in out[4] and "evan" not in out[4]


def test_the_tables_are_read_only_and_listed_as_plain_files(tmp_path, monkeypatch):
    outside = tmp_path / "cache"
    outside.mkdir()
    (outside / "real.csv").write_text("x\n1\n")
    inputs = tmp_path / "in"
    inputs.mkdir()
    (inputs / "t.csv").write_text("a\n1\n")
    (inputs / "linked.csv").symlink_to(outside / "real.csv")   # the jupyter-agent layout
    (inputs / ".complete").write_text("[]")                     # the download marker
    _, out = run_commands(tmp_path, monkeypatch, [
        "ls -a /home/user/input",
        "ls -l /home/user/input",
        "touch /home/user/input/new.csv; echo $?",
        "cat /home/user/input/linked.csv",
    ], inputs=inputs)
    assert out[0] == ".\n..\nlinked.csv\nt.csv\n"
    assert "->" not in out[1] and str(outside) not in out[1]
    assert out[2].endswith("1\n") and "Read-only file system" in out[2]
    assert out[3] == "x\n1\n"


def test_a_tool_free_reply_before_any_answer_ends_the_default_episode(tmp_path, monkeypatch):
    """`submit` policy (the default) is unchanged: no answer file, graded 0, no crash."""
    result, out = run_commands(tmp_path, monkeypatch, ["ls /home/user/input"])
    assert out == ["t.csv\n"]
    assert result["agent_status"] == "exit 0" and result["reward"] == 0.0
    assert result["stop_reason"] == "model_stopped" and result["bash_stop"] == "submit"


def test_the_model_policy_lets_the_model_rewrite_its_answer(tmp_path, monkeypatch):
    result, _ = run_commands(tmp_path, monkeypatch, [
        'echo -n "7" > /workdir/answer.txt', 'echo -n "42" > /workdir/answer.txt'],
        bash_stop="model")
    assert result["prediction"] == "42" and result["reward"] == 1.0
    assert result["bash_stop"] == "model"
