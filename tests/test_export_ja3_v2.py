"""Arm B's v2 export: the conversation the harness builds, proven by replaying it.

`train/export_ja3_v2.py` re-expresses the ja3 transcripts in upstream's conventions. What these
tests pin: (1) each translation rule on a hand-written recording, (2) every refusal, (3) that three
real exported rows (scrubbed, committed under tests/fixtures) replay byte-identically through
`or_agent.bash_loop`, and (4) with `-m slow`, that all of `data/train/ja3_sft_v2.jsonl` does.

    uv run --with pytest pytest -q tests/test_export_ja3_v2.py
    uv run --with pytest pytest -q tests/test_export_ja3_v2.py -m slow
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from smol_ladder import ladder as L
from smol_ladder.or_agent import EMPTY_OUTPUT, TRUNCATION_MARKER
from smol_ladder.upstream import BASH_SYSTEM, BASH_TOOL, answer_format_of, bash_prompt
from train import export_ja3 as X
from train import export_ja3_v2 as V
from train.replay import assert_replays

FIXTURES = Path(__file__).parent / "fixtures"
FIXTURE_ROWS = [json.loads(line) for line in
                (FIXTURES / "ja3_v2_rows.jsonl").read_text().splitlines()]
V2 = Path(__file__).resolve().parent.parent / "data" / "train" / "ja3_sft_v2.jsonl"
TID = "ja_0000_1_1.ipynb_qa_1"


def recorded(*steps: tuple[str, dict | str], final: str = "The answer is **7**.") -> list[dict]:
    """A transcript in the tools loop's wire shape: system, user, then (call, result) steps."""
    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    for i, (name, payload) in enumerate(steps):
        call = {"id": f"c{i}", "type": "function",
                "function": {"name": "run_shell" if name == "shell" else name,
                             "arguments": json.dumps(payload[0])}}
        messages.append({"role": "assistant", "content": "", "tool_calls": [call]})
        messages.append({"role": "tool", "tool_call_id": f"c{i}", "content": payload[1]})
    messages.append({"role": "assistant", "content": final})
    return messages


def shell(command: str, output: str):
    return ("shell", ({"command": command}, output))


# ── the translation rules ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("before,after", [
    ("head -3 ./input/a.csv", "head -3 /home/user/input/a.csv"),
    ("head -3 input/a.csv", "head -3 /home/user/input/a.csv"),
    ("ls -la ./input && wc -l x", "ls -la /home/user/input && wc -l x"),
    ("cd ./input && ls", "cd /home/user/input && ls"),
    ("cd input; ls", "cd /home/user/input; ls"),
    ("python3 -c \"import os; os.listdir('input')\"",
     "python3 -c \"import os; os.listdir('/home/user/input')\""),
    ("INPUT = './input'", "INPUT = '/home/user/input'"),
    ("pd.read_csv('./input/a.csv')", "pd.read_csv('/home/user/input/a.csv')"),
    ("df['input'].sum()", "df['input'].sum()"),                    # a column, not the directory
    ("x == 'input'", "x == 'input'"),
    ("input('x')", "input('x')"),                                  # the builtin
    ("../input/a.csv", "../input/a.csv"),                          # somebody else's directory
    ("/home/user/input/a.csv", "/home/user/input/a.csv"),          # idempotent
])
def test_input_paths_are_spelled_as_the_sandbox_has_them(before, after):
    assert V.world_text(before, TID, command=True)[0] == after


def test_the_trials_own_scratch_directory_becomes_workdir_and_nobody_elses_does_not():
    mine = f"File \"/var/tmp/smol-ladder/scratch/trials/{TID}/L1/solution.py\", line 3"
    assert V.world_text(mine, TID, False)[0] == 'File "/workdir/solution.py", line 3'
    other = "/var/tmp/smol-ladder/scratch/trials/ja_9_9_9.ipynb_qa_1/L1/solution.py"
    assert V.world_text(other, TID, False)[0] == other


def test_stderr_is_folded_into_the_stream_after_stdout():
    assert V.merged_result("out\n\n--- stderr ---\nTraceback\n") == ("out\nTraceback\n", True)
    assert V.merged_result("\n--- stderr ---\nbash: x\n") == ("bash: x\n", True)
    assert V.merged_result("plain\n") == ("plain\n", False)


def test_a_timeout_and_a_tail_cut_are_refused_not_translated():
    with pytest.raises(V.Refused, match="timed out"):
        V.merged_result("[timed out after 150s]")
    with pytest.raises(V.Refused, match="tail-cut"):
        V.merged_result("x" * 20_000)


def test_heredoc_never_collides_with_the_code_it_wraps():
    assert V.heredoc("/workdir/s.py", "print(1)\n") == "cat > /workdir/s.py << 'EOF'\nprint(1)\nEOF"
    tricky = V.heredoc("/workdir/s.py", "print(1)\nEOF\nprint(2)")
    assert tricky.startswith("cat > /workdir/s.py << 'EOF_1'") and tricky.endswith("\nEOF_1")


@pytest.mark.parametrize("value,command,quoting", [
    ("141", 'echo -n "141" > /workdir/answer.txt', "double"),
    ("New York", 'echo -n "New York" > /workdir/answer.txt', "double"),
    ("a, b", 'echo -n "a, b" > /workdir/answer.txt', "double"),
    ("$5", "echo -n '$5' > /workdir/answer.txt", "single"),
    ('say "hi"', "echo -n 'say \"hi\"' > /workdir/answer.txt", "single"),
    ("it's $", "echo -n 'it'\"'\"'s $' > /workdir/answer.txt", "single"),
])
def test_the_submission_is_the_prompts_own_echo_idiom_and_writes_the_value_exactly(
        value, command, quoting):
    assert V.submission_command(value) == (command, quoting)
    assert V.check_submission(command, value)


def test_a_value_echo_cannot_write_is_refused():
    for bad in ("a\nb", "-n", "-e"):
        with pytest.raises(V.Refused):
            V.submission_command(bad)


def test_the_input_listing_has_upstreams_layout():
    listing = V.input_listing([("b.csv", 10), ("a.csv", 5000)])
    assert listing.split("\n") == [
        "total 12",
        "drwxr-xr-x 2 user user 4096 Aug 30 23:00 .",
        "drwxr-xr-x 3 user user 4096 Aug 30 23:00 ..",
        "-rw-r--r-- 1 root root 5000 Aug 30 23:00 a.csv",
        "-rw-r--r-- 1 root root   10 Aug 30 23:00 b.csv"]


SYMLINK = ("lrwxrwxrwx 1 evan evan 64 Oct  1 17:17 ./input -> "
           f"/var/tmp/smol-ladder/kaggle/tasks/{TID}")


def translated(*steps, final="The answer is **7**.", prediction="7", files=lambda: [("t.csv", 9)]):
    return V.translate(recorded(*steps, final=final), TID, prediction, files)


def test_a_whole_recording_becomes_the_harness_conversation_and_replays():
    tail, n = translated(
        shell("ls -la ./input && head -2 ./input/t.csv", SYMLINK + "\na,b\n1,2\n"),
        shell("python3 -c 'print(1)' | grep zzz", ""),
        ("write_solution", ({"code": "print(open('./input/t.csv').read()[:1])"},
                            "written /app/solution.py")),
        shell("python3 solution.py", "7\n"),
        shell("python3 bad.py", "\n--- stderr ---\nValueError: x\n"))
    commands = [c["function"]["arguments"]["command"] for m in tail for c in m.get("tool_calls") or []]
    assert commands[0] == "ls -la /home/user/input && head -2 /home/user/input/t.csv"
    assert commands[2] == ("cat > /workdir/solution.py << 'EOF'\n"
                           "print(open('/home/user/input/t.csv').read()[:1])\nEOF")
    assert commands[-1] == 'echo -n "7" > /workdir/answer.txt'
    results = [m["content"] for m in tail if m["role"] == "tool"]
    assert results[0].startswith("total 4\n") and "lrwxrwxrwx" not in results[0]
    assert results[1:3] == [EMPTY_OUTPUT.format(rc=0)] * 2     # empty grep, and the heredoc write
    assert results[3] == "7\n" and results[4] == "ValueError: x\n"
    assert results[5] == EMPTY_OUTPUT.format(rc=0)               # the submission
    assert tail[-1] == {"role": "assistant", "content": "The answer is **7**."}
    assert n["submission_command"] == n["write_solution_wrapper"] == 1
    assert n["empty_result_rc0"] == 1 and n["stderr_appended"] == 1 and n["input_listing"] == 1
    assert all(m["name"] == "bash" for m in tail if m["role"] == "tool")
    row = {"messages": [{"role": "system", "content": BASH_SYSTEM},
                        {"role": "user", "content": "u"}, *tail], "tools": BASH_TOOL}
    assert_replays(row["messages"], strict=True)


def test_output_over_8000_characters_is_cut_by_the_harnesss_own_rule():
    tail, n = translated(shell("cat big", "7\n" + "y" * 9000))
    assert tail[1]["content"] == "7\n" + "y" * 7998 + TRUNCATION_MARKER and n["output_cut_8000"] == 1
    with pytest.raises(V.Refused, match="appears nowhere"):         # the value was in the cut part
        translated(shell("cat big", "y" * 9000 + "\n7\n"))


def test_a_bare_ls_drops_the_recordings_input_entry():
    tail, n = translated(shell("pwd && ls", f"/var/tmp/smol-ladder/scratch/trials/{TID}/L1\n"
                                            "input\nsolution.py\n7\n"))
    assert tail[1]["content"] == "/workdir\nsolution.py\n7\n" and n["cwd_input_entry_dropped"] == 1


@pytest.mark.parametrize("steps,reason", [
    ([shell("ls -la", "total 8\ndrwxrwxr-x 2 evan evan 4096 Oct  1 17:17 .\n7\n")],
     "local username"),
    ([shell("python3 x.py", "File \"/home/evan/proj/.venv/lib/python3.14/x.py\"\n7\n")],
     "home path"),
    ([shell("cat /var/tmp/smol-ladder/kaggle/datasets/o/n/versions/1/t.csv", "7\n")],
     "scratch or cache path"),
    ([shell("cd /home/user && ls", "7\n")], "/home/user or /workdir"),
    ([shell("python3 x.py", "bash: cd: /home/user: No such file or directory\n7\n")],
     "/home/user or /workdir"),
    ([shell("python3 x.py", "[timed out after 150s]")], "timed out"),
    ([("edit_solution", ({"old": "a"}, "ok")), shell("echo 7", "7\n")], "unsupported tool"),
    ([("write_solution", ({"code": "import os\np = os.path.join(here, 'input', 't.csv')"},
                          "written /app/solution.py")), shell("echo 7", "7\n")],
     "relative to the script"),
    ([shell("echo 8", "8\n")], "appears nowhere"),
])
def test_what_the_translation_cannot_make_true_is_refused_with_its_reason(steps, reason):
    with pytest.raises(V.Refused, match=re.escape(reason)):
        translated(*steps)


def test_a_transcript_that_does_not_end_in_text_is_refused():
    messages = recorded(shell("echo 7", "7\n"))
    messages[-1] = {"role": "assistant", "content": "", "tool_calls": [{"id": "z", "type": "function",
                    "function": {"name": "run_shell", "arguments": "{}"}}]}
    with pytest.raises(V.Refused, match="text-only final answer"):
        V.translate(messages, TID, "7")


# ── the scrub knows the intended path ─────────────────────────────────────────────────────────

def test_the_sandbox_path_is_not_a_leak_and_every_other_home_is():
    assert X.leaks_machine("pd.read_csv('/home/user/input/t.csv') in /workdir") == []
    assert X.scrub("/home/user/input/t.csv") == "/home/user/input/t.csv"
    assert X.leaks_machine("/home/somebody/x") == ["home path"]
    assert X.leaks_machine("/home/username/x") == ["home path"]
    assert X.leaks_machine("/Users/somebody/x") == ["home path"]


# ── the user turn is the harness's ────────────────────────────────────────────────────────────

def test_the_user_turn_is_what_the_harness_builds_for_the_same_task(monkeypatch):
    row = {"task_id": TID, "question": "How many rows?", "files": ["t.csv", "u.csv"]}
    built, _ = V.build_row(row, recorded(shell("echo 7", "7\n")), "7")
    assert built["messages"][0] == {"role": "system", "content": BASH_SYSTEM}
    assert built["messages"][1]["content"] == L.prompt_for(row, X.SPLIT, "L1", "bash")
    assert built["messages"][1]["content"] == bash_prompt("How many rows?", ["t.csv", "u.csv"],
                                                          answer_format_of(row))[1]["content"]
    assert built["tools"] == BASH_TOOL and set(built) == {"messages", "tools"}
    with pytest.raises(V.Refused, match="lists no files"):
        V.build_row({**row, "files": []}, recorded(shell("echo 7", "7\n")), "7")


# ── three real exported rows ──────────────────────────────────────────────────────────────────

def test_the_fixture_has_the_three_shapes_it_is_there_for():
    """Row 0: a directory listing, a failed `cd /app`, a bare `pwd && ls` (no `input` entry), a
    heredoc write. Row 1: an empty grep (rc 0) and two parallel calls in one turn. Row 2: a
    traceback whose library frames are at the jail's /usr/local/lib/python3.N."""
    assert len(FIXTURE_ROWS) == 3
    tool = [[m["content"] for m in r["messages"] if m["role"] == "tool"] for r in FIXTURE_ROWS]
    assert tool[0][0].startswith("total ") and "lrwxrwxrwx" not in tool[0][0]
    assert any("cd: /app: No such file or directory" in t for t in tool[0])
    assert any(t.startswith("/workdir\nsolution.py\n") for t in tool[0])
    assert EMPTY_OUTPUT.format(rc=0) in tool[1]
    assert any(len(m.get("tool_calls") or []) == 2 for m in FIXTURE_ROWS[1]["messages"])
    assert any(t.startswith("Traceback") and "/usr/local/lib/python3." in t for t in tool[2])
    assert all(any(c["function"]["arguments"]["command"].startswith("cat > /workdir/solution.py")
                   for m in r["messages"] for c in m.get("tool_calls") or []) for r in FIXTURE_ROWS)


@pytest.mark.parametrize("i", range(3))
def test_fixture_rows_replay_byte_identically_through_the_harness(i):
    assert assert_replays(FIXTURE_ROWS[i]["messages"], strict=True) >= 4


@pytest.mark.parametrize("i", range(3))
def test_fixture_rows_are_clean_and_end_the_way_upstream_rows_do(i):
    row = FIXTURE_ROWS[i]
    messages = row["messages"]
    assert set(row) == {"messages", "tools"} and row["tools"] == BASH_TOOL
    assert messages[0]["content"] == BASH_SYSTEM
    user = messages[1]["content"]
    assert "/home/user/input" in user and "./input" not in user and "\n\n\n" not in user
    assert X.leaks_machine(json.dumps(row)) == []
    assert "./input" not in json.dumps(messages[2:]) and "--- stderr ---" not in json.dumps(messages)
    assert "input/" not in re.sub(r"/home/user/input/", "", json.dumps(messages[2:]))
    submit, receipt, closing = messages[-3:]
    assert re.fullmatch(r'echo -n (".*"|\'.*\') > /workdir/answer\.txt',
                        submit["tool_calls"][0]["function"]["arguments"]["command"])
    assert receipt["content"] == EMPTY_OUTPUT.format(rc=0) and receipt["name"] == "bash"
    assert closing["role"] == "assistant" and closing["content"] and not closing.get("tool_calls")
    for m in messages:
        if m["role"] == "tool":
            assert len(m["content"]) <= 8000 + len(TRUNCATION_MARKER)
            assert m["content"].strip()
        for c in m.get("tool_calls") or []:
            assert isinstance(c["function"]["arguments"], dict)
            assert c["function"]["name"] == "bash"


# ── the whole set, when it is on disk ─────────────────────────────────────────────────────────

@pytest.mark.slow
def test_every_row_of_ja3_sft_v2_replays_and_is_clean():
    if not V2.exists():
        pytest.skip(f"{V2} is not on this machine")
    rows = [json.loads(line) for line in V2.read_text().splitlines()]
    index = [json.loads(line) for line in V2.with_suffix(".index.jsonl").read_text().splitlines()]
    manifest = json.loads(V2.with_suffix(".manifest.json").read_text())
    assert len(rows) == len(index) == manifest["rows"] > 0
    failed, dirty = [], []
    for row, entry in zip(rows, index):
        try:
            assert_replays(row["messages"], strict=True)
        except AssertionError:
            failed.append(entry["task_id"])
        if X.leaks_machine(json.dumps(row, ensure_ascii=False)):
            dirty.append(entry["task_id"])
    assert failed == [], f"{len(failed)} rows do not replay: {failed[:5]}"
    assert dirty == [], f"{len(dirty)} rows leak the machine: {dirty[:5]}"
    assert manifest["replay"]["rate"] == 1.0
    assert len({e["task_id"] for e in index}) == len(index)
    assert all(row["messages"][0]["content"] == BASH_SYSTEM and row["tools"] == BASH_TOOL
               for row in rows)


@pytest.mark.slow
def test_ja3_sft_v2_has_no_heldout_question_and_no_heldout_table():
    if not V2.exists():
        pytest.skip(f"{V2} is not on this machine")
    try:
        keys = X.heldout_keys_for()
    except Exception as e:  # noqa: BLE001 - the SmolDataEnvs splits are not cached here
        pytest.skip(f"held-out splits unavailable: {type(e).__name__}")
    pool = X.pool_rows(X.SPLIT)
    index = [json.loads(line) for line in V2.with_suffix(".index.jsonl").read_text().splitlines()]
    assert [e["task_id"] for e in index if X.row_is_heldout(pool[e["task_id"]], keys)] == []


@pytest.mark.slow
def test_the_fixture_rows_are_rows_of_the_exported_file():
    if not V2.exists():
        pytest.skip(f"{V2} is not on this machine")
    lines = {json.loads(line)["messages"][1]["content"] + json.dumps(json.loads(line)["messages"][2:])
             for line in V2.read_text().splitlines()}
    for row in FIXTURE_ROWS:
        assert row["messages"][1]["content"] + json.dumps(row["messages"][2:]) in lines
