"""The lean rewrite of arm B (train/export_ja3_v3.py): what it removes, folds and keeps."""
import json

from train import export_ja3_v3 as E

TOOLS = [{"type": "function", "function": {"name": "bash"}}]
WRITE = "cat > /workdir/solution.py << 'EOF'\nprint(3.95)\nEOF"


def row(*steps, closing="**3.95**"):
    """steps: (prose, [(command, output), ...]) per assistant turn."""
    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "q"}]
    n = 0
    for prose, calls in steps:
        ids = []
        for command, _ in calls:
            n += 1
            ids.append(f"c{n}")
        messages.append({"role": "assistant", "content": prose, "tool_calls": [
            {"id": i, "type": "function", "function": {"name": "bash", "arguments": {"command": c}}}
            for i, (c, _) in zip(ids, calls)]})
        messages += [{"role": "tool", "tool_call_id": i, "name": "bash", "content": o}
                     for i, (_, o) in zip(ids, calls)]
    return messages + [{"role": "assistant", "content": closing}]


def test_the_write_and_the_run_become_one_command_with_the_runs_output():
    messages, counts = E.lean(row(
        ("look", [("ls -la /home/user/input", "a.csv\n")]),
        ("Now the script.", [(WRITE, E.EMPTY)]),
        ("", [("cd . && python3 solution.py", "3.95\n")]),
        ("", [('echo -n "3.95" > /workdir/answer.txt', E.EMPTY)]),
        closing="The script prints 3.95."))
    assert E.commands(messages) == ["ls -la /home/user/input", WRITE + "\npython3 solution.py",
                                    'echo -n "3.95" > /workdir/answer.txt']
    folded = next(m for m in messages if m["role"] == "tool" and m["tool_call_id"] == "c2")
    assert folded["content"] == "3.95\n" and counts["write_and_run_folded"] == 1
    assert [m["role"] for m in messages[2:]] == ["assistant", "tool"] * 3 + ["assistant"]


def test_version_checks_and_the_failed_cd_app_are_dropped_and_prose_moves_on():
    messages, counts = E.lean(row(
        ("I'll start.", [("ls -la /home/user/input", "a.csv\n"),
                         ('cd . && python3 -c "import pandas; print(pandas.__version__)"', "3.0.6\n")]),
        ("", [(WRITE, E.EMPTY)]),
        ("Run it.", [("cd /app && python3 solution.py", "bash: line 1: cd: /app: No such file or directory\n")]),
        ("", [("pwd && ls && python3 solution.py", "/workdir\nsolution.py\n3.95\n")]),
        ("", [('echo -n "3.95" > /workdir/answer.txt', E.EMPTY)]),
        closing="The script prints 3.95."))
    assert E.commands(messages) == ["ls -la /home/user/input",
                                    WRITE + "\npwd && ls && python3 solution.py",
                                    'echo -n "3.95" > /workdir/answer.txt']
    assert counts["version_check_dropped"] == 1 and counts["failed_cd_app_dropped"] == 1
    assert "/app" not in json.dumps(messages)


def test_the_script_goes_only_when_an_earlier_command_ended_in_the_answer_and_nobody_mentions_it():
    steps = (("", [('python3 -c "print(3.95)"', "rows 10\n3.95\n")]),
             ("Writing the solution script.", [(WRITE, E.EMPTY)]),
             ("", [("python3 solution.py", "3.95\n")]),
             ("", [('echo -n "3.95" > /workdir/answer.txt', E.EMPTY)]))
    dropped, counts = E.lean(row(*steps))
    assert E.commands(dropped) == ['python3 -c "print(3.95)"', 'echo -n "3.95" > /workdir/answer.txt']
    assert counts["rows_solution_dropped"] == 1
    kept, _ = E.lean(row(*steps, closing="The script prints 3.95."))
    assert len(E.commands(kept)) == 3 and "solution.py" in E.commands(kept)[1]


def test_the_written_file_is_leaner_replays_and_leaves_v2_alone():
    v2 = (E.DATA / "train" / "ja3_sft_v2.jsonl")
    v3 = (E.DATA / "train" / "ja3_sft_v3.jsonl")
    if not (v2.exists() and v3.exists()):
        import pytest
        pytest.skip("the exports are not on this machine")
    rows = [json.loads(x) for x in v3.read_text().splitlines()]
    manifest = json.loads(v3.with_name("ja3_sft_v3.manifest.json").read_text())
    assert len(rows) == manifest["rows"] and manifest["source_rows"] == len(v2.read_text().splitlines())
    for r in rows:
        cmds = E.commands(r["messages"])
        assert max(map(len, cmds)) <= manifest["max_command"]
        assert not any(c.startswith("cd . && ") or E.is_version_check(c) for c in cmds)
        assert set(r) == {"messages", "tools"}
