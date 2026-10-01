"""The exporters, the firewall, and the rung scheduler. No GPU, no network, no model.

The tests that matter here are the ones about *refusal*: the firewall has to actually stop a
held-out trajectory, and the scheduler has to actually demote a task. Everything else is plumbing
that would fail loudly on its own.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from train.export_sft import split_train_val
from train.format import (answer_file_command, bash_row, closing_turn, heldout_keys, is_heldout,
                          read_jsonl, shell_turn, submission_turn, write_jsonl)
from train.grpo import RungScheduler, reward_for, shape_bonus
from train.rungs import rung_row
from train.traces import EMPTY_OUTPUT, fallback_turns, parse_transcript, write_program_command

ROW = {"task_id": "t1", "question": "How many?", "files": ["a.csv"], "answer": "4",
       "reward_mode": "numeric", "atol": 0.0, "rtol": 0.0, "bucket_prefix": "owner__ds"}


# ── the firewall ──────────────────────────────────────────────────────────────

def test_heldout_keys_cover_both_the_task_and_the_table():
    keys = heldout_keys({"test": [dict(ROW)], "eval": [{"task_id": "t2",
                                                       "bucket_prefix": "other__ds"}]})
    assert keys["task_id"] == {"t1", "t2"}
    assert keys["bucket_prefix"] == {"owner__ds", "other__ds"}


def test_a_heldout_task_is_refused_by_its_id():
    keys = heldout_keys({"test": [{"task_id": "t1", "bucket_prefix": "x"}]})
    assert is_heldout({"task_id": "t1", "bucket_prefix": "different"}, keys) == "task_id"


def test_a_sibling_question_over_a_heldout_table_is_also_refused():
    """The leak that matters for the ladder: a different question, same table.

    The ladder measures how much information a model needs, and knowing a table's shape and
    vocabulary is exactly the information L2 onwards hands over -- so a sibling question over a
    held-out table trains the model on the ladder's own signal.
    """
    keys = heldout_keys({"test": [{"task_id": "other", "bucket_prefix": "owner__ds"}]})
    assert is_heldout({"task_id": "brand_new", "bucket_prefix": "owner__ds"}, keys) == \
        "bucket_prefix"


def test_a_row_with_neither_key_is_kept():
    keys = heldout_keys({"test": [{"task_id": "t1", "bucket_prefix": "x"}]})
    assert is_heldout({"task_id": "t2", "bucket_prefix": "y"}, keys) is None


def test_the_firewall_does_not_fire_on_a_missing_key():
    """A row with no bucket_prefix (the SFT parquet) must not be treated as matching one."""
    keys = heldout_keys({"test": [{"task_id": "t1", "bucket_prefix": "x"}]})
    assert is_heldout({"task_id": "t2"}, keys) is None


# ── the target format ─────────────────────────────────────────────────────────

def test_a_row_is_messages_and_tools_in_upstreams_shape():
    row = bash_row("Q?", ["a.csv"], [
        {"content": "I'll look."},
        shell_turn("head a.csv", "a\n1\n", "c1"),
        submission_turn(answer_file_command("4")) | {
            "results": [{"tool_call_id": "call_submit", "content": EMPTY_OUTPUT}]},
        closing_turn("Done."),
    ])
    roles = [m["role"] for m in row["messages"]]
    assert roles == ["system", "user", "assistant", "assistant", "tool", "assistant", "tool",
                     "assistant"], roles
    assert row["tools"][0]["function"]["name"] == "bash"
    # arguments are a dict, as upstream's parquet stores them, not a JSON string
    call = row["messages"][3]["tool_calls"][0]
    assert isinstance(call["function"]["arguments"], dict)
    assert call["function"]["arguments"]["command"] == "head a.csv"
    # every tool_call_id is answered, which is what apply_chat_template assumes
    ids = [c["id"] for m in row["messages"] for c in m.get("tool_calls", [])]
    answered = [m["tool_call_id"] for m in row["messages"] if m["role"] == "tool"]
    assert sorted(ids) == sorted(answered)


def test_an_assistant_turn_with_nothing_in_it_is_dropped():
    """Not a shape the template can render; keeping it would put an empty block in the prompt."""
    row = bash_row("Q?", ["a.csv"], [{"content": "", "results": []}])
    assert [m["role"] for m in row["messages"]] == ["system", "user"]


def test_the_submission_command_quotes_the_answer():
    """shlex.quote, because an answer with a quote in it must survive the round trip."""
    assert answer_file_command("Male") == "echo -n Male > /workdir/answer.txt"
    tricky = answer_file_command("Bob's 5' pipe")
    assert "'" in tricky and tricky.endswith(" > /workdir/answer.txt")


def test_a_program_is_written_through_a_quoted_heredoc():
    """An unquoted heredoc would expand $ and backticks -- and we would be training that."""
    command = write_program_command("print('$HOME')")
    assert command.startswith("cat > solution.py << 'SMOL_EOF'")
    assert command.endswith("SMOL_EOF")
    assert "print('$HOME')" in command


def test_the_rung_row_has_no_tools_because_the_program_protocol_has_none():
    row = rung_row("Q?", ["a.csv"], "the prompt", "print(4)", "4")
    assert row["tools"] == []
    assert row["messages"][-1]["content"].startswith("```python\nprint(4)\n```")


# ── our traces ────────────────────────────────────────────────────────────────

def _event(**kwargs):
    return json.dumps({"type": "event", "event": kwargs})


def test_a_cmd_event_stream_becomes_bash_turns(tmp_path):
    """The real transcript.jsonl shape: deltas, a settled message_end, then partial tool output."""
    lines = [
        _event(type="run_start", sessionId="s"),
        _event(type="text_delta", delta="I'll st"),
        _event(type="message_end", content=[{"type": "text", "text": "I'll start."},
                                           {"type": "tool_use", "id": "t1", "name": "shell_command",
                                            "input": {"command": "ls input"}}]),
        _event(type="tool_update", toolCallId="t1", partial=[{"type": "text", "text": "a.csv\n"}]),
        _event(type="model_request_end", usage={"inputTokens": 5}),
    ]
    path = tmp_path / "transcript.jsonl"
    path.write_text("\n".join(lines) + "\n")
    turns = parse_transcript(path)
    assert len(turns) == 1
    assert turns[0]["tool_calls"][0]["function"]["arguments"]["command"] == "ls input"
    assert turns[0]["results"][0]["content"] == "a.csv\n"


def test_the_stream_assembles_from_settled_events_not_from_deltas(tmp_path):
    """A transcript built from text_delta would contain every streaming state of the answer."""
    path = tmp_path / "transcript.jsonl"
    path.write_text("\n".join([
        _event(type="text_delta", delta="Hel"),
        _event(type="text_delta", delta="Hello"),
        _event(type="message_end", content=[{"type": "text", "text": "Hello"}]),
    ]) + "\n")
    turns = parse_transcript(path)
    assert [t["content"] for t in turns] == ["Hello"]


def test_the_fallback_trajectory_writes_the_real_program_and_submits():
    turns = fallback_turns("print(6*7)\n", "42")
    commands = [t["tool_calls"][0]["function"]["arguments"]["command"]
                for t in turns if t.get("tool_calls")]
    assert len(commands) == 2, commands
    assert "print(6*7)" in commands[0], "the verified program is what the model is taught to write"
    assert commands[1] == "echo -n 42 > /workdir/answer.txt"
    # No invented exploration. Checked on the commands rather than on the serialised JSON, where
    # "answer.txt" would match a substring search for "ls".
    assert not any(c.strip().startswith(("ls", "head", "cat input", "python3"))
                   for c in commands)


def test_the_fallback_still_ends_with_a_submission_and_a_stop():
    turns = fallback_turns("print(1)", "1")
    assistant = [t for t in turns if t.get("tool_calls")]
    assert assistant[-1]["tool_calls"][0]["function"]["arguments"]["command"].endswith(
        " > /workdir/answer.txt")
    assert "tool_calls" not in turns[-1]


def test_a_solution_with_no_transcript_yields_the_fallback_not_nothing(tmp_path, monkeypatch):
    """The path that actually runs today: a verified trial with a program and no conversation."""
    from train import traces

    root = tmp_path / "runs" / "jupyter-agent"
    work = root / "ja_1" / "L1"
    work.mkdir(parents=True)
    (work / "result.json").write_text(json.dumps({"task_id": "ja_1", "reward": 1.0,
                                                  "prediction": "42"}))
    (work / "solution.py").write_text("print(42)\n")
    monkeypatch.setattr(traces, "SOURCES", (("runs/jupyter-agent", "jupyter-agent"),))
    monkeypatch.setattr(traces, "task_rows", lambda source: {"ja_1": dict(ROW)})
    rows, dropped = traces.collect_traces(tmp_path, keys={"task_id": set(), "bucket_prefix": set()})
    assert len(rows) == 1
    assert dropped == {}
    assert "print(42)" in json.dumps(rows[0])
    assert "echo -n 42 > /workdir/answer.txt" in json.dumps(rows[0])


def test_a_trial_that_failed_is_not_exported(tmp_path, monkeypatch):
    """Only verified trials: reward comes from the sealed offline pass, not the agent's claim."""
    from train import traces

    work = tmp_path / "runs" / "jupyter-agent" / "ja_1" / "L1"
    work.mkdir(parents=True)
    (work / "result.json").write_text(json.dumps({"task_id": "ja_1", "reward": 0.0,
                                                  "prediction": "wrong"}))
    (work / "solution.py").write_text("print(0)\n")
    monkeypatch.setattr(traces, "SOURCES", (("runs/jupyter-agent", "jupyter-agent"),))
    monkeypatch.setattr(traces, "task_rows", lambda source: {"ja_1": dict(ROW)})
    rows, _ = traces.collect_traces(tmp_path, keys={"task_id": set(), "bucket_prefix": set()})
    assert rows == []


def test_a_heldout_task_is_refused_and_counted(tmp_path, monkeypatch):
    from train import traces

    work = tmp_path / "runs" / "jupyter-agent" / "ja_1" / "L1"
    work.mkdir(parents=True)
    (work / "result.json").write_text(json.dumps({"task_id": "ja_1", "reward": 1.0,
                                                  "prediction": "42"}))
    (work / "solution.py").write_text("print(42)\n")
    monkeypatch.setattr(traces, "SOURCES", (("runs/jupyter-agent", "jupyter-agent"),))
    monkeypatch.setattr(traces, "task_rows", lambda source: {"ja_1": dict(ROW)})
    rows, dropped = traces.collect_traces(
        tmp_path, keys={"task_id": {"t1"}, "bucket_prefix": set()})
    assert rows == []
    assert dropped == {"task_id": 1}


def test_an_unprintable_trial_is_not_exported(tmp_path, monkeypatch):
    """No prediction means the sealed run printed nothing, so there is no value to submit."""
    from train import traces

    work = tmp_path / "runs" / "jupyter-agent" / "ja_1" / "L1"
    work.mkdir(parents=True)
    (work / "result.json").write_text(json.dumps({"task_id": "ja_1", "reward": 1.0,
                                                  "prediction": ""}))
    (work / "solution.py").write_text("pass\n")
    monkeypatch.setattr(traces, "SOURCES", (("runs/jupyter-agent", "jupyter-agent"),))
    monkeypatch.setattr(traces, "task_rows", lambda source: {"ja_1": dict(ROW)})
    rows, _ = traces.collect_traces(tmp_path, keys={"task_id": set(), "bucket_prefix": set()})
    assert rows == []


# ── the split ─────────────────────────────────────────────────────────────────

def test_the_train_val_split_is_deterministic_across_calls():
    rows = [{"i": i} for i in range(100)]
    a = split_train_val(rows, 0.1, 42)
    b = split_train_val([dict(r) for r in rows], 0.1, 42)
    assert a == b


def test_the_split_sizes_and_a_different_seed_gives_a_different_split():
    rows = [{"i": i} for i in range(100)]
    train, val = split_train_val(rows, 0.2, 42)
    assert len(train) == 80 and len(val) == 20
    train2, _ = split_train_val(rows, 0.2, 7)
    assert train2 != train


# ── the rung scheduler ────────────────────────────────────────────────────────

def test_a_fresh_task_starts_at_the_high_rung():
    s = RungScheduler(start="L3", enabled=True)
    assert s.rung("t1") == "L3"


def test_a_task_the_model_learns_is_demoted_toward_L1():
    """The curriculum: more help while failing, less once it passes."""
    s = RungScheduler(start="L3", window=2, enabled=True)
    assert s.update("t1", [1.0, 1.0]) == "L2"
    assert s.update("t1", [1.0, 1.0]) == "L1"
    assert s.update("t1", [1.0, 1.0]) == "L1", "L1 is the floor"


def test_a_task_it_cannot_solve_is_promoted_for_more_help():
    s = RungScheduler(start="L1", window=2, enabled=True)
    assert s.update("t1", [0.0, 0.0]) == "L2"
    assert s.update("t1", [0.0, 0.0]) == "L3"


def test_the_rung_does_not_flicker_on_a_task_halving_its_passes():
    """A 50% task would oscillate with thresholds on either side of 0.5; these are far apart."""
    s = RungScheduler(start="L3", window=4, enabled=True)
    for _ in range(6):
        rung = s.update("t1", [1.0, 0.0, 1.0, 0.0])
    assert rung == "L3", s.history["t1"]


def test_the_window_slides_so_a_regression_can_demote_again():
    """A rolling rate, not a lifetime one: a task that stops passing loses its rung again."""
    s = RungScheduler(start="L3", window=4, enabled=True)
    s.update("t1", [1.0] * 4)
    assert s.rate("t1") == 1.0
    s.update("t1", [0.0] * 4)
    assert s.rate("t1") == 0.0


def test_tasks_are_scheduled_independently():
    s = RungScheduler(start="L3", window=2, enabled=True)
    s.update("easy", [1.0, 1.0])
    assert s.rung("hard") == "L3"
    assert s.rung("easy") == "L2"


def test_the_schedule_can_be_switched_off():
    s = RungScheduler(start="L3", enabled=False)
    assert s.rung("t1") == "L1"
    assert s.update("t1", [1.0, 1.0]) == "L1"


def test_a_start_below_the_floor_is_refused():
    with pytest.raises(ValueError):
        RungScheduler(start="L1", floor="L3")


def test_an_unknown_rung_is_refused():
    with pytest.raises(ValueError):
        RungScheduler(start="L9")


# ── the reward ────────────────────────────────────────────────────────────────

def test_the_shaping_bonus_requires_output_not_a_clean_exit():
    """Upstream's first version paid for "no traceback", which an empty program satisfies."""
    assert shape_bonus("42") == 1.0
    assert shape_bonus("") == 0.0
    assert shape_bonus("   \n  ") == 0.0


def test_the_reward_is_the_last_line_of_the_sealed_run(monkeypatch):
    monkeypatch.setattr("smol_ladder.grade.grade", lambda row, prediction: 1.0)
    result = reward_for(ROW, "", "loading the table\n42\n")
    assert result["prediction"] == "42", result
    assert result["shaping"] == 1.0


def test_a_program_that_prints_nothing_scores_zero_on_both(monkeypatch):
    monkeypatch.setattr("smol_ladder.grade.grade", lambda row, prediction: 0.0)
    result = reward_for(ROW, "", "")
    assert result["reward"] == 0.0
    assert result["shaping"] == 0.0


def test_the_reward_funcs_grade_a_completion_with_a_stub_sandbox():
    """The wiring, without a GPU: a fenced program, a stubbed sandbox, our grader."""
    import train.grpo as grpo

    calls = []

    def sandbox(row, code):
        calls.append(code)
        return "42\n"

    funcs, shaping_log = grpo.build_reward_funcs(sandbox=sandbox)
    rewards = funcs[0]([[{"role": "assistant", "content": "```python\nprint(42)\n```"}]],
                      row=[dict(ROW)])
    # The grader is SmolDataEnvs' own and is not stubbed here: ROW's answer is "4" and the program
    # prints 42, so this asserts the *real* grade of a real mismatch is 0.0, and that the sandbox
    # saw the extracted program rather than the surrounding prose.
    assert rewards == [0.0], rewards
    assert calls == ["print(42)"]
    assert shaping_log == [1.0]
    assert funcs[1]([[{"role": "assistant", "content": "x"}]]) == [0.0]


def test_the_reward_funcs_score_a_correct_program_one():
    import train.grpo as grpo

    row = dict(ROW, answer="42")
    funcs, _ = grpo.build_reward_funcs(sandbox=lambda row, code: "42\n")
    rewards = funcs[0]([[{"role": "assistant", "content": "```python\nprint(42)\n```"}]], row=[row])
    assert rewards == [1.0], rewards


def test_the_shaping_diagnostic_is_weight_zero_by_default():
    """The guard against repeating upstream's reward hack: the bonus is logged, not optimised."""
    import train.grpo as grpo

    assert grpo.SHAPING_WEIGHT == 0.0
    # And the reward function that would carry it returns a constant zero, so even a caller who
    # sets --shaping-weight non-zero has to change this function to make the bonus mean anything.
    funcs, _ = grpo.build_reward_funcs(sandbox=lambda row, code: "42\n")
    assert funcs[1]([[{"role": "assistant", "content": "x"}]] * 3) == [0.0, 0.0, 0.0]