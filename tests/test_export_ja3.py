"""The four export rules, enforced at export time rather than reported.

`train/export_ja3.py` ships the ja3 transcript sweep as SFT data. The rules that make it usable
are not stylistic: a leaked `/home/evan/` teaches a model about this machine, a trajectory that
ends mid-exploration teaches it to stop early, a held-out table teaches it the benchmark, and a
hint in the prompt teaches it the answer. Each is checked in code, and each has a test here that
fails when the check is removed.

    uv run --with pytest pytest -q tests/test_export_ja3.py
"""

from __future__ import annotations

import json
import socket
from pathlib import Path

import pytest

from train import export_ja3 as X


# ── rule 2: the scrub ────────────────────────────────────────────────────────────────────────

def test_a_local_home_path_becomes_a_neutral_input_file():
    text = ("lrwxrwxrwx 1 evan evan 64 Oct  1 17:17 ./input -> "
            "/home/evan/Documents/smol-ladder/kaggle/tasks/ja_1\n")
    clean = X.scrub(text)
    assert X.leaks_machine(clean) == [], clean
    assert "./input" in clean


def test_the_ls_owner_column_does_not_keep_the_username():
    """The sweep's first command is always `ls -la ./input`, and its output is three columns of
    permissions, this account's name, and a date. Replacing a path cannot fix that column."""
    clean = X.scrub("lrwxrwxrwx 1 evan evan 64 Oct  1 17:17 ./input -> ./input")
    assert not X.leaks_machine(clean), clean


def test_the_kaggle_cache_layout_becomes_the_table_itself():
    text = "lrwxrwxrwx 1 evan evan 64 Jan 1 17:17 ./input/x.csv -> " \
           "/var/tmp/smol-ladder/kaggle/datasets/4quant/simplefoam/versions/2/bubble_volume.csv"
    clean = X.scrub(text)
    assert "/var/tmp/" not in clean, clean
    assert "kaggle/datasets" not in clean, clean
    assert "./input/bubble_volume.csv" in clean, clean


def test_the_worktree_and_scratch_layouts_are_removed():
    for path in ["/home/evan/Documents/smol-ladder-wt-release/data/runs/ja3",
                 "/tmp/smol-ladder/scratch/trials/t1/L1"]:
        assert not X.leaks_machine(X.scrub(f"see {path} for it")), path


def test_an_api_key_is_removed_whatever_it_is_spelled():
    for secret in ["sk-or-v1-abcdef0123456789abcdef01",
                   "hf_abcdefghijklmnopqrstuvwxyz0123",
                   "OPENROUTER_API_KEY=abcdef0123456789abcdef",
                   "Authorization: Bearer abcdef0123456789"]:
        clean = X.scrub(f"curl -H 'x: {secret}'")
        assert "api key" not in str(X.leaks_machine(clean)), f"{secret} survived: {clean}"
        assert "abcdef0123456789" not in clean, clean


def test_the_local_username_and_hostname_are_removed():
    for text in [f"owned by {X.LOCAL_USER}", f"host {socket.gethostname()}"]:
        clean = X.scrub(text)
        assert not X.leaks_machine(clean), f"{text!r} -> {clean!r}"


def test_a_clean_row_survives_the_scrub_unchanged():
    row = {"messages": [{"role": "assistant", "content": "**Answer: 141**"},
                        {"role": "tool", "content": "celebrity_deaths_4.csv 21458 rows"}],
           "tools": [{"type": "function", "function": {"name": "bash"}}]}
    clean, leaked = X.scrub_row(row)
    assert leaked == []
    assert clean == row


def test_the_export_fails_rather_than_writing_a_dirty_row():
    """The requirement is a hard stop, not a warning: the caller must not get a file to train on
    when the scrub could not do its job.

    The username is covered, so this pins the *mechanism* rather than a leak: `scrub_row` reports
    every rule the checker knows, and refuses the row when the checker finds anything. A leak the
    checker does not know about is a hole in `leaks_machine`, and this is where it would show up.
    """
    row = {"messages": [{"role": "tool", "content": "cat ./input/answer.csv"}]}
    clean, leaked = X.scrub_row(row)
    assert leaked == []
    assert clean == row
    assert X.leaks_machine("/home/somebody-else/x.csv") == ["home path"]
    assert X.leaks_machine("/var/tmp/smol-ladder/x") == ["scratch or cache path"]
    assert X.leaks_machine("kaggle/datasets/o/d/versions/1/x.csv") == ["kaggle cache layout"]
    assert X.leaks_machine("hf_abcdefghijklmnopqrstuvwxyz0123") == ["api key or token"]


def test_a_username_inside_an_absolute_path_is_scrubbed_by_the_path_rule():
    """The common case: the username only ever appears inside a path, and `/home/<user>` is
    replaced whole before the bare-name rule can see it."""
    row = {"messages": [{"role": "tool", "content": f"cat /home/{X.LOCAL_USER}/x/answer.csv"}]}
    clean, leaked = X.scrub_row(row)
    assert leaked == [], leaked
    assert X.LOCAL_USER not in json.dumps(clean)


def test_a_scrub_that_breaks_the_json_is_a_refusal_not_a_row():
    row = {"messages": [{"role": "tool", "content": '{"path": "/home/evan/x"}'}]}
    clean, leaked = X.scrub_row(row)
    assert leaked == ["row is no longer valid JSON after scrubbing"], leaked
    assert clean == {}


# ── rule 3: the trajectory ends with the graded answer ────────────────────────────────────────

def _conversation(prediction="141", last="**Answer: 141**"):
    """One tool-using exchange, then the model's closing answer -- the shape `or_agent.solve_loop`
    returns: every assistant message that made a call is followed by the tool message carrying that
    call's id and its real output."""
    return [
        {"role": "system", "content": "You are solving a data-analysis question."},
        {"role": "user", "content": "Question: How many celebrities died of accidents?"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c0", "type": "function",
             "function": {"name": "run_shell", "arguments": '{"command":"wc -l ./input/x.csv"}'}}]},
        {"role": "tool", "tool_call_id": "c0", "content": "21458 celebrity_deaths_4.csv"},
        {"role": "assistant", "content": "Now let me compute it."},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "run_shell", "arguments": '{"command":"python3 solution.py"}'}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "141\n"},
        {"role": "assistant", "content": last},
    ]


def test_the_final_answer_is_located_and_matches_the_graded_prediction():
    index, ok = X.final_answer_text(_conversation(), "141")
    assert ok
    assert index == 7


def test_a_prediction_the_transcript_never_states_is_refused():
    """The sealed pass graded 141; the model's own last words say 87. Ending the trajectory there
    would teach "stop after a wrong answer", and the submission turn would contradict it."""
    index, ok = X.final_answer_text(_conversation(last="The answer is 87."), "141")
    assert not ok
    assert index == 7, "the cut point is still known, so the caller can say where it stopped"


def test_a_trajectory_that_ends_mid_exploration_is_refused():
    truncated = _conversation()[:-1] + [{"role": "tool", "tool_call_id": "c1", "content": "141\n"}]
    index, ok = X.final_answer_text(truncated, "141")
    assert not ok


def test_a_conversation_with_no_assistant_turn_is_refused():
    assert X.final_answer_text([{"role": "user", "content": "q"}], "141") == (-1, False)


def test_the_row_ends_with_a_submission_of_the_graded_value():
    """Upstream's rows end with `echo -n "<value>" > /workdir/answer.txt` and nothing after it, so
    the model has to learn to stop there. The value is the one the sealed grader gave 1.0."""
    turns = X.to_bash_turns(_conversation(), 7, "141")
    last = turns[-1]
    command = last["tool_calls"][0]["function"]["arguments"]["command"]
    assert command == "echo -n 141 > /workdir/answer.txt", command
    assert last["tool_calls"][0]["function"]["name"] == "bash"
    assert len(last["results"]) == 1


def test_the_conversation_before_the_cut_is_kept_and_renamed():
    """A trace dataset is only worth training on if the exploration survives: the cut keeps every
    turn before the final answer, with the two tools collapsed onto the one the format has."""
    turns = X.to_bash_turns(_conversation(), 7, "141")
    names = {call["function"]["name"]
             for turn in turns for call in turn.get("tool_calls") or []}
    assert names == {"bash"}, names
    assert any("celebrity" in (result["content"] or "")
               for turn in turns for result in turn.get("results") or [])
    assert any("Now let me compute it." in (turn.get("content") or "") for turn in turns)


def test_a_json_string_argument_and_a_dict_argument_are_both_read():
    """`or_agent` logs what the endpoint returned (a JSON string); upstream's parquet stores a
    dict. Both have to produce the same row, and an unreadable one must not become a call with an
    empty command."""
    messages = _conversation()
    for arguments in ['{"command":"python3 solution.py"}', {"command": "python3 solution.py"}]:
        messages[5]["tool_calls"][0]["function"]["arguments"] = arguments
        turns = X.to_bash_turns(messages, 7, "141")
        command = turns[2]["tool_calls"][0]["function"]["arguments"]["command"]
        assert command == "python3 solution.py", (arguments, command)
    assert X.turn_arguments({"function": {"arguments": "not json"}}) == {}


def test_the_sweep_scratch_prefix_is_not_taught_to_the_model():
    messages = _conversation()
    messages[5]["tool_calls"][0]["function"]["arguments"] = \
        '{"command":"cd . && python3 solution.py"}'
    turns = X.to_bash_turns(messages, 7, "141")
    command = turns[2]["tool_calls"][0]["function"]["arguments"]["command"]
    assert command == "python3 solution.py", command


def test_a_tool_result_lands_under_the_call_that_produced_it():
    """Call ids are rewritten during the translation, so a result that matched its old id would
    otherwise attach to nothing -- and an unattached result is a command's output under a
    different command, which renders and trains and is simply false."""
    turns = X.to_bash_turns(_conversation(), 7, "141")
    outputs = [result["content"] for turn in turns for result in turn.get("results") or []]
    assert "21458 celebrity_deaths_4.csv" in outputs, outputs


# ── rule 4: no hints, no gold ────────────────────────────────────────────────────────────────

ROW = {"task_id": "ja_1", "answer": "141", "reward_mode": "numeric", "atol": 1e-4,
       "rtol": 1e-4, "question": "How many celebrities died as a result of accidents?",
       "kaggle_dataset_name": "someone/celebrity-deaths"}


def test_a_plain_l1_prompt_is_accepted():
    assert X.hints_absent(_conversation(), ROW) == []


def test_a_hint_in_the_prompt_is_refused():
    messages = _conversation()
    messages[1]["content"] += "\n\nNotes on the intended computation:\nColumns used: cause_of_death"
    assert any("hint marker" in reason for reason in X.hints_absent(messages, ROW))


def test_the_gold_answer_in_the_prompt_is_refused():
    """Delegated to `ladder.leaks`, which is the check every rung in this project is held to, so a
    row is refused on exactly the standard a hint would be. Its first guard is the answer's length:
    under four normalised characters it matches by chance ("3" is in every column dump), so this
    uses a task whose gold answer is long enough to be evidence."""
    row = {**ROW, "answer": "88.523142", "reward_mode": "numeric"}
    messages = _conversation()
    messages[1]["content"] += "\n\n(Across the rows the total is 88.523142.)"
    assert any("gold answer in the prompt" in reason
               for reason in X.hints_absent(messages, row))


def test_a_short_answer_in_prose_is_not_treated_as_a_leak():
    """`ladder.leaks` skips an answer under four characters, because "3" occurs in every column
    dump. The export delegates rather than reimplementing, so a row with "no" in the prompt is
    not refused for it -- and the export must not grow a second, stricter rule that would silently
    change which rows a ladder-trained set contains. It is *counted* instead."""
    messages = _conversation()
    messages[1]["content"] += "\n\nAnswer yes or no."
    assert X.hints_absent(messages, {**ROW, "answer": "no"}) == []


def test_rows_whose_answer_is_below_the_leak_floor_are_counted(tmp_path, monkeypatch):
    """So the manifest can state the fact rather than a reader discovering it from a training run.
    The fixture's answer is "141" -- three normalised characters, under `ladder.MIN_LEAK_LEN`."""
    _fixture(tmp_path, monkeypatch)

    rows, report = X.export_traces(tmp_path)

    assert len(rows) == 1
    assert report["gold_answer_below_the_leak_floor"] == 1, report


# ── rule 1: the firewall ──────────────────────────────────────────────────────────────────────

def test_the_bare_dataset_name_is_the_key_a_task_is_fired_on():
    """The same key `jtasks_v2` built v3 on: the bare Kaggle name, so a mirror under another
    owner still fires. A slug comparison is how the 471-vs-122 leak happened."""
    keys = {"kaggle_table": {"simplefoam"}}
    assert X.row_is_heldout({**ROW, "kaggle_dataset_name": "a-mirror/simplefoam"}, keys) == \
        "kaggle_table"
    assert X.row_is_heldout({**ROW, "kaggle_dataset_name": "other/table"}, keys) is None


def test_a_heldout_question_is_refused():
    keys = {"question": {ROW["question"].lower()}, "task_id": set()}
    assert X.row_is_heldout({**ROW, "kaggle_dataset_name": "other/table"}, keys) == "question"


def test_a_task_with_neither_is_kept():
    keys = {"question": {"something else"}, "task_id": {"t9"}}
    assert X.row_is_heldout(ROW, keys) is None


def test_the_firewall_reads_the_splits_and_not_the_pool_tag(tmp_path, monkeypatch):
    """A tag on a row is a claim about how the pool was built; the split is the fact. Re-deriving
    the keys from `load_split` means a pool built by older code is still checked."""
    keys = X.heldout_keys_for.__wrapped__ if hasattr(X.heldout_keys_for, "__wrapped__") else None
    assert keys is None, "heldout_keys_for must not be a cached fixture"

    calls: list[str] = []

    def fake_load_split(split):
        calls.append(split)
        return [{"task_id": "t1", "question": "Q?", "bucket_prefix": "x"}]

    def fake_datasets(splits=("test",)):
        return {"owner/held-out-table"}

    import smol_ladder.tasks as tasks

    monkeypatch.setattr(tasks, "load_split", fake_load_split)
    monkeypatch.setattr(X, "smoldataenvs_datasets", fake_datasets)

    keys = X.heldout_keys_for()

    assert calls == ["test", "eval"], calls
    assert "held-out-table" in keys["kaggle_table"]
    assert "q?" in keys["question"]


# ── the whole export, on a fixture tree ───────────────────────────────────────────────────────

def _fixture(tmp_path, monkeypatch, *, prediction="141", last="**Answer: 141**",
             reward=1.0, agent_status="exit 0", verify_status=None, task_id="ja_1"):
    """A one-task runs tree and pool, and the keys the firewall sees."""
    root = tmp_path / "runs" / "ja3" / "jupyter-agent-v3"
    trial = root / task_id / "L1"
    trial.mkdir(parents=True)
    result = {"task_id": task_id, "reward": reward, "agent_status": agent_status,
              "prediction": prediction, "rung": "L1", "sample": 0}
    if verify_status:
        result["verify_status"] = verify_status
    (trial / "result.json").write_text(json.dumps(result))
    (trial / "transcript.json").write_text(json.dumps(_conversation(prediction, last)))
    monkeypatch.setattr(X, "pool_rows", lambda split: {task_id: ROW})
    monkeypatch.setattr(X, "heldout_keys_for", lambda *a, **k: {
        "question": set(), "task_id": set(), "bucket_prefix": set(), "kaggle_table": set()})
    return root


def test_a_good_trial_is_exported_as_one_row(tmp_path, monkeypatch):
    _fixture(tmp_path, monkeypatch)

    rows, report = X.export_traces(tmp_path)

    assert len(rows) == 1, report
    assert set(rows[0]) == {"messages", "tools"}, sorted(rows[0])
    assert report["dropped"] == {}


def test_a_trial_that_did_not_re_run_sealed_is_not_exported(tmp_path, monkeypatch):
    _fixture(tmp_path, monkeypatch, verify_status="verify timeout")

    rows, report = X.export_traces(tmp_path)

    assert rows == []
    assert report["verified_trials"] == 0


def test_a_trial_with_no_transcript_is_dropped_and_counted(tmp_path, monkeypatch):
    root = _fixture(tmp_path, monkeypatch)
    (root / "ja_1" / "L1" / "transcript.json").unlink()

    rows, report = X.export_traces(tmp_path)

    assert rows == []
    assert report["dropped"] == {"no prediction or no transcript": 1}


def test_a_held_out_task_is_dropped_and_counted_by_key(tmp_path, monkeypatch):
    root = _fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(X, "heldout_keys_for", lambda *a, **k: {
        "question": set(), "task_id": set(), "bucket_prefix": set(),
        "kaggle_table": {"celebrity-deaths"}})

    rows, report = X.export_traces(tmp_path)

    assert rows == []
    assert report["dropped"] == {"heldout:kaggle_table": 1}, report["dropped"]


def test_a_transcript_that_never_states_the_answer_is_dropped_and_counted(tmp_path, monkeypatch):
    _fixture(tmp_path, monkeypatch, last="I could not work it out.")

    rows, report = X.export_traces(tmp_path)

    assert rows == []
    assert report["dropped"] == {"transcript does not end with the graded answer": 1}


def test_a_transcript_that_does_not_parse_is_dropped_and_counted(tmp_path, monkeypatch):
    root = _fixture(tmp_path, monkeypatch)
    (root / "ja_1" / "L1" / "transcript.json").write_text("{not json")

    rows, report = X.export_traces(tmp_path)

    assert rows == []
    assert report["dropped"] == {"transcript does not parse": 1}


def test_a_hinted_prompt_is_dropped_and_counted(tmp_path, monkeypatch):
    root = _fixture(tmp_path, monkeypatch)
    messages = _conversation()
    messages[1]["content"] += "\n\nNotes on the intended computation:\nColumns used: x"
    (root / "ja_1" / "L1" / "transcript.json").write_text(json.dumps(messages))

    rows, report = X.export_traces(tmp_path)

    assert rows == []
    assert any("hint marker" in key for key in report["dropped"]), report["dropped"]


def test_a_dirty_command_is_scrubbed_rather_than_refused(tmp_path, monkeypatch):
    """A leaked path in a command is fixable, so the row is fixed and kept; only a string the
    scrub cannot remove refuses the row."""
    root = _fixture(tmp_path, monkeypatch)
    messages = _conversation()
    messages[5]["tool_calls"][0]["function"]["arguments"] = \
        '{"command":"ls -la /var/tmp/smol-ladder/kaggle/datasets/owner/ds/versions/1"}'
    (root / "ja_1" / "L1" / "transcript.json").write_text(json.dumps(messages))

    rows, report = X.export_traces(tmp_path)

    assert len(rows) == 1, report
    commands = [call["function"]["arguments"]["command"]
                for message in rows[0]["messages"] for call in (message.get("tool_calls") or [])]
    assert "/var/tmp/" not in json.dumps(rows[0]), json.dumps(rows[0])[:400]
    assert "kaggle/datasets" not in json.dumps(rows[0])
    assert any(command.startswith("ls -la") for command in commands), commands


def test_the_manifest_states_what_was_refused_and_how_it_was_measured(tmp_path, monkeypatch):
    _fixture(tmp_path, monkeypatch)
    rows, report = X.export_traces(tmp_path)

    path = X.manifest("ja3_sft", rows, X.token_stats(rows),
                      {"dropped": report["dropped"],
                       "files": {"ja3_sft.jsonl": str(tmp_path / "ja3_sft.jsonl")},
                       "firewall_keys": {"kaggle_table": 170}}, tmp_path)

    record = json.loads(path.read_text())
    assert record["rows"] == 1
    assert record["files"]["ja3_sft.jsonl"].endswith("ja3_sft.jsonl")
    assert record["firewall"]["expected_drops"] == 0
    assert "/var/tmp/" in str(record["scrub"]["fails_the_export_if"])
    assert record["token_stats"]["method"], "a char/4 estimate must say so"
    assert record["smol_ladder_commit"]
    assert path.name == "ja3_sft.manifest.json"


def test_the_token_statistics_report_a_method_either_way(tmp_path, monkeypatch):
    _fixture(tmp_path, monkeypatch)
    rows, _ = X.export_traces(tmp_path)

    stats = X.token_stats(rows)

    assert stats["rows"] == 1
    assert stats["tokens"]["median"] > 0
    assert "chars/4" in stats["method"] or "chat template" in stats["method"]
    assert 0.0 <= stats["tokens"]["share_over_8192"] <= 1.0
    assert stats["turns"]["median"] >= 1


def test_the_fallback_export_is_the_contract_rows_and_labels_them(tmp_path, monkeypatch):
    """`solutions/jupyter-agent` has no conversations, so what it can yield is the contract
    trajectory. It has to be a separate file with a note, because a manifest that mixed them in
    with real traces without saying so would report a dataset that is not one."""
    solutions = tmp_path / "solutions" / "jupyter-agent" / "ja_1"
    solutions.mkdir(parents=True)
    (solutions / "result.json").write_text(json.dumps(
        {"task_id": "ja_1", "reward": 1.0, "prediction": "141"}))
    (solutions / "solution.py").write_text("print(141)\n")
    monkeypatch.setattr(X, "pool_rows", lambda split: {"ja_1": ROW})
    monkeypatch.setattr(X, "heldout_keys_for", lambda *a, **k: {
        "question": set(), "task_id": set(), "bucket_prefix": set(), "kaggle_table": set()})

    rows, report = X.export_fallbacks(tmp_path)

    assert len(rows) == 1, report
    commands = [call["function"]["arguments"]["command"]
                for message in rows[0]["messages"] for call in (message.get("tool_calls") or [])]
    assert any(command == "echo -n 141 > /workdir/answer.txt" for command in commands), commands
    assert report["dropped"] == {}


def test_the_fallback_export_refuses_a_held_out_table_too(tmp_path, monkeypatch):
    """The 383 held-out-table tasks are excluded from the fallback file by the same firewall, so
    the contract rows are no more trainable-on-held-out than the traces are."""
    solutions = tmp_path / "solutions" / "jupyter-agent" / "ja_1"
    solutions.mkdir(parents=True)
    (solutions / "result.json").write_text(json.dumps(
        {"task_id": "ja_1", "reward": 1.0, "prediction": "141"}))
    (solutions / "solution.py").write_text("print(141)\n")
    monkeypatch.setattr(X, "pool_rows", lambda split: {"ja_1": ROW})
    monkeypatch.setattr(X, "heldout_keys_for", lambda *a, **k: {
        "question": set(), "task_id": set(), "bucket_prefix": set(),
        "kaggle_table": {"celebrity-deaths"}})

    rows, report = X.export_fallbacks(tmp_path)

    assert rows == []
    assert report["dropped"] == {"heldout:kaggle_table": 1}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))