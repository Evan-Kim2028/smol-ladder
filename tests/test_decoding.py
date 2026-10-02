"""Decoding settings are request-level, recorded, and never pooled across conditions.

The default has to stay the harness's historical request (temperature 0 and no other key), so the
first tests pin it. The rest check that a requested setting reaches the wire in every agent mode,
that it lands in result.json and RUN.json, that it changes the prompt hash the summariser pools by,
and that the analysis-only repeat guard ends a loop without changing what the model was sent.
"""

import json
import os
import sys
from pathlib import Path

import pytest

import smol_ladder.run_ladder as runner
from smol_ladder.or_agent import BASH_MAX_REPEAT_ENV, DECODING_ENV, Decoding, Episode, bash_loop, call_model
from smol_ladder.or_agent import Endpoint
from tests.test_local_models import (Stub, _inputs_outside_tmp, _stub_env, assistant,
                                     runner_bash_prompt, runner_program_prompt)

ROW = {"task_id": "t1", "question": "Q?", "files": ["t.csv"], "answer": "42",
       "reward_mode": "numeric", "atol": 0.0, "rtol": 0.0}


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    # main() writes these straight into os.environ; setting then deleting through monkeypatch makes
    # it restore the pre-test (absent) state afterwards, so nothing leaks into other tests.
    for var in (*DECODING_ENV.values(), BASH_MAX_REPEAT_ENV):
        monkeypatch.setenv(var, "")
        monkeypatch.delenv(var)


def _call(stub, **kw):
    ep = Endpoint(stub.base_url)
    return call_model([{"role": "user", "content": "hi"}], "m", None, ep, **kw)


def test_the_default_request_is_unchanged():
    with Stub([assistant("x")]) as stub:
        _call(stub)
    assert stub.requests[0] == {"model": "m", "messages": [{"role": "user", "content": "hi"}],
                                "temperature": 0.0}


def test_requested_settings_reach_the_body(monkeypatch):
    monkeypatch.setenv("SMOL_LADDER_TEMPERATURE", "0.7")
    monkeypatch.setenv("SMOL_LADDER_TOP_P", "0.8")
    monkeypatch.setenv("SMOL_LADDER_TOP_K", "20")
    monkeypatch.setenv("SMOL_LADDER_REPETITION_PENALTY", "1.05")
    monkeypatch.setenv("SMOL_LADDER_PRESENCE_PENALTY", "1.5")
    monkeypatch.setenv("SMOL_LADDER_SEED", "7")
    with Stub([assistant("x")]) as stub:
        _call(stub)
    body = stub.requests[0]
    assert (body["temperature"], body["top_p"], body["top_k"], body["repetition_penalty"],
            body["presence_penalty"], body["seed"]) == (0.7, 0.8, 20, 1.05, 1.5, 7)
    assert isinstance(body["top_k"], int) and isinstance(body["seed"], int)


def test_an_unset_setting_adds_no_key(monkeypatch):
    monkeypatch.setenv("SMOL_LADDER_TOP_P", "0.9")
    with Stub([assistant("x")]) as stub:
        _call(stub)
    assert set(stub.requests[0]) == {"model", "messages", "temperature", "top_p"}
    assert stub.requests[0]["temperature"] == 0.0


def test_a_bad_value_is_an_error_not_a_silent_default(monkeypatch):
    monkeypatch.setenv("SMOL_LADDER_TEMPERATURE", "hot")
    with pytest.raises(ValueError):
        Decoding.from_env()


def test_default_hash_is_the_old_hash_and_any_setting_changes_it(monkeypatch):
    p = "the prompt"
    assert runner.condition_sha256(p) == runner.prompt_sha256(p)
    seen = {runner.condition_sha256(p)}
    for var, val in [("SMOL_LADDER_TEMPERATURE", "0.7"), ("SMOL_LADDER_TOP_P", "0.9"),
                     ("SMOL_LADDER_SEED", "1"), (BASH_MAX_REPEAT_ENV, "4")]:
        monkeypatch.setenv(var, val)
        seen.add(runner.condition_sha256(p))
        monkeypatch.delenv(var)
    assert len(seen) == 5
    monkeypatch.setenv("SMOL_LADDER_TEMPERATURE", "0.0")  # explicit zero is the default
    assert runner.condition_sha256(p) == runner.prompt_sha256(p)


def _once(tmp_path, monkeypatch, stub, agent, prompt, name):
    _stub_env(monkeypatch, stub)
    inputs = _inputs_outside_tmp(tmp_path)
    return runner.once(ROW, prompt, tmp_path / name, Path(sys.prefix), "m", 6,
                       inputs_of=lambda r: inputs, rung_label="L1", agent=agent)


@pytest.mark.parametrize("agent,prompt,replies", [
    ("bash", runner_bash_prompt, [assistant("done")]),
    ("program", runner_program_prompt, [assistant("```python\nprint(42)\n```")]),
    ("tools", runner_program_prompt, [assistant("done")]),
])
def test_every_agent_mode_sends_and_records_the_settings(tmp_path, monkeypatch, agent, prompt,
                                                        replies):
    monkeypatch.setenv("SMOL_LADDER_TEMPERATURE", "0.7")
    monkeypatch.setenv("SMOL_LADDER_TOP_K", "20")
    with Stub(replies) as stub:
        res = _once(tmp_path, monkeypatch, stub, agent, prompt(), "a")
    assert stub.requests[0]["temperature"] == 0.7 and stub.requests[0]["top_k"] == 20
    assert res["decoding"]["temperature"] == 0.7 and res["decoding"]["top_k"] == 20
    assert res["decoding"]["top_p"] is None
    assert res["prompt_sha256"] != runner.prompt_sha256(prompt())


def test_default_result_records_temperature_zero_and_the_plain_hash(tmp_path, monkeypatch):
    with Stub([assistant("done")]) as stub:
        res = _once(tmp_path, monkeypatch, stub, "bash", runner_bash_prompt(), "d")
    assert res["decoding"] == Decoding().as_dict() and res["decoding"]["temperature"] == 0.0
    assert res["prompt_sha256"] == runner.prompt_sha256(runner_bash_prompt())
    assert "bash_max_repeat" not in res
    assert set(stub.requests[0]) <= {"model", "messages", "temperature", "tools", "tool_choice",
                                     "max_tokens", "chat_template_kwargs"}


# ── the repeat guard ─────────────────────────────────────────────────────────────

def _same_call(n, cmd="ls"):
    return [assistant(calls=[("bash", {"command": cmd})]) for _ in range(n)]


def _loop(stub, **kw):
    ep = Endpoint(stub.base_url)
    m = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    ep_ep = Episode(None)
    from smol_ladder.or_agent import CommandResult
    run = lambda c: CommandResult("out", 0, False)  # noqa: E731
    bash_loop(m, run, lambda: None, "m", 8, ep=ep, episode=ep_ep, **kw)
    return m, ep_ep


def test_the_guard_is_off_by_default():
    with Stub(_same_call(8)) as stub:
        _, episode = _loop(stub)
    assert episode.stop_reason == "max_turns" and episode.turns == 8


def test_the_guard_ends_the_episode_on_the_nth_identical_call_and_changes_nothing_before():
    with Stub(_same_call(8)) as plain:
        _loop(plain)
    with Stub(_same_call(8)) as guarded:
        m, episode = _loop(guarded, max_repeat=3)
    assert episode.stop_reason == "repeat_loop" and episode.turns == 3
    assert len(guarded.requests) == 3
    assert guarded.requests == plain.requests[:3]  # what the model saw is untouched
    assert [x["role"] for x in m].count("tool") == 3  # the 3rd call still ran


def test_a_different_call_or_a_text_turn_resets_the_count():
    replies = (_same_call(2) + _same_call(1, "pwd") + _same_call(2) + [assistant("hmm")]
               + _same_call(2))
    with Stub(replies) as stub:
        _, episode = _loop(stub, max_repeat=3)
    assert episode.stop_reason == "max_turns" and episode.turns == 8


def test_the_guard_reads_its_env_and_is_recorded(tmp_path, monkeypatch):
    monkeypatch.setenv(BASH_MAX_REPEAT_ENV, "3")
    with Stub(_same_call(8)) as stub:
        res = _once(tmp_path, monkeypatch, stub, "bash", runner_bash_prompt(), "g")
    assert res["stop_reason"] == "repeat_loop" and res["turns_used"] == 3
    assert res["bash_max_repeat"] == 3
    assert res["prompt_sha256"] != runner.prompt_sha256(runner_bash_prompt())


# ── RUN.json and the run tag ─────────────────────────────────────────────────────

def _header(**kw):
    return {"run_tag": "x", "model": "m", **runner.decoding_record("bash"), **kw}


def test_a_run_tag_refuses_a_second_condition(monkeypatch):
    first = _header()
    existing = {"model": "m", "decoding": first["decoding"], "launches": [first]}
    assert runner.decoding_clash(existing, _header()) == ""
    monkeypatch.setenv("SMOL_LADDER_TEMPERATURE", "0.7")
    assert "never pooled" in runner.decoding_clash(existing, _header())
    monkeypatch.delenv("SMOL_LADDER_TEMPERATURE")
    monkeypatch.setenv(BASH_MAX_REPEAT_ENV, "4")
    assert "never pooled" in runner.decoding_clash(existing, _header())


def test_a_record_from_before_the_flags_counts_as_the_default(monkeypatch):
    old = {"model": "m", "launches": [{"model": "m"}]}
    assert runner.decoding_clash(old, _header()) == ""
    monkeypatch.setenv("SMOL_LADDER_TOP_P", "0.9")
    assert runner.decoding_clash(old, _header())


def test_run_json_carries_the_decoding(tmp_path, monkeypatch):
    monkeypatch.setenv("SMOL_LADDER_TEMPERATURE", "0.7")
    path = tmp_path / "RUN.json"
    runner.open_run_record(path, _header())
    rec = json.loads(path.read_text())
    assert rec["decoding"]["temperature"] == 0.7
    assert rec["launches"][0]["decoding"]["temperature"] == 0.7


def test_flags_set_the_env_and_need_a_run_tag(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_ladder", "--temperature", "0.7"])
    with pytest.raises(SystemExit) as e:
        runner.main()
    assert e.value.code == 2
    assert os.environ["SMOL_LADDER_TEMPERATURE"] == "0.7"  # the flag reached the environment
    monkeypatch.setattr(sys, "argv", ["run_ladder", "--bash-max-repeat", "4", "--run-tag", "t"])
    with pytest.raises(SystemExit):  # guard needs --agent bash
        runner.main()
