"""The SFT rows must reach the chat template with their tool calls, and the rendering must show it.

`sft_lora.prepare()` once rebuilt every message as {"role", "content"}. The rendered training text
then held no tool calls, 38% of assistant turns came out empty, and nothing errored. These tests
render the rows through Qwen3.5's own template and read the text back, so that failure cannot
return silently. They run in the quick loop: jinja2 is a dev dependency, and the template is a
committed copy (tests/fixtures/qwen3_5_chat_template.jinja) so no model download is involved.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from train.render import (RenderingError, assert_tool_calls_rendered, audit_row, audit_rows,
                          jinja_renderer, normalise_messages, render_row)
from train.sft_lora import prepare

jinja2 = pytest.importorskip("jinja2")

FIXTURES = Path(__file__).parent / "fixtures"
ROWS = [json.loads(line) for line in (FIXTURES / "bash_replay_rows.jsonl").read_text().splitlines()]
DATA = Path(__file__).resolve().parent.parent / "data" / "train"


@pytest.fixture(scope="module")
def render():
    return jinja_renderer((FIXTURES / "qwen3_5_chat_template.jinja").read_text())


def old_prepare(rows):
    """The behaviour this commit removes, kept here so the tests can show it is caught."""
    return [{"messages": [{"role": m["role"], "content": m["content"]} for m in r["messages"]],
             "chat_template_kwargs": {"enable_thinking": False}, "tools": r["tools"]}
            for r in rows]


def test_prepare_passes_tool_calls_and_tool_result_keys_through():
    fed = prepare(ROWS, "bash")
    for source, entry in zip(ROWS, fed):
        assert len(entry["messages"]) == len(source["messages"])
        for was, now in zip(source["messages"], entry["messages"]):
            assert now["role"] == was["role"] and now["content"] == was["content"]
            assert now.get("tool_calls") == was.get("tool_calls")
            if was["role"] == "tool":
                assert now["tool_call_id"] == was["tool_call_id"] and now["name"] == was["name"]
        assert entry["tools"] == source["tools"]
        assert entry["chat_template_kwargs"] == {"enable_thinking": False}


def test_the_program_protocol_still_sends_no_tools_key():
    assert "tools" not in prepare(ROWS, "program")[0]


def test_tool_call_arguments_are_a_dict_whichever_form_the_data_has_them_in():
    row = copy.deepcopy(ROWS[0])
    as_strings = copy.deepcopy(row["messages"])
    for m in as_strings:
        for call in m.get("tool_calls") or []:
            call["function"]["arguments"] = json.dumps(call["function"]["arguments"])
    assert normalise_messages(as_strings) == normalise_messages(row["messages"])
    with pytest.raises(ValueError, match="not JSON"):
        normalise_messages([{"role": "assistant", "content": "",
                             "tool_calls": [{"function": {"name": "bash", "arguments": "{oops"}}]}])


def test_the_template_cannot_render_a_string_arguments_call(render):
    """Why normalisation exists: Qwen3.5 iterates `arguments|items`, which a JSON string breaks."""
    row = copy.deepcopy(ROWS[0])
    for m in row["messages"]:
        for call in m.get("tool_calls") or []:
            call["function"]["arguments"] = json.dumps(call["function"]["arguments"])
    with pytest.raises(Exception):  # noqa: B017 - jinja's TypeError ("pairs from a mapping")
        render(row["messages"], row["tools"])


def test_the_rendered_text_contains_each_tool_call_and_result(render):
    row = ROWS[0]
    text = render_row(prepare([row], "bash")[0], render)
    first = row["messages"][2]["tool_calls"][0]["function"]["arguments"]["command"]
    assert f"<tool_call>\n<function=bash>\n<parameter=command>\n{first}\n</parameter>\n" in text
    assert text.count("<tool_call>\n<function=bash>") == sum(
        len(m.get("tool_calls") or []) for m in row["messages"])
    assert text.count("<tool_response>") == sum(m["role"] == "tool" for m in row["messages"])
    assert text.rstrip().endswith("<|im_end|>")


def test_the_audit_passes_every_fixture_row_as_prepare_feeds_it(render):
    result = audit_rows(ROWS, render, prepare(ROWS, "bash"))
    assert result["bad_rows"] == [] and result["empty_with_source_call"] == 0
    assert result["calls_rendered"] == result["calls_source"] > 0
    assert result["tool_results_rendered"] == result["tool_results_source"] > 0
    assert assert_tool_calls_rendered(ROWS, render, prepare(ROWS, "bash"))["rows"] == len(ROWS)


def test_the_old_prepare_is_caught_by_the_audit_and_by_the_guard(render):
    """The regression test. Auditing the old output against ITS OWN rows would pass vacuously;
    against the source rows it must fail, and the guard must refuse to proceed."""
    old = old_prepare(ROWS)
    result = audit_rows(ROWS, render, old)
    assert result["calls_rendered"] == 0 < result["calls_source"]
    assert len(result["bad_rows"]) == len(ROWS)
    with pytest.raises(RenderingError, match="0 rendered vs"):
        assert_tool_calls_rendered(ROWS, render, old, label="old prepare")
    # roughly the reported 38%: assistant turns that rendered with nothing in them
    text_turns = [m for r in ROWS for m in r["messages"] if m["role"] == "assistant"]
    empty = sum(1 for r in old for m in r["messages"] if m["role"] == "assistant"
                and not (m["content"] or "").strip())
    assert empty / len(text_turns) > 0.3


def test_an_empty_turn_where_the_source_had_a_call_is_flagged(render):
    row = copy.deepcopy(ROWS[0])
    fed = copy.deepcopy(prepare([row], "bash")[0])
    del fed["messages"][2]["tool_calls"]
    fed["messages"][2]["content"] = ""
    stats = audit_row(row, render_row(fed, render))
    assert stats["empty_with_source_call"] == 1 and stats["problems"]


def test_a_changed_argument_or_a_dropped_result_is_flagged(render):
    row = ROWS[0]
    fed = copy.deepcopy(prepare([row], "bash")[0])
    fed["messages"][2]["tool_calls"][0]["function"]["arguments"]["command"] += " # tampered"
    assert audit_row(row, render_row(fed, render))["problems"]
    fed = copy.deepcopy(prepare([row], "bash")[0])
    del fed["messages"][3]
    assert audit_row(row, render_row(fed, render))["problems"]


def test_two_calls_in_one_turn_render_as_two_blocks_and_two_results(render):
    two = next(r for r in ROWS if any(len(m.get("tool_calls") or []) == 2 for m in r["messages"]))
    stats = audit_row(two, render_row(prepare([two], "bash")[0], render))
    assert not stats["problems"]
    assert stats["calls_rendered"] == stats["calls_source"]


def test_the_fixture_has_no_host_paths_or_secrets():
    blob = (FIXTURES / "bash_replay_rows.jsonl").read_text()
    for needle in ("/home/evan", "/Users/", "hf_", "sk-", "api_key", "OPENROUTER"):
        assert needle not in blob, needle


def test_the_jinja_renderer_agrees_with_the_tokenizers_own_template(render):
    """The quick loop renders with jinja2 alone; where transformers and the cached tokenizer exist
    the two must produce the same text, or the quick loop is testing a different renderer."""
    pytest.importorskip("transformers")
    from train.render import load_tokenizer, tokenizer_renderer
    try:
        real = tokenizer_renderer(load_tokenizer())
    except Exception as e:  # noqa: BLE001 - tokenizer not cached and no network
        pytest.skip(f"no Qwen3.5 tokenizer: {type(e).__name__}")
    for row in ROWS:
        fed = prepare([row], "bash")[0]
        assert render_row(fed, render) == render_row(fed, real)


# ── the whole datasets, when they are on disk ─────────────────────────────────────────────────

@pytest.mark.slow
@pytest.mark.parametrize("name,expected_rows", [("sft_upstream/train.jsonl", 4439),
                                                ("sft_upstream/val.jsonl", 234),
                                                ("ja3_sft.jsonl", 2029)])
def test_every_row_of_both_datasets_renders_its_tool_calls(render, name, expected_rows):
    from train.format import read_jsonl

    path = DATA / name
    if not path.exists():
        pytest.skip(f"{path} is not on this machine")
    rows = read_jsonl(path)
    result = assert_tool_calls_rendered(rows, render, prepare(rows, "bash"), label=name)
    assert result["rows"] == expected_rows
    assert result["calls_rendered"] == result["calls_source"] > expected_rows
    assert result["tool_results_rendered"] == result["tool_results_source"]
    assert result["empty_with_source_call"] == 0


def test_the_staged_token_counts_are_those_of_the_rendering(tmp_path, render):
    """ops/amd/stage.py feeds the cost projection. It must count the text the trainer sees: the
    rendering with its tool calls, which the old per-message estimate and the old (call-less)
    rendering both undercount or misplace."""
    from ops.amd import stage

    path = tmp_path / "rows.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in ROWS))
    corrected = stage.count_file(path, 10**9, encode=len, render=lambda row: render_row(
        prepare([row], "bash")[0], render))
    assert corrected["raw_tokens"] == sum(len(render_row(prepare([r], "bash")[0], render))
                                          for r in ROWS)
    dropped = sum(len(render_row(old_prepare([r])[0], render)) for r in ROWS)
    assert corrected["raw_tokens"] > dropped + sum(
        len(c["function"]["arguments"]["command"]) for r in ROWS
        for m in r["messages"] for c in m.get("tool_calls") or [])
    capped = stage.count_file(path, 50, encode=len, render=lambda row: render_row(
        prepare([row], "bash")[0], render))
    assert capped["trained_tokens"] == 50 * len(ROWS) and capped["raw_tokens"] == corrected["raw_tokens"]
    sets = stage.build_token_counts([path], path, 10**9, len, lambda row: render_row(
        prepare([row], "bash")[0], render))["sets"]
    assert "chat template rendered" in sets["A"]["method"]
    assert sets["AB"]["trained_tokens"] == 2 * corrected["trained_tokens"]
