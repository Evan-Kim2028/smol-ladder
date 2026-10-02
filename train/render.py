"""Render SFT rows through the model's chat template, and audit what came out.

`sft_lora.prepare()` once rebuilt every message as `{"role", "content"}`. That deleted the
assistant's `tool_calls` and the tool messages' `tool_call_id`/`name`, so the chat template had
nothing to render for a tool call: 38% of assistant turns came out empty and the adapters never
saw a single tool-call target. Nothing errored, because an empty turn is a perfectly valid thing
to train on. The only defence is to look at the rendered text, which is what this module does.

Two things live here:

- `normalise_messages` -- the form the template needs. Qwen3.5's template iterates
  `tool_call.arguments|items`, so `arguments` must be a **dict**. The OpenAI wire format (and
  vLLM's responses) carry a JSON *string*; the SmolDataEnvs-sft parquet and our exports carry a
  dict. A string would not fail loudly either: `|items` on a str raises inside jinja, but a
  template that silently skips an undefined `arguments` renders an argument-less call.
- `audit_row` -- parses the rendered text back and checks it against the source row: one rendered
  tool call per source tool call with the same arguments, every tool result present and in
  order, no assistant turn that was empty in the render while the source had a call in it.

The audit is deliberately a *parse of the output*, not a second implementation of the template:
it knows the template's tag names (`<tool_call>`, `<function=...>`, `<parameter=...>`,
`<tool_response>`) and nothing about how it arrives at them.
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Callable

NON_THINKING = {"enable_thinking": False}
MODEL = "Qwen/Qwen3.5-2B"

Renderer = Callable[[list, list | None], str]


def normalise_messages(messages: list[dict]) -> list[dict]:
    """Messages exactly as given, except tool-call `arguments` become a dict.

    Everything else passes through untouched -- `tool_calls`, `tool_call_id`, `name` and any other
    key -- because the chat template is the only thing that decides how a message is rendered, and
    any reshaping here would be a second format to keep in sync with it.
    """
    out = copy.deepcopy(messages)
    for message in out:
        for call in message.get("tool_calls") or []:
            function = call.get("function")
            if not isinstance(function, dict):
                continue
            args = function.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args) if args.strip() else {}
                except json.JSONDecodeError as e:
                    raise ValueError(f"tool call arguments are not JSON: {args[:80]!r}") from e
            if args is None:
                args = {}
            if not isinstance(args, dict):
                raise ValueError(f"tool call arguments must be an object, got {type(args).__name__}")
            function["arguments"] = args
    return out


# ── renderers ─────────────────────────────────────────────────────────────────

def tokenizer_renderer(tokenizer, **template_kwargs) -> Renderer:
    """The real thing: `tokenizer.apply_chat_template`, non-thinking, as TRL calls it."""
    kwargs = {**NON_THINKING, **template_kwargs}

    def render(messages: list, tools: list | None) -> str:
        return tokenizer.apply_chat_template(messages, tools=tools or None, tokenize=False, **kwargs)

    return render


def load_tokenizer(model: str = MODEL):
    """The model's tokenizer from the local HF cache (downloaded only if it is not there)."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model)


def jinja_renderer(template: str, **template_kwargs) -> Renderer:
    """Render with jinja2 alone, configured the way transformers configures it.

    For environments without transformers (the quick test loop, a laptop without the `train`
    extra). Same environment as `transformers.utils.chat_template_utils`: trim_blocks and
    lstrip_blocks on, loopcontrols, and a `tojson` that does not HTML-escape. A test compares it
    with the tokenizer's own output when both are installed, so it cannot drift unnoticed.
    """
    import jinja2
    import jinja2.exceptions
    import jinja2.ext
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    def raise_exception(message):
        raise jinja2.exceptions.TemplateError(message)

    def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
        return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators,
                          sort_keys=sort_keys)

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                        extensions=[jinja2.ext.loopcontrols])
    env.filters["tojson"] = tojson
    env.globals["raise_exception"] = raise_exception
    compiled = env.from_string(template)
    kwargs = {**NON_THINKING, **template_kwargs}

    def render(messages: list, tools: list | None) -> str:
        return compiled.render(messages=messages, tools=tools or None, add_generation_prompt=False,
                               **kwargs)

    return render


def render_row(row: dict, render: Renderer) -> str:
    return render(normalise_messages(row["messages"]), row.get("tools") or None)


# ── the audit ─────────────────────────────────────────────────────────────────

_SEGMENT = re.compile(r"<\|im_start\|>(system|user|assistant)\n(.*?)<\|im_end\|>\n", re.S)
_THINK = re.compile(r"<think>\n(.*?)\n</think>\n\n(.*)\Z", re.S)
_CALL = re.compile(r"<tool_call>\n<function=([^>\n]+)>\n(.*?)</function>\n</tool_call>(?:\n|\Z)",
                   re.S)
_PARAM = re.compile(r"<parameter=([^>\n]+)>\n(.*?)\n</parameter>\n", re.S)
_FIRST_CALL = "<tool_call>\n<function="


def _param_text(value) -> str:
    """How the template spells one argument value."""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _groups(messages: list[dict]) -> list[tuple[str, list[dict]]]:
    """Source messages after the system turn, with runs of tool results merged: the template
    renders consecutive tool messages into one `user` segment."""
    groups: list[tuple[str, list[dict]]] = []
    for message in messages:
        if message["role"] == "system":
            continue
        if message["role"] == "tool" and groups and groups[-1][0] == "tool":
            groups[-1][1].append(message)
        else:
            groups.append((message["role"], [message]))
    return groups


def audit_row(row: dict, text: str) -> dict:
    """Compare a rendered row with its source. Returns counts and a list of `problems`.

    `problems` empty means: the text has exactly one segment per source turn, each assistant turn
    carries its content and exactly its tool calls (name and arguments), and each tool result
    appears, trimmed as the template trims it, in order.
    """
    messages = normalise_messages(row["messages"])
    segments = _SEGMENT.findall(text)
    if segments and segments[0][0] == "system":
        segments = segments[1:]
    stats = {"assistant_turns": 0, "calls_source": 0, "calls_rendered": 0, "tool_results_source": 0,
             "tool_results_rendered": 0, "empty_assistant_turns": 0, "empty_with_source_call": 0,
             "problems": []}
    problems = stats["problems"]
    groups = _groups(messages)
    if len(groups) != len(segments):
        problems.append(f"{len(groups)} source turns but {len(segments)} rendered segments")
    for (role, msgs), (seg_role, body) in zip(groups, segments):
        if role == "tool":
            stats["tool_results_source"] += len(msgs)
            expected = "\n".join(f"<tool_response>\n{(m.get('content') or '').strip()}\n"
                                 f"</tool_response>" for m in msgs)
            if seg_role != "user" or body != expected:
                problems.append("tool results differ from the source, or are out of order")
            else:
                stats["tool_results_rendered"] += len(msgs)
            continue
        if role != seg_role:
            problems.append(f"source {role} rendered as {seg_role}")
            continue
        if role != "assistant":
            continue
        message = msgs[0]
        source_calls = message.get("tool_calls") or []
        stats["assistant_turns"] += 1
        stats["calls_source"] += len(source_calls)
        think = _THINK.match(body)
        rest = think.group(2) if think else body
        cut = rest.find(_FIRST_CALL)
        prose, calls_text = (rest, "") if cut < 0 else (rest[:cut], rest[cut:])
        calls = _CALL.findall(calls_text)
        stats["calls_rendered"] += len(calls)
        if not rest.strip():
            stats["empty_assistant_turns"] += 1
            if source_calls:
                stats["empty_with_source_call"] += 1
        if prose.strip() != (message.get("content") or "").strip():
            problems.append("assistant prose differs from the source")
        if len(calls) != len(source_calls):
            problems.append(f"{len(source_calls)} source tool calls, {len(calls)} rendered")
            continue
        for (name, params), source in zip(calls, source_calls):
            function = source.get("function") or {}
            expected_args = {k: _param_text(v) for k, v in (function.get("arguments") or {}).items()}
            if name != function.get("name") or dict(_PARAM.findall(params)) != expected_args:
                problems.append("a rendered tool call differs from the source (name or arguments)")
    return stats


def audit_rows(source_rows: list[dict], render: Renderer,
               fed_rows: list[dict] | None = None) -> dict:
    """`audit_row` over a dataset: summed counts, and the indices of rows with problems.

    `source_rows` are the rows as exported; `fed_rows` are what the trainer is actually handed
    (the output of `prepare`, or the Arrow dataset built from it). Rendering the fed rows and
    comparing them with the *source* is the point: auditing a rendering against the rows it was
    rendered from would pass vacuously when `prepare` had already dropped the tool calls.
    """
    fed_rows = source_rows if fed_rows is None else fed_rows
    if len(fed_rows) != len(source_rows):
        raise ValueError(f"{len(fed_rows)} rows fed to the trainer for {len(source_rows)} source rows")
    total = {"rows": 0, "assistant_turns": 0, "calls_source": 0, "calls_rendered": 0,
             "tool_results_source": 0, "tool_results_rendered": 0, "empty_assistant_turns": 0,
             "empty_with_source_call": 0, "bad_rows": [], "first_problem": ""}
    for i, (row, fed) in enumerate(zip(source_rows, fed_rows)):
        stats = audit_row(row, render_row(fed, render))
        total["rows"] += 1
        for key in ("assistant_turns", "calls_source", "calls_rendered", "tool_results_source",
                    "tool_results_rendered", "empty_assistant_turns", "empty_with_source_call"):
            total[key] += stats[key]
        if stats["problems"]:
            total["bad_rows"].append(i)
            total["first_problem"] = total["first_problem"] or f"row {i}: {stats['problems'][0]}"
    return total


class RenderingError(RuntimeError):
    """The rendered training text does not carry the source row's tool calls."""


def assert_tool_calls_rendered(source_rows: list[dict], render: Renderer,
                               fed_rows: list[dict] | None = None, label: str = "dataset") -> dict:
    """The training guard: refuse to train on text that lost its tool calls.

    A source with assistant `tool_calls` whose rendering has fewer tool-call blocks, or has an
    empty assistant turn where the source had a call, is the exact failure this module exists
    for. Raises `RenderingError` with the counts; returns them when the rendering is faithful.
    """
    result = audit_rows(source_rows, render, fed_rows)
    if (result["bad_rows"] or result["calls_rendered"] != result["calls_source"]
            or result["empty_with_source_call"] or result["tool_results_rendered"]
            != result["tool_results_source"]):
        raise RenderingError(
            f"{label}: the rendered text does not carry the source's tool calls: "
            f"{result['calls_rendered']} rendered vs {result['calls_source']} in the source, "
            f"{result['empty_with_source_call']} assistant turns empty where the source had a call, "
            f"{result['tool_results_rendered']} of {result['tool_results_source']} tool results, "
            f"{len(result['bad_rows'])} of {result['rows']} rows differ ({result['first_problem']}). "
            "Messages must reach the chat template intact (see train/render.py).")
    return result


def main() -> None:
    import argparse

    from train.format import read_jsonl
    from train.sft_lora import prepare

    ap = argparse.ArgumentParser(description="Render SFT files through the Qwen3.5 chat template, "
                                 "as sft_lora.prepare feeds them to the trainer, and audit the "
                                 "tool calls against the source rows.")
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--model", default=MODEL)
    args = ap.parse_args()
    render = tokenizer_renderer(load_tokenizer(args.model))
    for path in args.files:
        rows = read_jsonl(path)
        result = audit_rows(rows, render, prepare(rows, "bash"))
        result["bad_rows"] = len(result["bad_rows"])
        print(path, json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
