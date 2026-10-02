"""Does the running vLLM turn model output into a real `tool_call`? Asked BEFORE any arm trains.

    python -m ops.amd.probe_tools --port 8001 --models amd-a-2b --base-port 8000 --base-model amd-base-2b

The bash agent protocol lives or dies on this: if `--tool-call-parser` does not match the model's
output format, every request returns plain text, the harness sees no tool call, every trial ends
in one turn, and the sweep reports a plausible-looking pass rate of about zero. That is a result
about a flag, and it costs the whole evaluation to discover it late. So: send the real `bash` tool
schema and system prompt (smol_ladder.upstream) to each model a few times and require at least one
parsed `bash` tool call with valid JSON arguments from the adapter. A raw `<tool_call>` left in
the content is the signature of a parser mismatch and is reported as such.

Prints `TOOL_CALLS_OK=1` if the first named adapter produced one, else 0. The base
is informational: an untuned base may legitimately answer in prose.

Second check, `ADAPTER_DIFFERS_FROM_BASE`: the first training row (up to its first assistant turn,
rendered with its own tools) is sent at temperature 0 to the adapter model and to the base. The
adapter's output must DIFFER from the base's. A merge once wrote the base's own weights into the
"merged" directory, three adapters were evaluated as the base, and nothing noticed because every
number looked plausible; identical temperature-0 output on a real training prompt is the one
symptom that cannot be a coincidence. It also prints each model's token-level agreement with the
row's own training target (`ADAPTER_TARGET_AGREEMENT`, `BASE_TARGET_AGREEMENT`: the fraction of the
target's tokens, split on word and punctuation boundaries, that the output has at the same
position), which is the first evidence of whether the adapter learned the format at all.

The last line is `ADAPTER_CHECK model=<name> differs=<0|1> tool_calls_ok=<0|1>`, the one line
serve.sh greps for and the driver records: an adapter that is identical to the base, or that returns
no parsed tool call, must not be evaluated.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def chat(port: int, body: dict, timeout: float = 120.0) -> dict:
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def classify(message: dict) -> str:
    """tool_call | leaked | prose | empty, for one assistant message."""
    calls = message.get("tool_calls") or []
    for call in calls:
        fn = call.get("function", {})
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except ValueError:
            continue
        if fn.get("name") == "bash" and isinstance(args.get("command"), str):
            return "tool_call"
    content = message.get("content") or ""
    if "<tool_call>" in content or "<function=" in content:
        return "leaked"
    return "prose" if content.strip() else "empty"


def probe(port: int, model: str, attempts: int, tools: list, messages: list) -> dict:
    counts: dict[str, int] = {}
    for _ in range(attempts):
        body = {"model": model, "messages": messages, "tools": tools, "tool_choice": "auto",
                "temperature": 0.0, "max_tokens": 512,
                "chat_template_kwargs": {"enable_thinking": False}}
        msg = chat(port, body)["choices"][0]["message"]
        kind = classify(msg)
        counts[kind] = counts.get(kind, 0) + 1
    return counts


_TOKEN = re.compile(r"\w+|[^\w\s]")


def first_training_prompt(rows_path: Path) -> tuple[list, list | None, str]:
    """(messages up to the first assistant turn, the row's tools, that turn's text) from the first
    row whose first assistant turn has something to say."""
    with open(rows_path) as f:
        for line in f:
            row = json.loads(line)
            msgs = row["messages"]
            cut = next((i for i, m in enumerate(msgs) if m["role"] == "assistant"), None)
            if cut is None:
                continue
            target = assistant_text(msgs[cut])
            if target.strip():
                prompt = [{"role": m["role"], "content": m["content"]} for m in msgs[:cut]]
                return prompt, row.get("tools") or None, target
    raise ValueError(f"no usable training row in {rows_path}")


def assistant_text(message: dict) -> str:
    """What a message says: its content, then each tool call's command (or raw arguments)."""
    parts = [message.get("content") or ""]
    for call in message.get("tool_calls") or []:
        args = (call.get("function") or call).get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except ValueError:
                pass
        parts.append(args.get("command", "") if isinstance(args, dict) else str(args or ""))
    return "\n".join(p for p in parts if p)


def token_agreement(output: str, target: str) -> float:
    """Fraction of the target's tokens that the output has at the same position."""
    want, got = _TOKEN.findall(target), _TOKEN.findall(output)
    return sum(a == b for a, b in zip(want, got)) / len(want) if want else 0.0


def generate(port: int, model: str, prompt: list, tools: list | None) -> dict:
    body = {"model": model, "messages": prompt, "temperature": 0.0, "max_tokens": 512,
            "logprobs": True, "chat_template_kwargs": {"enable_thinking": False}}
    if tools:
        body.update(tools=tools, tool_choice="auto")
    choice = chat(port, body)["choices"][0]
    lp = [t["logprob"] for t in ((choice.get("logprobs") or {}).get("content") or [])]
    return {"text": assistant_text(choice["message"]), "logprob_sum": sum(lp) if lp else None}


def adapter_check(port: int, adapter: str, base_port: int, base: str, rows_path: Path) -> dict:
    prompt, tools, target = first_training_prompt(rows_path)
    a, b = generate(port, adapter, prompt, tools), generate(base_port, base, prompt, tools)
    return {"differs": a["text"] != b["text"],
            "adapter_agreement": token_agreement(a["text"], target),
            "base_agreement": token_agreement(b["text"], target),
            "adapter_logprob_sum": a["logprob_sum"], "base_logprob_sum": b["logprob_sum"],
            "adapter_text": a["text"][:200], "base_text": b["text"][:200]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--attempts", type=int, default=3)
    ap.add_argument("--train-rows", type=Path, default=Path("data/train/sft_upstream/train.jsonl"),
                    help="training rows; the first one is the fixed prompt of the adapter check")
    ap.add_argument("--base-model", help="the base's served name when it is not the 2nd of --models")
    ap.add_argument("--base-port", type=int, help="the base's port when it is not --port (merged mode)")
    args = ap.parse_args()
    from smol_ladder.upstream import BASH_TOOL, bash_prompt

    messages = bash_prompt("How many rows does sales.csv have?", ["sales.csv"])
    ok = False
    for i, model in enumerate(args.models):
        try:
            counts = probe(args.port, model, args.attempts, BASH_TOOL, messages)
        except Exception as exc:  # noqa: BLE001 - a failed request is itself the finding
            print(f"  {model}: request failed: {exc}")
            continue
        print(f"  {model}: {counts}")
        if counts.get("leaked"):
            print(f"  {model}: raw <tool_call> text in the content: the tool-call parser does "
                  "not match this model's output format")
        if i == 0 and counts.get("tool_call"):
            ok = True
    print(f"TOOL_CALLS_OK={1 if ok else 0}")
    tool_ok = 1 if ok else 0
    base_model = args.base_model or (args.models[1] if len(args.models) > 1 else None)
    if base_model is None:
        return 0
    try:
        res = adapter_check(args.port, args.models[0], args.base_port or args.port, base_model,
                            args.train_rows)
    except Exception as exc:  # noqa: BLE001 - an unmeasured check must read as a failure
        print(f"  adapter check could not run: {exc}")
        print("ADAPTER_DIFFERS_FROM_BASE=0")
        print(f"ADAPTER_CHECK model={args.models[0]} differs=0 tool_calls_ok={tool_ok}")
        return 0
    print(f"  fixed training prompt, temperature 0: adapter {args.models[0]!r} vs base {base_model!r}")
    print(f"  adapter output: {res['adapter_text']!r}")
    print(f"  base output:    {res['base_text']!r}")
    print(f"  token agreement with the row's training target: adapter "
          f"{res['adapter_agreement']:.3f}, base {res['base_agreement']:.3f}; "
          f"sum logprob adapter {res['adapter_logprob_sum']}, base {res['base_logprob_sum']}")
    if not res["differs"]:
        print("  the adapter model's output is IDENTICAL to the base's: the adapter is not applied "
              "(a no-op merge or a LoRA module that did not attach)")
    print(f"ADAPTER_DIFFERS_FROM_BASE={1 if res['differs'] else 0}")
    print(f"ADAPTER_TARGET_AGREEMENT={res['adapter_agreement']:.4f}")
    print(f"BASE_TARGET_AGREEMENT={res['base_agreement']:.4f}")
    print(f"ADAPTER_CHECK model={args.models[0]} differs={1 if res['differs'] else 0} "
          f"tool_calls_ok={tool_ok}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
