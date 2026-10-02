"""Does the running vLLM turn model output into a real `tool_call`? Asked BEFORE any arm trains.

    python -m ops.amd.probe_tools --port 8000 --models amd-base-2b amd-probe-2b

The bash agent protocol lives or dies on this: if `--tool-call-parser` does not match the model's
output format, every request returns plain text, the harness sees no tool call, every trial ends
in one turn, and the sweep reports a plausible-looking pass rate of about zero. That is a result
about a flag, and it costs the whole evaluation to discover it late. So: send the real `bash` tool
schema and system prompt (smol_ladder.upstream) to each model a few times and require at least one
parsed `bash` tool call with valid JSON arguments from the adapter. A raw `<tool_call>` left in
the content is the signature of a parser mismatch and is reported as such.

Prints `TOOL_CALLS_OK=1` if the first named adapter (the probe) produced one, else 0. The base
is informational: an untuned base may legitimately answer in prose.
"""

from __future__ import annotations

import argparse
import json
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--attempts", type=int, default=3)
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
