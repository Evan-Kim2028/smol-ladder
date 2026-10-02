"""A scripted OpenAI-compatible endpoint that replays recorded bash trajectories.

    uv run python -m tools.oracle_server --data data/train/sft_upstream/train.jsonl
    # prints the base URL; then, in another shell:
    SMOL_LADDER_BASE_URL=http://127.0.0.1:PORT/v1 uv run python -m smol_ladder.run_ladder \\
        --split train --agent bash --task-ids a,b,c --run-tag oracle --model oracle

It answers a chat completion with the assistant turn the recorded trajectory had at that point:
the request is matched to a trajectory by the question in its user message, and the turn is picked
by how many assistant messages the request already carries. So a harness that sends the
conversation the model was trained on, and runs the commands in the environment the commands were
written for, ends with the answer the recorded run ended with. These trajectories were verified
upstream, so a mismatch is a fact about OUR side: a missing package, a path that does not exist, a
tool-output format the commands depended on.

No model is involved and nothing leaves the machine. It is also what makes the harness testable
without a GPU: the conversation each request carries is recorded (`Oracle.requests`), so a test can
assert what the harness sent.
"""

from __future__ import annotations

import argparse
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

_QUESTION = re.compile(r"\n\nQuestion:\n(.*?)\n\nWork it out", re.S)
_FILES = re.compile(r"Files \(in [^)]*\):\n(.*?)\n\nInstalled", re.S)

FALLBACK_REPLY = "No recorded trajectory covers this request."
EXHAUSTED_REPLY = "Done."


def question_of(messages: list[dict]) -> str | None:
    """The question in the first user message, or None when it is not the SFT template's."""
    for m in messages:
        if m.get("role") == "user":
            found = _QUESTION.search(m.get("content") or "")
            return found.group(1) if found else None
    return None


class Oracle:
    """The trajectories, indexed by question, and the rule that picks the next turn."""

    def __init__(self, rows: list[dict]):
        self.by_question: dict[str, list[dict]] = {}
        for row in rows:
            q = question_of(row["messages"])
            if q is not None:
                self.by_question.setdefault(q, []).append(row)
        self.requests: list[dict] = []
        self.unmatched = 0

    def row_for(self, messages: list[dict]) -> dict | None:
        candidates = self.by_question.get(question_of(messages) or "", [])
        if len(candidates) > 1:  # the same question over different tables: prefer the same files
            files = _FILES.search(next((m["content"] for m in messages if m["role"] == "user"), ""))
            for row in candidates:
                theirs = _FILES.search(row["messages"][1]["content"])
                if files and theirs and files.group(1) == theirs.group(1):
                    return row
        return candidates[0] if candidates else None

    def reply(self, messages: list[dict]) -> dict:
        """The assistant message to send back for a request carrying `messages`."""
        row = self.row_for(messages)
        if row is None:
            self.unmatched += 1
            return {"role": "assistant", "content": FALLBACK_REPLY}
        done = sum(1 for m in messages if m.get("role") == "assistant")
        turns = [m for m in row["messages"] if m["role"] == "assistant"]
        if done >= len(turns):
            return {"role": "assistant", "content": EXHAUSTED_REPLY}
        turn = turns[done]
        out = {"role": "assistant", "content": turn.get("content") or ""}
        if turn.get("tool_calls"):
            # `arguments` is a JSON string on the wire, whatever form the dataset stores it in.
            out["tool_calls"] = [
                {"id": c["id"], "type": "function",
                 "function": {"name": c["function"]["name"],
                              "arguments": c["function"]["arguments"]
                              if isinstance(c["function"]["arguments"], str)
                              else json.dumps(c["function"]["arguments"])}}
                for c in turn["tool_calls"]]
        return out


class OracleServer:
    """`Oracle` behind a loopback HTTP server. A context manager; `base_url` is the /v1 root."""

    def __init__(self, rows: list[dict], port: int = 0, max_model_len: int = 32768):
        self.oracle = Oracle(rows)
        outer = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_args):
                pass

            def _send(self, status: int, payload: dict) -> None:
                body = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                self._send(200, {"data": [{"id": "oracle", "max_model_len": max_model_len}]})

            def do_POST(self):
                request = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                outer.oracle.requests.append(request)
                message = outer.oracle.reply(request.get("messages") or [])
                self._send(200, {"choices": [{"message": message, "finish_reason": "stop"}]})

        self.server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self.base_url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"

    @property
    def requests(self) -> list[dict]:
        return self.oracle.requests

    def __enter__(self):
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *_exc):
        self.server.shutdown()
        self.server.server_close()


def read_rows(*paths: Path) -> list[dict]:
    return [json.loads(line) for p in paths for line in p.read_text().splitlines() if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, nargs="+", required=True, help="SFT jsonl file(s)")
    ap.add_argument("--port", type=int, default=0)
    args = ap.parse_args()
    server = OracleServer(read_rows(*args.data), args.port)
    print(f"oracle for {sum(map(len, server.oracle.by_question.values()))} trajectories at "
          f"{server.base_url}", flush=True)
    with server:
        threading.Event().wait()


if __name__ == "__main__":
    main()
