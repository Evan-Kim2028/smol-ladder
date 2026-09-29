"""A solver agent for Harbor that talks to OpenRouter directly.

`harbor_ext.cmd_agent` installs Command Code in the task container and uploads a login. That
works, but it makes every trial pay for a Node install and puts a credential in the container.
This agent instead runs the model loop in *our* process: the container gets nothing but the
task's tables, and the model is called over HTTPS from here.

Same contract as the cmd agent: write ./solution.py, print the answer as the last line.

    harbor run -p <tasks> --agent-import-path smol_ladder.or_agent:OpenRouter \
        -m stealth/space-bunny-alpha
"""

from __future__ import annotations

import json
import os
import shlex
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

MODEL = "stealth/space-bunny-alpha"
API = "https://openrouter.ai/api/v1/chat/completions"
ENV_KEY = "OPENROUTER_API_KEY"

SYSTEM = """You are solving a data-analysis question.

Work by running Python. The input tables are in ./input (read-only). Use the shell to explore
the data, then write ./solution.py: a self-contained script that reads only from ./input,
computes the answer, and prints the final answer as its LAST line of output. The final answer
is just the value: a number (no commas or units), a short label, yes/no, or a comma-separated
list. Run `python3 solution.py` to check it works.

Compute the answer from the files. Do not look it up online or in any dataset."""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_shell",
            "description": "Run a bash command in the task directory and return its output.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "the bash command to run"}
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_solution",
            "description": "Write ./solution.py.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "the full contents of solution.py"}
                },
                "required": ["code"],
            },
        },
    },
]

SHELL_TIMEOUT = 120


def api_key() -> str:
    key = os.environ.get(ENV_KEY)
    if not key:
        raise RuntimeError(f"{ENV_KEY} is not set")
    return key


def call_model(messages: list[dict], model: str, tools: list[dict]) -> dict:
    """One chat completion, with retries.

    OpenRouter intermittently answers 200 with a body that has no "choices" (a routed-provider
    error, or a rate limit surfaced as JSON). Retrying is the same call the cmd path already
    made, and losing a whole trial to a transient is worse than a few seconds of backoff.
    """
    body = json.dumps({
        "model": model,
        "messages": messages,
        "tools": tools,
        "tool_choice": "auto",
        "temperature": 0.0,
    }).encode()
    headers = {
        "Authorization": f"Bearer {api_key()}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/evan-kim2028/smol-ladder",
        "X-Title": "smol-ladder",
    }
    last = ""
    for attempt in range(5):
        try:
            req = urllib.request.Request(API, data=body, headers=headers)
            with urllib.request.urlopen(req, timeout=180) as resp:
                payload = json.load(resp)
            if payload.get("choices"):
                return payload
            last = json.dumps(payload)[:200]
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}: {e.read().decode()[:200]}"
        except Exception as e:  # noqa: BLE001 - any transport error is worth one more try
            last = f"{type(e).__name__}: {e}"
        time.sleep(min(60, 5 * 2**attempt))
    raise RuntimeError(f"OpenRouter gave no completion after 5 attempts: {last}")


def run_command(command: str, timeout: int = 150, cwd: str | None = None) -> str:
    """Run a shell command for the agent, never raising.

    Two things an agent does routinely used to hang the whole trial:

    - a command that outruns its timeout, and
    - a command that spawns workers of its own (joblib, multiprocessing) which keep the
      captured pipes open after the parent exits, so a plain communicate() waits on children
      the model never asked for. One trial sat on joblib for 16 minutes.

    So: own process group, real deadline, kill the group, then drain whatever was written.
    """
    import os
    import signal
    import subprocess
    proc = subprocess.Popen(["bash", "-c", command], stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, cwd=cwd, start_new_session=True)
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(os.getpgid(proc.pid), sig)
            except (ProcessLookupError, PermissionError):
                break
            try:
                proc.wait(timeout=5)
                break
            except subprocess.TimeoutExpired:
                continue
        # Close our ends before draining: a command that backgrounds a job leaves a
        # grandchild holding the write end, and communicate() would block on it forever.
        for stream in (proc.stdout, proc.stderr):
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    pass
        proc.wait(timeout=5)
        return f"[timed out after {timeout}s]"
    text = out.decode("utf-8", "replace") + \
        ("\n--- stderr ---\n" + err.decode("utf-8", "replace") if err else "")
    return text[-20_000:]


def _as_file(text: str) -> Path:
    """Harbor uploads from a path; the model gave us a string, so stage it on disk."""
    path = Path(tempfile.mkdtemp()) / "solution.py"
    path.write_text(text)
    return path


def solve_loop(instruction: str, run_shell, write_solution, model: str = MODEL,
               max_turns: int = 40) -> list[dict]:
    """The model loop, independent of Harbor.

    run_shell(command) -> str executes in the task container; write_solution(code) stages
    solution.py. Returns the per-turn log. Kept free of Harbor types so it can be tested
    against a stub and reused outside a trial.
    """
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": instruction},
    ]
    log: list[dict] = []
    for _ in range(max_turns):
        completion = call_model(messages, model, TOOLS)
        message = completion["choices"][0]["message"]
        calls = message.get("tool_calls") or []
        entry: dict = {"model": model, "message": message}
        messages.append({
            "role": "assistant",
            "content": message.get("content") or "",
            **({"tool_calls": calls} if calls else {}),
        })
        if not calls:
            log.append(entry)
            break
        results = []
        for call in calls:
            fn = call["function"]
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            if fn["name"] == "write_solution":
                write_solution(args.get("code", ""))
                output = "written /app/solution.py"
            else:
                output = run_shell(args.get("command", ""))
            entry.setdefault("tool_results", []).append(
                {"name": fn["name"], "output": output})
            results.append({"role": "tool", "tool_call_id": call["id"], "content": output})
        messages.extend(results)
        log.append(entry)
    return log


def _harbor_base():
    """Import Harbor lazily so this module imports (and is testable) without it."""
    from harbor.agents.installed.base import BaseInstalledAgent
    return BaseInstalledAgent


class OpenRouter:  # replaced below once Harbor is importable
    pass


def _build_agent_class():
    from typing import override

    from harbor.environments.base import BaseEnvironment
    from harbor.models.agent.context import AgentContext

    base = _harbor_base()

    class _OpenRouter(base):
        @staticmethod
        @override
        def name() -> str:
            return "openrouter"

        @override
        def get_version_command(self) -> str | None:
            return None  # nothing to install

        @override
        async def install(self, environment: BaseEnvironment) -> None:
            return None  # the task image already has the Python stack

        @override
        async def run(self, instruction: str, environment: BaseEnvironment,
                      context: AgentContext) -> None:
            import asyncio

            async def in_container(command: str) -> str:
                result = await environment.exec(
                    f"cd /app && timeout {SHELL_TIMEOUT} bash -c "
                    f"{shlex.quote(command)} 2>&1")
                return (getattr(result, "output", None) or str(result))[-20_000:]

            async def write_solution_async(code: str) -> None:
                await environment.upload_file(_as_file(code), "/app/solution.py")

            loop = asyncio.get_event_loop()
            # The agent's commands run in the task container, never on this host.
            run_sync = lambda c: loop.run_in_executor(  # noqa: E731
                None, lambda: asyncio.run(in_container(c)))
            write_sync = lambda c: loop.run_in_executor(  # noqa: E731
                None, lambda: asyncio.run(write_solution_async(c)))
            log = solve_loop(instruction, run_sync, write_sync, self.model_name or MODEL)
            await environment.exec("mkdir -p /logs/agent")
            await environment.upload_file(
                _as_file("\n".join(json.dumps(m) for m in log)),
                "/logs/agent/openrouter.jsonl")

    return _OpenRouter


try:  # Harbor is a separate tool; keep the import optional so tests run without it.
    OpenRouter = _build_agent_class()  # type: ignore[misc,assignment]
except Exception:  # pragma: no cover - only when Harbor is absent
    pass
