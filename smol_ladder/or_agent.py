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

from smol_ladder.upstream import BASH_TOOL, extract_code, is_program, localise_paths

MODEL = "stealth/space-bunny-alpha"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
API_KEY_ENV = "OPENROUTER_API_KEY"


class Endpoint:
    """Where the completions come from, resolved from the environment.

    Any OpenAI-compatible server speaks this shape: vLLM, llama.cpp's server, ollama, LM Studio,
    OpenRouter, a stub in a test. So the base URL, the name of the API-key variable and the
    model id are all configuration rather than constants, and a loopback URL is allowed to carry
    no key at all -- vLLM and llama.cpp both reject a bearer header they did not ask for, and
    making every local run invent a token to satisfy the remote path is how a local eval ends up
    not runnable at all.
    """

    def __init__(self, base_url: str, api_key: str = "", chat_template_kwargs: dict | None = None,
                 name: str = "smol-ladder"):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.chat_template_kwargs = chat_template_kwargs
        self.name = name

    @property
    def url(self) -> str:
        return f"{self.base_url}/chat/completions"

    def headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
            if "openrouter.ai" in self.base_url:
                headers["HTTP-Referer"] = "https://github.com/evan-kim2028/smol-ladder"
                headers["X-Title"] = self.name
        return headers


def _is_local(base_url: str) -> bool:
    return any(h in base_url for h in ("localhost", "127.0.0.1", "0.0.0.0", "[::1]"))


def endpoint(env: dict | None = None, model: str = MODEL) -> Endpoint:
    """Resolve the endpoint.

    SMOL_LADDER_BASE_URL    the server; a loopback one needs no key
    SMOL_LADDER_API_KEY_ENV which env var holds the key (default OPENROUTER_API_KEY)
    SMOL_LADDER_CHAT_TEMPLATE_KWARGS
                           JSON passed through as chat_template_kwargs. Defaults to
                           {"enable_thinking": False} because that is how every model in this
                           study was trained and how upstream scores it; set it to "" to send
                           none (a server whose template has no such flag rejects the kwarg).
    """
    env = os.environ if env is None else env
    base = (env.get("SMOL_LADDER_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
    key_env = env.get("SMOL_LADDER_API_KEY_ENV") or API_KEY_ENV
    key = env.get(key_env, "")
    if not key and not _is_local(base):
        raise RuntimeError(f"{key_env} is not set, and {base} is not a local server")
    raw = env.get("SMOL_LADDER_CHAT_TEMPLATE_KWARGS")
    kwargs = {"enable_thinking": False} if raw is None else (json.loads(raw) if raw else None)
    return Endpoint(base, key, kwargs)

SYSTEM = """You are solving a data-analysis question.

Work by running Python. The input tables are in ./input (read-only). Use the shell to explore
the data, then write ./solution.py: a self-contained script that reads only from ./input,
computes the answer, and prints the final answer as its LAST line of output. That last line is
graded on its own, so it must be the value alone: a number (no commas or units), a short label,
yes/no, or a comma-separated list. No label, no "Answer:" prefix, no trailing explanation --
"Answer: 42" grades as 0, "42" grades as 1. Run `python3 solution.py` to check it works.

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
    key = os.environ.get(API_KEY_ENV)
    if not key:
        raise RuntimeError(f"{API_KEY_ENV} is not set")
    return key


def call_model(messages: list[dict], model: str, tools: list[dict] | None,
               ep: Endpoint | None = None, max_tokens: int | None = None) -> dict:
    """One chat completion, with retries.

    OpenRouter intermittently answers 200 with a body that has no "choices" (a routed-provider
    error, or a rate limit surfaced as JSON). Retrying is the same call the cmd path already
    made, and losing a whole trial to a transient is worse than a few seconds of backoff. A local
    server does the same thing on an out-of-memory batch, so the retry is not OpenRouter-specific.

    `tools=None` means no tools at all: upstream's one-turn program protocol passes no `tools`
    key, and sending an empty list instead is not the same request (some servers reject it).
    """
    ep = ep or endpoint()
    body: dict = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
    }
    if tools is not None:
        body["tools"] = tools
        body["tool_choice"] = "auto"
    if max_tokens is not None:
        body["max_tokens"] = max_tokens
    if ep.chat_template_kwargs:
        body["chat_template_kwargs"] = ep.chat_template_kwargs
    payload_bytes = json.dumps(body).encode()
    headers = ep.headers()
    last = ""
    for attempt in range(5):
        try:
            req = urllib.request.Request(ep.url, data=payload_bytes, headers=headers)
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
    raise RuntimeError(f"{ep.base_url} gave no completion after 5 attempts: {last}")


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
               max_turns: int = 40, ep: Endpoint | None = None) -> list[dict]:
    """The model loop, independent of Harbor.

    run_shell(command) -> str executes in the task container; write_solution(code) stages
    solution.py. Kept free of Harbor types so it can be tested against a stub and reused outside
    a trial.

    **Returns the whole conversation, in order**, as the same message dicts that were sent:
    the system turn, the user turn, then each assistant message and each tool result. It used to
    return only the assistant turns, each wrapped with its own `tool_results`, which is enough to
    *count* turns and not enough to train on: the file this becomes is an SFT trajectory, and a
    trajectory that starts at the model's first reply has no question in it and no system prompt
    to have produced the reply that followed. Both of those are the scaffolding a chat template
    re-attaches anyway and silently gets wrong -- it has no way to know which prompt the turns
    answered, or that this run's contract was ours rather than the SFT arm's.

    The wire shape is kept verbatim rather than reshaped into the assistant/tool pairs upstream's
    dataset stores, because `train.traces` does that conversion and can do it once, from a file
    whose content is the record of what actually happened.
    """
    ep = ep or endpoint()
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": instruction},
    ]
    for _ in range(max_turns):
        completion = call_model(messages, model, TOOLS, ep)
        message = completion["choices"][0]["message"]
        calls = message.get("tool_calls") or []
        messages.append({
            "role": "assistant",
            "content": message.get("content") or "",
            **({"tool_calls": calls} if calls else {}),
        })
        if not calls:
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
            results.append({"role": "tool", "tool_call_id": call["id"], "content": output})
        messages.extend(results)
    return messages


def _harbor_base():
    """Import Harbor lazily so this module imports (and is testable) without it."""
    from harbor.agents.installed.base import BaseInstalledAgent
    return BaseInstalledAgent


# ── the upstream protocols, as loops ──────────────────────────────────────────
# MAX_NEW_TOKENS is upstream's eval_pass1.py default. Upstream caps the *program* at 1024
# generated tokens; anything longer is a 2B model repeating itself, and a truncated program has
# no closing fence so it cannot compile and scores zero anyway.
UPSTREAM_MAX_TOKENS = 1024


def program_once(messages: list[dict], model: str, ep: Endpoint | None = None,
                 max_tokens: int = UPSTREAM_MAX_TOKENS) -> dict:
    """One turn, no tools: the model writes a program, we hand back the extracted code.

    This is `eval_pass1.py`'s whole generation step. Nothing loops, because upstream's rollout
    is one turn; the loop in our `tools` agent is the part these models were not trained on.
    """
    ep = ep or endpoint()
    completion = call_model(messages, model, None, ep, max_tokens=max_tokens)
    message = completion["choices"][0]["message"]
    return {"model": model, "message": message,
            "code": extract_code(message.get("content") or ""),
            "ran": is_program(extract_code(message.get("content") or ""))}


def bash_loop(messages: list[dict], run_shell, read_answer, model: str = MODEL,
              max_turns: int = 16, ep: Endpoint | None = None) -> list[dict]:
    """The SFT protocol: one `bash` tool, and the loop ends when the answer is submitted.

    Stopping on submission is upstream's "then stop" made executable. Without it a 2B model that
    has answered correctly keeps calling bash, sometimes overwriting its own answer, and the
    trial measures its ability to stop talking rather than its ability to answer. The published
    trajectories show the pattern plainly: submit, then one short closing message, nothing more.

    Upstream's trajectories run 3-12 turns; 16 is a ceiling that no published row reaches.

    Returns the caller's `messages`, extended in place, so the transcript includes the system and
    user turns the caller built -- this is the protocol the SFT arm trains in, so its trajectory
    is the one whose completeness matters most. `messages` is mutated rather than copied because
    it is the live conversation; a caller that wants the opening turns must not lose them, and
    the submission is attached to the turn that made it.
    """
    ep = ep or endpoint()
    for _ in range(max_turns):
        completion = call_model(messages, model, BASH_TOOL, ep)
        message = completion["choices"][0]["message"]
        calls = message.get("tool_calls") or []
        turn = len(messages)
        messages.append({
            "role": "assistant",
            "content": message.get("content") or "",
            **({"tool_calls": calls} if calls else {}),
        })
        if not calls:
            break
        results = []
        for call in calls:
            fn = call["function"]
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            command = localise_paths(args.get("command", ""))
            output = run_shell(command)
            results.append({"role": "tool", "tool_call_id": call["id"], "content": output})
        messages.extend(results)
        submitted = read_answer()
        if submitted is not None:
            # On the assistant turn, so an exporter finds the submission without re-running the
            # trial -- and with the call's own arguments, which is where the value actually is.
            messages[turn]["submitted"] = submitted
            break
    return messages


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
