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
import re
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


class ClientError(RuntimeError):
    """A 4xx other than 429: the request itself is wrong, so it is never retried."""

    def __init__(self, message: str, status: int = 400, body: str = ""):
        super().__init__(message)
        self.status = status
        self.body = body


class ContextOverflow(ClientError):
    """The server refused the request because prompt + max_tokens exceeds the context length."""


def is_context_error(body: str) -> bool:
    low = body.lower()
    return "context length" in low or "context_length" in low or "maximum context" in low


def overflow_numbers(body: str) -> tuple[int | None, int | None]:
    """(input tokens, context length) as far as the error text says.

    vLLM: "You passed 16385 input tokens and requested 0 output tokens. However, the model's
    context length is only 16384 tokens" / "...your prompt contains at least 15500 input tokens".
    """
    m = re.search(r"(\d+)\s+input tokens", body)
    c = re.search(r"context length is (?:only )?(\d+)", body, re.I)
    return (int(m.group(1)) if m else None, int(c.group(1)) if c else None)


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
            body_text = e.read().decode("utf-8", "replace")
            last = f"HTTP {e.code}: {body_text[:300]}"
            if 400 <= e.code < 500 and e.code != 429:
                # The same request gets the same answer: retrying it five times only delays the
                # failure by a minute and hides what it was.
                cls = ContextOverflow if is_context_error(body_text) else ClientError
                raise cls(f"{ep.base_url} rejected the request: {last}", e.code, body_text) from None
        except Exception as e:  # noqa: BLE001 - any transport error is worth one more try
            last = f"{type(e).__name__}: {e}"
        time.sleep(min(60, 5 * 2**attempt))
    raise RuntimeError(f"{ep.base_url} gave no completion after 5 attempts: {last}")


def run_command(command: str, timeout: int = 150, cwd: str | None = None,
                limit: int | None = 20_000) -> str:
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
    return text if limit is None else text[-limit:]


def _as_file(text: str) -> Path:
    """Harbor uploads from a path; the model gave us a string, so stage it on disk."""
    path = Path(tempfile.mkdtemp()) / "solution.py"
    path.write_text(text)
    return path


# ── context budget ────────────────────────────────────────────────────────────
# Tool output is capped the way upstream's own SFT rows are. Measured on
# data/train/sft_upstream (4,439 train rows, 16,273 tool results): no tool result is longer than
# 8,016 characters, 126 are exactly 8,016, and every one of those is the first 8,000 characters
# of the output plus the 16 characters "\n... [truncated]". That is a head-only cut at 8,000, so
# it is the cut the models were trained on and the one evaluation reproduces. (Upstream's repo
# publishes no agent loop, so the rows are the only record of it.) p99 of tool results is 4,687
# characters and p95 1,541, so the cut touches about 1 result in 130.
TOOL_OUTPUT_MAX_CHARS = 8000
TRUNCATION_MARKER = "\n... [truncated]"

# Upstream's one published generation cap is eval_pass1's 1024 new tokens. A bash turn is a tool
# call or a closing sentence: p99 of assistant turns in the SFT rows is ~2,200 characters.
BASH_MAX_TOKENS = 1024
# Below this many tokens of room a tool call cannot even be completed, so the episode ends.
MIN_COMPLETION_TOKENS = 64
CONTEXT_MARGIN = 8
CONTEXT_ENV = "SMOL_LADDER_MAX_MODEL_LEN"


def truncate_output(text: str, limit: int = TOOL_OUTPUT_MAX_CHARS) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    return text[:limit] + TRUNCATION_MARKER, True


def context_length(ep: Endpoint, model: str, env: dict | None = None) -> int | None:
    """The served context window: the env override, else vLLM's `max_model_len` from GET /models.

    None when neither is known (OpenRouter's /models carries no such field per served model), in
    which case the loop relies on the server's own 400 to learn the limit.
    """
    env = os.environ if env is None else env
    raw = env.get(CONTEXT_ENV)
    if raw:
        try:
            return int(raw)
        except ValueError:
            pass
    try:
        req = urllib.request.Request(f"{ep.base_url}/models", headers=ep.headers())
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.load(resp).get("data") or []
        entry = next((d for d in data if d.get("id") == model), data[0] if data else {})
        n = entry.get("max_model_len")
        return int(n) if n else None
    except Exception:  # noqa: BLE001 - an unknown window is not a reason to fail the trial
        return None


class Episode:
    """What happened in one loop, for result.json. Mutated by the loops as they run."""

    def __init__(self, ctx: int | None = None):
        self.stop_reason: str | None = None
        self.turns = 0
        self.last_prompt_tokens: int | None = None
        self.truncated_outputs = 0
        self.context_length = ctx
        self._sent = 0

    def estimate_prompt_tokens(self, messages: list[dict], tools: list[dict] | None) -> int:
        """Prompt size for the next request. Exact usage of the previous request plus a
        conservative estimate of what was appended since; with no usage, an optimistic
        chars/4 (a too-big request is caught by the server's 400 and retried exactly)."""
        if self.last_prompt_tokens is not None:
            return self.last_prompt_tokens + len(json.dumps(messages[self._sent:])) // 3
        return (len(json.dumps(messages)) + len(json.dumps(tools or []))) // 4

    def as_dict(self) -> dict:
        return {"stop_reason": self.stop_reason, "turns": self.turns,
                "last_prompt_tokens": self.last_prompt_tokens,
                "truncated_outputs": self.truncated_outputs,
                "context_length": self.context_length}


def complete_within_context(messages: list[dict], model: str, tools: list[dict] | None,
                            ep: Endpoint, episode: Episode, cap: int | None) -> dict | None:
    """One completion whose max_tokens fits the remaining context; None when none can.

    None is "context exhausted", a property of the model's conversation, not a transport error.
    """
    max_tokens = cap
    if episode.context_length:
        room = (episode.context_length - episode.estimate_prompt_tokens(messages, tools)
                - CONTEXT_MARGIN)
        if room < MIN_COMPLETION_TOKENS:
            return None
        max_tokens = min(cap, room) if cap else room
    try:
        try:
            completion = call_model(messages, model, tools, ep, max_tokens=max_tokens)
        except ContextOverflow as e:
            used, window = overflow_numbers(e.body)
            window = window or episode.context_length
            if used is None or window is None:
                return None
            episode.context_length = window
            room = window - used - CONTEXT_MARGIN
            if room < MIN_COMPLETION_TOKENS:
                return None
            completion = call_model(messages, model, tools, ep,
                                    max_tokens=min(cap, room) if cap else room)
    except ContextOverflow:
        return None
    usage = completion.get("usage") or {}
    if usage.get("prompt_tokens") is not None:
        episode.last_prompt_tokens = usage["prompt_tokens"]
        episode._sent = len(messages)
    return completion


def dump_trial(messages: list[dict], episode: Episode | None = None) -> None:
    """Write transcript.json (and turns.json, episode.json) into the cwd. Called from a finally
    in the jailed script, so a crashed or exhausted trial still leaves its conversation."""
    Path("transcript.json").write_text(json.dumps(messages))
    Path("turns.json").write_text(json.dumps(
        sum(1 for m in messages if m.get("role") == "assistant")))
    if episode is not None:
        Path("episode.json").write_text(json.dumps(episode.as_dict()))


def solve_loop(instruction: str, run_shell, write_solution, model: str = MODEL,
               max_turns: int = 40, ep: Endpoint | None = None,
               messages: list[dict] | None = None,
               episode: Episode | None = None) -> list[dict]:
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
    episode = episode or Episode()
    if messages is None:
        messages = []
    # The caller may hand in the list so it can save it from a `finally` if the loop raises.
    messages.extend([
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": instruction},
    ])
    episode.stop_reason = "error"  # replaced on every clean exit; stays if a call raises
    for _ in range(max_turns):
        # No max_tokens and no budget here (prompts and caps unchanged); a context overflow
        # still ends the episode cleanly instead of crashing.
        completion = complete_within_context(messages, model, TOOLS, ep, episode, None)
        if completion is None:
            episode.stop_reason = "context_exhausted"
            break
        episode.turns += 1
        message = completion["choices"][0]["message"]
        calls = message.get("tool_calls") or []
        messages.append({
            "role": "assistant",
            "content": message.get("content") or "",
            **({"tool_calls": calls} if calls else {}),
        })
        if not calls:
            episode.stop_reason = "model_stopped"
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
    else:
        episode.stop_reason = "max_turns"
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
              max_turns: int = 16, ep: Endpoint | None = None,
              episode: Episode | None = None) -> list[dict]:
    """The SFT protocol: one `bash` tool, and the loop ends when the answer is submitted.

    Stopping on submission is upstream's "then stop" made executable. Without it a 2B model that
    has answered correctly keeps calling bash, sometimes overwriting its own answer, and the
    trial measures its ability to stop talking rather than its ability to answer. The published
    trajectories show the pattern plainly: submit, then one short closing message, nothing more.

    Upstream's trajectories run 3-12 turns; 16 is a ceiling that no published row reaches.

    Tool output is cut to TOOL_OUTPUT_MAX_CHARS the way upstream's rows are, each request's
    max_tokens is 1024 clamped to the room left in the context, and an exhausted context ends the
    episode (stop_reason "context_exhausted", graded on whatever answer.txt holds) rather than
    crashing it. `episode` is filled in with stop_reason, turns and token/truncation counts.

    Returns the caller's `messages`, extended in place, so the transcript includes the system and
    user turns the caller built -- this is the protocol the SFT arm trains in, so its trajectory
    is the one whose completeness matters most. `messages` is mutated rather than copied because
    it is the live conversation; a caller that wants the opening turns must not lose them, and
    the submission is attached to the turn that made it.
    """
    ep = ep or endpoint()
    episode = episode or Episode(context_length(ep, model))
    episode.stop_reason = "error"  # replaced on every clean exit; stays if a call raises
    for _ in range(max_turns):
        completion = complete_within_context(messages, model, BASH_TOOL, ep, episode,
                                             BASH_MAX_TOKENS)
        if completion is None:
            episode.stop_reason = "context_exhausted"
            break
        episode.turns += 1
        message = completion["choices"][0]["message"]
        calls = message.get("tool_calls") or []
        turn = len(messages)
        messages.append({
            "role": "assistant",
            "content": message.get("content") or "",
            **({"tool_calls": calls} if calls else {}),
        })
        if not calls:
            episode.stop_reason = "model_stopped"
            break
        results = []
        for call in calls:
            fn = call["function"]
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            command = localise_paths(args.get("command", ""))
            output, cut = truncate_output(run_shell(command))
            episode.truncated_outputs += cut
            results.append({"role": "tool", "tool_call_id": call["id"], "content": output})
        messages.extend(results)
        submitted = read_answer()
        if submitted is not None:
            messages[turn]["submitted"] = submitted
            episode.stop_reason = "answer_submitted"
            break
    else:
        episode.stop_reason = "max_turns"
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
