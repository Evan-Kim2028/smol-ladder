"""Supervise a set of harness processes so a stalled evaluation costs minutes, not hours.

Session 1's evaluation stalled for two hours: the engines hung at zero throughput, every trial
timed out slowly and was recorded as a harness error, so results kept "arriving" and nobody
noticed. About $5. The harness cannot see this from inside (each trial just times out), and the
reviewer cannot watch a log for two hours, so the driver watches, with three independent signals:

  * STALL    no result.json written by any live sweep for `stall_s` (default 5 minutes);
  * ERRORS   among a model's last `window` results written since the last (re)start, at least
             `error_share` are harness failures (agent_status != "exit 0");
  * HEALTH   a one-token chat completion to a model's port, once a minute with a short timeout,
             failed twice in a row.

Any of them stops the harness processes cleanly, restarts the servers ONCE, resumes with
`--retry-failed`, and the second time it stops the stage and exits non-zero with the reason. The
total waste is therefore bounded by two detection windows and one server start.

Stopping is by process group of a child this module started (`start_new_session`), never by
pattern-matching process names, and refuses to signal the group it is itself running in.

Every minute one line goes to the progress log and the terminal: trials done per model,
trials per minute, ETA, accrued dollars.

Everything with a side effect is injected (clock, sleep, probes, process start, server restart),
so the tests drive the whole state machine without a process, a socket or a sleep.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

EXIT_STALLED = 75            # EX_TEMPFAIL: the stage gave up after one recovery
DEFAULT_STALL_MIN = 5.0
DEFAULT_WINDOW = 10
DEFAULT_ERROR_SHARE = 0.5
HEALTH_FAILS = 2
HEALTH_TIMEOUT_S = 30.0
MIN_TICK_S = 60.0


@dataclass(frozen=True)
class Target:
    """One harness process and what it writes: read off its own command line, so the supervisor
    watches exactly the files and the port the command uses."""
    model: str
    tag: str
    split: str
    rung: str
    port: int
    expected: int

    @staticmethod
    def from_cmd(argv: tuple[str, ...], env: tuple[tuple[str, str], ...]) -> "Target":
        def arg(flag: str) -> str:
            return argv[argv.index(flag) + 1]
        url = dict(env)["SMOL_LADDER_BASE_URL"]
        return Target(model=arg("--model"), tag=arg("--run-tag"), split=arg("--split"),
                      rung=arg("--rungs").split(",")[0], port=int(url.rsplit(":", 1)[1].split("/")[0]),
                      expected=int(arg("--limit")) * int(arg("--samples")))


def result_files(root: Path, t: Target) -> list[Path]:
    base = Path(root) / t.tag / t.split
    if not base.exists():
        return []
    return list(base.glob(f"*/{t.rung}/result.json")) + list(base.glob(f"*/{t.rung}/s*/result.json"))


def is_failure(path: Path) -> bool:
    try:
        return json.loads(path.read_text()).get("agent_status") != "exit 0"
    except (OSError, ValueError):
        return True


def probe_http(port: int, model: str, timeout: float = HEALTH_TIMEOUT_S) -> bool:
    """A one-token chat completion through the tunnel. An engine that hung at zero throughput
    accepts the connection and never answers, so only a real completion proves it is alive."""
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": "ping"}],
                       "max_tokens": 1, "temperature": 0,
                       "chat_template_kwargs": {"enable_thinking": False}}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return 200 <= resp.status < 300 and bool(json.loads(resp.read()).get("choices"))
    except Exception:  # noqa: BLE001 - any failure to get a completion is "not healthy"
        return False


class Proc:
    """A harness process the supervisor started, in a session of its own so that the whole tree
    (uv, python, the bwrap jails) can be stopped by group and nothing else can be hit."""

    def __init__(self, argv: list[str], env: dict[str, str]):
        self.argv = argv
        self.popen = subprocess.Popen(argv, env=env, start_new_session=True)

    @property
    def pid(self) -> int:
        return self.popen.pid

    def poll(self) -> int | None:
        return self.popen.poll()

    def stop(self, grace: float = 15.0) -> None:
        """TERM the group, wait, KILL it, always reap. The group id is the child's pid (it leads
        its own session); if that ever equals our own group the signal is refused instead."""
        if self.popen.poll() is not None:
            return
        pgid = self.popen.pid
        if pgid == os.getpgrp() or pgid <= 1:
            self.popen.terminate()
        else:
            for sig, wait in ((signal.SIGTERM, grace), (signal.SIGKILL, 5.0)):
                try:
                    os.killpg(pgid, sig)
                except ProcessLookupError:
                    break
                try:
                    self.popen.wait(timeout=wait)
                    return
                except subprocess.TimeoutExpired:
                    continue
        try:
            self.popen.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            pass


@dataclass
class Outcome:
    code: int
    message: str = ""
    restarts: int = 0


@dataclass
class Supervisor:
    name: str
    targets: list[Target]
    results_root: Path
    launch: Callable[[bool], list]            # (retry_failed) -> list of handles (poll/stop/pid)
    restart_servers: Callable[[], bool]
    log: Callable[[str], None]
    accrued: Callable[[], float] = lambda: 0.0
    probe: Callable[[int, str], bool] = probe_http
    clock: Callable[[], float] = time.time     # wall time: result files carry wall-clock mtimes
    sleep: Callable[[float], None] = time.sleep
    tick_s: float = MIN_TICK_S
    stall_s: float = DEFAULT_STALL_MIN * 60.0
    window: int = DEFAULT_WINDOW
    error_share: float = DEFAULT_ERROR_SHARE
    max_restarts: int = 1
    restarts: int = 0
    handles: list = field(default_factory=list)
    started: float = 0.0
    launched: float = 0.0
    health_fails: dict[int, int] = field(default_factory=dict)
    final_rate: float = 0.0                   # trials per minute over the whole run, set on success

    # ── observation ───────────────────────────────────────────────────────────────
    def scan(self, t: Target) -> list[tuple[float, bool]]:
        """(mtime, is_failure) of every result of `t` written since the last (re)launch."""
        out = []
        for p in result_files(self.results_root, t):
            try:
                m = p.stat().st_mtime
            except OSError:
                continue
            if m > self.launched:
                out.append((m, is_failure(p)))
        return sorted(out)

    def totals(self, t: Target) -> tuple[int, int]:
        """(results on disk, of which harness failures), whenever they were written."""
        files = result_files(self.results_root, t)
        return len(files), sum(is_failure(p) for p in files)

    def rate_since_start(self, now: float) -> float:
        """Results written since the first launch, per minute of wall time (all models together)."""
        n = 0
        for t in self.targets:
            for p in result_files(self.results_root, t):
                try:
                    n += p.stat().st_mtime > self.started
                except OSError:
                    pass
        return n / max((now - self.started) / 60.0, 1e-9)

    def alive(self) -> list[int]:
        return [i for i, h in enumerate(self.handles) if h.poll() is None]

    def last_write(self, now: float) -> float:
        newest = self.launched
        for t in self.targets:
            for m, _ in self.scan(t):
                newest = max(newest, m)
        return newest

    def error_trigger(self, indices: list[int]) -> str:
        for i in indices:
            t, recent = self.targets[i], self.scan(self.targets[i])[-self.window:]
            need = max(3, self.window // 2)
            if len(recent) >= need:
                bad = sum(f for _, f in recent)
                if bad / len(recent) >= self.error_share:
                    return (f"{t.model}: {bad} of its last {len(recent)} results are harness "
                            f"failures (threshold {self.error_share:.0%})")
        return ""

    def health_trigger(self, indices: list[int]) -> str:
        if not indices:
            return ""
        with ThreadPoolExecutor(max_workers=len(indices)) as pool:
            oks = list(pool.map(lambda i: self.probe(self.targets[i].port, self.targets[i].model),
                                indices))
        bad = []
        for i, ok in zip(indices, oks):
            port = self.targets[i].port
            self.health_fails[port] = 0 if ok else self.health_fails.get(port, 0) + 1
            if self.health_fails[port] >= HEALTH_FAILS:
                bad.append(f"{self.targets[i].model} (:{port})")
        return (f"health probe failed {HEALTH_FAILS} times in a row: " + ", ".join(bad)) if bad else ""

    def heartbeat(self, now: float, note: str) -> str:
        parts, new = [], 0
        for t in self.targets:
            done, failed = self.totals(t)
            new += len(self.scan(t))
            parts.append(f"{t.model} {done}/{t.expected}" + (f" ({failed} failed)" if failed else ""))
        minutes = max((now - self.launched) / 60.0, 1e-9)
        rate = new / minutes
        left = sum(max(t.expected - self.totals(t)[0], 0) for t in self.targets)
        eta = f"{left / rate:.0f} min" if rate > 0 and left else ("-" if not left else "unknown")
        stamp = time.strftime("%H:%M:%S", time.localtime(now))
        return (f"[{stamp}] {self.name} | " + " | ".join(parts) +
                f" | {rate:.1f} trials/min | ETA {eta} | accrued ${self.accrued():.2f}"
                + (f" | {note}" if note else ""))

    # ── control ───────────────────────────────────────────────────────────────────
    def stop_all(self) -> None:
        for h in self.handles:
            h.stop()

    def start(self, retry: bool) -> None:
        self.launched = self.clock()
        self.health_fails = {}
        self.handles = self.launch(retry)

    def run(self) -> Outcome:
        self.started = self.clock()
        self.start(False)
        try:
            while True:
                self.sleep(self.tick_s)
                now = self.clock()
                live = self.alive()
                reason = ""
                if live:
                    if now - self.last_write(now) >= self.stall_s:
                        reason = (f"STALL: no new result.json for "
                                  f"{(now - self.last_write(now)) / 60.0:.1f} min (limit "
                                  f"{self.stall_s / 60.0:g})")
                    reason = reason or self.error_trigger(live)
                    reason = reason or self.health_trigger(live)
                else:
                    codes = [h.poll() for h in self.handles]
                    if any(c != 0 for c in codes):
                        reason = f"a harness process exited with status {codes}"
                    else:
                        reason = self.error_trigger(list(range(len(self.targets))))
                    if not reason:
                        self.log(self.heartbeat(now, "finished"))
                        self.final_rate = self.rate_since_start(now)
                        return Outcome(0, "all sweeps finished", self.restarts)
                self.log(self.heartbeat(now, reason or "ok"))
                if not reason:
                    continue
                self.stop_all()
                if self.restarts >= self.max_restarts:
                    msg = (f"{self.name}: STOPPED after {self.restarts} recovery attempt(s). {reason}. "
                           "The servers were restarted once and it happened again: not looping. "
                           "Check `driver.py status`, the vLLM logs on the droplet "
                           "(/var/log/smol-ladder/vllm_*.log) and the ssh tunnel; the droplet is "
                           "STILL BILLING.")
                    return Outcome(EXIT_STALLED, msg, self.restarts)
                self.restarts += 1
                self.log(f"{self.name}: {reason}. Harness processes stopped; restarting the servers "
                         f"(recovery {self.restarts} of {self.max_restarts})")
                if not self.restart_servers():
                    return Outcome(EXIT_STALLED, f"{self.name}: {reason}. The server restart failed "
                                   "too; the droplet is STILL BILLING.", self.restarts)
                self.start(True)
        finally:
            self.stop_all()      # an exception, Ctrl-C or a SIGTERM must not orphan sweeps
