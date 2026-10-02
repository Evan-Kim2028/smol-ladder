"""A thin DigitalOcean client. The token lives only inside the object, never in argv or a log.

The tests inject a fake with the same three methods, so nothing in this repo's tests touches the
network. Reads (GET) are what planning, status and verification use. The mutating calls
(POST create, DELETE destroy) exist because the reviewer-run `driver.py create --yes` and
`destroy --yes`, and the laptop dead-man switch, need them; nothing issues one without an explicit
flag from a human.
"""

from __future__ import annotations

import http.client
import json
import os
import urllib.error
import urllib.request
from pathlib import Path

BASE = "https://api.digitalocean.com/v2"


#: In preference order: the repo-specific name first, the generic one second.
DO_TOKEN_VARS = ("AMD_CLOUD_API_TOKEN", "DIGITALOCEAN_ACCESS_TOKEN")
#: Everything a child process (ssh, scp, the eval harness, uv) must never inherit. None of them
#: needs a token: the droplet has its own HF token from remote.env, and the harness talks to a
#: loopback tunnel with no key.
SECRET_VARS = (*DO_TOKEN_VARS, "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")


def load_dotenv(path: Path) -> None:
    """Read KEY=VALUE lines into os.environ. The file WINS over a variable already in the shell:
    a stale `export` from an old session must not outrank the file the owner just edited.

    If the file names a DigitalOcean token, the other DigitalOcean variable name is dropped from
    the environment even when the file does not set it, so the stale one cannot be picked up."""
    if not path.exists():
        return
    seen: set[str] = set()
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip().removeprefix("export ").strip()
        os.environ[key] = value.strip().strip("'\"")
        seen.add(key)
    if seen & set(DO_TOKEN_VARS):
        for name in DO_TOKEN_VARS:
            if name not in seen:
                os.environ.pop(name, None)


def token_from_env() -> str:
    for name in DO_TOKEN_VARS:
        if os.environ.get(name):
            return os.environ[name]
    return ""


def child_env(extra: dict[str, str] | None = None, base: dict[str, str] | None = None) -> dict:
    """The environment for a child process: the parent's minus every secret, plus `extra`."""
    env = {k: v for k, v in (os.environ if base is None else base).items()
           if k not in SECRET_VARS}
    env.update(extra or {})
    return env


class DoApi:
    def __init__(self, token: str, base: str = BASE, timeout: float = 30.0):
        if not token:
            raise SystemExit("no DigitalOcean token: set DIGITALOCEAN_ACCESS_TOKEN or "
                             "AMD_CLOUD_API_TOKEN (repo .env, git-ignored)")
        self._token = token
        self._base = base
        self._timeout = timeout

    def __repr__(self) -> str:  # never leak the token through a traceback or a log line
        return "DoApi(token=<hidden>)"

    def request(self, method: str, path: str, body: dict | None = None) -> tuple[int, dict]:
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(
            self._base + path, data=data, method=method,
            headers={"Authorization": f"Bearer {self._token}",
                     "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                status = resp.status
                raw = resp.read()          # a read timeout or IncompleteRead lands in the handler
            payload = json.loads(raw) if raw else {}
            if not isinstance(payload, dict):
                raise ValueError(f"expected a JSON object, got {type(payload).__name__}")
            return status, payload
        except urllib.error.HTTPError as exc:
            try:
                text = exc.read().decode(errors="replace")[:400]
            except Exception:  # noqa: BLE001 - the status code is the answer; the body is a bonus
                text = ""
            return exc.code, {"error": text}
        except urllib.error.URLError as exc:
            return 0, {"error": f"network: {exc.reason}"}
        except (OSError, http.client.HTTPException, ValueError) as exc:
            # Timeout, reset, IncompleteRead, a 200 whose body is not JSON. Status 0 means "the
            # outcome is UNKNOWN", and no caller may read it as "nothing is there": a listing that
            # failed to parse must not look like an empty listing.
            return 0, {"error": f"{type(exc).__name__}: {exc}"}

    def get(self, path: str) -> tuple[int, dict]:
        return self.request("GET", path)

    def post(self, path: str, body: dict) -> tuple[int, dict]:
        return self.request("POST", path, body)

    def delete(self, path: str) -> tuple[int, dict]:
        return self.request("DELETE", path)
