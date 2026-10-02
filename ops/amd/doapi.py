"""A thin DigitalOcean client. The token lives only inside the object, never in argv or a log.

The tests inject a fake with the same three methods, so nothing in this repo's tests touches the
network. Reads (GET) are what planning, status and verification use. The mutating calls
(POST create, DELETE destroy) exist because the reviewer-run `driver.py create --yes` and
`destroy --yes`, and the laptop dead-man switch, need them; nothing issues one without an explicit
flag from a human.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path

BASE = "https://api.digitalocean.com/v2"


def load_dotenv(path: Path) -> None:
    """Read KEY=VALUE lines into os.environ without overriding what is already set."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def token_from_env() -> str:
    for name in ("DIGITALOCEAN_ACCESS_TOKEN", "AMD_CLOUD_API_TOKEN"):
        if os.environ.get(name):
            return os.environ[name]
    return ""


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
                raw = resp.read()
                return resp.status, (json.loads(raw) if raw else {})
        except urllib.error.HTTPError as exc:
            return exc.code, {"error": exc.read().decode(errors="replace")[:400]}
        except urllib.error.URLError as exc:
            return 0, {"error": f"network: {exc.reason}"}

    def get(self, path: str) -> tuple[int, dict]:
        return self.request("GET", path)

    def post(self, path: str, body: dict) -> tuple[int, dict]:
        return self.request("POST", path, body)

    def delete(self, path: str) -> tuple[int, dict]:
        return self.request("DELETE", path)
