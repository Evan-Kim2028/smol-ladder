"""Run a Python script offline in bubblewrap, with the task's tables read-only at ./input."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Run:
    stdout: str
    stderr: str
    returncode: int
    timed_out: bool


def run_script(script: Path, inputs: Path, timeout: int = 120) -> Run:
    work = script.parent.resolve()
    cmd = [
        "bwrap",
        "--ro-bind", "/", "/",
        "--dev", "/dev",
        "--proc", "/proc",
        "--tmpfs", "/tmp",
        "--bind", str(work), "/tmp/work",
        "--ro-bind", str(inputs.resolve()), "/tmp/work/input",
        "--chdir", "/tmp/work",
        "--unshare-net",
        "--unshare-pid",
        "--die-with-parent",
        "--setenv", "MPLBACKEND", "Agg",
        sys.executable, script.name,
    ]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return Run(p.stdout, p.stderr, p.returncode, False)
    except subprocess.TimeoutExpired as e:
        out = e.stdout.decode(errors="replace") if isinstance(e.stdout, bytes) else e.stdout or ""
        err = e.stderr.decode(errors="replace") if isinstance(e.stderr, bytes) else e.stderr or ""
        return Run(out, err, -1, True)
