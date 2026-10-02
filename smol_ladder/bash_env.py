"""Which Python the `--agent bash` sandbox runs the model's commands under.

The SFT rows were recorded in a Python 3.12 container. Row 1852 of data/train/sft_upstream/train.jsonl
is a `pip list` taken there: pandas 3.0.3, numpy 2.4.6, scikit-learn 1.9.0, scipy 1.17.1,
matplotlib 3.11.0, seaborn 0.13.2, statsmodels 0.14.6, tabulate 0.10.0, pip 25.0.1 -- and no pyarrow,
openpyxl, plotly or xgboost (the rows show `No module named 'xgboost'` and the like). The repo's own
venv (3.14, a different numpy, pyarrow installed) differs in tracebacks, in string-array reprs and
in which imports fail, so commands run under a dedicated environment built to that list
(`tools/build_eval_env.sh`; `docs/LOCAL_MODELS.md` has the evidence), and every result records which
interpreter it was.

`SMOL_LADDER_BASH_ENV`: a path to such an environment; `current` to use the repo's venv on purpose;
unset to use DEFAULT_ENV when it exists. A missing or invalid environment falls back to the
repo's venv with a warning on stderr and in `result.json["bash_sandbox"]["warning"]`.
"""

from __future__ import annotations

import functools
import os
import platform
import re
import sys
from dataclasses import dataclass
from pathlib import Path

ENV_VAR = "SMOL_LADDER_BASH_ENV"
DEFAULT_ENV = Path("/var/tmp/smol-ladder/eval-env-py312")
KEY_PACKAGES = ("pandas", "numpy", "scipy", "scikit-learn", "statsmodels", "matplotlib", "seaborn",
                "tabulate", "pyarrow", "openpyxl", "plotly", "xgboost", "pip")


@dataclass(frozen=True)
class Sandbox:
    base: Path               # the interpreter prefix, mounted at /usr/local in the jail
    sites: tuple[Path, ...]  # site-packages directories, data stack first
    version: str             # "3.12"
    record: dict             # what result.json says about it


def _norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def package_versions(sites) -> dict[str, str]:
    """Versions of KEY_PACKAGES, read from the *.dist-info directory names (no import, no subprocess)."""
    found: dict[str, str] = {}
    for site in sites:
        for info in Path(site).glob("*.dist-info"):
            name, _, ver = info.name[:-len(".dist-info")].rpartition("-")
            if _norm(name) in KEY_PACKAGES:
                found.setdefault(_norm(name), ver)
    return {k: found[k] for k in KEY_PACKAGES if k in found}


def _current() -> tuple[Path, tuple[Path, ...], str]:
    sites = list(dict.fromkeys(Path(p).resolve() for p in sys.path
                               if p.endswith("site-packages") and Path(p).is_dir()))
    # The one that holds the data stack is the one that appears in tracebacks (under
    # `uv run --with X` there are two: X's overlay and the project's).
    sites.sort(key=lambda site: not (site / "pandas").is_dir())
    return Path(sys.base_prefix).resolve(), tuple(sites), f"{sys.version_info.major}.{sys.version_info.minor}"


def _from_env(env: Path):
    """(base, sites, version, full python version) of a venv, or a reason string."""
    py = env / "bin" / "python"
    if not py.exists():
        return f"{env} has no bin/python"
    sites = sorted((env / "lib").glob("python3.*/site-packages"))
    if len(sites) != 1:
        return f"{env} has no single lib/python3.x/site-packages"
    version = sites[0].parent.name.removeprefix("python")
    full = version
    cfg = env / "pyvenv.cfg"
    if cfg.exists():
        m = re.search(r"^version(?:_info)?\s*=\s*(\S+)", cfg.read_text(), re.M)
        full = m.group(1) if m else version
    return py.resolve().parent.parent, (sites[0],), version, full


def resolve(environ=None, default: Path | None = None) -> Sandbox:
    environ = os.environ if environ is None else environ
    default = DEFAULT_ENV if default is None else default
    asked = environ.get(ENV_VAR, "")
    cur_base, cur_sites, cur_ver = _current()

    def current(warning: str = "") -> Sandbox:
        rec = {"python": platform.python_version(), "env": "current venv", "fallback": bool(warning),
               "packages": package_versions(cur_sites)}
        if warning:
            rec["warning"] = warning
            print(f"WARNING: {warning}", file=sys.stderr)
        return Sandbox(cur_base, cur_sites, cur_ver, rec)

    if asked == "current":
        return current()
    target = Path(asked) if asked else default
    got = _from_env(target) if (asked or target.exists()) else f"{target} does not exist"
    if isinstance(got, str):
        return current(f"the bash sandbox is running under the repo's venv (Python "
                       f"{platform.python_version()}), NOT the recording environment: {got}. Build it "
                       f"with tools/build_eval_env.sh or set {ENV_VAR}; results are not comparable "
                       "with the recordings' tool output")
    base, sites, version, full = got
    return Sandbox(base, sites, version, {"python": full, "env": str(target), "fallback": False,
                                          "packages": package_versions(sites)})


@functools.lru_cache(maxsize=None)
def _cached(asked: str, default: str) -> Sandbox:
    return resolve({ENV_VAR: asked} if asked else {}, Path(default))


def cached() -> Sandbox:
    """Resolved once per process (and per variable value): a sweep must not change interpreter."""
    return _cached(os.environ.get(ENV_VAR, ""), str(DEFAULT_ENV))


cached.cache_clear = _cached.cache_clear  # type: ignore[attr-defined]
