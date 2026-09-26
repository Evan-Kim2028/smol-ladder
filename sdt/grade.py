"""SmolDataEnvs' own grader.py, so our pass/fail matches the dataset's definition."""

from __future__ import annotations

import importlib.util
import sys
from functools import cache

from huggingface_hub import hf_hub_download


@cache
def _grader():
    path = hf_hub_download("FineEnvs/SmolDataEnvs", "grader.py", repo_type="dataset")
    spec = importlib.util.spec_from_file_location("smoldataenvs_grader", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["smoldataenvs_grader"] = mod  # its dataclasses need this
    spec.loader.exec_module(mod)
    return mod


def last_line(stdout: str) -> str:
    lines = [ln.strip() for ln in (stdout or "").splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def grade(row: dict, prediction: str) -> float:
    if not prediction:
        return 0.0
    r = _grader().grade(
        row["answer"],
        prediction,
        reward_mode=row["reward_mode"],
        abs_tol=row["atol"],
        rel_tol=row["rtol"],
    )
    return float(r.reward)
