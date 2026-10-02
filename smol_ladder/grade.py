"""SmolDataEnvs' own grader.py, so our pass/fail matches the dataset's definition."""

from __future__ import annotations

import importlib.util
import sys
from functools import cache

from huggingface_hub import hf_hub_download


MATH_VERIFY_DEADLINE = 5.0


def _math_verify_match(gold: str, candidate: str) -> bool:
    """The dataset grader's tier 4, made to work off the main thread.

    Its own `_math_verify_match` calls `parse`/`verify` with their default timeouts, which arm
    `signal.alarm()` -- legal only on the main thread. The ladder grades inside a
    ThreadPoolExecutor, so there the call raised ValueError, the grader's bare `except` turned that
    into "no match", and the tier was silently off for every run: a gold of "28x28x1" graded
    "28x28" 1.0 on the main thread and 0.0 in a sweep. The deadline is kept, as a thread join
    instead of a signal; a call that outlives it is abandoned and counts as no match, as upstream's
    own timeout does.
    """
    import logging
    import threading

    # parse/verify warn on every call that their own timeout is off; the join below is the timeout
    logging.getLogger("math_verify").setLevel(logging.ERROR)
    verdict: list[bool] = []

    def run() -> None:
        try:
            from math_verify import parse, verify
            verdict.append(bool(verify(parse(gold, parsing_timeout=None),
                                       parse(candidate, parsing_timeout=None),
                                       timeout_seconds=None)))
        except Exception:  # noqa: BLE001 - upstream's grader treats any failure as no match
            pass

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    worker.join(MATH_VERIFY_DEADLINE)
    return bool(verdict and verdict[0])


@cache
def _grader():
    path = hf_hub_download("FineEnvs/SmolDataEnvs", "grader.py", repo_type="dataset")
    spec = importlib.util.spec_from_file_location("smoldataenvs_grader", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["smoldataenvs_grader"] = mod  # its dataclasses need this
    spec.loader.exec_module(mod)
    mod._math_verify_match = _math_verify_match
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
