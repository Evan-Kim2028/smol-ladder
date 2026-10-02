"""The dataset grader's math-verify tier must work where the ladder grades: in worker threads.

`math_verify.parse`/`verify` arm `signal.alarm()` timeouts, which Python allows only on the main
thread. The sweep grades inside a ThreadPoolExecutor, so the grader's own tier-4 call raised
ValueError there, its bare `except` turned that into "no match", and the tier was silently off:
the same prediction graded 1.0 on the main thread and 0.0 in a sweep. Found by the oracle run, where
a recorded answer graded 0.0 for exactly that reason.
"""

from __future__ import annotations

import threading

import pytest

pytest.importorskip("math_verify")
from smol_ladder.grade import grade  # noqa: E402

ROW = {"answer": "28x28x1", "reward_mode": "exact_short", "atol": 0.0, "rtol": 0.0}


def in_thread(fn):
    out = []
    t = threading.Thread(target=lambda: out.append(fn()))
    t.start()
    t.join()
    return out[0]


def test_a_math_verify_match_grades_the_same_on_a_worker_thread_as_on_the_main_thread():
    try:
        main = grade(ROW, "28x28")
    except Exception as e:  # noqa: BLE001 - the grader file is not cached and there is no network
        pytest.skip(f"dataset grader unavailable: {type(e).__name__}")
    assert main == 1.0, "tier 4 (math-verify) should match this on the main thread"
    assert in_thread(lambda: grade(ROW, "28x28")) == main


def test_a_non_match_is_still_a_non_match_in_a_thread():
    try:
        assert in_thread(lambda: grade(ROW, "banana")) == 0.0
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"dataset grader unavailable: {type(e).__name__}")
