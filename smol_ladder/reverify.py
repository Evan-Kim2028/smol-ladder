"""Re-verify trials whose offline grading pass failed, offline, without touching the model.

Why this exists. The v2 run lost 90 trials to a harness failure rather than a model failure: the
sealed offline pass timed out or crashed while the machine was at load average 60-80 with three
sweeps and two test suites running. `summarize` does the right thing with those -- they leave the
denominator instead of reading as 0.0 -- but that is not the same as learning what they were. A
program the model did write is sitting in `solution.py`, and the only thing missing is a grading
run that had the machine to itself. That run needs no model, so it can be repeated after the fact.

What it may and may not touch. It never writes `result.json`. The first pass is the measurement
that was made, and overwriting it would erase the evidence that the trial failed under load --
which is the finding, not a nuisance to be tidied away. The outcome goes in a NEW `reverify.json`
beside the trial: the status it replaced, the status now, the prediction and reward, the deadline
it ran under, and how many attempts it took. `summarize` prefers a re-verification that
succeeded, so the recovery shows up in the numbers instead of having to be applied by hand.

Scope. Only trials whose own offline pass failed AND that have a `solution.py` to re-run. An agent
timeout is excluded on purpose: that is the model loop failing, and the fix for it is the model,
not a second pass over a program it may never have finished writing.

    uv run python -m smol_ladder.reverify --run-tag v2 --split test --dry-run
    uv run python -m smol_ladder.reverify --run-tag v2 --split test --workers 4 --timeout 900
"""

from __future__ import annotations

import argparse
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from smol_ladder.grade import grade
from smol_ladder.run_ladder import _drop_verify_copy, runs_dir, sealed_grading_pass, source_for
from smol_ladder.tasks import DATA

#: Four at a time, niced. The machine that timed out the first pass is usually still running the
#: sweep that is using it, so a re-verification that competes for the same cores reproduces the
#: failure it was run to repair. 180s was the deadline the first pass had; a re-run gets a
#: generous multiple of it, because nothing here is a model and a slow program is worth waiting
#: out rather than re-booking as a failure.
DEFAULT_WORKERS = 4
DEFAULT_TIMEOUT = 900

#: The file written beside a trial. Never result.json: see the module docstring.
REVERIFY = "reverify.json"


@dataclass
class Trial:
    """One trial worth re-running: where it is, its task row, its tables, and its rung.

    The rung comes off the trial's own result, not the task row: the row is the task, which was
    run at five rungs, so a report grouped by it would file every trial under "?".
    """
    path: Path
    row: dict
    inputs: Path | None = None
    rung: str = "?"


def _failed_verify(result: dict) -> bool:
    """Did this trial's own offline pass fail? A missing key means the pass ran and was clean."""
    return result.get("verify_status", "exit 0") != "exit 0"


def _agent_finished(result: dict) -> bool:
    return result.get("agent_status") == "exit 0"


def candidates(root: Path, rows: dict[str, dict] | None = None) -> list[Trial]:
    """Every trial under `root` whose offline pass failed and which has a program to re-run.

    A trial with no `solution.py` is not a candidate: there is nothing to grade, and recording an
    attempt would book a harness failure as recovered without having graded anything.
    """
    out: list[Trial] = []
    for result_path in sorted(root.glob("*/*/**/result.json")):
        result = json.loads(result_path.read_text())
        if not (_failed_verify(result) and _agent_finished(result)):
            continue
        trial = result_path.parent
        if not (trial / "solution.py").exists():
            continue
        out.append(Trial(path=trial, row=(rows or {}).get(result.get("task_id"), {}),
                         rung=result.get("rung", "?")))
    return out


def _prepare(trial: Trial, inputs: Path | None) -> tuple[Path, str]:
    """Rebuild the trial's `verify/` directory the way once() left it: the program, a copy of the
    tables, and the sibling links that make a bare-filename read work.

    Rebuilt from the shared cache because the original copy is dropped after each pass (it is ~30
    MB of somebody else's tables per trial), and a re-verification is exactly the case that
    "reconstructible from the cache" was chosen for. Returns the verify dir and a status.
    """
    verify = trial.path / "verify"
    verify.mkdir(exist_ok=True)
    shutil.copyfile(trial.path / "solution.py", verify / "solution.py")
    if inputs is None or not inputs.is_dir():
        return verify, "no inputs"
    tables = verify / "input"
    if not tables.exists():
        try:
            shutil.copytree(inputs, tables, symlinks=True)
        except Exception:
            if tables.is_symlink() or tables.exists():
                if tables.is_dir() and not tables.is_symlink():
                    shutil.rmtree(tables, ignore_errors=True)
                else:
                    tables.unlink()
            tables.symlink_to(inputs.resolve())
    # The same relative links once() lays down, so a program that reads a bare `a.csv` resolves
    # it the same way here. An absolute link would dangle inside the jail.
    if tables.is_dir():
        for item in sorted(tables.iterdir()):
            if item.name.startswith("."):
                continue
            link = verify / item.name
            if link.exists() or link.is_symlink():
                continue
            try:
                link.symlink_to(Path("input") / item.name)
            except OSError:
                pass
    return verify, "exit 0"


def _previous_attempts(trial: Trial) -> int:
    """How many re-verifications this trial already has, so a repeat says it is a repeat."""
    path = trial.path / REVERIFY
    if not path.exists():
        return 0
    try:
        return int(json.loads(path.read_text()).get("attempts", 0))
    except (json.JSONDecodeError, OSError, TypeError, ValueError):
        return 0


def verify_trial(trial: Trial, timeout: int = DEFAULT_TIMEOUT) -> dict:
    """Re-run the sealed pass for one trial and write `reverify.json`. Never raises.

    Returns the record it wrote, so a caller can report on it without re-reading the file.
    """
    result = json.loads((trial.path / "result.json").read_text())
    record = {
        "task_id": result.get("task_id"),
        "rung": result.get("rung"),
        "sample": result.get("sample"),
        "old_verify_status": result.get("verify_status"),
        "old_prediction": result.get("prediction", ""),
        "old_reward": result.get("reward", 0.0),
        "timeout_seconds": timeout,
        "attempts": _previous_attempts(trial) + 1,
    }
    verify, status = _prepare(trial, trial.inputs)
    if status != "exit 0":
        record.update(new_verify_status=status, prediction="", reward=0.0, recovered=False)
    else:
        new_status, out = sealed_grading_pass(verify, timeout=timeout)
        lines = [line.strip() for line in out.splitlines() if line.strip()]
        prediction = lines[-1] if lines else ""
        reward = grade(trial.row, prediction) if new_status == "exit 0" else 0.0
        record.update(new_verify_status=new_status, prediction=prediction, reward=reward,
                      recovered=new_status == "exit 0")
        # The copy is dropped again, for the same reason once() drops it.
        _drop_verify_copy(verify)
    (trial.path / REVERIFY).write_text(json.dumps(record, indent=1))
    return record


def run(split: str, tag: str | None, workers: int = DEFAULT_WORKERS,
        timeout: int = DEFAULT_TIMEOUT, dry_run: bool = False) -> dict:
    """Re-verify every eligible trial under the run's tree, and summarise the outcome by rung."""
    rows, inputs_of = source_for(split)
    by_id = {r["task_id"]: r for r in rows}
    root = runs_dir(split, tag, data=DATA)
    todo = candidates(root, by_id)
    for trial in todo:
        if trial.row:
            trial.inputs = inputs_of(trial.row)

    by_rung: dict[str, dict] = {}
    def count(rung: str, key: str) -> None:
        by_rung.setdefault(rung, {"candidates": 0, "recovered": 0, "still_failing": 0})
        by_rung[rung][key] += 1

    for trial in todo:
        count(trial.rung, "candidates")
    if dry_run:
        return {"dry_run": True, "candidates": len(todo), "rungs": by_rung}

    def one(trial: Trial) -> dict:
        try:
            return verify_trial(trial, timeout=timeout)
        except Exception as e:  # noqa: BLE001 - one bad trial must not kill the pass
            return {"task_id": trial.row.get("task_id"), "rung": trial.rung,
                    "new_verify_status": f"error: {type(e).__name__}", "recovered": False}

    records: list[dict] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for record in pool.map(one, todo):
            records.append(record)
    for record in records:
        rung = record.get("rung") or "?"
        count(rung, "recovered" if record.get("recovered") else "still_failing")
    return {"candidates": len(todo), "recovered": sum(r.get("recovered", False) for r in records),
            "still_failing": sum(not r.get("recovered", False) for r in records),
            "rungs": by_rung}


def _print(got: dict) -> None:
    if got.get("dry_run"):
        print(f"{got['candidates']} trials would be re-verified (dry run, nothing written)")
    else:
        print(f"re-verified {got['candidates']} trials: {got['recovered']} recovered, "
              f"{got['still_failing']} still failing")
    print()
    print(f"  {'rung':<10} {'candidates':>11} {'recovered':>10} {'still failing':>14}")
    for rung in sorted(got["rungs"]):
        block = got["rungs"][rung]
        print(f"  {rung:<10} {block['candidates']:>11} {block['recovered']:>10} "
              f"{block['still_failing']:>14}")
    if got.get("dry_run"):
        print("\ndry run: nothing was written")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="test")
    ap.add_argument("--run-tag", default=None,
                    help="one run's own tree, data/runs/<tag>/<split>. Omit for the legacy tree.")
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                    help="re-verifications at a time (default 4)")
    ap.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT,
                    help="seconds per sealed pass (default 900)")
    ap.add_argument("--dry-run", action="store_true", help="list what would be re-verified")
    args = ap.parse_args()
    _print(run(args.split, args.run_tag, args.workers, args.timeout, args.dry_run))


if __name__ == "__main__":
    main()
