"""Summarise ladder runs: mean pass probability per rung, the buckets it implies, and monotonicity.

Four independent L1 runs on the same 244 test tasks disagreed on 18.9% of them, so a single
result per (task, rung) cannot say which rung a task first passes at, and stopping at the first
pass makes the per-rung numbers a partition with a composition-dependent denominator. With K
samples per (task, rung) the report is built on the per-task pass fraction instead, which is
monotone in the underlying pass probability and well defined at any K.

Five tables, and they are kept apart on purpose.

1. `rungs` — per rung: the mean of the per-task pass fractions over the tasks scored at that
   rung, with a bootstrap 95% CI, plus the raw counts. The mean is over tasks, never over trials,
   or a task that happened to get 4 samples would outweigh one that got 1.
2. `monotonicity` — for each adjacent pair, the tasks that pass at k and fail at k+1. A drop is
   the finding the ladder exists to detect, so it gets an exact paired sign test rather than
   being averaged into the curve.
3. `first_passing_rung` — the old bucket table, still a partition, now computed from the MAJORITY
   pass of a task's samples. `marginality` says how close each task was to the line, because a
   bucket whose membership is decided by a coin flip is exactly the bucket A7 warns about.
4. `control` — the L1+schema trials, on their own: it adds no information, so it is not a rung.
5. `harness_failures` — trials where the harness did not get a clean run, counted and kept out
   of every pass rate. A crash is not a model failure.

Results whose prompt hashes differ inside one rung are refused rather than pooled, because they
are measurements of different ladders. `--allow-mixed` overrides, and records what it pooled.

    uv run python -m smol_ladder.summarize --split test
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from smol_ladder.ladder import ladder_fingerprint, read_source
from smol_ladder.run_ladder import source_for
from smol_ladder.tasks import DATA

RUNGS = ["L1", "L2", "L3", "L4"]
CLIMBABLE = RUNGS[1:]
CONTROL = "L1+schema"
ALL = RUNGS + [CONTROL]
# The control's directory has no "+" in it; everything else is the rung name verbatim.
DIRS = {rung: rung.replace("+", "_") for rung in ALL}

# Every bucket `first_passing_rung` can hold, so the key set is fixed and a reader never has to
# guess whether a zero bucket is missing or empty.
BUCKETS = RUNGS + ["never", "not climbable (no reference)", "not scored (every trial was a "
                   "harness failure)", "not attempted"]

# Adjacent pairs the monotonicity test runs on. The control is excluded: it is not on the
# information axis, so "monotonic in rung" says nothing about it.
PAIRS = [(RUNGS[i], RUNGS[i + 1]) for i in range(len(RUNGS) - 1)]

BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20260101


class MixedPrompts(ValueError):
    """Results in one rung that were produced by different ladders."""


def rung_name(directory: str) -> str:
    """`L1_schema` -> `L1+schema`; anything else is the rung name verbatim."""
    for rung, name in DIRS.items():
        if name == directory:
            return rung
    return directory


def collect(split: str, tag: str | None = None) -> dict[str, dict[str, list[dict]]]:
    """task_id -> rung -> [result, ...], one entry per sample, sample 0 first.

    The unsuffixed <task>/<rung>/result.json is sample 0 and <task>/<rung>/s<k>/result.json is
    sample k, so a tree written before --samples existed reads as k=1 with no migration.

    `tag` reads one run's own tree (data/runs/<tag>/<split>) instead of the legacy shared one, so a
    summary can never average two ladder versions even if --allow-mixed was forgotten.
    """
    from smol_ladder.run_ladder import runs_dir

    root = runs_dir(split, tag, data=DATA)
    out: dict[str, dict[str, list[dict]]] = {}
    for path in root.glob("*/*/**/result.json"):
        parts = path.parent.relative_to(root).parts
        task, directory = parts[0], parts[1]
        result = json.loads(path.read_text())
        # The recorded index wins over the directory name; the name is only the fallback for
        # results written before there was an index to record.
        match = re.fullmatch(r"s(\d+)", parts[2]) if len(parts) > 2 else None
        index = result.get("sample")
        if not isinstance(index, int):
            index = int(match.group(1)) if match else 0
        result["_index"] = index
        # A re-verification lives beside the trial, never inside result.json, so it is read here
        # and attached for the counting below to prefer. A missing or unreadable file leaves the
        # trial exactly as recorded.
        reverified = path.parent / "reverify.json"
        if reverified.exists():
            try:
                result["reverify"] = json.loads(reverified.read_text())
            except (json.JSONDecodeError, OSError):
                pass
        out.setdefault(task, {}).setdefault(rung_name(directory), []).append(result)
    for rungs in out.values():
        for trials in rungs.values():
            trials.sort(key=lambda r: r["_index"])
    return out


def summary_path(split: str, tag: str | None = None) -> Path:
    """Where the report is written: beside the run it describes, inside the tag's own directory."""
    return DATA / "runs" / tag / f"summary_{split}.json" if tag \
        else DATA / "runs" / f"summary_{split}.json"


def read_run_record(tag: str | None) -> dict:
    """The RUN.json a tagged run wrote, or an empty record. Never raises on a missing or bad file."""
    if not tag:
        return {}
    path = DATA / "runs" / tag / "RUN.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def has_reference_at_launch(record: dict, live: Callable[[str], bool]) -> Callable[[str], bool]:
    """Did this task have a verified reference when the run started? Prefers the launch record.

    The `not climbable (no reference)` bucket is only true of the population the run was actually
    given. A task whose L2 was skipped because it had no reference then would, once a concurrent
    retry sweep lands one, be re-read as climbable and booked `not attempted` instead -- so the
    ladder's L2 pass rate would be read against a denominator that includes tasks the reference gate
    excluded. RUN.json fixes the set at launch; `live` is only the fallback for a run with no record.
    """
    ids = record.get("reference_task_ids_at_launch")
    if not isinstance(ids, list):
        return live
    allowed = set(ids)
    return lambda task: task in allowed


def _reverified(result: dict | None) -> dict | None:
    """The re-verification of a trial whose first offline pass failed, when one succeeded.

    `reverify.json` is written beside the trial, never into it, so `result.json` still says the
    first pass failed. Reading it here is what lets the recovered trial back into the denominator
    without rewriting the run: a program the model wrote and the first pass merely failed to
    grade is a real observation, and dropping it forever is how 90 trials went missing from v2.

    A re-verification that itself failed is ignored, so the trial stays a harness failure: a
    second timeout is an answer about how slow the program is, not a recovery.
    """
    if not result:
        return None
    record = result.get("reverify")
    if not isinstance(record, dict) or record.get("new_verify_status") != "exit 0":
        return None
    return record


def _effective(result: dict | None) -> dict | None:
    """The trial's verdict: the re-verification if one succeeded, else the trial as recorded."""
    reverified = _reverified(result)
    if reverified is None:
        return result
    merged = dict(result)
    merged["reward"] = reverified.get("reward", 0.0)
    merged["prediction"] = reverified.get("prediction", "")
    return merged


def _passed(result: dict | None) -> bool:
    effective = _effective(result)
    return bool(effective) and effective.get("reward", 0.0) >= 1.0


def _finished(result: dict | None) -> bool:
    """Did the harness get a clean run out of the trial? Anything else is not a model failure.

    A trial counts as finished only if the agent exited cleanly AND the offline grading pass
    produced a usable result. A `verify_status` other than "exit 0" means the solution could not be
    re-run under the sealed jail -- it timed out, or it crashed -- so there was no prediction to
    grade. The agent's own exit code says the model loop finished; it says nothing about whether the
    answer exists, and scoring the empty output of an unrunnable program as 0.0 books the harness's
    deadline in the model's pass rate.

    A later successful re-verification makes the trial finished after all: the pass is known to be
    able to grade this program, and it did. An agent timeout is never rescued this way -- there the
    model loop itself failed, and no amount of re-running the same program changes that.
    """
    if not result or result.get("agent_status") != "exit 0":
        return False
    if result.get("verify_status", "exit 0") == "exit 0":
        return True
    return _reverified(result) is not None


def _scored(trials: list[dict]) -> list[dict]:
    """Only trials the harness ran cleanly. A timeout is not evidence the model cannot solve it."""
    return [t for t in trials if _finished(t)]


def pass_fraction(trials: list[dict]) -> float | None:
    """This task's pass rate at this rung, over its scored samples. None if none were scored."""
    scored = _scored(trials)
    if not scored:
        return None
    return sum(_passed(t) for t in scored) / len(scored)


def check_prompts(runs: dict[str, dict[str, list[dict]]], allow_mixed: bool = False,
                  record: dict | None = None) -> dict:
    """Refuse to average two ladder versions into one number. Returns what was mixed, if allowed.

    A rung's results are pooled only when every trial of the SAME (task, rung) cell agrees on
    everything that defines the ladder it measured.

    Why the comparison is per task and not per rung. A rung prompt embeds the task's own question,
    the file list and -- above L1 -- that task's hint, so every task's `prompt_sha256` differs from
    every other task's by construction. Collecting a rung's hashes and calling a set of size > 1
    "mixed" therefore reports 250 hashes for 250 clean tasks: it read the design as the defect, made
    a clean sweep un-summarisable, and taught the reader to reach for --allow-mixed, which is the
    override that would hide a real one. The comparison belongs where "the same thing twice" means
    something -- inside one (task, rung) cell, across its samples and across a resume.

    Three axes are checked, because each catches a mixing the others cannot:

    1. `prompt_sha256` -- the prompt text itself. Two hashes in one cell mean a reword, a widened
       schema dump or a resume under new text.
    2. `ladder_sha256` -- a fingerprint of the code that builds and grades the ladder. The prompt
       can be byte-identical while the grader, the jail or the agent loop changed underneath it,
       and a per-result prompt hash cannot see that. It is a fingerprint rather than the commit
       because a resume across a commit that touched none of these files is still one ladder.
    3. `hint_prompt_version` -- the gen_hints version behind L2-L4. A rung text rebuilt from a hint
       of another version is a different ladder while its own hash merely looks per-task unique.

    A launch's fingerprint is also checked against the fingerprints the results carry, which
    catches a tree whose RUN.json claims a version that produced none of the trials in it.

    A result with no recorded value on an axis is not treated as matching a recorded one. Its
    provenance is unknown, not known-equal, and pooling it would silently average two ladders --
    the exact failure this check exists to prevent.
    """
    mixed: dict[str, list[str]] = {}
    detail: dict[str, list[str]] = {}

    def note(rung: str, values: list[str], line: str) -> None:
        bucket = mixed.setdefault(rung, [])
        lines = detail.setdefault(rung, [])
        lines.append(line)
        for value in values:
            if value not in bucket:
                bucket.append(value)

    for task, rungs in runs.items():
        for rung, trials in rungs.items():
            for field, label in (("prompt_sha256", "prompt"), ("ladder_sha256", "ladder version"),
                                 ("hint_prompt_version", "hint prompt version")):
                values = sorted({str(t[field]) if t.get(field) else "<unrecorded>"
                                 for t in trials})
                if len(values) > 1:
                    note(rung, values, f"task {task}: {len(values)} different {label}s "
                                       f"({', '.join(v[:12] for v in values)})")
        # The hint version is task-wide, not cell-wide: L2, L3 and L4 of one task all read the
        # same cached hint, so a task whose L2 came from one hint version and whose L3 came from
        # another is a ladder whose rungs are not cumulative -- and the ordering claim the
        # "lowest rung that passes" rests on is void for exactly that task.
        versions = sorted({str(t["hint_prompt_version"])
                           for rung in CLIMBABLE
                           for t in rungs.get(rung, []) if t.get("hint_prompt_version")})
        if len(versions) > 1:
            note("/".join(rung for rung in CLIMBABLE if rung in rungs), versions,
                 f"task {task}: its rungs were built from {len(versions)} different hint prompt "
                 f"versions ({', '.join(v[:12] for v in versions)}), so they are not cumulative")

    for rung, fingerprint in _stray_ladder_versions(runs, record):
        note(rung, [fingerprint], f"RUN.json launch ladder version {fingerprint[:12]} produced "
                                  f"none of the results on disk")

    if mixed and not allow_mixed:
        raise MixedPrompts(
            "these results are measurements of more than one ladder and cannot be pooled: "
            + "; ".join(f"rung {rung}: " + " | ".join(lines) for rung, lines in detail.items())
            + ". Re-run the affected cells on one version, or pass --allow-mixed to pool them "
              "anyway and record it in the report.")
    return mixed


def _stray_ladder_versions(runs: dict[str, dict[str, list[dict]]],
                           record: dict | None) -> list[tuple[str, str]]:
    """Launch fingerprints in RUN.json that no result in the tree carries, as (rung, fingerprint).

    A resumed run legitimately spans code, so a fingerprint the results share is not a finding:
    the per-cell check above is what says whether those two versions measured the same thing.
    What cannot be legitimate is a recorded launch whose version produced no trial at all -- a
    tree assembled from somewhere else, or one whose record describes a sweep that never ran.
    """
    if not record:
        return []
    results = {str(t["ladder_sha256"]) for rungs in runs.values()
               for trials in rungs.values() for t in trials if t.get("ladder_sha256")}
    launches = record.get("launches") or []
    if isinstance(launches, dict):
        launches = [launches]
    return [("run", str(launch["ladder_sha256"]))
            for launch in launches
            if isinstance(launch, dict) and launch.get("ladder_sha256")
            and str(launch["ladder_sha256"]) not in results]


def _bootstrap_ci(values: list[float], draws: int = BOOTSTRAP_DRAWS,
                  seed: int = BOOTSTRAP_SEED) -> list[float]:
    """Percentile bootstrap over tasks. Seeded, so two runs of the summary agree exactly.

    Resampling tasks, not trials: the between-task spread is the dominant term (the audit put
    the task-to-task sd at 0.394 against a rerun sd of similar size), and resampling trials
    would ignore it and report an interval far too narrow.
    """
    if not values:
        return [float("nan"), float("nan")]
    if len(values) == 1:
        # Every resample is the same task, so the bootstrap distribution is a point mass. The
        # honest interval is the sampling error of one task, which bootstrap cannot estimate.
        return [values[0], values[0]]
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(values), size=(draws, len(values)))
    means = np.asarray(values, dtype=float)[samples].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def rung_stats(runs: dict[str, dict[str, list[dict]]],
               rungs: Sequence[str] | None = None) -> dict[str, dict]:
    """Per rung: mean pass probability over tasks, its bootstrap CI, and the raw counts.

    `runs` is the task set, so passing a subset restricts every rung to it. That is how the
    common-set table is built: the same measurement, over one population, rather than L1 read on
    250 tasks and L2 on the 213 that have a reference.
    """
    out: dict[str, dict] = {}
    for rung in (ALL if rungs is None else rungs):
        fractions: list[float] = []
        trials = scored = harness = 0
        attempted = 0
        for rungs in runs.values():
            block = rungs.get(rung)
            if block is None:
                continue
            attempted += 1
            trials += len(block)
            harness += sum(not _finished(t) for t in block)
            scored += len(_scored(block))
            fraction = pass_fraction(block)
            if fraction is not None:
                fractions.append(fraction)
        mean = float(np.mean(fractions)) if fractions else float("nan")
        out[rung] = {
            # `tasks` is how many tasks were attempted here, and `scored_tasks` how many
            # contributed a pass fraction. They differ when a task's every trial crashed: it was
            # attempted, it contributes nothing, and dropping it from the denominator is what
            # keeps a harness outage from reading as a pass-rate drop.
            "tasks": attempted,
            "scored_tasks": len(fractions),
            "trials": trials,
            "trials_scored": scored,
            "harness_failures": harness,
            "mean_pass_probability": mean,
            "ci95": _bootstrap_ci(fractions),
        }
    return out


def pass_fractions(runs: dict[str, dict[str, list[dict]]]) -> dict[str, dict[str, float]]:
    """rung -> task -> that task's pass fraction. The raw material behind the mean."""
    out: dict[str, dict[str, float]] = {}
    for rung in ALL:
        per_task = {}
        for task, rungs in runs.items():
            fraction = pass_fraction(rungs[rung]) if rung in rungs else None
            if fraction is not None:
                per_task[task] = fraction
        out[rung] = per_task
    return out


def _sign_test(down: int, up: int) -> float | None:
    """Two-sided exact sign test on the discordant pairs. None when there are none.

    Zero discordant pairs is perfect agreement, which is not evidence about ordering, and the
    test is undefined there: reporting p=1.0 would read as a measurement.
    """
    n = down + up
    if n == 0:
        return None
    from scipy import stats
    return float(stats.binomtest(min(down, up), n, 0.5, alternative="two-sided").pvalue)


def monotonicity(runs: dict[str, dict[str, list[dict]]]) -> dict[str, dict]:
    """Per adjacent rung pair, the tasks whose pass probability went DOWN, and an exact test.

    Paired on tasks, so a task that was measured at one rung and not the other is excluded and
    counted in `excluded_unpaired` rather than quietly inflating the denominator. That case is
    the normal one under climbing, which is why the exclusion is reported.
    """
    out: dict[str, dict] = {}
    for lower, upper in PAIRS:
        violations = discordant = up = paired = 0
        for rungs in runs.values():
            low = pass_fraction(rungs[lower]) if lower in rungs else None
            high = pass_fraction(rungs[upper]) if upper in rungs else None
            if low is None or high is None:
                continue
            paired += 1
            if high < low:
                violations += 1
                discordant += 1
            elif high > low:
                up += 1
        out[f"{lower}->{upper}"] = {
            "paired": paired,
            "violations": violations,
            "improved": up,
            "discordant": discordant,
            "excluded_unpaired": len(runs) - paired,
            "p_value": _sign_test(violations, up),
        }
    return out


def first_passing_rung(rungs: dict[str, list[dict]], has_reference: bool) -> str:
    """The one bucket this task belongs to, out of BUCKETS, decided by the MAJORITY of its
    samples. A tie is not a pass: at k=1 every task ties, and letting ties fall upward is the
    coin-flip bucket that four reruns disagreed on 18.9% of the time.

    The control cannot appear here. A task that the control rescued is booked by the rung that
    passed it, if any, and by `never` if no rung did -- the honest reading, because the control
    result says nothing about the ladder rungs.
    """
    if "L1" not in rungs:
        return "not attempted"
    fraction = pass_fraction(rungs["L1"])
    if fraction is None:
        # Every L1 trial crashed. The task was attempted and nothing was learned, which is not
        # the same as never having been run: booking it as "not attempted" would understate the
        # sweep's coverage, and booking it "never" would blame the model for a harness failure.
        return "not scored (every trial was a harness failure)"
    if fraction > 0.5:
        return "L1"
    if not has_reference:
        # Nothing above L1 was ever built for this task, so it cannot have passed one. It is
        # not a ladder failure; it is outside the ladder.
        return "not climbable (no reference)"
    for rung in CLIMBABLE:
        fraction = pass_fraction(rungs[rung]) if rung in rungs else None
        if fraction is not None and fraction > 0.5:
            return rung
    return "never"


def partition(runs: dict[str, dict[str, list[dict]]],
              has_reference: Callable[[str], bool]) -> dict[str, int]:
    """task -> exactly one bucket. Sums to len(runs) by construction, one task per iteration."""
    hist = dict.fromkeys(BUCKETS, 0)
    for task, rungs in runs.items():
        hist[first_passing_rung(rungs, has_reference(task))] += 1
    return hist


def marginality(runs: dict[str, dict[str, list[dict]]]) -> dict[str, float]:
    """How far each task's L1 pass fraction is from the majority line at 0.5.

    A bucket decided at 0.25 is a different kind of claim from one decided at 1.0, and the
    bucket counts alone cannot tell a reader which tasks those are. This is the number A7's
    18.9% was about, measured on the current run rather than reconstructed from logs.

    Reported as the distance from 0.5, so 0.5 is the *least* decisive task: exactly the coin
    flip that four reruns disagreed on. A task at 1.0 scores 0.5, the most decisive.
    """
    out = {}
    for task, rungs in runs.items():
        fraction = pass_fraction(rungs["L1"]) if "L1" in rungs else None
        if fraction is not None:
            out[task] = abs(fraction - 0.5)
    return out


def control_block(runs: dict[str, dict[str, list[dict]]],
                  has_reference: Callable[[str], bool]) -> dict:
    """The L1+schema trials, on the tasks they were actually run on, split by reference status.

    Split by reference because the control is gated on nothing: it runs on every L1 failure, so
    most of its rescues are on tasks the ladder never climbs, and a single blended rate hides
    that. Reported on the tasks' own sample counts, so a control sampled 4 times is not counted
    as 4 separate tasks.
    """
    block: dict = {"attempted": 0, "rescued": 0,
                   "with reference": {"attempted": 0, "rescued": 0},
                   "without reference": {"attempted": 0, "rescued": 0}}
    harness = trials = 0
    for task, rungs in runs.items():
        results = rungs.get(CONTROL)
        if not results:
            continue
        group = "with reference" if has_reference(task) else "without reference"
        block["attempted"] += 1
        block[group]["attempted"] += 1
        trials += len(results)
        harness += sum(not _finished(r) for r in results)
        # A task whose every control trial crashed still counts as attempted: it was run, and
        # dropping it would make a harness outage look like the control was never tried.
        fraction = pass_fraction(results)
        if fraction is not None and fraction > 0.5:
            block["rescued"] += 1
            block[group]["rescued"] += 1
    block["trials"] = trials
    block["harness_failures"] = harness
    return block


def reverified_block(runs: dict[str, dict[str, list[dict]]]) -> dict:
    """How many trials a re-verification recovered, and how many it did not.

    Reported because the pass rates below silently include the recovered trials. Without this
    number a reader cannot tell a rung measured on more trials from one measured on fewer, and
    the whole point of re-verifying was to make that difference visible.
    """
    recovered = still = attempts = 0
    for rungs in runs.values():
        for trials in rungs.values():
            for trial in trials:
                record = trial.get("reverify")
                if not isinstance(record, dict):
                    continue
                attempts += 1
                if _reverified(trial) is not None:
                    recovered += 1
                else:
                    still += 1
    return {"recovered": recovered, "still_failing": still, "attempts": attempts}


# --- the analyses a ladder curve cannot be read off a pass-rate table ----------------------------

def common_set(runs: dict[str, dict[str, list[dict]]], keep: Callable[[str], bool],
               inside: bool = True) -> dict[str, dict]:
    """The per-rung table restricted to one task set, so every rung shares a denominator.

    L2-L4 are gated on a verified reference, so they run on 213 of the 250 test tasks while L1
    runs on all 250. Reading a curve off each rung's own denominator compares two populations, and
    the difference between them is the reference gate rather than the ladder. The referenced set
    is the one population every rung was measured on, so it is the only one a curve can be drawn
    on; `inside=False` reports the 37 tasks outside it, which is the only place the control and
    L1 have anything to say.
    """
    subset = {task: rungs for task, rungs in runs.items() if keep(task) == inside}
    return rung_stats(subset)


def control_effect(runs: dict[str, dict[str, list[dict]]]) -> dict:
    """L1+schema minus L1, paired per task, with a sign test and a bootstrap CI.

    The control adds no information -- only a schema dump, which is cheaper reading, not a hint.
    So a gain here is not the ladder working: it is the model getting better at reading a table it
    could already read, and the number is what tells us how much of any L2 gain is really just
    that. Paired because both arms ran on the same tasks, and the question is per task.

    A task whose control trials all crashed is dropped from the pair rather than counted as a
    regression: the harness's failure is not an effect of the schema dump.
    """
    deltas: list[float] = []
    paired = rescued = hurt = unpaired = 0
    for rungs in runs.values():
        base = pass_fraction(rungs["L1"]) if "L1" in rungs else None
        ctrl = pass_fraction(rungs[CONTROL]) if CONTROL in rungs else None
        if base is None or ctrl is None:
            unpaired += 1
            continue
        paired += 1
        delta = ctrl - base
        deltas.append(delta)
        if delta > 0:
            rescued += 1
        elif delta < 0:
            hurt += 1
    mean = float(np.mean(deltas)) if deltas else float("nan")
    return {
        "paired": paired,
        "rescued": rescued,
        "hurt": hurt,
        "discordant": rescued + hurt,
        "excluded_unpaired": unpaired,
        "mean_delta": mean,
        "ci95": _bootstrap_ci(deltas),
        "p_value": _sign_test(hurt, rescued),
    }


def consistency(runs: dict[str, dict[str, list[dict]]]) -> dict[str, dict]:
    """Per rung, the share of tasks whose samples of the same cell agree on pass or fail.

    The noise floor. At k=2 two samples either agree or they do not, and a rung whose samples
    disagree often is a rung whose pass probability carries most of its width in sampling error
    rather than in the difference between rungs. A task needs two *scored* samples to say
    anything: one sample cannot agree with itself, and a crashed partner is a harness failure
    rather than a disagreeing verdict.
    """
    out: dict[str, dict] = {}
    for rung in ALL:
        agree = disagree = 0
        for rungs in runs.values():
            block = _scored(rungs[rung]) if rung in rungs else []
            if len(block) < 2:
                continue
            verdicts = [_passed(t) for t in block]
            if all(v == verdicts[0] for v in verdicts):
                agree += 1
            else:
                disagree += 1
        total = agree + disagree
        out[rung] = {
            "tasks": total, "agree": agree, "disagree": disagree,
            "agreement": agree / total if total else float("nan"),
        }
    return out


def hint_source_split(runs: dict[str, dict[str, list[dict]]]) -> dict[str, dict]:
    """L2-L4 pass probability split by which hand wrote the rung's text.

    A rung's prose is either a validated model hint or, where none was cached, what the AST
    extractor could say. Those are different instruments: one names the computation in the task's
    own terms, the other lists operations. A curve that blends them averages over the instrument
    as well as the rung, so if the AST rungs are weaker the blended number says neither. Results
    that never recorded a hand -- gen_refs drives once() without the split -- are booked under
    `none` rather than dropped, so the counts still sum to the rung's task count.
    """
    out: dict[str, dict] = {}
    for rung in CLIMBABLE:
        groups: dict[str, dict[str, dict[str, list[dict]]]] = {}
        trials = harness = 0
        for task, rungs in runs.items():
            block = rungs.get(rung)
            if not block:
                continue
            source = next((t.get("hint_source") for t in block if t.get("hint_source")), "none")
            groups.setdefault(source, {})[task] = rungs
            trials += len(block)
            harness += sum(not _finished(t) for t in block)
        out[rung] = {"hands": {source: rung_stats(tasks, rungs=[rung])[rung]
                               for source, tasks in sorted(groups.items())},
                     "trials": trials, "harness_failures": harness}
    return out


def ceiling(runs: dict[str, dict[str, list[dict]]]) -> dict:
    """How many referenced tasks the model already passes at L1 in BOTH samples.

    A task it solves from the question alone has no headroom: L2 cannot improve on a pass, so it
    contributes a flat 1.0 to every rung and hides whatever the rungs do on the tasks that
    actually need them. The "failed L1 in at least one sample" count is the population where a
    hint can show an effect at all, and its size is the first thing to check before reading a
    curve as evidence about hints.

    A task with a single scored L1 sample is `unmeasured`: it has not been shown to pass reliably,
    so calling it ceiling would overstate the headroom left.
    """
    both = at_least_one = unmeasured = 0
    for rungs in runs.values():
        scored = _scored(rungs["L1"]) if "L1" in rungs else []
        if not scored:
            continue
        verdicts = [_passed(t) for t in scored]
        if len(verdicts) < 2:
            unmeasured += 1
        elif all(verdicts):
            both += 1
        else:
            at_least_one += 1
    headroom = at_least_one / (both + at_least_one) if both + at_least_one else float("nan")
    return {"passed_l1_both": both, "failed_l1_at_least_once": at_least_one,
            "unmeasured": unmeasured, "headroom": headroom}


def _has_headroom(rungs: dict[str, list[dict]]) -> bool:
    """Did this task fail L1 in at least one scored sample? Over a task's whole rung dict."""
    scored = _scored(rungs["L1"]) if "L1" in rungs else []
    return len(scored) >= 2 and not all(_passed(t) for t in scored)


def headroom_curve(runs: dict[str, dict[str, list[dict]]]) -> dict[str, dict]:
    """The rung curve restricted to the tasks that failed L1 in at least one sample.

    This is the only version of the curve on which a hint can move a number. On the full set the
    tasks the model already solves contribute a constant 1.0 at every rung, so the curve is
    partly a measure of how many tasks were easy to begin with; here each task started from a
    failure, and any rise is the ladder working.
    """
    return rung_stats({task: rungs for task, rungs in runs.items() if _has_headroom(rungs)})


def summarise(split: str, runs: dict[str, dict[str, list[dict]]],
              has_reference: Callable[[str], bool], allow_mixed: bool = False,
              record: dict | None = None, exclude_ids: set[str] | None = None) -> dict:
    """The whole report, as a dict. `summarise` asserts nothing; this asserts the partition."""
    mixed = check_prompts(runs, allow_mixed, record)
    hist = partition(runs, has_reference)
    assert sum(hist.values()) == len(runs), (hist, len(runs))
    report = {
        "split": split,
        "tasks": len(runs),
        "mixed_prompts": mixed,
        "rungs": rung_stats(runs),
        "monotonicity": monotonicity(runs),
        "first_passing_rung": hist,
        "control": control_block(runs, has_reference),
        "marginality": marginality(runs),
        "reverified": reverified_block(runs),
        "common_set": common_set(runs, has_reference),
        "outside_common_set": common_set(runs, has_reference, inside=False),
        "control_effect": control_effect(runs),
        "consistency": consistency(runs),
        "hint_source": hint_source_split(runs),
        "ceiling": ceiling(runs),
        "headroom_curve": headroom_curve(runs),
    }
    if exclude_ids:
        # The same measurement over the tasks NOT in `exclude_ids` (for arm A: the test tasks whose
        # notebook is in its training set, ops/amd/overlap.py), so the held-out number is its own table.
        kept = {t: r for t, r in runs.items() if t not in exclude_ids}
        report["without_listed"] = {"excluded_tasks": len(runs) - len(kept), "tasks": len(kept),
                                    "rungs": rung_stats(kept)}
    return report


def _rows_for(split: str) -> tuple[list[dict], Callable[[str], bool]]:
    rows, _ = source_for(split)
    by_id = {r["task_id"]: r for r in rows}
    return rows, lambda task: task in by_id and read_source(by_id[task], split) is not None


def _default_exclusions(split: str) -> set[str]:
    """The test split's overlap-with-arm-A list, committed beside the AMD code."""
    path = Path(__file__).resolve().parent.parent / "ops" / "amd" / "overlap_with_arm_a.json"
    if split != "test" or not path.exists():
        return set()
    return set(json.loads(path.read_text())["ids"])


def _pct(value: float) -> str:
    return "   n/a" if value != value else f"{value:>6.1%}"


def _print(report: dict) -> None:
    print(f"split={report['split']}  tasks={report['tasks']}")
    if report["mixed_prompts"]:
        print(f"  WARNING pooled mixed prompts: "
              f"{ {k: [h[:12] for h in v] for k, v in report['mixed_prompts'].items()} }")

    print()
    print("mean pass probability per rung, over tasks scored there (bootstrap 95% CI)")
    for rung in ALL:
        block = report["rungs"][rung]
        if not block["tasks"]:
            print(f"  {rung:<10} not run")
            continue
        low, high = block["ci95"]
        print(f"  {rung:<10} {_pct(block['mean_pass_probability'])}"
              f"  [{_pct(low)}, {_pct(high)}]  "
              f"{block['trials_scored']}/{block['trials']} trials scored, "
              f"{block['tasks']} tasks, {block['harness_failures']} harness failures")

    block = report.get("without_listed")
    if block:
        print()
        print(f"the same, WITHOUT the {block['excluded_tasks']} tasks whose notebook is in arm A's "
              f"training set ({block['tasks']} tasks left; the held-out table)")
        for rung in ALL:
            row = block["rungs"][rung]
            if not row["tasks"]:
                continue
            low, high = row["ci95"]
            print(f"  {rung:<10} {_pct(row['mean_pass_probability'])}  [{_pct(low)}, {_pct(high)}]  "
                  f"{row['trials_scored']}/{row['trials']} trials scored, {row['tasks']} tasks")

    print()
    print("monotonicity: tasks whose pass probability fell as information was added")
    for pair in report["monotonicity"].values():
        p = "   n/a" if pair["p_value"] is None else f" p={pair['p_value']:.2g}"
        print(f"  {pair['paired']:>4} paired, {pair['violations']:>3} fell, "
              f"{pair['improved']:>3} rose, {pair['excluded_unpaired']:>3} unpaired{p}")

    print()
    print("first passing rung by majority pass (mutually exclusive, control excluded)")
    for key in BUCKETS:
        value = report["first_passing_rung"][key]
        if value:
            print(f"  {key:<30} {value:>4}")
    print(f"  {'sum':<30} {sum(report['first_passing_rung'].values()):>4}")

    marginal = report["marginality"]
    if marginal:
        # marginality is |pass fraction - 0.5|, so small means undecided, not close.
        undecided = sum(1 for v in marginal.values() if v < 0.25)
        print(f"  {undecided} of {len(marginal)} tasks pass on a fraction within 0.25 of the "
              f"majority line at L1, so their bucket is a coin flip")

    control = report["control"]
    print()
    print("L1+schema control (not a rung: adds no information, only cheaper reading)")
    for group in ("with reference", "without reference"):
        block = control[group]
        rate = block["rescued"] / block["attempted"] if block["attempted"] else float("nan")
        print(f"  {group:<20} {block['rescued']:>3}/{block['attempted']:<3} = {_pct(rate)}")
    rate = control["rescued"] / control["attempted"] if control["attempted"] else float("nan")
    print(f"  {'all':<20} {control['rescued']:>3}/{control['attempted']:<3} = {_pct(rate)}"
          f"   (harness failures {control['harness_failures']})")

    effect = report["control_effect"]
    low, high = effect["ci95"]
    p = "   n/a" if effect["p_value"] is None else f" p={effect['p_value']:.3g}"
    print(f"  paired effect     {effect['paired']:>3} paired: {effect['rescued']} rescued, "
          f"{effect['hurt']} hurt, delta {_pct(effect['mean_delta'])} "
          f"[{_pct(low)}, {_pct(high)}]{p}")

    reverified = report["reverified"]
    if reverified["attempts"]:
        print()
        print(f"re-verification: {reverified['recovered']} of {reverified['attempts']} trials "
              f"recovered by a later offline pass, {reverified['still_failing']} still failing")

    common = report["common_set"]
    outside = report["outside_common_set"]
    print()
    print("every rung on the SAME task set (the referenced tasks), so the curve is comparable")
    for rung in ALL:
        block = common[rung]
        if not block["scored_tasks"]:
            print(f"  {rung:<10} not run on this set")
            continue
        low, high = block["ci95"]
        print(f"  {rung:<10} {_pct(block['mean_pass_probability'])}"
              f"  [{_pct(low)}, {_pct(high)}]  "
              f"{block['trials_scored']}/{block['trials']} trials scored, "
              f"{block['scored_tasks']} tasks, {block['harness_failures']} harness failures")
    if any(b["scored_tasks"] for b in outside.values()):
        print()
        print("the tasks outside that set (no reference, so no rung above L1 was ever run)")
        for rung in ALL:
            block = outside[rung]
            if not block["scored_tasks"]:
                continue
            low, high = block["ci95"]
            print(f"  {rung:<10} {_pct(block['mean_pass_probability'])}"
                  f"  [{_pct(low)}, {_pct(high)}]  "
                  f"{block['trials_scored']}/{block['trials']} trials scored, "
                  f"{block['scored_tasks']} tasks, {block['harness_failures']} harness failures")

    print()
    print("rerun consistency: share of tasks whose two samples agree on pass or fail")
    for rung in ALL:
        block = report["consistency"][rung]
        if not block["tasks"]:
            print(f"  {rung:<10} not measured twice")
            continue
        print(f"  {rung:<10} {block['agree']:>3}/{block['tasks']:<3} = {_pct(block['agreement'])}"
              f"   ({block['disagree']} disagreed)")

    print()
    print("L2-L4 by which hand wrote the rung's text")
    for rung in CLIMBABLE:
        block = report["hint_source"][rung]
        parts = [f"{source} {hand['scored_tasks']} tasks "
                 f"{_pct(hand['mean_pass_probability'])}"
                 for source, hand in block["hands"].items()]
        if parts:
            print(f"  {rung:<10} " + "; ".join(parts)
                  + f"   ({block['harness_failures']} harness failures)")

    ceiling = report["ceiling"]
    headroom = report["headroom_curve"]
    print()
    print("ceiling: how many tasks the model already passes at L1, where a hint has no room")
    print(f"  passed L1 in both samples: {ceiling['passed_l1_both']}; "
          f"failed in at least one: {ceiling['failed_l1_at_least_once']}; "
          f"L1 measured once: {ceiling['unmeasured']}")
    print("  the curve restricted to the tasks that failed L1 in at least one sample:")
    for rung in ALL:
        block = headroom[rung]
        if not block["scored_tasks"]:
            continue
        low, high = block["ci95"]
        print(f"  {rung:<10} {_pct(block['mean_pass_probability'])}"
              f"  [{_pct(low)}, {_pct(high)}]  {block['scored_tasks']} tasks, "
              f"{block['harness_failures']} harness failures")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test",
                    choices=["test", "eval", "train", "jupyter-agent", "synthetic"])
    ap.add_argument("--run-tag", default=None,
                    help="summarise one run's own tree, data/runs/<tag>/<split>, and write the "
                         "report to data/runs/<tag>/summary_<split>.json. Omit it for the legacy "
                         "data/runs/<split>/ tree.")
    ap.add_argument("--out", type=Path, help="where to write the JSON report")
    ap.add_argument("--no-overlap-table", action="store_true",
                    help="skip the second table without the test tasks whose notebook is in arm A's "
                         "training set (ops/amd/overlap_with_arm_a.json; test split only)")
    ap.add_argument("--allow-mixed", action="store_true",
                    help="pool results in one rung that have different prompt hashes, and record "
                         "in the report that they were pooled")
    args = ap.parse_args()

    rows, has_reference = _rows_for(args.split)
    runs = collect(args.split, args.run_tag)
    record = read_run_record(args.run_tag)
    # Pinned to the launch when RUN.json has it, so a reference that landed mid-run cannot move the
    # `not climbable` denominator out from under the numbers computed above it.
    reference_at_launch = has_reference_at_launch(record, has_reference)
    n_ref = sum(reference_at_launch(row["task_id"]) for row in rows)
    total = len(rows)
    pinned = record.get("reference_tasks_at_launch")
    print(f"split={args.split}  tasks in the source={total}  "
          f"with verified reference={n_ref} ({n_ref/total:.0%})"
          f"{f'  [pinned at launch: {pinned}]' if pinned is not None else ''}"
          f"{f'  run tag={args.run_tag}' if args.run_tag else ''}")

    try:
        report = summarise(args.split, runs, reference_at_launch, args.allow_mixed, record,
                           None if args.no_overlap_table else _default_exclusions(args.split))
    except MixedPrompts as e:
        raise SystemExit(f"refusing to pool results from different prompts: {e}")
    report["source_tasks"] = total
    report["with_reference"] = n_ref
    report["reference_pinned_at_launch"] = pinned is not None
    report["run_tag"] = args.run_tag
    _print(report)

    dest = args.out or summary_path(args.split, args.run_tag)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, indent=1))
    print(f"\nwrote {dest}")
    missing = {r["task_id"] for r in rows} - set(runs)
    if missing:
        print(f"{len(missing)} source tasks have no trial on disk, "
              f"counted as 'not attempted', e.g. {sorted(missing)[:3]}")


if __name__ == "__main__":
    main()
