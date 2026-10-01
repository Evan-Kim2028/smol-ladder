"""--run-tag: two ladder versions get two directories, and each one is a run you can name.

Results from different ladder versions used to land in the same data/runs/<split>/ tree, where the
only thing keeping them apart was the prompt hash, and the hash check is a refusal at summary time --
after the second version has already been written on top of the first, and a resumed sweep will
happily reuse the first version's cached result.json as if it were its own. So the separation has to
be in the path, not in a check that runs after the damage.

`--run-tag TAG` puts trials under data/runs/<TAG>/<split>/ and the summary at
data/runs/<TAG>/summary_<split>.json. No tag is exactly the old tree, because existing results live
there and the whole point is that they are not moved.

RUN.json is the other half: one file per tagged run recording the code, the command, the model and
the shape of the sweep, written before the first trial so a run that dies an hour in still says what
it was, and closed out with the counts and the end time when it finishes.

    uv run --with pytest pytest -q tests/test_run_tag.py
"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from smol_ladder import run_ladder as runner
from smol_ladder import summarize as S
from smol_ladder import regrade as R
from smol_ladder.tasks import DATA


# --- the runs tree ---------------------------------------------------------------------------

def test_no_tag_reads_and_writes_the_legacy_tree():
    assert runner.runs_dir("test") == DATA / "runs" / "test"


def test_a_tag_puts_the_split_under_the_tag():
    assert runner.runs_dir("test", "v2") == DATA / "runs" / "v2" / "test"


def test_a_tag_and_no_tag_are_different_directories():
    """The property the flag exists for. Sharing a directory is what let two ladder versions pool."""
    assert runner.runs_dir("test", "v2") != runner.runs_dir("test")
    assert runner.runs_dir("eval", "v2") != runner.runs_dir("test", "v2")


def test_a_tag_may_not_escape_the_runs_directory():
    """A tag is a path segment, and a path segment that can climb out of data/runs can also be
    aimed at another run's tree, or at anything else on the box."""
    for bad in ["..", "v2/../../v2", "a/b", "/abs", "."]:
        with pytest.raises(ValueError):
            runner.runs_dir("test", bad)


def test_summary_path_follows_the_tag():
    assert S.summary_path("test", "v2") == DATA / "runs" / "v2" / "summary_test.json"
    assert S.summary_path("test") == DATA / "runs" / "summary_test.json"


def test_summarize_reads_the_tagged_tree(monkeypatch, tmp_path):
    root = tmp_path / "runs" / "v2" / "test"
    (root / "t1" / "L1").mkdir(parents=True)
    (root / "t1" / "L1" / "result.json").write_text(json.dumps(
        {"reward": 1.0, "agent_status": "exit 0", "prediction": "1", "sample": 0}))
    # a legacy result in the same split, which the tagged summary must not see
    legacy = tmp_path / "runs" / "test" / "t2" / "L1"
    legacy.mkdir(parents=True)
    (legacy / "result.json").write_text(json.dumps(
        {"reward": 0.0, "agent_status": "exit 0", "prediction": "0", "sample": 0}))
    monkeypatch.setattr(S, "DATA", tmp_path)

    runs = S.collect("test", "v2")

    assert set(runs) == {"t1"}, f"the tagged summary pooled the legacy tree: {set(runs)}"
    assert runs["t1"]["L1"][0]["prediction"] == "1"


def test_collect_keeps_the_sample_layout_inside_a_tag(tmp_path, monkeypatch):
    root = tmp_path / "runs" / "v2" / "test"
    for name in ["L1", "L1/s1"]:
        (root / "t1" / name).mkdir(parents=True)
        (root / "t1" / name / "result.json").write_text(json.dumps(
            {"reward": 1.0, "agent_status": "exit 0", "prediction": "1"}))
    monkeypatch.setattr(S, "DATA", tmp_path)

    trials = S.collect("test", "v2")["t1"]["L1"]

    assert len(trials) == 2, "the sample axis was lost inside the tagged tree"
    assert [t["_index"] for t in trials] == [0, 1]


def test_regrade_reads_the_tagged_tree(monkeypatch, tmp_path):
    root = tmp_path / "runs" / "v2" / "test"
    (root / "t1" / "L1").mkdir(parents=True)
    (root / "t1" / "L1" / "result.json").write_text(json.dumps(
        {"reward": 1.0, "agent_status": "exit 0", "prediction": "1"}))
    legacy = tmp_path / "runs" / "test" / "t2" / "L1"
    legacy.mkdir(parents=True)
    (legacy / "result.json").write_text(json.dumps(
        {"reward": 0.0, "agent_status": "exit 0", "prediction": "0"}))
    monkeypatch.setattr(R, "DATA", tmp_path)

    assert set(R.collect("test", "v2")) == {"t1"}


# --- RUN.json -------------------------------------------------------------------------------

def launch(tmp_path, tag="v2", rungs=("L1", "L3"), samples=1, limit=None, **flags):
    """Run main() with the model loop stubbed, and return the RUN.json it wrote.

    main() is driven rather than unit-called because RUN.json's contract is about the launch: the
    command line has to be the real argv, and the end-of-run counts have to be there whether the
    sweep returned or blew up. The flags go through the real argparse, so a RUN.json that recorded
    a field argparse never accepted would fail here rather than in a live sweep.
    """
    ran: list = []
    monkey = pytest.MonkeyPatch()
    monkey.setattr(runner, "DATA", tmp_path)
    monkey.setattr(runner, "source_for", lambda split: ([
        {"task_id": f"t{i}", "question": "q", "files": [], "answer": "1"}
        for i in range(limit or 2)
    ], lambda r: tmp_path / "in"))
    monkey.setattr(runner, "task_trials",
                    lambda row, split, rungs, *a, **k: (ran.append(row["task_id"]),
                                                        _trials(row, rungs, k.get("samples", 1)))[1])
    argv = ["run_ladder", "--run-tag", tag, "--split", "test", "--rungs", ",".join(rungs),
            "--samples", str(samples)]
    if not flags.get("climb", True):
        argv.append("--no-climb")
    for name, value in flags.items():
        if name == "climb":
            continue
        argv += [f"--{name.replace('_', '-')}"] if value is True else \
            [f"--{name.replace('_', '-')}", str(value)]
    monkey.setattr(sys, "argv", argv)
    try:
        runner.main()
    finally:
        monkey.undo()
    return json.loads((tmp_path / "runs" / tag / "RUN.json").read_text()), ran


def _trials(row, rungs, samples=1):
    return [{"task_id": row["task_id"], "rung": rung, "sample": k, "reward": 0.0,
             "agent_status": "exit 0", "prediction": ""}
            for rung in rungs for k in range(samples)]


def test_run_json_records_the_launch(tmp_path):
    run, _ = launch(tmp_path, model="stealth/space-bunny-alpha", agent="tools",
                    climb=False, workers=4)

    assert run["split"] == "test"
    assert run["run_tag"] == "v2"
    assert run["model"] == "stealth/space-bunny-alpha"
    assert run["agent"] == "tools"
    assert run["rungs"] == ["L1", "L3"]
    assert run["samples"] == 1
    assert run["climb"] is False
    assert run["workers"] == 4
    assert run["start_time"]
    assert run["git_commit"]
    assert "git_dirty" in run
    assert "--run-tag" in run["command_line"] and "v2" in run["command_line"]


def test_run_json_is_written_before_any_trial_runs(tmp_path):
    """A run that dies an hour in must still say what it was. main() writes the file, then the
    pool starts, so the existence of RUN.json at the first trial is the check."""
    seen: list[bool] = []

    def check_then_run(row, *a, **k):
        seen.append((tmp_path / "runs" / "v2" / "RUN.json").exists())
        return [{"task_id": row["task_id"], "rung": "L1", "sample": 0, "reward": 0.0,
                 "agent_status": "exit 0", "prediction": ""}]

    monkey = pytest.MonkeyPatch()
    monkey.setattr(runner, "DATA", tmp_path)
    monkey.setattr(runner, "source_for", lambda split: ([
        {"task_id": "t1", "question": "q", "files": [], "answer": "1"}], lambda r: tmp_path / "in"))
    monkey.setattr(runner, "task_trials", check_then_run)
    monkey.setattr(sys, "argv", ["run_ladder", "--run-tag", "v2", "--split", "test", "--rungs", "L1"])
    try:
        runner.main()
    finally:
        monkey.undo()

    assert seen == [True], "RUN.json was written after the first trial, not at launch"


def test_run_json_records_the_counts_and_the_end_time(tmp_path):
    run, ran = launch(tmp_path, rungs=("L1", "L3"), samples=2, limit=3)

    assert run["end_time"], "the run finished but never wrote an end time"
    assert run["tasks"] == 3
    assert run["trials"] == 6, "3 tasks x 2 rungs x 2 samples is six trials, not something else"
    assert len(ran) == 3


def test_run_json_counts_only_trials_that_really_ran(tmp_path):
    """A skipped rung was not attempted, so booking it would inflate the denominator."""
    def mixed(row, *a, **k):
        return [{"task_id": row["task_id"], "rung": "L1", "sample": 0, "reward": 0.0,
                 "agent_status": "exit 0", "prediction": "x"},
                {"task_id": row["task_id"], "rung": "L2", "sample": 0, "reward": 0.0,
                 "skipped": "no verified reference"}]

    monkey = pytest.MonkeyPatch()
    monkey.setattr(runner, "DATA", tmp_path)
    monkey.setattr(runner, "source_for", lambda split: ([
        {"task_id": f"t{i}", "question": "q", "files": [], "answer": "1"} for i in range(2)
    ], lambda r: tmp_path / "in"))
    monkey.setattr(runner, "task_trials", mixed)
    monkey.setattr(sys, "argv", ["run_ladder", "--run-tag", "v2", "--split", "test",
                                 "--rungs", "L1,L2"])
    try:
        runner.main()
    finally:
        monkey.undo()

    run = json.loads((tmp_path / "runs" / "v2" / "RUN.json").read_text())
    assert run["trials"] == 2, "a rung skipped for want of a reference was counted as a trial"
    assert run["skipped"] == 2


def test_a_resumed_launch_appends_and_keeps_the_earlier_records(tmp_path):
    """RUN.json is append-safe: a resumed sweep is a second launch of the same run, and the first
    launch's start time and code are the ones that produced the results already on disk."""
    first, _ = launch(tmp_path, limit=2)
    second, _ = launch(tmp_path, limit=3)

    assert len(second["launches"]) == 2
    assert second["launches"][0]["start_time"] == first["start_time"]
    assert second["launches"][0]["tasks"] == 2
    assert second["launches"][1]["tasks"] == 3
    # the header is the first launch's, so it still describes the results on disk
    assert second["start_time"] == first["start_time"]
    assert second["tasks"] == 3, "the header must carry the latest counts"
    assert second["end_time"] >= first["end_time"]


def test_a_dead_launch_still_closes_run_json_out(tmp_path):
    """A sweep that dies has to leave the end time behind, or "when did this stop" is unanswerable.

    Raised as a `KeyboardInterrupt` rather than an `Exception`: since one task raising became a
    harness failure for that task (`tests/test_task_isolation.py`), the only things that still end
    a run are the ones that are not a task -- a kill, an out-of-memory abort, a closed pipe on the
    way out. An `except Exception` around the loop would swallow all of them.
    """
    def explode(row, *a, **k):
        if row["task_id"] == "t1":
            raise KeyboardInterrupt
        return [{"task_id": row["task_id"], "rung": "L1", "sample": 0, "reward": 0.0,
                 "agent_status": "exit 0", "prediction": ""}]

    monkey = pytest.MonkeyPatch()
    monkey.setattr(runner, "DATA", tmp_path)
    monkey.setattr(runner, "source_for", lambda split: ([
        {"task_id": "t1", "question": "q", "files": [], "answer": "1"},
        {"task_id": "t0", "question": "q", "files": [], "answer": "1"}], lambda r: tmp_path / "in"))
    monkey.setattr(runner, "task_trials", explode)
    monkey.setattr(sys, "argv", ["run_ladder", "--run-tag", "v2", "--split", "test", "--rungs", "L1"])
    try:
        with pytest.raises(KeyboardInterrupt):
            runner.main()
    finally:
        monkey.undo()

    run = json.loads((tmp_path / "runs" / "v2" / "RUN.json").read_text())
    assert run["end_time"], "a crashed sweep left no end time"
    assert run["error"].startswith("KeyboardInterrupt")


def test_no_tag_writes_no_run_json(tmp_path):
    """The legacy tree is shared and half-written by every run ever; stamping it with one run's
    provenance would be a lie about which run owns it."""
    monkey = pytest.MonkeyPatch()
    monkey.setattr(runner, "DATA", tmp_path)
    monkey.setattr(runner, "source_for", lambda split: ([
        {"task_id": "t1", "question": "q", "files": [], "answer": "1"}], lambda r: tmp_path / "in"))
    monkey.setattr(runner, "task_trials", lambda row, *a, **k: [
        {"task_id": "t1", "rung": "L1", "sample": 0, "reward": 0.0,
         "agent_status": "exit 0", "prediction": ""}])
    monkey.setattr(sys, "argv", ["run_ladder", "--split", "test", "--rungs", "L1"])
    try:
        runner.main()
    finally:
        monkey.undo()

    assert not (tmp_path / "runs" / "test" / "RUN.json").exists()
    assert not (tmp_path / "runs" / "RUN.json").exists()


def test_the_runs_tree_the_tag_creates_is_where_trials_land(tmp_path, monkeypatch):
    """The end-to-end shape: trials under data/runs/<tag>/<split>/<task>/<rung>."""
    monkeypatch.setattr(runner, "DATA", tmp_path)
    ran = []
    monkeypatch.setattr(runner, "once", _fake_once(ran))
    monkeypatch.setattr(runner, "prompt_for", lambda row, split, rung: f"prompt:{rung}")
    monkeypatch.setattr(runner, "read_source", lambda row, split: "REF")
    row = {"task_id": "t1", "question": "q", "files": [], "answer": "1"}

    runner.task_trials(row, "test", ["L1_schema"], tmp_path, "m", 5, runs_root=runner.runs_dir("test", "v2"))

    assert [str(Path(w["work"]).relative_to(tmp_path / "runs" / "v2" / "test")) for w in ran] \
        == ["t1/L1_schema"]
    assert (tmp_path / "runs" / "v2" / "test" / "t1" / "L1_schema" / "result.json").exists()
    assert not (tmp_path / "runs" / "test").exists(), "the tagged run wrote into the legacy tree"


# --- the reference denominator ---------------------------------------------------------------

def test_the_launch_records_which_tasks_had_a_reference(monkeypatch):
    """References arrive while a sweep runs, so the set of tasks that could attempt L2 is a
    property of the launch, not of the summary. A summary that reads it off disk at the end is
    reading a different denominator than the one the run was given."""
    monkeypatch.setattr(runner, "read_source",
                        lambda row, split: "REF" if row["task_id"] in {"t0", "t2"} else None)
    rows = [{"task_id": f"t{i}"} for i in range(3)]

    state = runner.reference_state(rows, "test", ["L1", "L2", "L3", "L4"])

    assert state["reference_task_ids_at_launch"] == ["t0", "t2"]
    assert state["reference_tasks_at_launch"] == 2
    assert state["tasks_at_launch"] == 3


def test_a_run_with_no_reference_needing_rung_records_no_denominator(monkeypatch):
    """L1 and the control need no reference, so the ids would be a fact about the corpus rather
    than about this run, and 250 lines of it would be noise in every L1-only RUN.json."""
    monkeypatch.setattr(runner, "read_source", lambda row, split: "REF")
    assert runner.reference_state([{"task_id": "t0"}], "test", ["L1", "L1_schema"]) == {}


def test_a_reference_arriving_after_launch_does_not_join_the_denominator(monkeypatch):
    """The point of the record: the retry sweep built a reference for t1 while this run was going.
    L2 was skipped for t1, so t1 is not in this run's L2 population, whatever read_source says now."""
    had = {"t0"}
    monkeypatch.setattr(runner, "read_source", lambda row, split: "REF" if row["task_id"] in had
                        else None)
    rows = [{"task_id": "t0"}, {"task_id": "t1"}]
    at_launch = runner.reference_state(rows, "test", ["L1", "L2"])["reference_task_ids_at_launch"]

    had.add("t1")   # the concurrent retry sweep lands a reference for t1
    later = runner.reference_state(rows, "test", ["L1", "L2"])["reference_task_ids_at_launch"]

    assert at_launch == ["t0"]
    assert later == ["t0", "t1"], "the test does not reproduce a moving denominator"


def test_the_summary_pins_the_not_climbable_bucket_to_the_launch(monkeypatch):
    """A task with no L2 result is only 'not climbable' if it had no reference at launch. With one
    now on disk, read_source says yes and the task is booked 'not attempted' -- which reads as a
    ladder gap rather than as a rung the reference gate excluded."""
    record = {"reference_task_ids_at_launch": ["t0"]}
    # read_source now finds a reference for both tasks, because one arrived after the launch
    has_ref = {"t0": True, "t1": True}
    runs = {"t0": {"L1": [{"reward": 0.0, "agent_status": "exit 0"}]},
            "t1": {"L1": [{"reward": 0.0, "agent_status": "exit 0"}]}}

    pinned = S.has_reference_at_launch(record, lambda _t: True)
    hist = S.partition(runs, pinned)

    assert hist["not climbable (no reference)"] == 1, \
        "t1 had no reference at launch, so it belongs in the gated bucket"
    assert hist["not attempted"] == 0
    assert has_ref["t1"] is True, "the test no longer reproduces the moving denominator"


def test_the_pinned_reference_set_covers_every_launch(monkeypatch, tmp_path):
    """A smoke run is usually the first launch and a full run the second, and the smoke's six
    tasks are a subset of the full run's 250.

    Keeping the FIRST launch's reference set -- which is right for provenance, where the earliest
    code really did produce the earliest trials -- would pin the summary to those six tasks and
    report the other 244 climbable tasks as never attempted, when all of them were in fact run at
    L2. The reference set is a property of the tree, not of a launch: a task could attempt L2 if
    any launch had its reference, and references only accumulate.
    """
    monkey = pytest.MonkeyPatch()
    monkey.setattr(runner, "DATA", tmp_path)
    ids = {"t0", "t1"}
    monkey.setattr(runner, "read_source", lambda row, split: "REF" if row["task_id"] in ids else None)
    monkey.setattr(runner, "git_provenance", lambda: {"git_commit": "abc", "git_dirty": False})
    rows = [f"t{i}" for i in range(4)]
    rows[3] = "t3"

    def run_for(selected, limit=None):
        keep = set(selected)
        monkey.setattr(runner, "source_for", lambda split: (
            [{"task_id": t} for t in (rows[:limit] if limit else rows) if t in keep],
            lambda r: tmp_path / "in"))
        monkey.setattr(runner, "task_trials", lambda row, split, rungs, *a, **k: [
            {"task_id": row["task_id"], "rung": "L1", "sample": 0, "reward": 0.0,
             "agent_status": "exit 0", "prediction": ""}])
        monkey.setattr(sys, "argv", ["run_ladder", "--run-tag", "v2", "--split", "test",
                                     "--rungs", "L1,L2", "--limit", str(limit or 4)])
        runner.main()

    try:
        run_for(["t0", "t1"], limit=2)          # the smoke
        run_for(rows)                           # the full run
        record = json.loads((tmp_path / "runs" / "v2" / "RUN.json").read_text())
    finally:
        monkey.undo()

    assert len(record["launches"]) == 2
    # each launch keeps its own scope
    assert record["launches"][0]["reference_task_ids_at_launch"] == ["t0", "t1"]
    assert record["launches"][0]["tasks_at_launch"] == 2
    # and the tree-level set is the union, which is what the summary is pinned to
    assert record["reference_task_ids_at_launch"] == ["t0", "t1"]
    assert record["reference_tasks_at_launch"] == 2
    assert record["tasks_at_launch"] == 4, "the tree holds four tasks even though the smoke saw two"


def test_a_reference_pinned_by_an_earlier_launch_still_counts_after_a_wider_one(monkeypatch, tmp_path):
    """The union has to keep a task the early launch saw, in case that launch's trials are the
    only ones that ran it -- union the wrong way round and a real L2 population shrinks."""
    path = tmp_path / "runs" / "v2" / "RUN.json"
    runner.open_run_record(path, {"run_tag": "v2", "reference_task_ids_at_launch": ["t0", "t9"],
                                  "reference_tasks_at_launch": 2, "tasks_at_launch": 2})
    runner.open_run_record(path, {"run_tag": "v2", "reference_task_ids_at_launch": ["t1"],
                                  "reference_tasks_at_launch": 1, "tasks_at_launch": 3})
    record = json.loads(path.read_text())
    assert record["reference_task_ids_at_launch"] == ["t0", "t1", "t9"]
    assert record["reference_tasks_at_launch"] == 3
    assert record["tasks_at_launch"] == 3


def _fake_once(ran: list):
    def once(row, prompt, work, venv, model, max_turns, retry_failed=False, inputs_of=None,
             rung_label="run", provenance=None, was_run=None, agent="tools",
             save_transcript=True):
        work = Path(work)
        if was_run is not None:
            was_run.clear()
        if (work / "result.json").exists():
            return json.loads((work / "result.json").read_text())
        if was_run is not None:
            was_run.append(True)
        work.mkdir(parents=True, exist_ok=True)
        (work / "prompt.txt").write_text(prompt)
        ran.append({"work": work})
        return {"task_id": row["task_id"], "reward": 0.0, "agent_status": "exit 0",
                "prediction": "", "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()}
    return once


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
