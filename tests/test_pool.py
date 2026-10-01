"""Selecting the v3 ladder-grade pool, without touching the v1 pool on disk.

`data/jtasks.jsonl` is the 2,000-task v1 extract, and it is what every existing jupyter-agent run
was measured against -- so it stays exactly as it is. This module reads `data/jtasks_v3.jsonl`
instead and applies the ladder-grade filter (`jtasks_v2.is_ladder_grade`: not nondeterministic,
not ambiguous, and it names at least one input file), which is the 4,217-task subset.

Two things have to be true of the selection and neither is about the rows themselves:

- **`data/jtasks.jsonl` is not written.** It is read by other checkouts' live sweeps, and a pool
  is an input to a measurement, not an output of one.
- **Results never mix with the v1 sweep.** A run tag gives the run its own tree, and a reference
  split name gives it its own solutions directory, so a task verified under v3 is never mistaken
  for the same task verified under v1's rules.
"""

import json
from pathlib import Path

import pytest

from smol_ladder import jtasks_v2, pool
from smol_ladder.jtasks_v2 import is_ladder_grade


V3_SPLIT = "jupyter-agent-v3"


def _write_pool(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def _row(task_id: str, **kw) -> dict:
    base = {"task_id": task_id, "question": "How many rows?", "answer": "3", "reward_mode": "numeric",
            "atol": 1e-3, "rtol": 1e-3, "files": ["t.csv"], "source": "jupyter-agent/jupyter-agent-dataset",
            "kaggle_dataset_name": "owner/ds", "edu_score": 1.0, "sde_overlap": "none",
            "shares_table_with_sde_train": False, "op_family": "count", "answer_type": "numeric",
            "nondeterminism_reasons": [], "nondeterministic": False, "ambiguity_reasons": [],
            "ambiguous": False, "n_files": 1, "input_bytes": 100, "inputs_cached": True}
    return {**base, **kw}


def test_the_v3_split_reads_v3_and_keeps_only_the_ladder_grade_rows(tmp_path, monkeypatch):
    """The whole point of the split name: the corrected pool, filtered, and v1 untouched."""
    rows = [
        _row("keep_me"),
        _row("nondeterministic", nondeterministic=True, nondeterminism_reasons=["model fit"]),
        _row("ambiguous", ambiguous=True, ambiguity_reasons=["unstated threshold"]),
        _row("no_files", files=[], n_files=0),
    ]
    _write_pool(tmp_path / "jtasks_v3.jsonl", rows)
    monkeypatch.setattr(pool, "DATA", tmp_path)

    selected, inputs_of = pool.source_for(V3_SPLIT)

    assert [r["task_id"] for r in selected] == ["keep_me"]
    # And every selected row is one the shared filter agrees with, so this module cannot drift
    # from the definition it claims to apply.
    assert all(is_ladder_grade(r) for r in selected)
    assert callable(inputs_of)


def test_selecting_v3_never_writes_the_v1_pool(tmp_path, monkeypatch):
    """`data/jtasks.jsonl` is an input to every live jupyter-agent sweep, not this module's to
    regenerate. Reading v3 and rewriting v1 would change the population under a running sweep
    while looking like a no-op."""
    v1 = tmp_path / "jtasks.jsonl"
    _write_pool(v1, [_row("v1_task")])
    before = v1.read_bytes()
    _write_pool(tmp_path / "jtasks_v3.jsonl", [_row("v3_task")])
    monkeypatch.setattr(pool, "DATA", tmp_path)

    selected, _ = pool.source_for(V3_SPLIT)

    assert [r["task_id"] for r in selected] == ["v3_task"]
    assert v1.read_bytes() == before, "selecting v3 rewrote the v1 pool"


def test_the_v1_split_still_reads_v1(tmp_path, monkeypatch):
    """The v1 split name must keep meaning v1. A sweep that asked for `jupyter-agent` and got v3
    rows would silently change every number it had already published."""
    _write_pool(tmp_path / "jtasks.jsonl", [_row("v1_task")])
    _write_pool(tmp_path / "jtasks_v3.jsonl", [_row("v3_task"), _row("v3_b")])
    monkeypatch.setattr(pool, "DATA", tmp_path)

    selected, _ = pool.source_for("jupyter-agent")

    assert [r["task_id"] for r in selected] == ["v1_task"]


def test_the_split_is_a_different_split_and_therefore_a_different_tree(tmp_path, monkeypatch):
    """`--split jupyter-agent-v3` has to land under its own directory, or the v3 trials overwrite
    the v1 ones in `data/runs/jupyter-agent/` and neither sweep is readable afterwards."""
    import smol_ladder.run_ladder as runner

    tagged = runner.runs_dir(V3_SPLIT, "ja3", data=tmp_path)
    assert tagged == tmp_path / "runs" / "ja3" / "jupyter-agent-v3"
    assert runner.runs_dir("jupyter-agent", "ja3", data=tmp_path) != tagged
    assert runner.runs_dir(V3_SPLIT, None, data=tmp_path) == \
        tmp_path / "runs" / "jupyter-agent-v3"
    # and nothing about it can escape the runs tree, same as any other tag
    with pytest.raises(ValueError):
        runner.runs_dir(V3_SPLIT, "../escape", data=tmp_path)


def test_references_go_to_a_split_named_directory(tmp_path, monkeypatch):
    """`gen_refs --split jupyter-agent-v3` writes `data/solutions/jupyter-agent-v3/<task>/`, and
    `read_source` looks there too -- so a v3 reference can never be served as a v1 reference to a
    rung built from v1's gold."""
    import smol_ladder.ladder as ladder
    from smol_ladder import gen_refs

    monkeypatch.setattr(gen_refs, "DATA", tmp_path)
    monkeypatch.setattr(ladder, "DATA", tmp_path)

    row = _row("t1")
    assert ladder.read_source(row, V3_SPLIT) is None

    task = tmp_path / "solutions" / V3_SPLIT / "t1"
    task.mkdir(parents=True)
    (task / "result.json").write_text(json.dumps({"reward": 1.0, "keep": True}))
    (task / "solution.py").write_text("print(3)\n")

    assert ladder.read_source(row, V3_SPLIT) == "print(3)\n"
    assert ladder.read_source(row, "jupyter-agent") is None, \
        "a v3 reference leaked into the v1 split"
    # and gen_refs considers it verified, so a re-run of v3 is not re-rolled
    assert gen_refs.verified(task) is not None


def test_tasks_without_cached_inputs_are_separated_and_recorded(tmp_path, monkeypatch):
    """7,018 of 7,518 v3 tasks have their tables; the other 500 would each cost a Kaggle
    download, and the day's downloads were returning 403s.

    They are *skipped*, not attempted and not counted as failures -- and the ids are written down,
    because "we skipped 500 tasks" is only auditable if the 500 are named. A sweep that silently
    drops them reports a pass rate over a population nobody chose.
    """
    cached = tmp_path / "cache"
    cached.mkdir()
    rows = [_row("cached_a"), _row("cached_b"), _row("uncached", inputs_cached=False,
                                                    input_bytes=None)]
    _write_pool(tmp_path / "jtasks_v3.jsonl", rows)
    monkeypatch.setattr(pool, "DATA", tmp_path)

    selected, inputs_of = pool.source_for(V3_SPLIT)
    # The two cached rows point at a directory that exists; the third does not. inputs_of is the
    # seam: in production it is jtasks.input_dir, which resolves the Kaggle cache by task id.
    present = {"cached_a": cached, "cached_b": cached, "uncached": tmp_path / "gone"}
    runnable, skipped = pool.without_cached_inputs(
        selected, inputs_of=lambda r: present[r["task_id"]])

    assert [r["task_id"] for r in runnable] == ["cached_a", "cached_b"]
    assert skipped == ["uncached"]

    out = tmp_path / "skipped.json"
    pool.record_skipped(V3_SPLIT, skipped, out, planned=len(selected))
    saved = json.loads(out.read_text())
    assert saved["split"] == V3_SPLIT
    assert saved["tasks_planned"] == 3
    assert saved["skipped_for_inputs"] == 1
    assert saved["skipped_task_ids"] == ["uncached"]
    # The v3 pool file itself is never touched by any of this.
    assert len((tmp_path / "jtasks_v3.jsonl").read_text().splitlines()) == 3


def test_a_task_whose_flag_says_cached_but_whose_table_is_gone_is_still_skipped(tmp_path,
                                                                               monkeypatch):
    """The flag records what was true when the pool was built; the cache is what is true now.

    7,018 is a snapshot. If a dataset has since been evicted from /var/tmp, the row still says
    `inputs_cached: true` and `input_dir` would try to download it -- the exact 403 path this
    skip exists to avoid. So the flag is never trusted on its own: it has to be corroborated by a
    directory that exists.
    """
    monkeypatch.setenv("SMOL_LADDER_CACHE", str(tmp_path))
    assert pool.inputs_are_cached(_row("flag_only", inputs_cached=True),
                                  inputs_of=lambda r, **kw: tmp_path / "nope") is False
    present = tmp_path / "there"
    present.mkdir()
    assert pool.inputs_are_cached(_row("flag_only", inputs_cached=True),
                                  inputs_of=lambda r, **kw: present) is True


def test_a_cached_dataset_archive_counts_as_cached_without_a_per_task_directory(tmp_path,
                                                                              monkeypatch):
    """The cache has two levels, and only the looser one answers the question being asked.

    `kagglehub` keeps dataset archives under `kaggle/datasets/<owner>/<name>/`; `input_dir` then
    builds a per-task symlink directory under `kaggle/tasks/<task_id>/` the first time that task
    runs. On this machine 3,880 of the 4,217 ladder-grade tasks have the archive and only 1,623
    have the per-task directory. Testing for the per-task directory alone would skip 2,257 tasks
    whose tables are already on disk and cost nothing to read -- and the sweep would report a
    population nobody chose.
    """
    cache = tmp_path / "kaggle"
    (cache / "datasets" / "owner" / "ds").mkdir(parents=True)
    monkeypatch.setenv("SMOL_LADDER_CACHE", str(tmp_path))
    row = _row("t", inputs_cached=True)

    # No per-task directory exists, so input_dir(fetch=False) says no...
    assert not (cache / "tasks" / "t").exists()
    # ...but the archive the task would be built from is here, so no download is needed.
    assert pool.inputs_are_cached(row, inputs_of=lambda r, **kw: cache / "tasks" / "t") is True


def test_the_archive_is_matched_on_the_full_owner_name(tmp_path, monkeypatch):
    """A mirror of a cached dataset is a different dataset.

    Two owners uploading the same table is how 121 tasks reached a held-out SmolDataEnvs table
    past the overlap firewall, which compared full slugs while the tag comparison used bare names.
    Reading a mirror's files when the owner's archive is what is on disk would be a wrong answer
    to a question about the wrong table, so the key here is the whole `owner/name`.
    """
    (tmp_path / "kaggle" / "datasets" / "owner" / "ds").mkdir(parents=True)
    monkeypatch.setenv("SMOL_LADDER_CACHE", str(tmp_path))
    missing = lambda r, **kw: tmp_path / "kaggle" / "tasks" / r["task_id"]

    assert pool.inputs_are_cached(_row("t", inputs_cached=True), missing) is True
    assert pool.inputs_are_cached(
        _row("t", inputs_cached=True, kaggle_dataset_name="mirror/ds"), missing) is False


def test_a_split_that_is_not_a_kaggle_pool_skips_nothing(tmp_path, monkeypatch):
    """The skip is a property of a Kaggle-backed pool, not of every split.

    SmolDataEnvs rows come from one HuggingFace bucket and synthetic rows point at a table that is
    already local, so neither carries `inputs_cached` or has a kagglehub archive to look for.
    Applying the test to them would skip the entire corpus and report a population of zero -- a
    sweep that ran nothing and said 0%, which reads as a model that cannot answer anything.
    """
    rows = [_row("a"), _row("b")]
    gone = lambda r, **kw: tmp_path / "nowhere"

    assert pool.without_cached_inputs(rows, gone, enabled=False) == (rows, [])
    assert pool.without_cached_inputs(rows, gone, enabled=True)[1] == ["a", "b"]


def test_the_same_filter_applies_to_v1_so_the_two_subsets_are_comparable(tmp_path, monkeypatch):
    """A selection is only meaningful if it is the shared definition, not a private one."""
    rows = [_row("keep"), _row("drop", nondeterministic=True)]
    _write_pool(tmp_path / "jtasks_v3.jsonl", rows)
    monkeypatch.setattr(pool, "DATA", tmp_path)

    selected, _ = pool.source_for(V3_SPLIT)
    assert [r["task_id"] for r in selected] == ["keep"]
    # the filter itself is jtasks_v2's, not a copy of it
    assert pool.LADDER_GRADE is jtasks_v2.is_ladder_grade
    assert not pool.LADDER_GRADE(rows[1])
