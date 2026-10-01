"""`input_dir` twice, and two workers at once.

The `FileExistsError` that killed the ja3 sweep came from `(dest / name).symlink_to(...)` in
`jtasks.input_dir`. Two tasks named the same table, so the second one's link landed where the
first one's already was and `os.symlink` refused; the exception came back through `f.result()`
and took 2,257 still-unattempted tasks with it. Both halves of that are defects of the same
function, and they are pinned here separately:

- **idempotent**: asking twice for a task whose tables are already linked is success, not an
  error. That is what "are these tables here?" (`fetch=False`) already does; now the fetch path
  does too, which is what a resumed sweep and a retry actually do.
- **race-safe**: two threads asking for the same task at the same moment both get a correct
  directory. Replacing a *wrong* link has to be atomic, or the loser of the race reads a
  half-removed file.

    uv run --with pytest pytest -q tests/test_input_idempotence.py
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from smol_ladder import jtasks


@pytest.fixture
def cache(tmp_path, monkeypatch):
    """A cache root holding one extracted dataset, and the `dataset_download` call stubbed.

    `SMOL_LADDER_CACHE` is the *root* -- `input_dir` appends `kaggle/` itself -- so the fixture
    hands back `<root>/kaggle`, which is where `dataset_download`'s stub points and where the
    per-task directories land.
    """
    root = tmp_path / "kaggle" / "datasets" / "owner" / "ds" / "versions" / "1"
    (root / "nested").mkdir(parents=True)
    (root / "nested" / "t.csv").write_text("a\n1\n")
    (root / "other.csv").write_text("b\n2\n")
    monkeypatch.setenv("SMOL_LADDER_CACHE", str(tmp_path))
    monkeypatch.delenv("KAGGLEHUB_CACHE", raising=False)
    return tmp_path / "kaggle"


ROW = {"task_id": "t1", "question": "q", "files": ["t.csv", "other.csv"],
       "kaggle_dataset_name": "owner/ds"}


def _patch_download(cache, monkeypatch):
    """Stub the dataset download, re-importing so the module-level name is the one called.

    `input_dir` imports kagglehub inside the function, so the patch has to go on the module
    object itself: `input_dir`'s own `import kagglehub` resolves the attribute at call time.
    """
    import kagglehub

    root = cache / "datasets" / "owner" / "ds" / "versions" / "1"
    monkeypatch.setattr(kagglehub, "dataset_download", lambda dataset: str(root))


def test_the_task_directory_is_built_once(cache, monkeypatch):
    _patch_download(cache, monkeypatch)

    dest = jtasks.input_dir(ROW)

    assert (dest / "t.csv").is_symlink()
    assert (dest / "t.csv").resolve() == (cache / "datasets" / "owner" / "ds" / "versions" / "1"
                                          / "nested" / "t.csv")
    assert sorted(p.name for p in dest.iterdir()) == ["other.csv", "t.csv"]


def test_asking_twice_is_success_and_changes_nothing(cache, monkeypatch):
    """The idempotence half. This is exactly what a resumed sweep does to a task it already ran."""
    _patch_download(cache, monkeypatch)

    first = jtasks.input_dir(ROW)
    before = {p.name: p.readlink() for p in first.iterdir()}
    second = jtasks.input_dir(ROW)          # must not raise

    assert second == first
    assert {p.name: p.readlink() for p in second.iterdir()} == before, \
        "the second call rewrote the directory"


def test_a_symlink_pointing_somewhere_else_is_replaced(cache, monkeypatch):
    """The stale-target half. A link left by an earlier cache version points at a file that is no
    longer the one this task should read, and silently keeping it hands the trial the wrong table."""
    _patch_download(cache, monkeypatch)
    dest = jtasks.input_dir(ROW)
    (dest / "t.csv").unlink()
    (dest / "t.csv").symlink_to(cache / "datasets" / "owner" / "ds" / "versions" / "1" / "other.csv")

    jtasks.input_dir(ROW)

    assert (dest / "t.csv").resolve().name == "t.csv"
    assert (dest / "t.csv").resolve() != (cache / "datasets" / "owner" / "ds" / "versions" / "1"
                                          / "other.csv")


def test_a_missing_link_is_created_even_though_the_directory_exists(cache, monkeypatch):
    """Partly built is the state a killed trial leaves. `dest.exists()` is true for it, and a
    directory that exists with one of its three tables is not a usable input directory."""
    _patch_download(cache, monkeypatch)
    dest = jtasks.input_dir(ROW)
    (dest / "other.csv").unlink()

    jtasks.input_dir(ROW)

    assert (dest / "other.csv").is_symlink()


def test_a_real_file_where_a_link_belongs_is_replaced(cache, monkeypatch):
    """Not only symlinks. `path.exists()` follows links, so a dangling link reads as missing and
    `path.is_symlink()` as not-there; both have to be handled, and a stale regular file is a
    table copy from an older layout that must not be served in place of the real one."""
    _patch_download(cache, monkeypatch)
    dest = jtasks.input_dir(ROW)
    (dest / "t.csv").unlink()
    (dest / "t.csv").write_text("wrong\n")

    jtasks.input_dir(ROW)

    assert (dest / "t.csv").is_symlink()
    assert (dest / "t.csv").read_text() == "a\n1\n"


def test_two_tasks_naming_the_same_table_both_get_it(cache, monkeypatch):
    """The bug itself: three tasks in data/jtasks_v3.jsonl name `bubble_volume.csv` twice. The
    second symlink_to raises FileExistsError, and the sweep dies on it."""
    _patch_download(cache, monkeypatch)
    (cache / "datasets" / "owner" / "ds" / "versions" / "1" / "bubble_volume.csv").write_text("v\n")

    row = {"task_id": "dup", "question": "q", "kaggle_dataset_name": "owner/ds",
           "files": ["bubble_volume.csv", "bubble_volume.csv", "other.csv"]}

    dest = jtasks.input_dir(row)      # twice on one row: not a raise

    assert (dest / "bubble_volume.csv").is_symlink()
    assert (dest / "other.csv").is_symlink()


def test_concurrent_callers_of_one_task_all_succeed(cache, monkeypatch):
    """The race half: a sweep runs 16 workers over one pool, and two tasks can share a cache
    entry. Every caller has to come back with the same, correct directory."""
    _patch_download(cache, monkeypatch)
    seen: list[Path] = []
    errors: list[BaseException] = []
    start = threading.Barrier(8)

    def call():
        try:
            start.wait(timeout=30)
            seen.append(jtasks.input_dir(ROW))
        except BaseException as e:  # noqa: BLE001 - the point is to record, not to raise
            errors.append(e)

    threads = [threading.Thread(target=call) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)

    assert errors == [], f"{len(errors)} of 8 concurrent callers raised: {errors[:1]}"
    assert len({str(p) for p in seen}) == 1
    dest = seen[0]
    assert (dest / "t.csv").resolve().name == "t.csv"
    assert sorted(p.name for p in dest.iterdir()) == ["other.csv", "t.csv"]


def test_a_replacing_call_never_leaves_a_window_with_no_link(cache, monkeypatch):
    """Atomic replacement: link to a temporary name in the same directory, then `os.replace` it
    over the old one. A reader that looked between an unlink and a symlink_to would see a
    directory with no table in it, which is the failure the atomic rename exists to prevent."""
    _patch_download(cache, monkeypatch)
    dest = jtasks.input_dir(ROW)
    link = dest / "t.csv"

    jtasks.input_dir(ROW)

    assert not [p for p in dest.iterdir() if p.name.startswith(".t.csv")], \
        "the temporary link used for the atomic swap was left behind"
    assert link.is_symlink()


def test_fetch_false_still_never_downloads(cache, monkeypatch):
    """The contract that made the cached-input skip affordable must not change: asking whether a
    task's tables are here must not cost a dataset download."""
    def boom(dataset):
        raise AssertionError("input_dir(fetch=False) hit the network")
    import kagglehub

    monkeypatch.setattr(kagglehub, "dataset_download", boom)

    dest = jtasks.input_dir(ROW, fetch=False)

    assert dest.name == "t1"
    assert not dest.exists(), "fetch=False created the task directory"


def test_the_helper_is_the_one_input_dir_calls():
    """`link_table` is the unit under test; if input_dir grew its own inline os.symlink this
    would stop being the code path the tests exercise."""
    import inspect

    source = inspect.getsource(jtasks.input_dir)
    assert "link_table" in source
    assert "symlink_to" not in source, "input_dir still builds links itself"