"""Reclaim disk from a results tree, without deleting a single result.

Trial directories accumulate whatever the agent wrote into them, and $HOME pointed at the trial
dir, so one `pip install xgboost` left 660 MB of wheels there. TensorFlow and nvidia-nccl were
worse. Across the SmolDataEnvs runs that was 22 GB of vendored site-packages, pip HTTP cache and
wheels — none of it a result.

**The verifier's copies of the tables.** The offline grading pass copies a task's input tables
into `<trial>/verify/input` so the sealed pass can bind-mount a self-contained directory. Those
copies were left behind, and they are the largest thing in a results tree: ~30 MB per trial, and
24 GB across the v2 run — more than every result in it put together. `once()` now drops each copy
as soon as the pass finishes, but the ones already written are still on disk, so this module
collects those too.

The guard is the trial's own `result.json`: a copy beside a result is finished work whose tables
are reconstructible from the shared cache by path, and a copy with *no* result beside it belongs
to a trial that was killed mid-grading and may be the only record of what it was doing. So that
one is left alone, and it is the reason this is not a blind `rm -rf`.

    uv run python -m smol_ladder.reclaim --dry-run
    uv run python -m smol_ladder.reclaim
    uv run python -m smol_ladder.reclaim --split v2/test       # one tree of a tagged run
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from smol_ladder.tasks import DATA

# Directories the agent creates by installing or caching things. Never a result: result.json,
# solution.py, the transcript and the verify copy are what we keep.
# The prefixes are there because agents invent directory names for the same thing: whl310 for a
# wheel per interpreter, libs310 for an unpacked wheel, and .local for a user install. Between
# them that was 22 GB in the SmolDataEnvs tree and 4.9 GB in a single jupyter-agent trial.
JUNK_DIRS = {".local", ".cache", ".tmp", "site-packages", "_vendor_xgb", ".ipynb_checkpoints",
             "__pycache__", ".npm", ".config"}
JUNK_DIR_PREFIXES = ("whl", "libs", "vendor", "pkgs")
# Only archives, never .so: a solution is free to ship a compiled extension, and deleting one
# would silently change what a re-run of that trial does.
JUNK_SUFFIX = {".whl", ".tar.gz"}


def is_junk_dir(path: Path) -> bool:
    return path.name in JUNK_DIRS or path.name.startswith(JUNK_DIR_PREFIXES)


# ── the verifier's copies of the task tables ──────────────────────────────────

#: Where `once()` puts them. Matched by name and by position rather than by any recorded path,
#: because there is no index: the trials are whatever the sweep happened to write, at
#: <task>/<rung>/, <task>/<rung>/s<k>/ and <task>/<rung>/<attempt>/, and the reclaim must work on
#: a tree written by code that is no longer running.
VERIFY = "verify"
VERIFY_INPUT = "input"


def _dir_bytes(path: Path) -> int:
    """Bytes under `path`, counting the files a symlink points at.

    A copy is made with `symlinks=True`, so its members are links into the shared cache and their
    own `st_size` is the length of a path, not of a table. Reporting that would make the
    reclaimable figure read as a few kilobytes for 24 GB of tables, which is the one number this
    module exists to be right about.
    """
    total = 0
    for f in path.rglob("*"):
        if f.is_file():
            try:
                total += f.stat().st_size
            except OSError:
                continue
    return total


def verify_copies(root: Path) -> tuple[list[Path], int]:
    """Every `verify/input` under `root` whose trial has a result, and their total size.

    The result.json guard is the whole safety argument: it is what distinguishes a finished
    trial from one a sweep was killed in the middle of. It is checked on the trial directory --
    `verify`'s parent -- rather than inside `verify`, because result.json is written beside
    `verify`, not inside it.
    """
    found: list[Path] = []
    total = 0
    for verify in sorted(root.rglob(VERIFY)):
        tables = verify / VERIFY_INPUT
        if not (tables.is_dir() and not tables.is_symlink()):
            continue
        if not (verify.parent / "result.json").exists():
            continue
        found.append(tables)
        total += _dir_bytes(tables)
    return found, total


def reclaim_verify_copies(root: Path) -> tuple[int, int]:
    """Delete each finished trial's `verify/input`, and the links that resolve through it.

    Returns `(copies removed, bytes freed)`. The sibling links are removed with it because they
    are relative targets into the copy, and a dangling `a.csv -> input/a.csv` beside a surviving
    solution.py is worse than no link: it reads as evidence and resolves to nothing.

    Never raises. A tree shared with a running sweep will have trials appear and vanish under the
    walk, and reclaim failing halfway leaves every copy it already removed reclaimed, which is
    strictly better than aborting the pass.
    """
    copies, freed = 0, 0
    for tables in verify_copies(root)[0]:
        freed += _dir_bytes(tables)
        verify = tables.parent
        try:
            shutil.rmtree(tables, ignore_errors=True)
            copies += 1
        except OSError:
            continue
        for item in verify.glob("*"):
            if item.name == VERIFY_INPUT:
                continue
            try:
                if item.is_symlink() and str(item.readlink()).startswith(
                        VERIFY_INPUT + os.sep):
                    item.unlink()
            except OSError:
                pass
    return copies, freed


def scan(root: Path) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    """Every removable path with its size: (directories, files).

    Returns all of them, not a sample. A previous version returned only the largest 15, so the
    non-dry run would have deleted those and stopped, leaving the rest to accumulate again.
    """
    dirs: list[tuple[str, int]] = []
    files: list[tuple[str, int]] = []
    for path in sorted(root.rglob("*"), key=lambda p: -len(p.parts)):
        try:
            if path.is_dir() and is_junk_dir(path):
                # Skip if an ancestor is already listed; removing the ancestor removes this.
                if any(str(path.relative_to(root)).startswith(d + os.sep) for d, _ in dirs):
                    continue
                size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
                dirs.append((str(path.relative_to(root)), size))
            elif path.is_file() and path.suffix in JUNK_SUFFIX:
                files.append((str(path.relative_to(root)), path.stat().st_size))
        except OSError:
            continue
    dirs.sort(key=lambda x: -x[1])
    files.sort(key=lambda x: -x[1])
    return dirs, files


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="test",
                    help="the tree to reclaim, relative to data/runs/. A tagged run's tree is "
                         "v2/test, so a run can be reclaimed without touching any other.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verify-inputs-only", action="store_true",
                    help="only collect the verifier's copies of the task tables, and leave the "
                         "agent's wheels and caches alone")
    args = ap.parse_args()

    root = DATA / "runs" / args.split
    if not root.exists():
        print(f"{root} does not exist")
        return
    before = sum(f.stat().st_size for f in root.rglob("*") if f.is_file())

    # The copies are handled first and on their own, because they are the bulk of the tree and
    # because they are the one category whose guard depends on a file the other scan does not
    # look at. Counting them in both passes would report each byte twice.
    copies, copy_bytes = verify_copies(root)
    print(f"{root}: {before/1e9:.1f} GB")
    print(f"  {len(copies)} verify/input copies, {copy_bytes/1e9:.1f} GB "
          f"(trials with a result.json; in-flight trials are left alone)")

    dirs, files = ([], []) if args.verify_inputs_only else scan(root)
    reclaimable = sum(s for _, s in dirs) + sum(s for _, s in files)
    if not args.verify_inputs_only:
        print(f"  reclaimable {reclaimable/1e9:.1f} GB: {len(dirs)} dirs, "
              f"{len(files)} wheel/archive files")
        for name, size in (dirs + files)[:10]:
            print(f"    {size/1e6:8.0f} MB  {name}")

    if args.dry_run:
        print("dry run: nothing removed")
        return

    removed = 0
    if copies:
        removed_copies, freed = reclaim_verify_copies(root)
        print(f"removed {removed_copies} verify/input copies, freed {freed/1e9:.1f} GB")
        removed += removed_copies
    for name, _ in dirs:
        try:
            shutil.rmtree(root / name)
            removed += 1
        except OSError as e:
            print(f"  could not remove {name}: {e}")
    for name, _ in files:
        try:
            (root / name).unlink()
            removed += 1
        except OSError as e:
            print(f"  could not remove {name}: {e}")
    after = sum(f.stat().st_size for f in root.rglob("*") if f.is_file())
    print(f"removed {removed} paths; {before/1e9:.1f} GB -> {after/1e9:.1f} GB "
          f"(freed {(before-after)/1e9:.1f} GB)")


if __name__ == "__main__":
    main()
