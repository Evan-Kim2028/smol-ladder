"""Reclaim disk from a results tree, without deleting a single result.

Trial directories accumulate whatever the agent wrote into them, and $HOME pointed at the trial
dir, so one `pip install xgboost` left 660 MB of wheels there. TensorFlow and nvidia-nccl were
worse. Across the SmolDataEnvs runs that was 22 GB of vendored site-packages, pip HTTP cache and
wheels — none of it a result.

    uv run python -m smol_ladder.reclaim --dry-run
    uv run python -m smol_ladder.reclaim
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    root = DATA / "runs" / args.split
    if not root.exists():
        print(f"{root} does not exist")
        return
    before = sum(f.stat().st_size for f in root.rglob("*") if f.is_file())
    dirs, files = scan(root)
    reclaimable = sum(s for _, s in dirs) + sum(s for _, s in files)

    print(f"{root}: {before/1e9:.1f} GB, reclaimable {reclaimable/1e9:.1f} GB")
    print(f"  {len(dirs)} dirs, {len(files)} wheel/archive files")
    for name, size in (dirs + files)[:10]:
        print(f"    {size/1e6:8.0f} MB  {name}")

    if args.dry_run:
        return
    removed = 0
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
