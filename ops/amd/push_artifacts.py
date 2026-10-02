"""Push the droplet's run artifacts to a private Hugging Face dataset repo.

    python -m ops.amd.push_artifacts --repo <ns>/smol-ladder-runs --run-tag-prefix amd1

Why this exists rather than a shell loop: an interrupted upload is the common case (the watchdog
is racing the droplet's shutdown), so every upload is skipped when the remote already has the file
at the same size. That makes it safe to re-run and cheap to re-run, which is the only property that
matters when it is called from a teardown path.
"""

from __future__ import annotations

import argparse
from pathlib import Path

ARTIFACT_GLOBS = ("runs/sft_*/**/*", "logs/*")


def wanted(root: Path, run_tag_prefix: str) -> list[Path]:
    """Adapter files, logs and the results tree for the tags this driver owns.

    Checkpoints are included deliberately. `--resume` needs the newest one to be complete, and a
    half-copied checkpoint is a resume that fails on load.
    """
    files: set[Path] = set()
    for pattern in ARTIFACT_GLOBS:
        for path in root.glob(pattern):
            if path.is_file() and path.stat().st_size:
                files.add(path)
    if run_tag_prefix:
        for path in (root / "data" / "runs").glob(f"{run_tag_prefix}*/**/*"):
            if path.is_file() and path.stat().st_size:
                files.add(path)
    return sorted(files)


def remote_name(path: Path, root: Path) -> str:
    return str(path.relative_to(root))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True, help="private dataset repo, e.g. myns/smol-ladder-runs")
    ap.add_argument("--root", type=Path, default=Path("/opt/smol-ladder"))
    ap.add_argument("--run-tag-prefix", default="")
    ap.add_argument("--dry-run", action="store_true", help="list what would be uploaded, upload nothing")
    args = ap.parse_args()

    if not args.repo.count("/") == 1:
        raise SystemExit(f"--repo must be <namespace>/<name>, got {args.repo!r}")

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(args.repo, repo_type="dataset", private=True, exist_ok=True)
    files = wanted(args.root, args.run_tag_prefix)
    if not files:
        print("nothing to upload")
        return
    if args.dry_run:
        for path in files:
            print(f"{path.stat().st_size:>12}  {remote_name(path, args.root)}")
        return
    api.upload_folder(args.repo, str(args.root), repo_type="dataset",
                      path_in_repo="", allow_patterns=[remote_name(p, args.root) for p in files])


if __name__ == "__main__":
    main()
