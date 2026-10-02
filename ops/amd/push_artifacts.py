"""Push the droplet's logs and measurements to the private Hub dataset repo, and verify adapters.

    python ops/amd/push_artifacts.py --repo ns/smol-ladder-runs --log-dir /var/log/smol-ladder
    python ops/amd/push_artifacts.py --verify ns/smol-ladder-sft-a ns/smol-ladder-sft-b

Runs on the teardown path, possibly while racing a shutdown, so it is written to be re-runnable: a
file the Hub already has at the same size is skipped. Adapters and checkpoints do not go through
here: the trainer pushes those to each arm's own private model repo (hub_strategy="checkpoint")
and run_sft.sh pushes the final adapter, so this only carries what the trainer does not: logs,
the smoke's measurements, the checksum list.

`--verify` is the check that matters before a destroy: each named model repo must list both
adapter files. A push that returned success but wrote nothing readable is invisible until the
adapter is needed, which is on a droplet that no longer exists.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

LOG_SUFFIXES = (".log", ".json", ".tsv", ".trips", ".artifacts")


def wanted(log_dir: Path) -> list[Path]:
    return sorted(p for p in log_dir.rglob("*")
                  if p.is_file() and p.stat().st_size and "adapters" not in p.parts
                  and (p.suffix in LOG_SUFFIXES or p.name.startswith("SHA256SUMS")))


def already_there(api, repo: str, name: str, size: int) -> bool:
    try:
        info = api.get_paths_info(repo, [name], repo_type="dataset")
    except Exception:  # noqa: BLE001 - "unknown" means upload
        return False
    return any(getattr(i, "size", None) == size for i in info)


def push(api, repo: str, log_dir: Path) -> int:
    api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
    # exist_ok leaves a pre-existing repo as it was; logs of our own traces must not go to a public one.
    if not api.dataset_info(repo).private:
        raise SystemExit(f"{repo} is not private: refusing to push logs to it")
    n = 0
    for path in wanted(log_dir):
        name = str(path.relative_to(log_dir))
        if already_there(api, repo, name, path.stat().st_size):
            continue
        api.upload_file(path_or_fileobj=str(path), path_in_repo=f"droplet-logs/{name}",
                        repo_id=repo, repo_type="dataset")
        n += 1
    return n


def verify(api, repos: list[str]) -> list[str]:
    """Repos that do NOT list both adapter files."""
    bad = []
    for repo in repos:
        try:
            files = set(api.list_repo_files(repo))
        except Exception:  # noqa: BLE001
            files = set()
        if not {"adapter_config.json", "adapter_model.safetensors"} <= files:
            bad.append(repo)
    return bad


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", help="private dataset repo for logs")
    ap.add_argument("--log-dir", type=Path, default=Path("/var/log/smol-ladder"))
    ap.add_argument("--verify", nargs="*", default=[], help="model repos that must hold an adapter")
    args = ap.parse_args()
    from huggingface_hub import HfApi
    api = HfApi()
    if args.repo:
        print(f"uploaded {push(api, args.repo, args.log_dir)} log files to {args.repo}")
    if args.verify:
        bad = verify(api, args.verify)
        for repo in args.verify:
            print(f"  {'FAIL' if repo in bad else 'PASS'}  adapter readable on the Hub: {repo}")
        return 1 if bad else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
