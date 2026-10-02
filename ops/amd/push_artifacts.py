"""Push the droplet's logs and measurements to the private Hub dataset repo, and verify adapters.

    python ops/amd/push_artifacts.py --repo ns/smol-ladder-runs-s2 --log-dir /var/log/smol-ladder
    python ops/amd/push_artifacts.py --verify ns/smol-ladder-sft-a-s2=runs/sft_a/adapter_model.safetensors

Runs on the teardown path, possibly while racing a shutdown, so it is written to be re-runnable: a
file the Hub already has at the same size is skipped. Adapters and checkpoints do not go through
here: the trainer pushes those to each arm's own private model repo (hub_strategy="checkpoint")
and run_sft.sh pushes the final adapter, so this only carries what the trainer does not: logs,
the smoke's measurements, the checksum list.

`--verify` is the check that matters before a destroy: each named model repo must hold the FINAL
adapter (final.done marker, sha256 equal to the local file's), not merely two files named like it. A push that returned success but wrote nothing readable is invisible until the
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


def sha256_of(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(api, adapters: dict[str, Path]) -> list[str]:
    """Repos whose Hub adapter is not THE final one: `adapters` maps repo -> the local final
    adapter file. A repo passes only with both adapter files, the `final.done` marker (pushed last,
    after a verified upload) and a root adapter whose sha256 equals the local file's. Two files
    alone prove nothing: a checkpoint's adapter, or an earlier session's invalid one, has them too."""
    bad = []
    for repo, local in adapters.items():
        try:
            files = set(api.list_repo_files(repo))
            ok = {"adapter_config.json", "adapter_model.safetensors", "final.done"} <= files
            ok = ok and api.get_paths_info(repo, ["adapter_model.safetensors"])[0].lfs.sha256 == sha256_of(local)
        except Exception:  # noqa: BLE001 - unreadable means unverified
            ok = False
        if not ok:
            bad.append(repo)
    return bad


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", help="private dataset repo for logs")
    ap.add_argument("--log-dir", type=Path, default=Path("/var/log/smol-ladder"))
    ap.add_argument("--verify", nargs="*", default=[],
                    help="REPO=LOCAL_ADAPTER_FILE pairs: each repo must hold that final adapter")
    args = ap.parse_args()
    from huggingface_hub import HfApi
    api = HfApi()
    if args.repo:
        print(f"uploaded {push(api, args.repo, args.log_dir)} log files to {args.repo}")
    if args.verify:
        pairs = dict(spec.split("=", 1) for spec in args.verify)
        bad = verify(api, {repo: Path(local) for repo, local in pairs.items()})
        for repo in pairs:
            print(f"  {'FAIL' if repo in bad else 'PASS'}  adapter readable on the Hub: {repo}")
        return 1 if bad else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
