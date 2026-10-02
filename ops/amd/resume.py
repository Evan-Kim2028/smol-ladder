"""Decide how an SFT arm starts: finished, resume from disk, resume from the Hub, or fresh.

    python -m ops.amd.resume status --out runs/sft_a --repo ns/smol-ladder-sft-a
    python -m ops.amd.resume verdict --killed-at 50 --log resume.log

A spot reclaim costs a few minutes only if a restarted arm picks up where it was. Three things
have to be true and none of them is established by reading the trainer:

  1. a checkpoint that was half-written when the process died must not be resumed from (the
     trainer's own `--resume` takes the highest-numbered directory without looking inside it);
  2. on a fresh droplet the disk is gone, so the newest checkpoint is restored from the Hub's
     `last-checkpoint/` folder, which `hub_strategy="every_save"` keeps current;
  3. a finished arm is not trained again.

Everything below takes the Hub as an injected object so the tests run it against a stub.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import struct
import sys
from pathlib import Path

DONE = ".done"
CKPT = re.compile(r"^checkpoint-(\d+)$")
ADAPTER = "adapter_model.safetensors"
# Files a resumable checkpoint must have. rng_state is written last by the trainer, so its
# presence means the directory was finished; any of the other three missing means it was not.
REQUIRED = ("trainer_state.json", ADAPTER, "optimizer.pt", "scheduler.pt")


def safetensors_ok(path: Path) -> bool:
    """A truncated .safetensors fails here: 8-byte header length, JSON header, then tensor data
    whose end offset must equal the file size."""
    try:
        size = path.stat().st_size
        with path.open("rb") as fh:
            (n,) = struct.unpack("<Q", fh.read(8))
            if n <= 0 or 8 + n > size:
                return False
            header = json.loads(fh.read(n))
    except (OSError, struct.error, ValueError):
        return False
    ends = [v["data_offsets"][1] for k, v in header.items() if k != "__metadata__"]
    return bool(ends) and 8 + n + max(ends) == size


def is_complete(path: Path) -> bool:
    if not all((path / name).is_file() and (path / name).stat().st_size > 0 for name in REQUIRED):
        return False
    if not any(path.glob("rng_state*.pth")):
        return False
    return safetensors_ok(path / ADAPTER)


def checkpoints(out: Path) -> list[tuple[int, Path]]:
    found = []
    for p in out.glob("checkpoint-*"):
        m = CKPT.match(p.name)
        if m and p.is_dir():
            found.append((int(m.group(1)), p))
    return sorted(found)


def prune_incomplete(out: Path) -> list[str]:
    """Move incomplete checkpoint directories out of the way so the trainer cannot pick one."""
    moved = []
    for step, path in checkpoints(out):
        if not is_complete(path):
            target = out / f"partial-{step}"
            shutil.rmtree(target, ignore_errors=True)
            path.rename(target)
            moved.append(path.name)
    return moved


def adapter_finished(out: Path) -> bool:
    return (out / DONE).exists() and safetensors_ok(out / ADAPTER) and (out / "adapter_config.json").exists()


def restore_from_hub(out: Path, repo: str, hub) -> int | None:
    """Fetch `last-checkpoint/` into out/checkpoint-<step>. Returns the step, or None if the
    repo has no usable checkpoint. `hub.download(repo, subfolder, dest)` returns the directory."""
    try:
        files = set(hub.list_files(repo))
    except Exception:  # noqa: BLE001 - an unreadable repo means "no checkpoint there"
        return None
    if "last-checkpoint/trainer_state.json" not in files:
        return None
    tmp = out / ".hub-restore"
    shutil.rmtree(tmp, ignore_errors=True)
    src = Path(hub.download(repo, "last-checkpoint", tmp))
    state = json.loads((src / "trainer_state.json").read_text())
    step = int(state["global_step"])
    dest = out / f"checkpoint-{step}"
    shutil.rmtree(dest, ignore_errors=True)
    shutil.move(str(src), str(dest))
    shutil.rmtree(tmp, ignore_errors=True)
    if not is_complete(dest):
        dest.rename(out / f"partial-{step}")
        return None
    return step


def status(out: Path, repo: str = "", hub=None) -> dict:
    """One of: done | resume-local | resume-hub | fresh, after pruning and restoring."""
    out.mkdir(parents=True, exist_ok=True)
    if adapter_finished(out):
        return {"state": "done", "step": None}
    pruned = prune_incomplete(out)
    local = [c for c in checkpoints(out)]
    if local:
        return {"state": "resume-local", "step": local[-1][0], "pruned": pruned}
    if repo and hub is not None:
        step = restore_from_hub(out, repo, hub)
        if step is not None:
            return {"state": "resume-hub", "step": step, "pruned": pruned}
    return {"state": "fresh", "step": None, "pruned": pruned}


def verdict(killed_at: int, log_text: str) -> dict:
    """Did the second run resume from the checkpoint the first one left, and not from zero?
    `train/sft_lora.py` prints `resuming from <path>/checkpoint-<step>` when it does."""
    m = re.findall(r"resuming from .*checkpoint-(\d+)", log_text)
    resumed = int(m[-1]) if m else None
    return {"killed_at": killed_at, "resumed_from": resumed,
            "ok": resumed is not None and resumed > 0 and resumed == killed_at,
            "note": ("resumed from the checkpoint the kill left" if resumed == killed_at
                     else "no resume message" if resumed is None
                     else f"resumed from {resumed}, not {killed_at}")}


class HubStub:  # pragma: no cover - documentation of the interface the tests implement
    def list_files(self, repo: str) -> list[str]: ...
    def download(self, repo: str, subfolder: str, dest: Path) -> Path: ...


class RealHub:
    def __init__(self):
        from huggingface_hub import HfApi, snapshot_download
        self._api, self._snap = HfApi(), snapshot_download

    def list_files(self, repo: str) -> list[str]:
        return self._api.list_repo_files(repo)

    def download(self, repo: str, subfolder: str, dest: Path) -> Path:
        self._snap(repo, allow_patterns=[f"{subfolder}/*"], local_dir=str(dest))
        return Path(dest) / subfolder


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("status")
    s.add_argument("--out", required=True, type=Path)
    s.add_argument("--repo", default="")
    v = sub.add_parser("verdict")
    v.add_argument("--killed-at", type=int, required=True)
    v.add_argument("--log", required=True, type=Path)
    v.add_argument("--out", type=Path)
    args = ap.parse_args()
    if args.cmd == "status":
        res = status(args.out, args.repo, RealHub() if args.repo else None)
        print(json.dumps(res))
        return 0
    res = verdict(args.killed_at, args.log.read_text(errors="replace"))
    if args.out:
        args.out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res))
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
