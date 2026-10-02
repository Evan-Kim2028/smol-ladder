"""A fresh droplet resumes from the Hub only if the trainer put the checkpoint there.

`ops/amd/resume.py` restores `last-checkpoint/` from the arm's Hub repo. Which hub_strategy makes
the Trainer write that folder is not a guess: transformers 5.18's `Trainer._push_from_checkpoint`
uploads the checkpoint folder to `last-checkpoint/` only for "checkpoint" (and to
`checkpoint-<step>/` for "all_checkpoints"); "every_save" pushes the model files and nothing else.
The file lists below were captured from a real `Trainer` (transformers 5.18.0, peft 0.21.2, a
one-layer model, save_steps=2, `upload_folder` replaced by a copy into a directory) for each
strategy; the last test in this file reproduces the capture when torch is installed.
"""

from __future__ import annotations

import json
import shutil
import struct
import sys
import types
from pathlib import Path

import pytest

from ops.amd import resume

# Captured: what the Hub repo holds after training with each strategy.
LAYOUT_EVERY_SAVE = ["adapter_config.json", "adapter_model.safetensors", "training_args.bin"]
LAYOUT_CHECKPOINT = LAYOUT_EVERY_SAVE + [
    "last-checkpoint/README.md", "last-checkpoint/adapter_config.json",
    "last-checkpoint/adapter_model.safetensors", "last-checkpoint/optimizer.pt",
    "last-checkpoint/rng_state.pth", "last-checkpoint/scheduler.pt",
    "last-checkpoint/trainer_state.json", "last-checkpoint/training_args.bin"]


def safetensors(path: Path) -> None:
    header = json.dumps({"w": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}}).encode()
    path.write_bytes(struct.pack("<Q", len(header)) + header + b"\0" * 8)


class LayoutHub:
    """A Hub repo holding exactly `layout`, served the way RealHub serves it: snapshot_download with
    allow_patterns=["last-checkpoint/*"] and local_dir=dest puts the files at dest/last-checkpoint/."""

    def __init__(self, layout: list[str], step: int, tmp: Path):
        self.layout, self.step, self.tmp, self.downloads = layout, step, tmp, 0

    def list_files(self, repo):
        return list(self.layout)

    def download(self, repo, subfolder, dest):
        self.downloads += 1
        for name in self.layout:
            if not name.startswith(subfolder + "/"):
                continue
            target = Path(dest) / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if name.endswith(".safetensors"):
                safetensors(target)
            elif name.endswith("trainer_state.json"):
                target.write_text(json.dumps({"global_step": self.step}))
            else:
                target.write_bytes(b"x")
        return Path(dest) / subfolder


def test_a_fresh_droplet_resumes_from_the_layout_the_checkpoint_strategy_produces(tmp_path):
    hub = LayoutHub(LAYOUT_CHECKPOINT, 250, tmp_path)
    res = resume.status(tmp_path / "fresh-disk", "ns/sft-a", hub)
    assert res["state"] == "resume-hub" and res["step"] == 250 and hub.downloads == 1
    assert resume.is_complete(tmp_path / "fresh-disk" / "checkpoint-250")


def test_the_every_save_layout_has_nothing_to_resume_from(tmp_path):
    # the bug: with hub_strategy="every_save" every arm restarted from step 0 after a reclaim
    hub = LayoutHub(LAYOUT_EVERY_SAVE, 250, tmp_path)
    assert resume.status(tmp_path / "fresh-disk", "ns/sft-a", hub)["state"] == "fresh"
    assert hub.downloads == 0


def test_the_trainer_defaults_to_the_strategy_that_pushes_last_checkpoint():
    from train import sft_lora
    ap_help = __import__("subprocess").run([sys.executable, "-m", "train.sft_lora", "--help"],
                                           capture_output=True, text=True,
                                           cwd=Path(__file__).resolve().parent.parent).stdout
    assert "--hub-strategy" in ap_help
    assert sft_lora.HUB_STRATEGY == "checkpoint"


def _run_wrapper(monkeypatch, **cfg_kwargs):
    seen = {}

    class Cfg:
        def __init__(self, **kw):
            seen.update(kw)
    trl = types.ModuleType("trl")
    trl.SFTConfig = Cfg
    trainer = types.ModuleType("train.sft_lora")
    trainer.main = lambda: __import__("trl").SFTConfig(save_steps=100, **cfg_kwargs)
    pkg = types.ModuleType("train")
    pkg.sft_lora = trainer
    monkeypatch.setitem(sys.modules, "trl", trl)
    monkeypatch.setitem(sys.modules, "train", pkg)
    monkeypatch.setitem(sys.modules, "train.sft_lora", trainer)
    from ops.amd import sft_run
    sft_run.main(["--save-steps", "50"])
    return seen


def test_the_wrapper_lets_a_resumable_strategy_through(monkeypatch):
    seen = _run_wrapper(monkeypatch, push_to_hub=True, hub_strategy="checkpoint")
    assert seen["hub_strategy"] == "checkpoint" and seen["save_steps"] == 50


def test_the_wrapper_refuses_a_pushing_config_whose_strategy_cannot_resume(monkeypatch):
    with pytest.raises(SystemExit, match="last-checkpoint"):
        _run_wrapper(monkeypatch, push_to_hub=True, hub_strategy="every_save")


def test_the_wrapper_does_not_care_when_nothing_is_pushed(monkeypatch):
    assert _run_wrapper(monkeypatch, push_to_hub=False, hub_strategy="every_save")["save_steps"] == 50


def test_the_real_trainer_writes_last_checkpoint_only_for_the_checkpoint_strategy(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("peft")
    import transformers
    assert transformers.__version__.startswith("5."), "captured against transformers 5.x"
    script = Path(__file__).parent / "fixtures" / "trainer_hub_layout.py"
    import subprocess
    for strategy, expected in (("every_save", LAYOUT_EVERY_SAVE), ("checkpoint", LAYOUT_CHECKPOINT)):
        hub = tmp_path / strategy
        hub.mkdir()
        out = subprocess.run([sys.executable, str(script), str(hub), strategy],
                             capture_output=True, text=True)
        assert out.returncode == 0, out.stderr[-800:]
        assert sorted(str(p.relative_to(hub)) for p in hub.rglob("*") if p.is_file()) == sorted(expected)
    shutil.rmtree(tmp_path, ignore_errors=True)
