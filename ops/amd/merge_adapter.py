"""Merge a LoRA adapter into the base weights and save a standalone model.

    python ops/amd/merge_adapter.py --base Qwen/Qwen3.5-2B --adapter runs/sft_a --out runs/merged_amd-a-2b

Only the FALLBACK serve mode (`serve.sh --merged`) uses this, for the case where the vLLM build on
the image cannot load LoRA adapters for Qwen3.5. A merged 2B model is 4.6 GB, so four of them on
one 288 GB card is no hardship; the cost is one extra minute per adapter and four server
processes instead of one.
"""

from __future__ import annotations

import argparse


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText

    # The full multimodal class, not AutoModelForCausalLM: that loads the text tower alone, and
    # saving it writes model_type qwen3_5_text with different weight names, which vLLM 0.17 (which
    # runs on transformers 4.x) cannot load. The adapter's keys (model.language_model.*) match this one.
    model = AutoModelForImageTextToText.from_pretrained(args.base, dtype=torch.bfloat16)
    merged = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
    merged.save_pretrained(args.out)
    # Everything that is not a weight (config, tokenizer, preprocessor files) comes from the base
    # as published: the transformers 5 copies of those files are not readable by vLLM's 4.x.
    import shutil
    from pathlib import Path
    from huggingface_hub import snapshot_download
    base_dir = snapshot_download(args.base, ignore_patterns=["*.safetensors", "*.bin", "*.pt", "*.msgpack"])
    for f in Path(base_dir).iterdir():
        if f.is_file():
            shutil.copyfile(f, Path(args.out) / f.name)
    print("merged ->", args.out)


if __name__ == "__main__":
    main()
