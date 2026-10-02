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
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16)
    merged = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
    merged.save_pretrained(args.out)
    AutoTokenizer.from_pretrained(args.base).save_pretrained(args.out)
    print("merged ->", args.out)


if __name__ == "__main__":
    main()
