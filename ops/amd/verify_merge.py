"""Prove a merged checkpoint equals base + adapter, independently of how it was merged.

    python ops/amd/verify_merge.py --base Qwen/Qwen3.5-2B --adapter runs/sft_a --merged runs/merged_a \
        --prompt-rows data/train/sft_upstream/train.jsonl

Needs torch, transformers and peft (the training venv; a CPU is enough for a 2B model). Checks:

1. every LoRA-targeted tensor equals base + (alpha/r) * B @ A, computed here from the raw adapter
   safetensors in float32 (no peft), to within bfloat16 rounding;
2. every other tensor is bit-identical to the base, and names, shapes and dtypes are the base's;
3. the merged checkpoint's logits on a real training prompt (float32, CPU) match those of base +
   adapter loaded through peft on the multimodal class the adapter was trained on (its keys are
   `model.language_model.*` and `model.visual.*`). Two comparisons: the same merge without the
   rounding of each weight to bfloat16 must equal peft to float32 noise (this checks the formula),
   and the merged checkpoint itself must be an order of magnitude closer to peft than the base is
   (what is left is that storage rounding).

Prints one JSON object and exits non-zero on any failed check.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ops.amd import merge_adapter as M  # noqa: E402


def check_tensors(base_dir: Path, merged: Path, adapter: Path) -> dict:
    import numpy as np

    cfg, mods = M.load_adapter(adapter)
    scale = cfg["lora_alpha"] / cfg["r"]
    base, out = M.tensor_map(base_dir), M.tensor_map(merged)
    targeted, worst, worst_name = {}, 0.0, ""
    for mod, ab in mods.items():
        t = M.resolve_target(mod, set(base))
        if t is None:
            raise SystemExit(f"adapter module {mod} has no base tensor")
        targeted[t] = ab
    res = {"key_set_equal": set(base) == set(out), "targeted": len(targeted), "untargeted_identical": True,
           "max_abs_err_vs_formula": 0.0, "max_rel_err_vs_formula": 0.0}

    def raw(table, n):
        sh, p = table[n]
        with open(p, "rb") as f:
            a, b = sh.span(n)
            f.seek(a)
            return f.read(b - a)

    for n in base:
        if n not in targeted:
            if raw(base, n) != raw(out, n):
                res["untargeted_identical"] = False
                res.setdefault("untargeted_changed", []).append(n)
            continue
        dt, shp = base[n][0].signature(n)
        assert out[n][0].signature(n) == (dt, shp), n
        w = M.decode(raw(base, n), dt, shp)
        got = M.decode(raw(out, n), dt, shp)
        want = w + scale * (targeted[n]["B"] @ targeted[n]["A"])
        err = float(np.abs(got - want).max())
        # one bfloat16 ulp of the largest value is the most rounding can cost
        bound = float(np.abs(want).max()) * 2 ** -8
        res["max_abs_err_vs_formula"] = max(res["max_abs_err_vs_formula"], err)
        res["max_rel_err_vs_formula"] = max(res["max_rel_err_vs_formula"], err / max(float(np.abs(want).max()), 1e-30))
        if err > bound:
            worst, worst_name = err, n
    res["formula_ok"] = not worst_name
    if worst_name:
        res["formula_worst"] = [worst_name, worst]
    return res


def prompt_ids(tok, rows_path: Path, max_tokens: int):
    """First training row, cut before its first assistant turn, rendered like training."""
    row = json.loads(next(open(rows_path)))
    msgs = [m for m in row["messages"]]
    cut = next(i for i, m in enumerate(msgs) if m["role"] == "assistant")
    prompt = [{"role": m["role"], "content": m["content"]} for m in msgs[:cut]]
    ids = tok.apply_chat_template(prompt, tools=row.get("tools") or None, add_generation_prompt=True,
                                  enable_thinking=False, return_tensors="pt", return_dict=True)["input_ids"]
    return ids[:, :max_tokens]


def logits_of(model, ids):
    import torch
    with torch.no_grad():
        return model(input_ids=ids).logits.float()


def check_logits(base: str, adapter: Path, merged: Path, rows: Path) -> dict:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base)
    ids = prompt_ids(tok, rows, 1024)
    res = {"prompt_tokens": int(ids.shape[1])}
    # float32 activations and float32 copies of the (bfloat16-stored) weights: the only difference
    # left between the merge and base+adapter is the merge's rounding of each weight to bfloat16
    dtype = torch.float32
    base_m = AutoModelForImageTextToText.from_pretrained(base, dtype=dtype).eval()
    l_base = logits_of(base_m, ids)
    peft_m = PeftModel.from_pretrained(base_m, str(adapter)).eval()
    injected = sum(1 for n, m in peft_m.named_modules() if n.endswith("lora_A.default"))
    res["peft_lora_modules_injected"] = injected
    l_peft = logits_of(peft_m, ids)
    del peft_m, base_m
    mm = AutoModelForImageTextToText.from_pretrained(str(merged), dtype=dtype).eval()
    l_merged = logits_of(mm, ids)

    def stats(a, b):
        return {"max_abs": float((a - b).abs().max()), "mean_abs": float((a - b).abs().mean()),
                "argmax_agree": float((a.argmax(-1) == b.argmax(-1)).float().mean()),
                "allclose_atol_0.25": bool(torch.allclose(a, b, atol=0.25, rtol=0.02))}

    del mm
    # The same merge with no bfloat16 rounding of the result (base + scale * B @ A, float32, from
    # the raw adapter): this isolates the formula from the storage format, and should equal peft's
    # forward pass to float32 noise.
    cfg, mods = M.load_adapter(adapter)
    scale = cfg["lora_alpha"] / cfg["r"]
    ex = AutoModelForImageTextToText.from_pretrained(base, dtype=dtype).eval()
    params = dict(ex.named_parameters())
    for mod, ab in mods.items():
        p = params[M.resolve_target(mod, set(params))]
        p.data += torch.from_numpy(scale * (ab["B"] @ ab["A"]))
    res["unrounded_merge_vs_peft"] = stats(logits_of(ex, ids), l_peft)
    res["merged_vs_peft"] = stats(l_merged, l_peft)
    res["base_vs_peft"] = stats(l_base, l_peft)
    res["merged_vs_base"] = stats(l_merged, l_base)
    # What remains is the rounding of each merged weight to bfloat16 (2**-9 relative), so the merge
    # is accepted when it is an order of magnitude closer to base+adapter than the base is, and
    # picks the same next token almost everywhere
    res["logits_ok"] = (res["merged_vs_peft"]["mean_abs"] < 0.1 * res["base_vs_peft"]["mean_abs"]
                        and res["merged_vs_peft"]["argmax_agree"] > 0.95
                        and res["unrounded_merge_vs_peft"]["max_abs"] < 0.05)
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True, type=Path)
    ap.add_argument("--merged", required=True, type=Path)
    ap.add_argument("--prompt-rows", type=Path, default=Path("data/train/sft_upstream/train.jsonl"))
    ap.add_argument("--skip-logits", action="store_true")
    args = ap.parse_args()
    base_dir = M.resolve_base(args.base)
    out = {"tensors": check_tensors(base_dir, args.merged, args.adapter)}
    ok = out["tensors"]["key_set_equal"] and out["tensors"]["untargeted_identical"] and out["tensors"]["formula_ok"]
    if not args.skip_logits:
        out["logits"] = check_logits(args.base, args.adapter, args.merged, args.prompt_rows)
        ok = ok and out["logits"]["logits_ok"]
    out["ok"] = bool(ok)
    print(json.dumps(out, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
