"""Merge a LoRA adapter into a copy of the base checkpoint, and PROVE that it took.

    python ops/amd/merge_adapter.py --base Qwen/Qwen3.5-2B --adapter runs/sft_a --out runs/merged_amd-a-2b
    python ops/amd/merge_adapter.py --check runs/merged_amd-a-2b      # exit 0 only for a passing merge

Only the FALLBACK serve mode (`serve.sh --merged`) uses this, for the case where the vLLM build on
the image cannot load LoRA adapters for Qwen3.5.

## How it merges, and why not through peft

The output is the base checkpoint, file for file. Each base weight file is copied, and the bytes
of exactly the tensors a LoRA module targets are overwritten in place with

    W' = round_to_dtype(W + (lora_alpha / r) * B @ A)        (computed in float32)

so the header of every shard (tensor names, shapes, dtypes, order, offsets), the index, the config
and the tokenizer are the base's own, every untargeted tensor (including the `mtp.*` ones that
`save_pretrained` drops) is bit-identical, and vLLM loads the result exactly as it loads the base.
Nothing here needs torch, transformers or peft: the adapter is read as the raw safetensors it is.

## The failure this replaces

The previous version merged with peft and `save_pretrained`, then copied "everything that is not a
weight" from the base's Hub snapshot with `iterdir()`. The snapshot directory also holds the base's
OWN weights (`model.safetensors-00001-of-00001.safetensors` and `model.safetensors.index.json`,
because from_pretrained had just downloaded them), so those were copied into the output. vLLM
trusts the index, so it loaded the base weights from the merged directory: a no-op that looked like
a merge. The merge itself had worked (204 tensors differed in the file it wrote); the directory
served the wrong file. Hence the checks below read back what the LOADER will read, not what was
written, and `serve.sh --merged` refuses a directory without a passing merge_report.json.

## Exit status

Non-zero if no LoRA module was applied, if any adapter module found no tensor in the base, if the
output differs from the base in tensor names, shapes or dtypes, if any tensor the adapter does not
target changed, or if the merged weights the loader reads equal the base's.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import mmap
import shutil
import sys
from pathlib import Path

import numpy as np

REPORT = "merge_report.json"
INDEX = "model.safetensors.index.json"
_DTYPES = {"F32": np.float32, "F16": np.float16, "BF16": np.uint16}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


# ── safetensors, read raw (the format is an 8-byte length, a JSON header, then the bytes) ──────

class Shard:
    def __init__(self, path: Path):
        self.path = Path(path)
        with open(self.path, "rb") as f:
            n = int.from_bytes(f.read(8), "little")
            self.header = json.loads(f.read(n))
        self.start = 8 + n
        self.meta = self.header.pop("__metadata__", None)

    def names(self) -> list[str]:
        return list(self.header)

    def span(self, name: str) -> tuple[int, int]:
        a, b = self.header[name]["data_offsets"]
        return self.start + a, self.start + b

    def raw(self, name: str, mm: mmap.mmap) -> bytes:
        a, b = self.span(name)
        return mm[a:b]

    def signature(self, name: str) -> tuple:
        h = self.header[name]
        return (h["dtype"], tuple(h["shape"]))


def decode(raw: bytes, dtype: str, shape) -> np.ndarray:
    """float32 array from a tensor's raw bytes."""
    if dtype not in _DTYPES:
        raise ValueError(f"cannot merge into a {dtype} tensor")
    a = np.frombuffer(raw, dtype=_DTYPES[dtype])
    if dtype == "BF16":
        a = (a.astype(np.uint32) << 16).view(np.float32)
    return a.astype(np.float32).reshape(shape)


def encode(a: np.ndarray, dtype: str) -> bytes:
    """Raw bytes of a float32 array in `dtype` (bfloat16 rounds to nearest even)."""
    a = np.ascontiguousarray(a, dtype=np.float32)
    if dtype == "BF16":
        u = a.view(np.uint32).astype(np.uint64)
        return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16).tobytes()
    return a.astype(_DTYPES[dtype]).tobytes()


def weight_files(model_dir: Path) -> list[Path]:
    """The weight files a loader reads: the index's shards if there is an index, else every one."""
    model_dir = Path(model_dir)
    if (model_dir / INDEX).exists():
        names = sorted(set(json.loads((model_dir / INDEX).read_text())["weight_map"].values()))
        return [model_dir / n for n in names]
    return sorted(model_dir.glob("*.safetensors"))


def stray_weight_files(model_dir: Path) -> list[str]:
    """Weight files in the directory that the index does not name (a loader ignores or trips on them)."""
    model_dir = Path(model_dir)
    if not (model_dir / INDEX).exists():
        return []
    used = {p.name for p in weight_files(model_dir)}
    return sorted(p.name for p in model_dir.glob("*.safetensors") if p.name not in used)


def tensor_map(model_dir: Path) -> dict[str, tuple[Shard, Path]]:
    out: dict[str, tuple[Shard, Path]] = {}
    for p in weight_files(model_dir):
        sh = Shard(p)
        for n in sh.names():
            out[n] = (sh, p)
    return out


# ── the adapter ────────────────────────────────────────────────────────────────────────────

def load_adapter(adapter_dir: Path) -> tuple[dict, dict[str, dict[str, np.ndarray]]]:
    """(config, {module: {"A": ..., "B": ...}}) with the peft prefix stripped from the module names."""
    adapter_dir = Path(adapter_dir)
    cfg = json.loads((adapter_dir / "adapter_config.json").read_text())
    if cfg.get("peft_type", "LORA") != "LORA":
        raise ValueError(f"not a LoRA adapter: peft_type={cfg.get('peft_type')}")
    for key in ("use_dora", "fan_in_fan_out", "modules_to_save", "rank_pattern", "alpha_pattern",
                "trainable_token_indices", "target_parameters", "lora_bias"):
        if cfg.get(key):
            raise ValueError(f"adapter_config {key}={cfg[key]!r} is not supported by this merge")
    if cfg.get("bias", "none") != "none":
        raise ValueError(f"adapter_config bias={cfg['bias']!r} is not supported by this merge")
    sh = Shard(adapter_dir / "adapter_model.safetensors")
    mods: dict[str, dict[str, np.ndarray]] = {}
    with open(sh.path, "rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        for name in sh.names():
            for tag in ("lora_A", "lora_B"):
                suffix = f".{tag}.weight"
                if name.endswith(suffix):
                    mod = name[:-len(suffix)].removeprefix("base_model.model.")
                    dtype, shape = sh.signature(name)
                    mods.setdefault(mod, {})[tag[-1]] = decode(sh.raw(name, mm), dtype, shape)
                    break
            else:
                raise ValueError(f"adapter tensor {name!r} is not a LoRA A/B weight")
    for mod, ab in mods.items():
        if set(ab) != {"A", "B"}:
            raise ValueError(f"module {mod} has only {sorted(ab)}")
    return cfg, mods


def resolve_target(module: str, names: set[str]) -> str | None:
    """The base tensor a LoRA module's weight lands on, or None.

    An adapter trained on the text-only class names its modules `model.layers.N...`, the
    checkpoint (and the multimodal class) `model.language_model.layers.N...`; both are accepted.
    """
    cands = [module, module.replace("model.", "model.language_model.", 1),
             module.replace("model.language_model.", "model.", 1)]
    for c in cands:
        if c + ".weight" in names:
            return c + ".weight"
    return None


# ── verify what the loader will read ───────────────────────────────────────────────────────────

def compare(base_dir: Path, out_dir: Path, targets: set[str]) -> dict:
    """Read both directories the way a loader does and compare tensor by tensor."""
    base, out = tensor_map(base_dir), tensor_map(out_dir)
    res = {"key_set_equal": set(base) == set(out), "missing_in_output": sorted(set(base) - set(out))[:10],
           "extra_in_output": sorted(set(out) - set(base))[:10], "shapes_dtypes_equal": True,
           "tensors_total": len(base), "tensors_changed": 0, "changed_untargeted": [],
           "targets_unchanged": [], "max_relative_delta": 0.0, "mean_relative_delta": 0.0,
           "stray_weight_files": stray_weight_files(out_dir)}
    mms: dict[Path, mmap.mmap] = {}
    handles = []

    def raw(table, p, name):
        if p not in mms:
            fh = open(p, "rb")
            handles.append(fh)
            mms[p] = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
        return table[name][0].raw(name, mms[p])

    rels = []
    try:
        for name in sorted(set(base) & set(out)):
            sb, pb = base[name]
            so, po = out[name]
            if sb.signature(name) != so.signature(name):
                res["shapes_dtypes_equal"] = False
                continue
            rb, ro = raw(base, pb, name), raw(out, po, name)
            if rb == ro:
                if name in targets:
                    res["targets_unchanged"].append(name)
                continue
            res["tensors_changed"] += 1
            if name not in targets:
                res["changed_untargeted"].append(name)
                continue
            dtype, shape = sb.signature(name)
            w, w2 = decode(rb, dtype, shape), decode(ro, dtype, shape)
            rels.append(float(np.linalg.norm(w2 - w) / max(float(np.linalg.norm(w)), 1e-30)))
    finally:
        for m in mms.values():
            m.close()
        for fh in handles:
            fh.close()
    if rels:
        res["max_relative_delta"] = max(rels)
        res["mean_relative_delta"] = float(np.mean(rels))
    return res


def judge(rep: dict) -> list[str]:
    """Why a merge must not be served; empty means it passes."""
    why = []
    if rep["modules_applied"] == 0:
        why.append("zero LoRA modules were applied")
    if rep["modules_unmatched"]:
        why.append(f"{len(rep['modules_unmatched'])} adapter modules matched no base tensor "
                   f"(e.g. {rep['modules_unmatched'][:3]})")
    if not rep["key_set_equal"]:
        why.append("output tensor names differ from the base's")
    if not rep["shapes_dtypes_equal"]:
        why.append("output tensor shapes or dtypes differ from the base's")
    if rep["changed_untargeted"]:
        why.append(f"{len(rep['changed_untargeted'])} tensors the adapter does not target changed")
    if rep["stray_weight_files"]:
        why.append(f"weight files the index does not name: {rep['stray_weight_files']}")
    if rep["tensors_changed"] == 0:
        why.append("the merged weights the loader reads are IDENTICAL to the base: the merge is a no-op")
    elif rep["max_relative_delta"] <= 0.0:
        why.append("max relative delta is zero")
    return why


# ── merge ────────────────────────────────────────────────────────────────────────────────────

def resolve_base(base: str) -> Path:
    if Path(base).is_dir():
        return Path(base)
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(base, ignore_patterns=["*.bin", "*.pt", "*.msgpack", "*.gguf"]))


def merge(base: str, adapter_dir: Path, out_dir: Path) -> dict:
    base_dir = resolve_base(base).resolve()
    adapter_dir, out_dir = Path(adapter_dir), Path(out_dir)
    if out_dir.resolve() == base_dir:
        raise SystemExit("--out must not be the base directory")
    cfg, mods = load_adapter(adapter_dir)
    r_cfg, alpha = cfg["r"], cfg["lora_alpha"]
    scale = alpha / (r_cfg ** 0.5) if cfg.get("use_rslora") else alpha / r_cfg

    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in [*out_dir.glob("*.safetensors"), out_dir / INDEX, out_dir / REPORT]:
        stale.unlink(missing_ok=True)   # a previous merge's weights must not outlive this one
    # Non-weight files from the base, verbatim. Weight files are NOT copied here: they are copied
    # below, once, from the very files the base's index names.
    for f in base_dir.iterdir():
        if f.is_file() and not f.name.endswith(".safetensors") and f.name != INDEX:
            shutil.copyfile(f, out_dir / f.name)
    if (base_dir / INDEX).exists():
        shutil.copyfile(base_dir / INDEX, out_dir / INDEX)

    base_map = tensor_map(base_dir)
    names = set(base_map)
    plan: dict[Path, list[tuple[str, str]]] = {}   # base shard -> [(module, tensor name)]
    unmatched = []
    for mod in sorted(mods):
        t = resolve_target(mod, names)
        if t is None:
            unmatched.append(mod)
        else:
            plan.setdefault(base_map[t][1], []).append((mod, t))
    targets = {t for v in plan.values() for _, t in v}

    for p in weight_files(base_dir):
        dst = out_dir / p.name
        shutil.copyfile(p, dst)
        todo = plan.get(p, [])
        if not todo:
            continue
        sh = Shard(dst)
        with open(dst, "r+b") as f:
            for mod, t in todo:
                dtype, shape = sh.signature(t)
                a, b = sh.span(t)
                f.seek(a)
                w = decode(f.read(b - a), dtype, shape)
                A, B = mods[mod]["A"], mods[mod]["B"]
                if B.shape[0] != shape[0] or A.shape[1] != shape[1] or A.shape[0] != B.shape[1]:
                    raise SystemExit(f"{mod}: LoRA shapes A{A.shape} B{B.shape} do not fit {t} {tuple(shape)}")
                f.seek(a)
                f.write(encode(w + scale * (B @ A), dtype))

    rep = {"base": base, "base_dir": str(base_dir), "adapter": str(adapter_dir),
           "adapter_sha256": sha256_file(adapter_dir / "adapter_model.safetensors"),
           "lora_r": r_cfg, "lora_alpha": alpha, "scale": scale,
           "lora_modules_in_adapter": len(mods), "modules_applied": len(targets),
           "modules_unmatched": unmatched}
    rep.update(compare(base_dir, out_dir, targets))
    # A targeted tensor can stay equal to the base because its B is exactly zero: LoRA's B starts
    # at zero and only moves under gradient, so the vision tower's modules never train on text-only
    # data and merge to themselves. That is correct, and the report says how many it was.
    same = rep.pop("targets_unchanged")
    zero_b = {t for v in plan.values() for m, t in v if not mods[m]["B"].any()}
    rep["targets_unchanged"] = len(same)
    rep["targets_unchanged_with_zero_B"] = len(set(same) & zero_b)
    rep["targets_unchanged_examples"] = same[:3]
    rep["output_weight_sha256"] = {p.name: sha256_file(p) for p in weight_files(out_dir)}
    rep["failures"] = judge(rep)
    rep["ok"] = not rep["failures"]
    (out_dir / REPORT).write_text(json.dumps(rep, indent=1) + "\n")
    return rep


def check(out_dir: Path, adapter_dir: Path | None = None) -> list[str]:
    """Why `out_dir` may not be served; empty means it holds a passing merge of the files it has now.

    With `adapter_dir`, also that the merge was made FROM that adapter: the report's adapter_sha256
    must equal the adapter file's, or a retrained adapter would be served as the old merge.
    """
    out_dir = Path(out_dir)
    try:
        rep = json.loads((out_dir / REPORT).read_text())
    except (OSError, ValueError):
        return [f"no readable {REPORT} in {out_dir}: not a verified merge"]
    why = []
    if rep.get("ok") is not True:
        why.append(f"{REPORT} records a failed merge: {rep.get('failures')}")
    if not rep.get("modules_applied") or not rep.get("tensors_changed"):
        why.append(f"{REPORT} records no applied modules or no changed tensors")
    have = {p.name: sha256_file(p) for p in weight_files(out_dir) if p.exists()}
    if not have or have != rep.get("output_weight_sha256"):
        why.append("the weight files the loader reads are not the ones the report was written for")
    if stray_weight_files(out_dir):
        why.append(f"weight files the index does not name: {stray_weight_files(out_dir)}")
    if adapter_dir is not None:
        try:
            current = sha256_file(Path(adapter_dir) / "adapter_model.safetensors")
        except OSError as exc:
            why.append(f"cannot read the adapter to compare the merge with: {exc}")
        else:
            if rep.get("adapter_sha256") != current:
                why.append(f"{REPORT} was written from a different adapter (sha256 "
                           f"{str(rep.get('adapter_sha256'))[:12]}, the adapter now is {current[:12]})")
    return why


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base")
    ap.add_argument("--adapter", help="the adapter directory to merge; with --check, the one the "
                                      "merge must have been made from (its sha256 is compared)")
    ap.add_argument("--out")
    ap.add_argument("--check", metavar="DIR", help="verify an existing merged directory and exit")
    ap.add_argument("--label", help="with --check: the served model's name, for the MERGE_OK line "
                                    "the driver reads (it wires the report into the evaluation guard)")
    args = ap.parse_args(argv)
    if args.check:
        why = check(Path(args.check), Path(args.adapter) if args.adapter else None)
        for w in why:
            print(f"merge check FAILED: {w}", file=sys.stderr)
        if not why:
            rep = json.loads((Path(args.check) / REPORT).read_text())
            print(f"merge check ok: {args.check}")
            if args.label:
                print(f"MERGE_OK model={args.label} modules_applied={rep['modules_applied']} "
                      f"tensors_changed={rep['tensors_changed']} "
                      f"max_relative_delta={rep['max_relative_delta']:.3g}")
        return 1 if why else 0
    if not (args.base and args.adapter and args.out):
        ap.error("--base, --adapter and --out are required (or --check DIR)")
    rep = merge(args.base, Path(args.adapter), Path(args.out))
    keys = ("modules_applied", "lora_modules_in_adapter", "tensors_changed", "tensors_total",
            "max_relative_delta", "adapter_sha256")
    print("merge", " ".join(f"{k}={rep[k]}" for k in keys))
    for w in rep["failures"]:
        print(f"MERGE FAILED: {w}", file=sys.stderr)
    if rep["ok"]:
        print("merged ->", args.out)
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
