"""ops/amd/merge_adapter.py on a tiny random checkpoint and a tiny random LoRA, built here.

No downloads and no torch: the merge reads and writes raw safetensors, so a few KB of numpy is
enough to exercise the same code path the 2B checkpoint takes.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from ops.amd import merge_adapter as M

R, ALPHA = 4, 8
rng = np.random.default_rng(0)


def write_safetensors(path: Path, tensors: dict[str, tuple[str, np.ndarray]]) -> None:
    """tensors: name -> (safetensors dtype, float32 array); stored as that dtype."""
    header, blobs, off = {"__metadata__": {"format": "pt"}}, [], 0
    for name, (dtype, a) in tensors.items():
        raw = M.encode(a, dtype)
        header[name] = {"dtype": dtype, "shape": list(a.shape), "data_offsets": [off, off + len(raw)]}
        blobs.append(raw)
        off += len(raw)
    h = json.dumps(header).encode()
    path.write_bytes(len(h).to_bytes(8, "little") + h + b"".join(blobs))


def make_base(d: Path) -> dict[str, np.ndarray]:
    """Two shards + index, like the real one: language tower, a vision tensor, an mtp tensor."""
    d.mkdir(parents=True)
    W = {
        "model.language_model.layers.0.mlp.gate_proj.weight": rng.normal(size=(12, 8)).astype(np.float32),
        "model.language_model.layers.0.self_attn.q_proj.weight": rng.normal(size=(8, 8)).astype(np.float32),
        "model.language_model.layers.0.input_layernorm.weight": rng.normal(size=(8,)).astype(np.float32),
        "model.language_model.embed_tokens.weight": rng.normal(size=(16, 8)).astype(np.float32),
        "model.visual.blocks.0.attn.qkv.weight": rng.normal(size=(24, 8)).astype(np.float32),
        "mtp.fc.weight": rng.normal(size=(8, 16)).astype(np.float32),
    }
    names = list(W)
    shards = {"model-00001-of-00002.safetensors": names[:3], "model-00002-of-00002.safetensors": names[3:]}
    for f, ns in shards.items():
        write_safetensors(d / f, {n: ("BF16", W[n]) for n in ns})
    wm = {n: f for f, ns in shards.items() for n in ns}
    (d / M.INDEX).write_text(json.dumps({"metadata": {}, "weight_map": wm}))
    (d / "config.json").write_text(json.dumps({"architectures": ["Qwen3_5ForConditionalGeneration"]}))
    (d / "tokenizer.json").write_text("{}")
    # what the base's own weights decode to, i.e. the values the merge starts from
    base = M.tensor_map(d)
    out = {}
    for n, (sh, p) in base.items():
        with open(p, "rb") as fh:
            data = fh.read()
        a, b = sh.span(n)
        dt, shp = sh.signature(n)
        out[n] = M.decode(data[a:b], dt, shp)
    return out


def make_adapter(d: Path, base: dict[str, np.ndarray], modules: list[str], *, prefix="base_model.model.",
                 zero_b=False) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    d.mkdir(parents=True)
    ab, tensors = {}, {}
    for mod in modules:
        out_f, in_f = base[mod + ".weight"].shape
        A = rng.normal(size=(R, in_f)).astype(np.float32)
        B = np.zeros((out_f, R), np.float32) if zero_b else rng.normal(size=(out_f, R)).astype(np.float32)
        ab[mod] = (A, B)
        tensors[f"{prefix}{mod}.lora_A.weight"] = ("F32", A)
        tensors[f"{prefix}{mod}.lora_B.weight"] = ("F32", B)
    write_safetensors(d / "adapter_model.safetensors", tensors)
    (d / "adapter_config.json").write_text(json.dumps(
        {"peft_type": "LORA", "r": R, "lora_alpha": ALPHA, "bias": "none", "target_modules": ["gate_proj"]}))
    return ab


TARGETS = ["model.language_model.layers.0.mlp.gate_proj", "model.language_model.layers.0.self_attn.q_proj",
           "model.visual.blocks.0.attn.qkv"]


@pytest.fixture()
def env(tmp_path):
    base_dir = tmp_path / "base"
    base = make_base(base_dir)
    return tmp_path, base_dir, base


def tensors_of(d: Path) -> dict[str, bytes]:
    out = {}
    for n, (sh, p) in M.tensor_map(d).items():
        data = p.read_bytes()
        a, b = sh.span(n)
        out[n] = data[a:b]
    return out


def test_a_correct_merge_adds_scaled_ba_to_exactly_the_targeted_tensors(env):
    tmp, base_dir, base = env
    ab = make_adapter(tmp / "ad", base, TARGETS)
    out = tmp / "out"
    assert M.main(["--base", str(base_dir), "--adapter", str(tmp / "ad"), "--out", str(out)]) == 0
    rep = json.loads((out / M.REPORT).read_text())
    assert rep["ok"] and rep["modules_applied"] == 3 and rep["lora_modules_in_adapter"] == 3
    assert rep["tensors_changed"] == 3 and rep["tensors_total"] == 6 and rep["max_relative_delta"] > 0
    assert rep["modules_unmatched"] == [] and rep["scale"] == ALPHA / R
    assert len(rep["adapter_sha256"]) == 64

    got, want_base = tensors_of(out), tensors_of(base_dir)
    for mod, (A, B) in ab.items():
        n = mod + ".weight"
        expect = base[n] + (ALPHA / R) * B @ A
        merged = M.decode(got[n], "BF16", expect.shape)
        # bfloat16 rounds to a relative 2**-8 per element
        np.testing.assert_allclose(merged, expect, rtol=2 ** -7, atol=1e-6)
        assert got[n] != want_base[n]
    for n in set(want_base) - {m + ".weight" for m in TARGETS}:
        assert got[n] == want_base[n], n          # bit-identical, mtp.* and embeddings included


def test_the_output_has_the_bases_headers_index_and_config(env):
    tmp, base_dir, base = env
    make_adapter(tmp / "ad", base, TARGETS)
    out = tmp / "out"
    assert M.main(["--base", str(base_dir), "--adapter", str(tmp / "ad"), "--out", str(out)]) == 0
    for f in base_dir.iterdir():
        if f.suffix == ".safetensors":
            a, b = M.Shard(f), M.Shard(out / f.name)
            assert (a.header, a.start, a.meta) == (b.header, b.start, b.meta)
        else:
            assert (out / f.name).read_bytes() == f.read_bytes()
    assert {p.name for p in out.glob("*.safetensors")} == {p.name for p in base_dir.glob("*.safetensors")}


def test_a_text_only_adapter_with_model_layers_names_lands_on_the_checkpoints_language_model_names(env):
    tmp, base_dir, base = env
    d = tmp / "adapter_short"
    d.mkdir()
    A = rng.normal(size=(R, 8)).astype(np.float32)
    B = rng.normal(size=(12, R)).astype(np.float32)
    write_safetensors(d / "adapter_model.safetensors", {
        "base_model.model.model.layers.0.mlp.gate_proj.lora_A.weight": ("F32", A),
        "base_model.model.model.layers.0.mlp.gate_proj.lora_B.weight": ("F32", B)})
    (d / "adapter_config.json").write_text(json.dumps({"peft_type": "LORA", "r": R, "lora_alpha": ALPHA}))
    out = tmp / "out"
    assert M.main(["--base", str(base_dir), "--adapter", str(d), "--out", str(out)]) == 0
    n = "model.language_model.layers.0.mlp.gate_proj.weight"
    expect = base[n] + (ALPHA / R) * B @ A
    np.testing.assert_allclose(M.decode(tensors_of(out)[n], "BF16", expect.shape), expect, rtol=2 ** -7, atol=1e-6)


def test_an_adapter_none_of_whose_modules_match_exits_non_zero_and_reports_it(env, capsys):
    tmp, base_dir, base = env
    d = tmp / "ad"
    make_adapter(d, {"ghost.layers.0.gate_proj.weight": base[TARGETS[0] + ".weight"]}, ["ghost.layers.0.gate_proj"])
    out = tmp / "out"
    assert M.main(["--base", str(base_dir), "--adapter", str(d), "--out", str(out)]) == 1
    rep = json.loads((out / M.REPORT).read_text())
    assert rep["ok"] is False and rep["modules_applied"] == 0
    assert any("zero LoRA modules" in f for f in rep["failures"])
    assert "MERGE FAILED" in capsys.readouterr().err
    assert M.check(out)   # and the directory is not servable


def test_a_merge_that_leaves_the_weights_equal_to_the_base_is_a_failure(env):
    tmp, base_dir, base = env
    make_adapter(tmp / "ad", base, TARGETS, zero_b=True)   # B = 0: every tensor merges to itself
    out = tmp / "out"
    assert M.main(["--base", str(base_dir), "--adapter", str(tmp / "ad"), "--out", str(out)]) == 1
    rep = json.loads((out / M.REPORT).read_text())
    assert rep["modules_applied"] == 3 and rep["tensors_changed"] == 0
    assert any("no-op" in f for f in rep["failures"])


def test_partial_matches_fail_too(env):
    tmp, base_dir, base = env
    d = tmp / "ad"
    make_adapter(d, base, TARGETS)
    # rename one module so it matches nothing
    st = d / "adapter_model.safetensors"
    st.write_bytes(st.read_bytes().replace(b"layers.0.self_attn.q_proj", b"layers.9.self_attn.q_proj"))
    assert M.main(["--base", str(base_dir), "--adapter", str(d), "--out", str(tmp / "out")]) == 1
    rep = json.loads((tmp / "out" / M.REPORT).read_text())
    assert rep["modules_applied"] == 2 and len(rep["modules_unmatched"]) == 1


def test_the_old_failure_a_base_weight_shard_beside_the_merge_is_caught_by_the_loader_view(env):
    """A directory whose index points at the BASE's shards (what the old script produced)."""
    tmp, base_dir, base = env
    make_adapter(tmp / "ad", base, TARGETS)
    out = tmp / "out"
    assert M.main(["--base", str(base_dir), "--adapter", str(tmp / "ad"), "--out", str(out)]) == 0
    assert M.check(out) == []
    # reproduce it: the merged tensors live in a file the index does not name, the base's are named
    shutil.copyfile(out / "model-00001-of-00002.safetensors", out / "model.safetensors")
    for f in base_dir.glob("*.safetensors"):
        shutil.copyfile(f, out / f.name)
    why = M.check(out)
    assert why and any("not the ones the report was written for" in w for w in why)
    assert any("index does not name" in w for w in why)
    rep = M.compare(base_dir, out, set())
    assert rep["tensors_changed"] == 0 and rep["stray_weight_files"] == ["model.safetensors"]


def test_a_rerun_removes_the_previous_merges_weights_from_the_output(env):
    tmp, base_dir, base = env
    make_adapter(tmp / "ad", base, TARGETS)
    out = tmp / "out"
    out.mkdir()
    (out / "model.safetensors").write_bytes(b"stale")
    assert M.main(["--base", str(base_dir), "--adapter", str(tmp / "ad"), "--out", str(out)]) == 0
    assert not (out / "model.safetensors").exists()


def test_check_requires_a_report_and_the_files_it_was_written_for(env):
    tmp, base_dir, base = env
    make_adapter(tmp / "ad", base, TARGETS)
    out = tmp / "out"
    assert M.main(["--base", str(base_dir), "--adapter", str(tmp / "ad"), "--out", str(out)]) == 0
    assert M.main(["--check", str(out)]) == 0
    assert M.main(["--check", str(base_dir)]) == 1            # a plain checkpoint has no report
    shard = out / "model-00002-of-00002.safetensors"
    shard.write_bytes(shard.read_bytes() + b"\0")             # touched after the merge
    assert M.main(["--check", str(out)]) == 1
    (out / M.REPORT).unlink()
    assert M.main(["--check", str(out)]) == 1


def test_unsupported_adapter_features_are_refused_rather_than_ignored(env):
    tmp, base_dir, base = env
    make_adapter(tmp / "ad", base, TARGETS)
    cfg = json.loads((tmp / "ad" / "adapter_config.json").read_text())
    cfg["use_dora"] = True
    (tmp / "ad" / "adapter_config.json").write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="use_dora"):
        M.merge(str(base_dir), tmp / "ad", tmp / "out")


def test_bf16_rounding_is_round_to_nearest_even():
    one = np.float32(1.0)
    just_above = np.float32(1.0 + 2 ** -8)         # a tie between 1.0 and 1 + 2**-7: rounds to even (1.0)
    assert M.decode(M.encode(np.array([just_above]), "BF16"), "BF16", (1,))[0] == one
    x = np.array([0.1234567, -3.5, 1e-3], np.float32)
    back = M.decode(M.encode(x, "BF16"), "BF16", x.shape)
    np.testing.assert_allclose(back, x, rtol=2 ** -8)


def test_check_refuses_a_merge_made_from_a_different_adapter_than_the_one_given(env):
    # a retrained adapter must not be served as the old merge
    tmp, base_dir, base = env
    make_adapter(tmp / "ad", base, TARGETS)
    out = tmp / "out"
    assert M.main(["--base", str(base_dir), "--adapter", str(tmp / "ad"), "--out", str(out)]) == 0
    assert M.check(out, tmp / "ad") == []
    make_adapter(tmp / "retrained", base, TARGETS)             # same shapes, new weights
    why = M.check(out, tmp / "retrained")
    assert why and "different adapter" in why[0]
    assert M.main(["--check", str(out), "--adapter", str(tmp / "retrained")]) == 1
    assert M.main(["--check", str(out), "--adapter", str(tmp / "ad")]) == 0
    assert M.check(out) == []                                   # without an adapter, as before


def test_check_refuses_when_the_adapter_to_compare_with_is_unreadable(env):
    tmp, base_dir, base = env
    make_adapter(tmp / "ad", base, TARGETS)
    out = tmp / "out"
    assert M.main(["--base", str(base_dir), "--adapter", str(tmp / "ad"), "--out", str(out)]) == 0
    assert M.check(out, tmp / "nowhere")
