"""The gate's 60 tasks: a fixed seeded sample stratified by difficulty tier, not the first 60.

The first 60 ids of the test split are 25% easy against the split's 13%, which made the gate easier
than the evaluation it vouches for.
"""

from __future__ import annotations

import collections
import json
import os
from pathlib import Path

import pytest

from ops.amd import gate as G
from ops.amd import plan as P

REPO = Path(__file__).resolve().parent.parent


def fake_rows():
    tiers = ["easy"] * 33 + ["medium"] * 118 + ["hard"] * 99
    # interleave so "the first 60" is not already representative
    return [{"task_id": f"t{i:03d}", "difficulty_tier": t} for i, t in enumerate(sorted(tiers, key=lambda t: t != "easy"))]


def test_the_subset_matches_the_splits_tier_proportions_by_largest_remainder():
    ids, meta = G.stratified_subset(fake_rows(), 60, seed=42)
    assert len(ids) == len(set(ids)) == 60
    assert meta["allocation"] == {"easy": 8, "hard": 24, "medium": 28}
    assert meta["split_counts"] == {"easy": 33, "hard": 99, "medium": 118}


def test_the_subset_is_deterministic_and_changes_with_the_seed():
    a, _ = G.stratified_subset(fake_rows(), 60, seed=42)
    assert a == G.stratified_subset(fake_rows(), 60, seed=42)[0]
    assert a != G.stratified_subset(fake_rows(), 60, seed=43)[0]


def test_the_subset_is_in_split_order_and_independent_of_row_order():
    rows = fake_rows()
    ids, _ = G.stratified_subset(rows, 60, seed=42)
    assert ids == sorted(ids)
    assert G.stratified_subset(list(reversed(rows)), 60, seed=42)[0] == ids


def test_a_request_larger_than_the_split_is_refused():
    with pytest.raises(ValueError):
        G.stratified_subset(fake_rows()[:10], 60, seed=1)


def test_the_committed_subset_file_is_what_the_generator_produces_and_the_gate_uses_it():
    meta = json.loads((REPO / G.SUBSET_JSON).read_text())
    ids = (REPO / G.SUBSET_IDS).read_text().split()
    assert ids == meta["ids"] and len(ids) == P.GATE_TASKS == 60
    cmd = P.gate_cmds(P.Config(host="h", fingerprint="f"))[0]
    assert cmd.argv[cmd.argv.index("--task-ids") + 1].endswith(G.SUBSET_IDS)
    assert cmd.argv[cmd.argv.index("--limit") + 1] == "60"


def test_the_committed_subset_reproduces_from_the_dataset_when_it_is_cached():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    try:
        from smol_ladder.tasks import load_split
        rows = load_split("test")
    except Exception:  # noqa: BLE001
        pytest.skip("the SmolDataEnvs test split is not in the local Hub cache")
    meta = json.loads((REPO / G.SUBSET_JSON).read_text())
    ids, again = G.stratified_subset(rows, meta["n"], seed=meta["seed"])
    assert ids == meta["ids"]
    first60 = collections.Counter(r["difficulty_tier"] for r in rows[:60])["easy"]
    chosen = collections.Counter(r["difficulty_tier"] for r in rows if r["task_id"] in set(ids))
    assert chosen["easy"] == 8 and first60 == 15           # 13% of the split, not the first 60's 25%


def test_the_full_l1_run_reuses_the_gates_trials_because_the_run_tags_are_the_same():
    cfg = P.Config(host="h", fingerprint="f")
    gate = {c.argv[c.argv.index("--run-tag") + 1] for c in P.gate_cmds(cfg)}
    l1 = {c.argv[c.argv.index("--run-tag") + 1] for _, cmds in P.eval_phases(cfg)[:1] for c in cmds}
    assert gate <= l1


def test_the_plans_copy_of_the_subset_path_is_the_gates():
    assert P.GATE_SUBSET_IDS == G.SUBSET_IDS


def test_the_gate_records_the_subset_it_ran_beside_its_output(tmp_path):
    import shutil
    from ops.amd import driver
    runs = tmp_path / "runs"
    for tag in ("amd2-base", "amd2-r"):
        for i in ("a", "b"):
            d = runs / tag / "test" / i / "L1"
            d.mkdir(parents=True)
            (d / "result.json").write_text("{}")
    cfg = P.Config(host="h", fingerprint="f", local_logs=str(tmp_path / "logs"))
    driver.record_gate_tasks(cfg, runs)
    rec = json.loads((tmp_path / "logs" / "gate_subset.json").read_text())
    assert rec["seed"] == 42 and rec["allocation"] == {"easy": 8, "hard": 24, "medium": 28}
    assert json.loads((tmp_path / "logs" / "gate_tasks.json").read_text()) == ["a", "b"]
