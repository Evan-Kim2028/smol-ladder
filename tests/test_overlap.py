"""The L1 table that leaves out the test tasks arm A has seen notebooks of.

A test task's question can come from a notebook that also has questions in arm A's training set;
arm A learned from the same data tables and the same author's analysis, so its L1 on those tasks
is not a clean held-out number. The list is computed by code and committed
(ops/amd/overlap_with_arm_a.json); the checkpoint report prints a second table without them.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from ops.amd import overlap as O
from smol_ladder import summarize as S

REPO = Path(__file__).resolve().parent.parent


def test_overlap_is_by_source_notebook_not_by_question():
    test = [{"task_id": "t1", "source_row_id": "0001/1/1.ipynb_qa_1"},
            {"task_id": "t2", "source_row_id": "0002/2/2.ipynb_qa_1"},
            {"task_id": "t3", "source_row_id": "0003/3/3.ipynb_qa_2"}]
    train = {"a": {"source_row_id": "0001/1/1.ipynb_qa_3"}, "b": {"source_row_id": "0009/9/9.ipynb_qa_1"}}
    assert O.overlapping(test, train, ["a", "b"]) == ["t1"]
    assert O.overlapping(test, train, ["b"]) == []          # only arm A's rows count


def test_the_committed_list_has_27_ids_and_says_how_it_was_derived():
    rec = json.loads((REPO / O.FILE).read_text())
    assert len(rec["ids"]) == 27 and rec["ids"] == sorted(rec["ids"])
    assert "source_row_id" in rec["derivation"] and "index.jsonl" in rec["derivation"]


def test_the_committed_list_reproduces_from_the_data_when_it_is_present():
    index = REPO / "data/train/sft_upstream/index.jsonl"
    if not index.exists():
        pytest.skip("data/train/sft_upstream is not present")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    try:
        ids = O.compute(REPO)
    except Exception:  # noqa: BLE001
        pytest.skip("the SmolDataEnvs splits are not in the local Hub cache")
    assert ids == json.loads((REPO / O.FILE).read_text())["ids"]


def trial(reward):
    return {"agent_status": "exit 0", "reward": reward, "prompt_sha256": "p", "ladder_sha256": "l"}


def test_the_summary_adds_a_second_l1_table_without_the_listed_tasks():
    runs = {"t1": {"L1": [trial(1.0)]}, "t2": {"L1": [trial(0.0)]}, "t3": {"L1": [trial(1.0)]}}
    report = S.summarise("test", runs, lambda t: True, exclude_ids={"t1"})
    block = report["without_listed"]
    assert block["excluded_tasks"] == 1 and block["tasks"] == 2
    assert block["rungs"]["L1"]["mean_pass_probability"] == 0.5
    assert report["rungs"]["L1"]["mean_pass_probability"] == pytest.approx(2 / 3)


def test_no_list_no_second_table():
    runs = {"t1": {"L1": [trial(1.0)]}}
    assert "without_listed" not in S.summarise("test", runs, lambda t: True)
