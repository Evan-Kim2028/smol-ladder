"""The test tasks whose source notebook also has questions in arm A's training set.

    python -m ops.amd.overlap            # rewrites ops/amd/overlap_with_arm_a.json

Arm A trains on SmolDataEnvs-sft's `train.jsonl` (4,439 rows). `data/train/sft_upstream/index.jsonl`
lists the task id of every exported row in file order (train rows first, then the 234 validation
rows); the first 4,439 are arm A's. A test task overlaps when its notebook, the `source_row_id`
without its `_qa_N` suffix, is the notebook of any of those train tasks. The same notebook means
the same tables and the same author's analysis, so arm A's L1 on those tasks is not clean held-out
data. The checkpoint report prints a second L1 table without them (smol_ladder/summarize.py).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

FILE = "ops/amd/overlap_with_arm_a.json"
ARM_A_ROWS = 4439


def notebook(row: dict) -> str:
    return row["source_row_id"].rsplit("_qa_", 1)[0]


def overlapping(test_rows: list[dict], train_by_id: dict[str, dict], arm_a_ids: list[str]) -> list[str]:
    seen = {notebook(train_by_id[t]) for t in arm_a_ids if t in train_by_id}
    return sorted(r["task_id"] for r in test_rows if notebook(r) in seen)


def compute(root: Path) -> list[str]:
    from smol_ladder.tasks import load_split
    index = [json.loads(line)["task_id"] for line in
             (root / "data/train/sft_upstream/index.jsonl").read_text().splitlines() if line.strip()]
    train_rows = {r["task_id"]: r for r in load_split("train")}
    return overlapping(load_split("test"), train_rows, index[:ARM_A_ROWS])


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    ids = compute(root)
    (root / FILE).write_text(json.dumps({
        "n": len(ids), "ids": ids,
        "derivation": "test tasks whose notebook (source_row_id without _qa_N) is the notebook of one of "
                      f"the first {ARM_A_ROWS} task ids of data/train/sft_upstream/index.jsonl "
                      "(arm A's train.jsonl rows, in file order), from the SmolDataEnvs train and test "
                      "splits; ops.amd.overlap.compute"}, indent=1) + "\n")
    print(len(ids), "overlapping test tasks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
