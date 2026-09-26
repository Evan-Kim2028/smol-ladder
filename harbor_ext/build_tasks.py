"""Turn SmolDataEnvs rows into Harbor tasks.

The agent's container holds only the task's tables at /app/input. The gold answer and the
grader live in tests/, which Harbor mounts only for the verifier, after the agent finishes.
The agent's network is limited to the Command Code API; the verifier has no network.

    uv run python -m harbor_ext.build_tasks --split test --task-ids a,b,c
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

from smol_ladder.gen_solutions import PROMPT
from smol_ladder.tasks import DATA, input_dir, load_split

BASE_IMAGE = "smol-ladder-base:1"
OUT = DATA / "harbor_tasks"

TASK_TOML = """schema_version = "1.3"

[metadata]
category = "data-analysis"
tags = ["smoldataenvs", "{split}", "{tier}"]

[agent]
network_mode = "allowlist"
allowed_hosts = ["commandcode.ai", "*.commandcode.ai"]
timeout_sec = 900.0

[verifier]
network_mode = "no-network"
timeout_sec = 300.0

[environment]
build_timeout_sec = 600.0
network_mode = "public"
cpus = 2
memory_mb = 4096
"""

DOCKERFILE = f"""FROM {BASE_IMAGE}
COPY input/ /app/input/
RUN chmod -R a-w /app/input
WORKDIR /app
"""

TEST_SH = """#!/bin/bash
# Re-run the agent's solution.py offline and grade its last printed line.
mkdir -p /logs/verifier
cd /app
if [ -f solution.py ]; then
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 timeout 120 python solution.py \\
    > /logs/verifier/solution_stdout.txt 2> /logs/verifier/solution_stderr.txt
fi
python /tests/grade.py
"""

GRADE_PY = """import importlib.util, json, sys
from pathlib import Path

spec = importlib.util.spec_from_file_location("smoldataenvs_grader", "/tests/grader.py")
mod = importlib.util.module_from_spec(spec)
sys.modules["smoldataenvs_grader"] = mod
spec.loader.exec_module(mod)

row = json.loads(Path("/tests/gold.json").read_text())
out = Path("/logs/verifier/solution_stdout.txt")
lines = [l.strip() for l in (out.read_text() if out.exists() else "").splitlines() if l.strip()]
pred = lines[-1] if lines else ""
reward = 0.0
if pred:
    r = mod.grade(row["answer"], pred, reward_mode=row["reward_mode"],
                  abs_tol=row["atol"], rel_tol=row["rtol"])
    reward = float(r.reward)
Path("/logs/verifier/reward.txt").write_text(str(reward))
Path("/logs/verifier/prediction.txt").write_text(pred)
print(f"prediction={pred!r} reward={reward}")
"""


def build(row: dict, split: str) -> Path:
    task = OUT / split / row["task_id"]
    if task.exists():
        shutil.rmtree(task)
    (task / "environment").mkdir(parents=True)
    (task / "tests").mkdir()

    (task / "task.toml").write_text(TASK_TOML.format(split=split, tier=row["difficulty_tier"]))
    files = "\n".join(f"- {f}" for f in row["files"])
    (task / "instruction.md").write_text(PROMPT.format(question=row["question"], files=files))

    # Hard links: the build context costs no extra disk.
    src = input_dir(row)
    dst = task / "environment" / "input"
    dst.mkdir()
    for f in src.iterdir():
        if not f.name.startswith("."):
            os.link(f, dst / f.name)
    (task / "environment" / "Dockerfile").write_text(DOCKERFILE)

    tests = task / "tests"
    (tests / "test.sh").write_text(TEST_SH)
    (tests / "test.sh").chmod(0o755)
    (tests / "grade.py").write_text(GRADE_PY)
    shutil.copy(hf_hub_download("FineEnvs/SmolDataEnvs", "grader.py", repo_type="dataset"),
                tests / "grader.py")
    gold = {k: row[k] for k in ("task_id", "answer", "reward_mode", "atol", "rtol")}
    (tests / "gold.json").write_text(json.dumps(gold))
    return task


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--task-ids", help="comma-separated; default all")
    args = ap.parse_args()
    rows = load_split(args.split)
    if args.task_ids:
        want = set(args.task_ids.split(","))
        rows = [r for r in rows if r["task_id"] in want]
    for r in rows:
        print(build(r, args.split))


if __name__ == "__main__":
    main()
