"""Tasks from jupyter-agent: a second, non-overlapping source for the ladder.

SmolDataEnvs gives 250 test tasks, 181 of which have a verified reference, and the critics were
right that the ladder is underpowered at that size. jupyter-agent is a completely separate
source (51,389 rows, its own Kaggle datasets), so tasks from it cannot overlap SmolDataEnvs by
construction. It is also much noisier, and the noise is the whole problem to solve here.

This module builds the v1 pool (2,000 tasks, 8 shards). `smol_ladder.jtasks_v2` supersedes it
with all 103 shards (9,187 tasks) plus the quality tags the audit asked for; v1 is kept because
data/jtasks.jsonl is what the existing runs on the jupyter-agent split were measured against.
The two agree on ids, so a v1 run stays comparable to a v2 one.

Two filters decide what survives:

- ``executor_type == "e2b"``. The ``llm`` rows have their outputs simulated, so their answers
  are fiction. That is 66% of the data.
- A gradable answer. jupyter-agent answers are free text ("61,257 USD (70,187 for failed minus
  8,930 for successful)"). SmolDataEnvs' grader needs a value plus a reward_mode, so an answer
  has to reduce to a number, a short label, or yes/no. Measured over 1,704 e2b rows: 945
  numeric, 324 label, 43 bool, 388 rejected.

Also rejected, because they look gradable and are not:

- "Not explicitly stated in the notebook outputs" and its relatives. That is the dataset
  recording that *no* answer exists, and it matches a label pattern.
- Answers carrying units or a derivation, e.g. "61,257 USD (...)", "area_se (5.447186)".
  The value is there but the intended answer is a sentence.
- Values that are the question restated back, e.g. "Y=3 with 95,293 instances".

    uv run python -m smol_ladder.jtasks --limit 400 --shards 8
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from huggingface_hub import hf_hub_download

from smol_ladder.tasks import DATA

DATASET = "jupyter-agent/jupyter-agent-dataset"
SHARDS = 103
COLUMNS = ["id", "question", "answer", "executor_type", "files_used", "packages_used",
           "kaggle_dataset_name", "edu_score"]

# "Not explicitly stated in the notebook outputs": the dataset's way of saying there is no
# answer. It is a sentence, so it survives a naive label pattern and must be excluded by name.
NO_ANSWER = re.compile(
    r"not (explicitly )?(stated|mentioned|provided|specified|available|given)"
    r"|no (answer|information|explicit)|does not (state|specify)|unknown|n/?a$",
    re.I)
NUMERIC = re.compile(r"^-?\$?\s*\d[\d,]*(?:\.\d+)?\s*%?$")
LABEL = re.compile(r"^[A-Za-z][\w \-/&'.]{0,48}$")
# A trailing unit or parenthetical means the intended answer is a sentence, not a value.
NOISY = re.compile(r"[(/]|\b(usd|eur|kg|km|per|times|with|and)\b", re.I)


def classify(answer: str) -> tuple[str, str] | None:
    """Return (reward_mode, normalised value) if the answer is gradable, else None."""
    raw = (answer or "").strip()
    if not raw or len(raw) > 60 or NO_ANSWER.search(raw) or NOISY.search(raw):
        return None
    if raw.lower() in {"yes", "no"}:
        return "exact_bool", raw.lower()
    if NUMERIC.match(raw):
        value = raw.replace("$", "").replace(",", "").replace("%", "").strip()
        try:
            number = float(value)
        except ValueError:
            return None
        # SmolDataEnvs grades short integers exactly, which makes 1-of-N guessing a pass. Give
        # numeric answers a tolerance in proportion to their own precision instead.
        decimals = len(value.split(".")[1]) if "." in value else 0
        return "numeric", value
    if LABEL.match(raw):
        return "exact_short", raw
    return None


def grade_params(mode: str, value: str) -> tuple[float, float]:
    """atol/rtol for the grader.

    A tolerance in proportion to the answer's own printed precision, so the metric stays about
    the computation rather than the last printed digit — but never tighter than 1e-4 relative.
    An answer stored as 8.89663104713 has 11 decimals, and a model that prints 8.896631 is
    right; a 1e-12 tolerance calls that wrong, which measures the grader, not the agent.

    Capped at the other end too, because an integer printed as "453" would otherwise get a
    whole-unit tolerance and admit 453.4.
    """
    if mode != "numeric":
        return 0.0, 0.0
    decimals = len(value.split(".")[1]) if "." in value else 0
    tol = min(max(10 ** (-(decimals + 1)), 1e-4), 0.05)
    return tol, tol


def load_shard(index: int) -> list[dict]:
    """Rows from a shard, preferring the local copy so a sweep never re-fetches from the Hub."""
    from smol_ladder.fetch_shards import read_shard, shard_path

    if shard_path(index).exists():
        return read_shard(index)
    path = hf_hub_download(DATASET, f"data/non_thinking-{index:05d}-of-{SHARDS:05d}.parquet",
                           repo_type="dataset")
    import pyarrow.parquet as pq
    return pq.ParquetFile(path).read(columns=COLUMNS).to_pylist()


def smoldataenvs_datasets() -> set[str]:
    """Kaggle dataset slugs SmolDataEnvs already uses, so we can exclude them by construction."""
    from smol_ladder.tasks import load_split
    return {r.get("kaggle_dataset") for r in load_split("test") if r.get("kaggle_dataset")}


def collect(shards: int, limit: int | None, exclude_overlap: bool = True) -> tuple[list[dict], Counter]:
    """Build tasks from the first `shards` local shards.

    Overlap with SmolDataEnvs is excluded explicitly rather than assumed. The two datasets are
    separate releases and share no task ids, but both are built from public Kaggle datasets, so
    "non-overlapping" has to mean "not the same underlying table" — otherwise a jupyter-agent
    task could ask about a table a SmolDataEnvs rung already described, and the two curves would
    not be independent. Checked on kaggle_dataset slug.
    """
    out, stats = [], Counter()
    seen_ids: set[str] = set()
    banned = smoldataenvs_datasets() if exclude_overlap else set()
    for index in range(shards):
        for row in load_shard(index):
            if row.get("executor_type") != "e2b":
                stats["not e2b"] += 1
                continue
            if row.get("kaggle_dataset_name") in banned:
                stats["overlaps SmolDataEnvs"] += 1
                continue
            files = row.get("files_used") or []
            if not files:
                stats["no files"] += 1
                continue
            graded = classify(row.get("answer", ""))
            if graded is None:
                stats["ungradable answer"] += 1
                continue
            mode, value = graded
            stats[mode] += 1
            # The raw id contains slashes ("0016/712/16712977.ipynb_qa_5"), which would turn
            # one task into a directory tree under data/runs.
            slug = re.sub(r"[^A-Za-z0-9_.-]", "_", f"{row['id']}")
            if slug in seen_ids:
                stats["duplicate id"] += 1
                continue
            seen_ids.add(slug)
            out.append({
                "task_id": f"ja_{slug}",
                "question": (row.get("question") or "").strip(),
                "answer": value,
                "reward_mode": mode,
                "atol": grade_params(mode, value)[0],
                "rtol": grade_params(mode, value)[1],
                "files": [Path(f).name for f in files],
                "source": DATASET,
                "kaggle_dataset_name": row.get("kaggle_dataset_name"),
                "edu_score": row.get("edu_score"),
            })
            if limit and len(out) >= limit:
                return out, stats
    return out, stats


def load_rows(path: Path | str | None = None) -> list[dict]:
    """The extracted tasks, in the same shape load_split returns for SmolDataEnvs."""
    src = Path(path or DATA / "jtasks.jsonl")
    if not src.exists():
        raise FileNotFoundError(
            f"{src} not found; run: uv run python -m smol_ladder.jtasks --shards 8")
    return [json.loads(line) for line in src.read_text().splitlines() if line.strip()]


def load_synthetic() -> list[dict]:
    """Synthetic tasks, in the same row shape as every other source."""
    src = DATA / "synthetic.jsonl"
    if not src.exists():
        raise FileNotFoundError(
            f"{src} not found; run: uv run python -m smol_ladder.synthetic --max-tables 60")
    return [json.loads(line) for line in src.read_text().splitlines() if line.strip()]


def synthetic_input_dir(row: dict) -> Path:
    """Synthetic tasks point at an already-cached table, so this is a lookup, not a download.

    The bucket_prefix is the table's parent directory, which is exactly what
    tasks.input_dir(row) expects for a SmolDataEnvs row.
    """
    from smol_ladder.tasks import input_dir
    return input_dir(row)


def input_dir(row: dict) -> Path:
    """The task's tables, fetched from Kaggle and cached.

    Cached outside $HOME. The trial jail mounts a tmpfs over the home directory, so anything
    under it disappears inside the sandbox: a cache there downloads successfully and then
    presents the agent with an empty ./input. Symlinks are resolved and bound directly, so the
    files are reachable wherever they live.

    One directory per task, holding only the files that task names, so a question never sees
    columns it was not asked about.
    """
    import os

    import kagglehub

    dataset = row.get("kaggle_dataset_name")
    if not dataset:
        raise ValueError(f"{row['task_id']}: no kaggle_dataset_name")
    cache = Path(os.environ.get("SMOL_LADDER_CACHE", "/var/tmp/smol-ladder")) / "kaggle"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("KAGGLEHUB_CACHE", str(cache))
    dest = cache / "tasks" / row["task_id"]
    if not dest.exists():
        root = Path(kagglehub.dataset_download(dataset))
        dest.mkdir(parents=True)
        for name in row["files"]:
            matches = list(root.rglob(Path(name).name))
            if not matches:
                raise FileNotFoundError(f"{row['task_id']}: {name} not in {dataset}")
            (dest / name).symlink_to(matches[0].resolve())
    return dest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=int, default=8)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out", default=str(DATA / "jtasks.jsonl"))
    ap.add_argument("--allow-overlap", action="store_true",
                    help="keep rows whose Kaggle dataset SmolDataEnvs also uses")
    args = ap.parse_args()

    rows, stats = collect(args.shards, args.limit, not args.allow_overlap)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    print(f"wrote {len(rows)} tasks to {out}")
    for key, value in stats.most_common():
        print(f"  {key:18} {value}")
    if rows:
        print("modes:", Counter(r["reward_mode"] for r in rows).most_common())


if __name__ == "__main__":
    main()
