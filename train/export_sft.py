"""Export training data in upstream's SmolDataEnvs-sft format.

    uv run python -m train.export_sft --source smoldataenvs-sft --out data/train/sft_upstream
    uv run python -m train.export_sft --source traces --out data/train/sft_traces
    uv run python -m train.export_sft --source rungs --rung L2 --out data/train/sft_rungs_L2

Three sources, because the audit of what we actually have came first:

1. `--source smoldataenvs-sft` -- upstream's 4,677 verified trajectories, through the firewall.
   This is arm A of the experiment design, so it is the one that must be the faithful format.
2. `--source traces` -- **our** verified traces, translated from our `run_shell` protocol into
   upstream's bash protocol. See `traces.py` for exactly what that translation preserves.
3. `--source rungs` -- each task at one chosen rung, for the curriculum arm.

The firewall is not a report here: `--heldout-columns` decides which keys are matched and the
export refuses any trajectory that trips one, counting the drops by column. Run it with
`--dry-run` to see the counts without writing anything.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from smol_ladder.tasks import DATA, load_split

from train.format import heldout_keys, is_heldout, read_jsonl, write_jsonl

SFT_DATASET = "FineEnvs/SmolDataEnvs-sft"
DEFAULT_HELDOUT_COLUMNS = "task_id,question"
SOURCES = ("smoldataenvs-sft", "traces", "rungs")


def question_of(row: dict) -> str:
    """The question an SFT trajectory is about, lifted out of its `user` turn.

    `SmolDataEnvs-sft` has no `question` column -- only `task_id`, difficulty, `messages`, `tools` --
    so the question-key firewall has to read it out of the rendered prompt, and it has to do so the
    same way every time or it silently never matches. The `user` turn is upstream's `BASH_USER`
    verbatim, whose `Question:` line is followed by the question and a blank line.
    """
    for message in row.get("messages") or []:
        if message.get("role") != "user":
            continue
        content = message.get("content") or ""
        if "Question:\n" not in content:
            return ""
        return content.split("Question:\n", 1)[1].split("\n\n", 1)[0].strip()
    return ""


def with_firewall_keys(row: dict) -> dict:
    """A training row carrying the keys the firewall matches on, wherever they came from."""
    out = dict(row)
    out.setdefault("question", question_of(row))
    return out


def load_sft_rows() -> list[dict]:
    """Upstream's SFT trajectories, as plain dicts.

    `load_dataset` rather than the parquet directly because the parquet's `messages` column is a
    list of JSON blobs, which is one more conversion between us and the text we actually train on.
    """
    from datasets import load_dataset

    return load_dataset(SFT_DATASET, split="train").to_list()


def task_index(splits=("test", "eval")) -> dict[str, list[dict]]:
    """The held-out rows we firewall against, fetched once."""
    return {split: load_split(split) for split in splits}


def export_upstream(rows: list[dict], keys: dict[str, set[str]], limit: int | None = None
                    ) -> tuple[list[dict], dict]:
    """Upstream's rows, minus everything the firewall catches.

    `keys` decides what is compared. `task_id` and `question` are read off the trajectory itself;
    `bucket_prefix` has to be *joined in* from the task dataset, because the SFT parquet does not
    carry it -- and that join is the one that catches the 110 rows that share a held-out table.
    """
    kept, dropped = [], {}
    for row in rows:
        column = is_heldout(with_firewall_keys(row), keys)
        if column:
            dropped[column] = dropped.get(column, 0) + 1
            continue
        kept.append(row)
        if limit and len(kept) >= limit:
            break
    return kept, dropped


def export_traces(data: Path | None = None, limit: int | None = None,
                  keys: dict[str, set[str]] | None = None) -> tuple[list[dict], dict]:
    """Our verified traces, translated to bash format. See `traces.collect_traces`."""
    from train.traces import collect_traces

    return collect_traces(data or DATA, limit=limit, keys=keys)


def export_rungs(rung: str, data: Path | None = None, limit: int | None = None,
                 keys: dict[str, set[str]] | None = None) -> tuple[list[dict], dict]:
    """Each task at one rung of the ladder, as a single-turn program transcript."""
    from train.rungs import collect_rungs

    return collect_rungs(rung, data or DATA, limit=limit, keys=keys)


def split_train_val(rows: list[dict], val_fraction: float, seed: int) -> tuple[list, list]:
    """A deterministic train/val split by row index.

    Deterministic on purpose: a resume or a re-run has to produce the same split, and a random
    split re-drawn from the OS entropy each time makes a loss curve incomparable between runs.
    A hash of the index rather than a PRNG so the split is stable even if the exporter's
    row order changes -- which it does when the firewall drops rows.
    """
    import hashlib

    scored = [(hashlib.sha256(f"{seed}:{i}".encode()).hexdigest(), i) for i in range(len(rows))]
    scored.sort()
    n_val = int(round(len(rows) * val_fraction))
    val_indices = {i for _, i in scored[:n_val]}
    train_rows = [r for i, r in enumerate(rows) if i not in val_indices]
    val_rows = [r for i, r in enumerate(rows) if i in val_indices]
    return train_rows, val_rows


def report(name: str, kept: list[dict], dropped: dict, out: Path | None,
           val_fraction: float, seed: int, extra: dict | None = None) -> dict:
    """Write the split and print the counts. Returns the record for the caller's own summary.

    `extra` carries whatever the source knows that the caller cannot see from the rows -- for
    traces, how many rows came from a real conversation and how many from the contract fallback.
    Both are `messages` + `tools` rows and TRL cannot tell them apart, so a dataset that is mostly
    invented exploration would train fine and be reported as ours. The numbers are in the record
    rather than only on the console because the record is what a run gets quoted from.
    """
    if out is not None:
        train_rows, val_rows = split_train_val(kept, val_fraction, seed)
        write_jsonl(train_rows, out / "train.jsonl")
        write_jsonl(val_rows, out / "val.jsonl")
        # The sidecar is what lets a trained checkpoint be traced back to tasks. It is written
        # beside the data, never into it: TRL reads train.jsonl and nothing else, and a row with
        # a "task_id" field is still a row TRL might one day start rendering into the prompt.
        write_jsonl([{"task_id": (r.get("task_id") or ""), "source": name}
                     for r in train_rows + val_rows], out / "index.jsonl")
        counts = {"train": len(train_rows), "val": len(val_rows)}
    else:
        counts = {}
    record = {"source": name, "kept": len(kept), "dropped_by_column": dropped,
              "dropped_total": sum(dropped.values()), "written": counts,
              "out": str(out) if out else None}
    if extra:
        record.update(extra)
    print(json.dumps(record, indent=1))
    return record


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, choices=SOURCES)
    ap.add_argument("--out", type=Path, default=None,
                    help="directory for train.jsonl / val.jsonl / index.jsonl; omit to only count")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--val-fraction", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rung", default="L1", choices=["L1", "L2", "L3", "L4"],
                    help="which rung --source rungs emits")
    ap.add_argument("--heldout-columns", default=DEFAULT_HELDOUT_COLUMNS,
                    help="comma-separated keys the firewall matches on. Default "
                         "bucket_prefix,task_id,question: the held-out *table*, the held-out "
                         "*task*, and the held-out *question text*. Narrow it to task_id to keep "
                         "more data at the cost of a leaky benchmark.")
    args = ap.parse_args()

    columns = tuple(c.strip() for c in args.heldout_columns.split(",") if c.strip())
    tasks = task_index()
    keys = heldout_keys(tasks)
    keys = {c: keys.get(c, set()) for c in columns}
    print(f"firewall on {list(keys)}: "
          f"{ {c: len(v) for c, v in keys.items()} } held-out keys")

    if args.source == "smoldataenvs-sft":
        rows = load_sft_rows()
        if "bucket_prefix" in keys:
            # The join that makes the table check possible: the SFT parquet has no table id, so it
            # is looked up by task_id from the task dataset. Report how many rows could not be
            # resolved, because a table check silently skipped on 0% of rows is a check that did
            # not run and would otherwise read as a clean result.
            table_of = {r["task_id"]: r.get("bucket_prefix")
                        for split in ("train", "test", "eval")
                        for r in load_split(split)}
            resolved = sum(1 for r in rows if r["task_id"] in table_of)
            print(f"joined bucket_prefix for {resolved}/{len(rows)} trajectories from the task "
                  f"dataset ({len(rows) - resolved} unresolved)")
            rows = [{**r, "bucket_prefix": table_of.get(r["task_id"])} for r in rows]
        kept, dropped = export_upstream(rows, keys, args.limit)
        extra = None
    elif args.source == "traces":
        # The traces exporter returns a report rather than a bare drop count, so the record can
        # say how many rows are real trajectories and how many are the contract fallback. Both
        # are `messages` + `tools` rows, so nothing downstream can tell them apart.
        kept, report_fields = export_traces(DATA, args.limit, keys)
        dropped = report_fields["dropped"]
        extra = {k: v for k, v in report_fields.items() if k != "dropped"}
    else:
        kept, dropped = export_rungs(args.rung, DATA, args.limit, keys)
        extra = None
    report(args.source, kept, dropped, args.out, args.val_fraction, args.seed, extra)


if __name__ == "__main__":
    main()


# `read_jsonl` is re-exported for the tests, which check a written export against the same reader
# the trainer will use rather than against a second implementation of it.
_ = read_jsonl