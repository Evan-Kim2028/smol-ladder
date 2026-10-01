"""Audit a seeded sample of hints against the reference code they were written from.

A hint that names a real column and passes the leak check can still describe the wrong
computation — nothing mechanical catches that, because faithfulness to the reference is a
judgement, not a predicate. So this prints the hint and the reference side by side for a human
(here: a model) to mark faithful or unfaithful, and totals the result.

The audit also re-derives, mechanically, the two things that can be checked: does the hint's
column set survive a second, independent read of the headers, and does the L3 text add anything
the L2 block does not already say.

    uv run python -m smol_ladder.hint_audit --split test --n 20 --seed 0
"""

from __future__ import annotations

import argparse
import json
import random

from smol_ladder import gen_hints as G
from smol_ladder import ladder as L
from smol_ladder.run_ladder import source_for


def sample(rows: list[dict], split: str, n: int, seed: int) -> list[dict]:
    pool = [r for r in rows
            if L.read_source(r, split) and G.output_file(split, r["task_id"]).exists()]
    random.Random(seed).shuffle(pool)
    return pool[:n]


def show(rows: list[dict], split: str, n: int, seed: int) -> None:
    for i, row in enumerate(sample(rows, split, n, seed), 1):
        record = json.loads(G.output_file(split, row["task_id"]).read_text())
        source = L.read_source(row, split)
        print(f"\n{'#' * 70}\n[{i}] {row['task_id']}\n{'#' * 70}")
        print(f"QUESTION: {row['question']}\n")
        print(f"REFERENCE:\n{source}\n")
        if record.get("failed"):
            print(f"HINT: none (failed: {record.get('reason')}) — ladder uses the AST")
            continue
        l2 = record["l2"]
        print(f"HINT L2 files:   {l2.get('files')}")
        print(f"HINT L2 columns: {l2.get('columns')}")
        print(f"HINT L2 filters: {l2.get('filters')}")
        print(f"HINT L3: {record['l3']}")


def mechanical_checks(rows: list[dict], split: str, n: int, seed: int) -> dict:
    """The part of the audit that can be counted rather than judged."""
    chosen = sample(rows, split, n, seed)
    out = {"n": len(chosen), "failed": 0, "columns_outside_headers": 0, "l3_adds_nothing": 0,
           "leaks": 0}
    for row in chosen:
        record = json.loads(G.output_file(split, row["task_id"]).read_text())
        if record.get("failed"):
            out["failed"] += 1
            continue
        headers = G.headers_of(row, split)
        files = G.input_files(row, split)
        for column in record["l2"].get("columns") or []:
            if not G.column_is_real(column, headers, record["l2"].get("files") or files):
                out["columns_outside_headers"] += 1
        if not G.l3_adds_information(G._hint_text({"l2": record["l2"]}), record.get("l3", "")):
            out["l3_adds_nothing"] += 1
        if G._leak_hits(row, G._hint_text(record), split):
            out["leaks"] += 1
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()
    rows = source_for(args.split)[0]
    stats = mechanical_checks(rows, args.split, args.n, args.seed)
    print(f"mechanical checks over {stats['n']} sampled hints ({args.split}, seed {args.seed}):")
    print(f"  hint failed (AST fallback): {stats['failed']}")
    print(f"  columns outside the real headers: {stats['columns_outside_headers']}")
    print(f"  L3 that adds nothing beyond L2: {stats['l3_adds_nothing']}")
    print(f"  hints that still leak the answer: {stats['leaks']}")
    if args.show:
        show(rows, args.split, args.n, args.seed)


if __name__ == "__main__":
    main()