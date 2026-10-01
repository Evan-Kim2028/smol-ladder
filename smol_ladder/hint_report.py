"""Report on the plain-language hints: coverage, content, and the rungs they build.

Answers the questions the design raises, from the cache on disk and from the prompts ladder.py
actually builds:

- coverage: how many tasks with a verified reference got a usable hint, and how many fell back
- content: the share with real columns, real filters and a real method, against the AST
- hallucinations: how many named columns that are not in the table headers
- leaks: how many hints were rejected for naming the answer
- cost: how many characters each rung adds, since a rung spends context as well as information
- examples: a seeded random sample, printed in full, for a human to judge against the reference

    uv run python -m smol_ladder.hint_report --split test --examples 8
"""

from __future__ import annotations

import argparse
import json
import random

from smol_ladder import gen_hints as G
from smol_ladder import ladder as L
from smol_ladder.run_ladder import source_for

# The AST block for the same task, so every number has its before-and-after partner.
AST_FILTER_SENTINEL = "(none)"


def _stats(rows: list[dict], split: str) -> dict:
    out = {
        "tasks": len(rows),
        "with_reference": 0,
        "hint_ok": 0,
        "hint_failed": 0,
        "no_hint_file": 0,
        "ast_fallback": 0,
        "nl_columns": 0,
        "ast_columns": 0,
        "nl_filters": 0,
        "ast_filters": 0,
        "nl_method": 0,
        "ast_method": 0,
        "nl_chars": 0,
        "ast_chars": 0,
        "hallucinated_columns": 0,
        "leak_failures": 0,
        "regenerated": 0,
    }
    for row in rows:
        source = L.read_source(row, split)
        if source is None:
            continue
        out["with_reference"] += 1
        path = G.output_file(split, row["task_id"])
        if not path.exists():
            out["no_hint_file"] += 1
            continue
        record = json.loads(path.read_text())
        out["hallucinated_columns"] += len(record.get("hallucinated_columns") or [])
        out["regenerated"] += int(record.get("regenerations") or 0)
        if record.get("failed"):
            out["hint_failed"] += 1
            if str(record.get("reason", "")).startswith("leak:"):
                out["leak_failures"] += 1
            out["ast_fallback"] += 1
            continue
        out["hint_ok"] += 1

        l2 = record.get("l2") or {}
        if [c for c in (l2.get("columns") or [])]:
            out["nl_columns"] += 1
        if str(l2.get("filters") or "").strip().lower() not in {"", "none"}:
            out["nl_filters"] += 1
        if str(record.get("l3") or "").strip():
            out["nl_method"] += 1

        facts = L.code_facts(source)
        if facts["columns"]:
            out["ast_columns"] += 1
        if facts["filters"]:
            out["ast_filters"] += 1
        if L.method_hint(source).strip():
            out["ast_method"] += 1

        out["nl_chars"] += len(_hint_chars(row, split, record))
        out["ast_chars"] += len(_ast_chars(row, split, source))
    return out


def _hint_chars(row: dict, split: str, record: dict) -> str:
    l2 = record.get("l2") or {}
    return (
        f"Files read: {', '.join(l2.get('files') or []) or 'the tables above'}\n"
        f"Columns used: {', '.join(l2.get('columns') or []) or '(discover them yourself)'}\n"
        f"Filters applied: {l2.get('filters') or '(none)'}\n"
        f"Method: {record.get('l3') or ''}"
    )


def _ast_chars(row: dict, split: str, source: str) -> str:
    facts = L.code_facts(source)
    return (
        f"Files read: {', '.join(facts['files']) or 'the tables above'}\n"
        f"Columns used: {', '.join(facts['columns']) or '(discover them yourself)'}\n"
        f"Filters applied: {'; '.join(facts['filters']) or '(none)'}\n"
        f"Method: {L.method_hint(source)}"
    )


def _pct(n: int, d: int) -> str:
    return f"{100.0 * n / d:.1f}%" if d else "n/a"


def print_examples(rows: list[dict], split: str, n: int, seed: int = 0) -> None:
    """A seeded random sample, printed in full, so a reviewer can check it against the code."""
    pool = [r for r in rows
            if L.read_source(r, split) and G.output_file(split, r["task_id"]).exists()]
    random.Random(seed).shuffle(pool)
    print(f"\n{'=' * 78}\n{n} random hints (seed {seed})\n{'=' * 78}")
    for i, row in enumerate(pool[:n], 1):
        source = L.read_source(row, split)
        record = json.loads(G.output_file(split, row["task_id"]).read_text())
        print(f"\n--- [{i}] {row['task_id']} ---")
        print(f"QUESTION: {row['question']}")
        print("REFERENCE CODE:")
        print("\n".join("  " + ln for ln in source.splitlines() if ln.strip()))
        if record.get("failed"):
            print(f"HINT: FAILED ({record.get('reason')}) -> the ladder uses the AST extraction")
            print(f"AST L2/L3:\n  {_ast_chars(row, split, source)}")
            continue
        print("L2 BLOCK (as the rung shows it):")
        print("\n".join("  " + ln for ln in _hint_chars(row, split, record).splitlines()))
        print("REFERENCE (full):")
        print("\n".join("  " + ln for ln in source.splitlines() if ln.strip()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--examples", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rows = source_for(args.split)[0]
    s = _stats(rows, args.split)
    n_ref = s["with_reference"]
    print(f"{args.split}: {s['tasks']} tasks, {n_ref} with a verified reference")
    print(f"  hints cached and valid: {s['hint_ok']} ({_pct(s['hint_ok'], n_ref)})")
    print(f"  hint failed after 3 attempts: {s['hint_failed']} "
          f"({_pct(s['hint_failed'], n_ref)}) -> AST fallback")
    print(f"  no hint file at all: {s['no_hint_file']} -> AST fallback")
    print(f"  total on AST extraction: {s['ast_fallback'] + s['no_hint_file']} "
          f"({_pct(s['ast_fallback'] + s['no_hint_file'], n_ref)})")
    print("\n  share WITH non-empty content, out of all tasks with a reference")
    for label, key, ast in (("columns", "nl_columns", "ast_columns"),
                            ("filters", "nl_filters", "ast_filters"),
                            ("method", "nl_method", "ast_method")):
        print(f"    {label:8s} plain-language {_pct(s[key], n_ref):>6s}"
              f"   (was {_pct(s[ast], n_ref):>6s} from the AST)")
    print("\n  validation")
    print(f"    hallucinated columns dropped: {s['hallucinated_columns']}")
    print(f"    hints rejected for leaking the answer (final failure): {s['leak_failures']}")
    print(f"    regenerations across all tasks: {s['regenerated']}")
    print("\n  cost (mean characters added above L1, tasks with a plain-language hint)")
    used = s["hint_ok"] or 1
    print(f"    plain language: {s['nl_chars'] / used:.0f}")
    print(f"    AST:            {s['ast_chars'] / used:.0f}")
    if args.examples:
        print_examples(rows, args.split, args.examples, args.seed)


if __name__ == "__main__":
    main()