"""Plain-language L2/L3 hints, written by the model from the verified reference solution.

L2 (which files, columns and filters the computation uses) and L3 (the method) were being read
off the reference's AST. That is deterministic, but it is empty or thin on most tasks: 98/181
test references are hand-written csv/sqlite code rather than pandas, and the extractor returns an
empty filter list on 71% of them and an L3 like "get, items, values" — a bag of method names
rather than a method. So the hint is now written by a model, in plain language, from the
verified reference. That is a deliberate trade: it is no longer a pure function of the reference,
so it buys faithfulness by validating every hint and falling back to the AST whenever a hint
cannot be trusted.

Validation, per hint, three ways. (1) Any file it names must exist in ./input, and any column it
names must exist in that table's header — a fabricated column is dropped and the rate recorded.
(2) The existing leak checks from ladder.py: no gold answer by normalised text match, and no
numeric literal in the hint that the dataset's own grader accepts, differenced against L1 so an
answer the question already names is not blamed on the hint. (3) L3 must add information beyond
L2, or it is not a rung. A hint that fails any check is regenerated up to three times; after that
the task is marked failed and the ladder uses the AST extraction for it.

Every hint is cached to data/hints/<split>/<task_id>.json together with the model, the prompt
version and a hash of the reference, so a rerun is resumable and, once the reference is
unchanged, deterministic.

    uv run python -m smol_ladder.gen_hints --split test --workers 16
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from smol_ladder.ladder import input_files, inputs_of, normalise, read_source
from smol_ladder.or_agent import MODEL, endpoint
from smol_ladder.tasks import DATA

# Bump when the prompt or the output contract changes, so a stale cache is regenerated rather
# than silently reused with the wrong shape.
PROMPT_VERSION = "nl-hints-v1"

SYSTEM = """You write short hints that describe how a data-analysis question's answer is \
computed. You are given the question, the exact list of files available, the real column headers \
of each file, and a verified reference program that computes the correct answer.

Write the hint from the reference program. Do not run it and do not state its output. Your job \
is to say what computation the question wants, in plain English, precisely enough that someone \
could implement it from your words alone — but WITHOUT giving away the answer.

Rules:
- Never state or imply the numeric or label answer. Do not include the computed result, even in \
an example. Only describe the procedure.
- Only name columns that appear verbatim in the headers you were given, and only files that \
appear in the file list. Do not invent names.
- Describe the method step by step, including the aggregation, how missing values are handled, \
how ties or ordering are broken, and any rounding or formatting, whenever the reference does any \
of those.
- If the reference filters rows, say which column and which values in plain words. If it filters \
nothing, say "none".
- Be concrete and brief. Aim for at most three sentences for the method.

Reply with exactly this JSON object and nothing else:
{"l2": {"files": ["exact file names used"], "columns": ["exact column names used"],
        "filters": "plain words describing which rows are kept, or the string \\"none\\""},
 "l3": "the method, one to three plain sentences"}"""

# Keys we accept for the two fields, because the model sometimes nests them under a wrapper
# object ({"hint": {...}}) or returns the method under a synonym. Rejecting the whole call for a
# key name cost us the first task's hint three times over for output that was perfectly good.
_L2_KEYS = ("l2", "L2", "l2_hint", "columns_filters")
_L3_KEYS = ("l3", "L3", "l3_hint", "method")


def prompt_for_hint(row: dict, source: str, files: list[str],
                    headers: dict[str, list[str]]) -> str:
    head = []
    for name in files:
        cols = headers.get(name)
        head.append(f"- {name}: " + (", ".join(cols) if cols else "(header not readable)"))
    return (
        f"Question:\n{row['question']}\n\n"
        f"Files available in the input directory:\n" + "\n".join(f"- {f}" for f in files) + "\n\n"
        f"Column headers:\n" + "\n".join(head) + "\n\n"
        f"Verified reference program (its final print has been removed; it computes the correct "
        f"answer, but you must describe the method, not report the result):\n\n"
        f"```python\n{source}\n```\n\n"
        "Return the hint as JSON."
    )


def call_model(row: dict, source: str, files: list[str], headers: dict[str, list[str]],
               model: str = MODEL) -> dict:
    """One JSON-mode chat completion, with retries and backoff on rate limits.

    Reuses or_agent's endpoint and key. The tool-free JSON mode is what makes the output parseable
    without scraping prose; the retry/backoff is needed because the free model 429s under 16
    concurrent threads.
    """
    import urllib.error
    import urllib.request

    ep = endpoint()
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": prompt_for_hint(row, source, files, headers)},
    ]
    body = json.dumps({
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
    }).encode()
    headers_req = ep.headers()
    last = ""
    for attempt in range(5):
        try:
            req = urllib.request.Request(ep.url, data=body, headers=headers_req)
            with urllib.request.urlopen(req, timeout=180) as resp:
                payload = json.load(resp)
            content = payload["choices"][0]["message"].get("content") or ""
            return json.loads(content)
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}: {e.read().decode()[:200]}"
            if e.code == 429:
                time.sleep(min(60, 5 * 2**attempt))
                continue
        except Exception as e:  # noqa: BLE001 - any transport error is worth one more try
            last = f"{type(e).__name__}: {e}"
        time.sleep(min(60, 5 * 2**attempt))
    raise RuntimeError(f"no completion after 5 attempts: {last}")


def normalise_output(raw: dict) -> dict:
    """Coerce whatever the model returned into {"l2": {...}, "l3": str}.

    JSON mode guarantees valid JSON, not the schema we asked for. In practice the same model
    answers with the two fields nested under a wrapper ("hint", "l2_l3", "output") about as often
    as it honours the shape, and returning all empty made `_validate` report l3-adds-nothing three
    times over for a hint that was actually correct. So unwrap rather than reject: a name mismatch
    is a formatting problem, not evidence that the hint is wrong.
    """
    if not isinstance(raw, dict):
        return {"l2": {}, "l3": ""}
    l2 = next((raw[k] for k in _L2_KEYS if isinstance(raw.get(k), dict)), None)
    l3 = next((raw[k] for k in _L3_KEYS if isinstance(raw.get(k), str)), None)
    if l2 is None or l3 is None:
        # Look one level down, under whatever wrapper key it used.
        for key, value in raw.items():
            if not isinstance(value, dict):
                continue
            nested_l2 = next((value[k] for k in _L2_KEYS if isinstance(value.get(k), dict)), None)
            nested_l3 = next((value[k] for k in _L3_KEYS if isinstance(value.get(k), str)), None)
            if nested_l2 is None and nested_l3 is None:
                continue
            l2 = l2 or nested_l2
            l3 = l3 or nested_l3
    result = {"l2": dict(l2) if isinstance(l2, dict) else {}, "l3": str(l3 or "")}
    # The other common shape is the bare {"files": [...], "columns": [...], "filters": "..."}
    # with no wrapper at all, or nested one level down under a wrapper key. Recognise it wherever
    # it sits, otherwise a perfectly good hint validates as empty.
    for candidate in [raw, *[v for v in raw.values() if isinstance(v, dict)]]:
        if result["l2"] and result["l3"]:
            break
        if not result["l2"] and any(k in candidate for k in ("files", "columns", "filters")):
            result["l2"] = {k: candidate[k] for k in ("files", "columns", "filters") if k in candidate}
        if not result["l3"]:
            result["l3"] = next(
                (str(v) for k, v in candidate.items()
                 if k not in ("files", "columns", "filters") and isinstance(v, str)), "")
    return result


def reference_hash(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8", "replace")).hexdigest()[:16]


def output_file(split: str, task_id: str) -> Path:
    return DATA / "hints" / split / f"{task_id}.json"


def headers_of(row: dict, split: str) -> dict[str, list[str]]:
    """The real column headers of each file in the task's input directory.

    Read from the tables, not from the reference, so a column the model names is checked against
    the actual environment and not against whatever the program happens to subscript. This is
    also why a header named after the answer is not special-cased here: a hint naming it would be
    caught by the leak check, not by hiding it.
    """
    out: dict[str, list[str]] = {}
    try:
        base = inputs_of(split)(row)
    except Exception:
        return out
    for name in input_files(row, split):
        path = base / name
        try:
            if path.suffix.lower() in {".sqlite", ".db"}:
                out[name] = _sqlite_columns(path)
            else:
                head = pd.read_csv(path, nrows=50)
                out[name] = [str(c) for c in head.columns]
        except Exception:
            continue
    return out


def _sqlite_columns(path: Path) -> list[str]:
    """Every column of every table in a sqlite file, deduplicated.

    13 test references are hand-written SQL, and their columns live in table_info rather than in
    a csv header, so a header check that only knew about delimited files would reject every real
    column they name.
    """
    cols: list[str] = []
    try:
        conn = sqlite3.connect(path)
        try:
            tables = [r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'")]
            for table in tables:
                try:
                    cols += [r[1] for r in conn.execute(f'PRAGMA table_info("{table}")')]
                except Exception:
                    continue
        finally:
            conn.close()
    except Exception:
        return []
    return sorted({c for c in cols if isinstance(c, str)})


def column_is_real(column: str, headers: dict[str, list[str]], files: list[str]) -> bool:
    """Does this column name appear in the header of one of the task's files?

    A hint that names a column the solver cannot find is worse than one that names none, so an
    unverified name is treated as fabricated. Case-insensitive, since csv headers and SQL
    identifiers vary.
    """
    want = normalise(column)
    pool = [c for f in (files or list(headers)) for c in headers.get(f, [])]
    if not pool:
        return False
    return any(normalise(c) == want for c in pool)


def file_is_real(name: str, files: list[str]) -> bool:
    return any(normalise(name) == normalise(f) for f in files)


def l3_adds_information(l2_text: str, l3_text: str) -> bool:
    """Does the method sentence add anything the L2 block does not already say?

    The ladder forbids a rung that adds nothing above the one below it. An L3 that merely restates
    the columns and filters is L2 wearing a label, and would re-measure L2 on a paid trial.
    """
    a, b = normalise(l2_text), normalise(l3_text)
    if not b:
        return False
    if a and b == a:
        return False
    return True


def _hint_text(record: dict) -> str:
    """Everything the hint says, as one string, for the leak and add-information checks."""
    l2 = record.get("l2", {})
    return " ".join([
        " ".join(l2.get("files", [])), " ".join(l2.get("columns", [])),
        str(l2.get("filters", "")), str(record.get("l3", "")),
    ])


def _validate(row: dict, record: dict, split: str, headers: dict[str, list[str]],
              files: list[str]) -> tuple[bool, str]:
    """Check one generated hint. Returns (ok, reason). Empty reason means accepted."""
    l2 = record.get("l2") or {}
    if not isinstance(l2, dict):
        return False, "l2-not-a-dict"
    columns = [str(c) for c in (l2.get("columns") or [])]
    hint_files = [str(f) for f in (l2.get("files") or [])]
    filters = str(l2.get("filters") or "")
    l3 = str(record.get("l3") or "")

    # (1) hallucination: drop files and columns that are not in the real environment.
    real_files = [f for f in hint_files if file_is_real(f, files)]
    real_columns = [c for c in columns if column_is_real(c, headers, real_files or files)]
    l2["files"] = real_files
    l2["columns"] = real_columns

    # An L3 that says nothing, or that only restates L2, is not a rung.
    if not l3_adds_information(_hint_text({"l2": l2}), l3):
        return False, "l3-adds-nothing"

    # (2) leak: the answer must not be in the hint, and the grader must not accept any number in
    # it that L1 does not already contain.
    hint_text = _hint_text({"l2": l2, "l3": l3})
    hits = _leak_hits(row, hint_text, split)
    if hits:
        return False, "leak:" + ",".join(sorted(set(hits)))

    record["l2"] = l2
    record["l3"] = l3
    record["hallucinated_columns"] = sorted(set(columns) - set(real_columns))
    return True, ""


def _leak_hits(row: dict, hint_text: str, split: str) -> list[str]:
    """Leak hits under ladder.py's own rule, differenced against L1.

    Delegated rather than reimplemented so a hint is held to exactly the same standard as the
    AST extraction: normalised substring match first, then every numeric literal put through the
    dataset's grader. The differential matters because 52/250 test tasks are multiple choice,
    where the answer is already in the question — blaming the hint for it would reject a
    perfectly good hint on two out of every five tasks that could have one.
    """
    from smol_ladder.ladder import PROMPT, leaks as ladder_leaks

    hits = ladder_leaks(row, hint_text)
    if not hits:
        return []
    l1 = PROMPT.format(question=row["question"],
                       files="\n".join(f"- {f}" for f in input_files(row, split)))
    return [h for h in hits if h not in ladder_leaks(row, l1)]


def generate_one(row: dict, split: str, source: str, model: str = MODEL,
                 attempts: int = 3) -> dict:
    """Generate (or reuse) the cached hint for one task.

    Returns the cached record. On total failure the record is written with `failed: True` and no
    usable l2/l3, and ladder.py falls back to the AST extraction for that task. The record is
    cached even when it fails, so a rerun does not re-roll a genuinely bad task forever; the
    reference hash still forces a regen if the reference itself changes.
    """
    out = output_file(split, row["task_id"])
    want_hash = reference_hash(source)
    if out.exists():
        try:
            cached = json.loads(out.read_text())
        except json.JSONDecodeError:
            cached = {}
        if (cached.get("reference_hash") == want_hash
                and cached.get("prompt_version") == PROMPT_VERSION
                and cached.get("model") == model):
            return cached

    files = input_files(row, split)
    headers = headers_of(row, split)
    record: dict = {
        "task_id": row["task_id"], "split": split, "model": model,
        "prompt_version": PROMPT_VERSION, "reference_hash": want_hash,
        "hallucinated_columns": [], "regenerations": 0,
    }
    reason = "no attempt"
    for attempt in range(1, attempts + 1):
        try:
            raw = normalise_output(call_model(row, source, files, headers, model))
        except Exception as e:  # noqa: BLE001 - a bad call is one failed attempt, not a crash
            reason = f"call-error:{type(e).__name__}"
            record["regenerations"] = attempt - 1
            continue
        ok, why = _validate(row, raw, split, headers, files)
        if ok:
            record.update({
                "l2": raw.get("l2"), "l3": raw.get("l3"),
                "hallucinated_columns": raw.get("hallucinated_columns", []),
                "regenerations": attempt - 1, "failed": False,
            })
            break
        reason = why
        record["regenerations"] = attempt
    else:
        record.update({"failed": True, "reason": reason,
                       "l2": {"files": [], "columns": [], "filters": "none"}, "l3": ""})

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=1))
    return record


def _generate_row(row: dict, split: str, model: str, attempts: int) -> dict:
    source = read_source(row, split)
    if source is None:
        return {"task_id": row["task_id"], "skipped": "no verified reference"}
    return generate_one(row, split, source, model=model, attempts=attempts)


def main() -> None:
    from smol_ladder.run_ladder import source_for

    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--attempts", type=int, default=3)
    args = ap.parse_args()

    rows = source_for(args.split)[0][: args.limit]
    done = ok = failed = skipped = 0
    hallucinated = 0
    leak_rejects = 0
    with ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(_generate_row, row, args.split, args.model, args.attempts)
                   for row in rows]
        for f in as_completed(futures):
            done += 1
            rec = f.result()
            if "skipped" in rec:
                skipped += 1
            elif rec.get("failed"):
                failed += 1
                if str(rec.get("reason", "")).startswith("leak:"):
                    leak_rejects += 1
            else:
                ok += 1
                hallucinated += len(rec.get("hallucinated_columns") or [])
            if done % 25 == 0:
                print(f"  [{done}/{len(rows)}] ok={ok} failed={failed}", flush=True)
    print(f"{args.split}: {len(rows)} tasks")
    print(f"  no verified reference: {skipped}")
    print(f"  hints cached ok: {ok}")
    print(f"  hints failed after {args.attempts} attempts: {failed} "
          f"(of which leak rejections at the end: {leak_rejects})")
    print(f"  hallucinated columns dropped (total across attempts): {hallucinated}")


if __name__ == "__main__":
    main()