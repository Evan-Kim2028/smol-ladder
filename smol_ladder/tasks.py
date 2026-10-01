"""SmolDataEnvs rows and their input files, cached under data/."""

from __future__ import annotations

import csv
import io
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import download_bucket_files, hf_hub_download, list_bucket_tree

# Delimiters, in the order the sniffer's verdict is doubted. Every table in both splits uses
# one of these four, so the fallback ladder terminates on the whole corpus.
DELIMITERS = (",", ";", "\t", "|")
ENCODINGS = ("utf-8", "latin-1")  # latin-1 never raises, so it is always the last resort
SNIFF_BYTES = 64_000  # enough rows for a stable delimiter guess on a wide table

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DATASET = "FineEnvs/SmolDataEnvs"


def load_split(split: str) -> list[dict]:
    path = hf_hub_download(DATASET, f"data/{split}-00000-of-00001.parquet", repo_type="dataset")
    rows = pd.read_parquet(path).to_dict("records")
    # Normalise to the same row shape jupyter-agent tasks use. to_dict on a row with a list
    # column yields a numpy array, so `files` is not a list and every consumer that treats it
    # as one (a conformance test, a prompt builder, JSON round-tripping) has to special-case
    # the source. One shape for both sources is the point.
    for row in rows:
        for key, value in row.items():
            if isinstance(value, np.ndarray):
                row[key] = value.tolist()
            elif isinstance(value, np.generic):
                row[key] = value.item()
        for key in ("files", "tags"):
            if key in row and row[key] is not None and not isinstance(row[key], list):
                row[key] = list(row[key])
    return rows


def _delimited_frames(path: Path, nrows: int) -> list[pd.DataFrame]:
    """The file as a table, guessed delimiter first.

    A wrong separator still parses, so "it parsed" cannot accept a guess: pandas splits a csv on
    a semicolon into one long column. A candidate is therefore only accepted when it yields more
    than one column, and the sniffer's verdict is merely the first candidate to try -- the whole
    DELIMITERS ladder is behind it, which is what rescues a table the sniffer gets wrong.
    Single-column files are handled by taking any parse that succeeds, preferring the sniffer's.
    """
    raw = path.read_bytes()[: SNIFF_BYTES * 64]
    for encoding in ENCODINGS:
        try:
            text = raw.decode(encoding)
        except UnicodeDecodeError:
            continue
        candidates: list[str] = []
        try:
            sniffed = csv.Sniffer().sniff(text[: SNIFF_BYTES], delimiters="".join(DELIMITERS))
            candidates.append(sniffed.delimiter)
        except csv.Error:
            pass
        candidates.extend(d for d in DELIMITERS if d not in candidates)
        parsed: list[pd.DataFrame] = []
        for delimiter in candidates:
            try:
                frame = pd.read_csv(io.StringIO(text), sep=delimiter, nrows=nrows)
            except Exception:
                continue
            if len(frame.columns) > 1:
                return [frame]
            parsed.append(frame)
        if parsed:  # genuinely single-column, or every delimiter folded it to one column
            return [parsed[0]]
    return []


def _json_frame(path: Path, nrows: int) -> list[pd.DataFrame]:
    """A JSON document as tables: the top-level dict of records, else the nested dict flattened.

    Both are real shapes in the corpus -- a list of row objects, and a keyed object of row
    objects. Anything else is not tabular, and returning nothing is correct rather than a guess.
    """
    try:
        doc = json.loads(path.read_text(errors="replace"))
    except (OSError, ValueError):
        return []
    if isinstance(doc, list):
        rows = [r for r in doc if isinstance(r, dict)]
    elif isinstance(doc, dict):
        rows = [v for v in doc.values() if isinstance(v, dict)]
    else:
        return []
    if not rows:
        return []
    try:
        return [pd.DataFrame(rows).head(nrows)]
    except Exception:
        return []


def _sqlite_frames(path: Path, nrows: int) -> list[pd.DataFrame]:
    """Every user table in the database, read-only and in a stable order.

    Opened through a URI in read-only mode because these are shared cached downloads: a plain
    connect() can create a -wal beside them, and the tasks' own runs must not write here.
    """
    frames: list[pd.DataFrame] = []
    try:
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    except sqlite3.Error:
        return []
    try:
        names = [r[0] for r in con.execute(
            "select name from sqlite_master where type='table' order by name")]
        for name in names:
            try:
                quoted = '"' + name.replace('"', '""') + '"'
                frames.append(pd.read_sql_query(f"select * from {quoted} limit {nrows}", con))
            except Exception:
                continue
    finally:
        con.close()
    return frames


def read_tables(path: Path, nrows: int) -> list[pd.DataFrame]:
    """Every table a task's file holds, by format. Empty list means genuinely unreadable.

    One reader for the whole project. schema_dump and synthetic.iter_tables both need "read this
    task file into frames", and each having its own version is how the two drifted: iter_tables
    knew about .tsv and read_csv-with-a-raised-ValueError looked like an unreadable file.
    """
    suffix = path.suffix.lower()
    if suffix in {".sqlite", ".db", ".sqlite3"}:
        return _sqlite_frames(path, nrows)
    if suffix in {".json", ".jsonl", ".ndjson"}:
        frames = _json_frame(path, nrows)
        if frames:
            return frames
    if suffix in {".xlsx", ".xls"}:
        try:
            return list(pd.read_excel(path, sheet_name=None, nrows=nrows).values())
        except Exception:
            return []
    if suffix in {".parquet", ".pq"}:
        try:
            return [pd.read_parquet(path).head(nrows)]
        except Exception:
            return []
    return _delimited_frames(path, nrows)


def input_dir(row: dict) -> Path:
    """Download the task's tables once per bucket prefix; many tasks share one."""
    dest = DATA / "inputs" / row["bucket_prefix"]
    done = dest / ".complete"
    if done.exists():
        return dest
    prefix = row["bucket_prefix"].rstrip("/") + "/"
    items = [
        i
        for i in list_bucket_tree(row["hf_bucket"], prefix=prefix, recursive=True)
        if getattr(i, "type", None) == "file"
    ]
    dest.mkdir(parents=True, exist_ok=True)
    download_bucket_files(
        row["hf_bucket"], files=[(i.path, str(dest / i.path.split("/")[-1])) for i in items]
    )
    done.write_text(json.dumps([i.path for i in items]))
    return dest
