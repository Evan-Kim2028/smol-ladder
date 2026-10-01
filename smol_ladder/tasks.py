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

# The point past which a delimited file is read in full rather than through the nrows-capped
# profile path. A schema dump needs enough rows for a stable shape, so a cap is right there; but
# the cap must not be reachable, because a short read in the middle of a file is indistinguishable
# from a small file and that is the bug this project shipped 38 unpassable synthetic tasks over.
MAX_READ_BYTES = 8_000_000

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


def detect_separator(path: Path, sniff: int = 64_000) -> str:
    """How to split a table's fields, decided from the bytes rather than from the suffix.

    The suffix is a hint, not evidence: a ".csv" full of semicolons parses into one garbage
    column, silently, and anything computed over that frame is not an answer to the file's
    question. Each candidate delimiter is tried and the first that gives more than one column
    wins, so a mis-suffixed file is read the way a reference implementation would read it.
    Nothing is returned until one of them splits the file, because a single-column parse is a
    guess about the data.
    """
    preferred = "\t" if path.suffix.lower() == ".tsv" else ","
    for sep in dict.fromkeys([preferred, *DELIMITERS]):
        with path.open(encoding="utf-8", errors="replace") as handle:
            if len(pd.read_csv(handle, sep=sep, nrows=sniff).columns) > 1:
                return sep
    return preferred


def read_shipped(path: Path, on_bad_lines: str = "error") -> pd.DataFrame:
    """The whole shipped file as one table: every row, pandas' own dtype and NA inference.

    Nothing here may approximate the file, because a gold computed from the result and an agent
    computing from the file have to describe the same computation. `on_bad_lines="error"` is the
    default for the same reason: pandas drops the offending row and warns, so a 239k-row table
    with one ragged row would give a gold over 239,177 rows and a task unpassable for that reason
    alone. The suffix is a hint, so the separator is sniffed off the bytes.
    """
    return pd.read_csv(path, sep=detect_separator(path), on_bad_lines=on_bad_lines)


def _delimited_frames(path: Path, nrows: int) -> list[pd.DataFrame]:
    """The file as a table, guessed delimiter first, capped at `nrows`.

    The cap is what makes this the *profile* reader rather than the gold's: a schema dump wants
    enough rows for a stable shape, not the file. It reads at most MAX_READ_BYTES, so a table
    beyond that is not silently short-read in the middle -- it falls to read_shipped and reads
    the file exactly.
    """
    if path.stat().st_size > MAX_READ_BYTES:
        try:
            return [read_shipped(path)]
        except Exception:  # noqa: BLE001 - an unreadable file is named by the caller
            return []
    raw = path.read_bytes()[:SNIFF_BYTES * 64]
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

    `nrows` is a profiling cap, not a read limit, and it must not become one: a file too large to
    read through the capped path falls to read_shipped, which returns every row. Nothing that
    computes an answer may go through here — the gold path is read_shipped.
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
