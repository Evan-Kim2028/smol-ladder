"""SmolDataEnvs rows and their input files, cached under data/."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import download_bucket_files, hf_hub_download, list_bucket_tree

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
