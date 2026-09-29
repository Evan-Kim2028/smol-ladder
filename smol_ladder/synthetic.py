"""Synthetic data-analysis tasks with a known answer by construction.

The two existing sources are both secondhand. SmolDataEnvs' gold answers came from Kaggle
notebooks, and jupyter-agent's came from an LLM. We keep only 72-73% of the former and 8% of the
latter, and the ladder's population is gated on that yield, which is the circularity an
adversarial review flagged as the design's worst problem: the tasks we can measure are the
tasks the model under evaluation happened to solve.

A synthetic task removes the gate. We pick a table, instantiate a *specification* over it, and
execute the specification to get the answer. The gold is correct by construction, so the yield
is 100% and nothing is conditioned on the model.

The specification, not the question, is the ground truth. Each task records the ops it applied
so the ladder rungs can be built from it — the same verified reference the other sources get
from a model-written solution.py, except here it is the generator's own program and it cannot
leak, because we authored the question from the spec and not the reverse.

Two rules borrowed from the openswe pipeline, which is the only prior art here that took
solvability seriously:

- **ARBITRARY vs DERIVABLE.** A task is only fair if a competent analyst holding the question
  would produce the graded answer, not merely guess it. A question whose answer depends on a
  tie-break we chose (which of two equal maxima, which arbitrary seed) is ARBITRARY and gets
  rejected, because it measures our choice rather than the analyst's. This is the single most
  useful idea to carry over, and it is a gate here too: tasks with a tie or a free parameter
  are dropped, not published.
- **Prove it three ways before trusting it.** The answer must be computable, the question must
  not name the answer, and a second independent implementation of the same spec must agree.
    uv run python -m smol_ladder.synthetic --limit 400
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from smol_ladder.tasks import DATA, input_dir, load_split

# A column name that is itself the answer would make the task a lookup, so those are skipped
# for label answers and never used as the target of an "which X" question.
_ID_LIKE = re.compile(r"^(id|index|idx|unnamed:? ?\d*|uuid|key|hash|no|no\.|s/n|unnamed)$", re.I)
# Columns with no human meaning. "What is the mean of #" is a task nobody would be graded on,
# and a good fraction of these tables carry an index column under some such name.
_JUNK = re.compile(r"^[\W_]*$|^(unnamed|no|no\.|s/n|na|n/a|null|index|level|of|total rows)$", re.I)
_NUMERIC_HINT = re.compile(r"(count|sum|mean|avg|average|median|total|rate|ratio|percent|"
                           r"score|price|age|year|number|amount|qty|quantity|length|size|"
                           r"weight|height|temp|income|salary)", re.I)
_CATEGORICAL_HINT = re.compile(r"(name|type|category|class|label|status|group|kind|genre|"
                               r"state|country|city|brand|model|method|region|dept)", re.I)


@dataclass
class Spec:
    """One instantiated task: what to compute, and the answer we computed.

    The spec is the ground truth. The question is rendered from it, and the ladder rungs read
    its ops, so a rung can never describe a computation the answer did not come from.
    """

    task_id: str
    source_table: str
    question: str
    answer: str
    reward_mode: str
    atol: float
    rtol: float
    files: list[str]
    ops: list[str] = field(default_factory=list)
    columns: list[str] = field(default_factory=list)
    tier: str = "synthetic"


def _clean_column(name: str) -> str:
    return re.sub(r"\s+", " ", str(name)).strip()


def usable_columns(df: pd.DataFrame) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Columns usable as a numeric measure and as a grouping key, as (original, display) pairs.

    The original key is what indexes the frame; the display form is what goes in the question.
    Real tables have columns like " Total Discharges", so the two cannot be the same string.

    Rejects blanks, id-like columns, and anything with too few distinct values to be worth
    asking about, which is the mechanical form of "the question must have an answer".
    """
    numeric, categorical = [], []
    for raw in df.columns:
        name = _clean_column(raw)
        if not name or _ID_LIKE.match(name) or _JUNK.match(name):
            continue
        series = df[raw]
        if pd.api.types.is_numeric_dtype(series) and series.notna().sum() > 20:
            if series.nunique(dropna=True) > 3:
                numeric.append((raw, name))
        elif not pd.api.types.is_numeric_dtype(series) and series.notna().sum() > 20:
            # Not `dtype == object`: pandas 3 gives text columns a dedicated str dtype, and the
            # object check silently matched nothing, so every table produced numeric-only tasks.
            if 1 < series.nunique(dropna=True) <= 60 and series.notna().mean() > 0.5:
                categorical.append((raw, name))
    return numeric, categorical


def _fmt(value) -> str:
    """Render a value the way the grader can compare it, and the model can print."""
    if isinstance(value, (bool, np.bool_)):
        return "yes" if value else "no"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return ""
        return f"{float(value):.6g}"
    text = str(value).strip()
    return "" if len(text) > 48 else text


def _tolerance(answer: str) -> tuple[float, float]:
    if re.fullmatch(r"-?\d+", answer):
        return 0.0, 0.0
    if re.fullmatch(r"-?\d*\.\d+", answer):
        return 1e-6, 1e-6
    return 0.0, 0.0


def _unique_max(series: pd.Series) -> bool:
    """Is the maximum unique? A tie means the answer depends on a tie-break we did not state,
    which is exactly the ARBITRARY case: fair only if the question tells the model how to break
    it, and we do not."""
    top = series.max()
    return int((series == top).sum()) == 1


def build_task(df: pd.DataFrame, table: str, stem: str, index: int) -> Spec | None:
    """One task from one table, or None if the table cannot support a fair question."""
    numeric, categorical = usable_columns(df)
    if not numeric and not categorical:
        return None

    # Alternate families by index so one table yields a mix, and so a categorical-only table
    # still produces tasks. Family 2 needs a categorical column; family 1 needs a numeric one.
    want_categorical = index % 2 == 1

    if not want_categorical and numeric:
        raw_col, col = numeric[(index // 2) % len(numeric)]
        series = pd.to_numeric(df[raw_col], errors="coerce").dropna()
        if len(series) < 20:
            return None
        kinds = [
            ("mean", "What is the mean of {col} across all rows?", series.mean()),
            ("median", "What is the median of {col} across all rows?", series.median()),
            ("sum", "What is the total of {col} summed over all rows?", series.sum()),
            ("max", "What is the largest value of {col} in the table?",
             series.max() if _unique_max(series) else None),
        ]
        for offset, (op, template, value) in enumerate(kinds):
            if offset != index % len(kinds):
                continue
            if value is None or not np.isfinite(value):
                return None
            answer = _fmt(value)
            if not answer:
                return None
            return Spec(
                task_id=f"syn_{stem}_{index}_{op}",
                source_table=table,
                question=template.format(col=col),
                answer=answer,
                reward_mode="numeric",
                atol=_tolerance(answer)[0], rtol=_tolerance(answer)[1],
                files=[Path(table).name],
                ops=[f"{op}({raw_col})"],
                columns=[col],
            )

    # Family 3: an aggregate *within* one group of a categorical column. Two columns, so the
    # task is not a single-call lookup, and the group is named in the question, so nothing is
    # left to guess. Requires the group to hold enough rows for a mean to be meaningful.
    if not want_categorical and numeric and categorical and index >= 2:
        raw_num, col_num = numeric[(index // 3) % len(numeric)]
        raw_grp, col_grp = categorical[(index // 3) % len(categorical)]
        frame = df[[raw_grp, raw_num]].dropna()
        counts = frame[raw_grp].value_counts()
        if counts.empty:
            return None
        # Only the most common group, and only when it is unique: "the rows where X is the
        # most common value" is fully determined, "a row where X is Y" is not.
        if not _unique_max(counts) or counts.iloc[0] < 20:
            return None
        group = counts.index[0]
        value = pd.to_numeric(frame.loc[frame[raw_grp] == group, raw_num],
                              errors="coerce").dropna().mean()
        if not np.isfinite(value) or len(value) == 0:
            return None
        answer = _fmt(value)
        if not answer:
            return None
        return Spec(
            task_id=f"syn_{stem}_{index}_grouped_mean",
            source_table=table,
            question=(f"Considering only the rows where {col_grp} is {group}, "
                      f"what is the mean of {col_num}?"),
            answer=answer,
            reward_mode="numeric",
            atol=_tolerance(answer)[0], rtol=_tolerance(answer)[1],
            files=[Path(table).name],
            ops=[f"filter({raw_grp}=={group})", f"mean({raw_num})"],
            columns=[col_grp, col_num],
        )

    # Family 2: the mode of a categorical column. The computation is fully determined by the
    # question, and the maximum must be unique — a tie means the answer depends on a tie-break
    # we never stated, which is the ARBITRARY case.
    if not categorical:
        return None
    raw_col, col = categorical[(index // 2) % len(categorical)]
    counts = df[raw_col].value_counts(dropna=True)
    if counts.empty or not _unique_max(counts):
        return None
    answer = _fmt(counts.index[0])
    if not answer:
        return None
    return Spec(
        task_id=f"syn_{stem}_{index}_mode",
        source_table=table,
        question=f"Which value of {col} appears most often in the table?",
        answer=answer,
        reward_mode="exact_short",
        atol=0.0, rtol=0.0,
        files=[Path(table).name],
        ops=[f"value_counts({raw_col})", "argmax"],
        columns=[col],
    )


def iter_tables(limit_tables: int) -> list[tuple[str, pd.DataFrame]]:
    """Every readable table we already have cached, from both sources."""
    seen: set[str] = set()
    out: list[tuple[str, pd.DataFrame]] = []
    for row in load_split("test")[:limit_tables * 2]:
        try:
            src = input_dir(row)
        except Exception:
            continue
        for path in sorted(src.glob("*")):
            if path.suffix.lower() not in {".csv", ".tsv"} or path.stat().st_size > 60_000_000:
                continue
            key = str(path.resolve())
            if key in seen:
                continue
            seen.add(key)
            try:
                sep = "\t" if path.suffix.lower() == ".tsv" else ","
                out.append((str(path), pd.read_csv(path, sep=sep, nrows=50_000)))
            except Exception:
                continue
            if len(out) >= limit_tables:
                return out
    return out


def verify(spec: Spec, df: pd.DataFrame) -> bool:
    """Re-derive the answer by executing the recorded ops, independently of build_task.

    This is the synthetic analogue of the openswe three-way proof. build_task computed the
    answer as a side effect of choosing the question; verify recomputes it from the ops alone,
    so a spec whose ops do not actually produce its answer is caught rather than published.
    """
    try:
        frame: pd.DataFrame | pd.Series = df
        value: object = None
        for op in spec.ops:
            name, _, arg = op.partition("(")
            arg = arg.rstrip(")")
            if name == "filter":
                column, _, want = arg.partition("==")
                value = frame[frame[column] == want]
            elif name == "value_counts":
                # value_counts operates on a COLUMN of whatever the previous op produced, not
                # on the frame itself: frame[arg] is the column, and the result is a Series.
                value = frame[arg].value_counts(dropna=True)
                if not _unique_max(value):
                    return False
            elif name == "argmax":
                value = value.index[0]
            else:
                source = value if value is not None else frame
                series = pd.to_numeric(source[arg], errors="coerce").dropna()
                if name == "mean":
                    value = series.mean()
                elif name == "median":
                    value = series.median()
                elif name == "sum":
                    value = series.sum()
                elif name == "max":
                    if not _unique_max(series):
                        return False
                    value = series.max()
                else:
                    return False
        return _fmt(value) == spec.answer
    except Exception:
        return False


def leaks(spec: Spec) -> bool:
    """The question must not contain its own answer. Cheap, but it catches the obvious."""
    q = spec.question.lower()
    a = spec.answer.lower().strip()
    return bool(a) and len(a) >= 3 and a in q


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=200, help="tasks per table")
    ap.add_argument("--max-tables", type=int, default=60)
    ap.add_argument("--out", default=str(DATA / "synthetic.jsonl"))
    args = ap.parse_args()

    tables = iter_tables(args.max_tables)
    print(f"{len(tables)} readable tables")
    out: list[Spec] = []
    used: set[str] = set()
    stats = {"no fair task": 0, "failed verification": 0, "answer leaked": 0,
             "duplicate id": 0, "kept": 0}
    for path, df in tables:
        # The id must key on the *file*, not its parent directory. SmolDataEnvs keeps several
        # tables under one bucket_prefix, so keying on the parent gave four different CSVs the
        # same id and 275 tasks collapsed to 201.
        stem = re.sub(r"[^A-Za-z0-9]+", "_",
                      Path(path).stem or Path(path).parent.name)[:40]
        for index in range(args.limit):
            spec = build_task(df, path, stem, index)
            if spec is None:
                stats["no fair task"] += 1
                continue
            if leaks(spec):
                stats["answer leaked"] += 1
                continue
            if not verify(spec, df):
                stats["failed verification"] += 1
                continue
            if spec.task_id in used:
                stats["duplicate id"] += 1
                continue
            used.add(spec.task_id)
            stats["kept"] += 1
            out.append(spec)

    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w") as fh:
        for spec in out:
            fh.write(json.dumps({
                "task_id": spec.task_id, "question": spec.question, "answer": spec.answer,
                "reward_mode": spec.reward_mode, "atol": spec.atol, "rtol": spec.rtol,
                "files": [str(Path(spec.source_table).name)],
                "bucket_prefix": str(Path(spec.source_table).parent),
                "difficulty_tier": 0, "tier": spec.tier, "ops": spec.ops,
                "columns": spec.columns, "source": "synthetic",
            }) + "\n")
    print(f"wrote {len(out)} synthetic tasks to {dest}")
    for key, value in stats.items():
        print(f"  {key:20} {value}")
    if out:
        from collections import Counter
        print("  modes:", Counter(s.reward_mode for s in out).most_common())


if __name__ == "__main__":
    main()
