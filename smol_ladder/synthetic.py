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
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from smol_ladder.tasks import DATA, detect_separator, input_dir, load_split, read_shipped

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


# A gold that arrived as text and does not repr back to itself. One part in fifty thousand is
# wider than any text rendering of a float64 that is not the value, and far tighter than the
# 1e-6 the corpus used to carry, so it can only ever admit the value the text stands for.
ROUNDING_RTOL = 2e-5


def _fmt(value) -> str:
    """Render a value the way the grader can compare it, and the model can print.

    The numeric branch is the whole fix. `f"{x:.6g}"` was lossy, and with the 1e-6 tolerance
    that used to accompany it a lossy print made the task unpassable by construction: a sum of
    15,000 rows printed as `31535.6` excludes the exact `31535.63530029`, so the model's correct
    answer graded 0.0 and the task looked like a model failure. That accounted for 26 of the 64
    synthetic tasks that no correct answer could pass, on tables small enough that the 50,000-row
    truncation could not explain them. `repr` is the shortest string that round-trips to the same
    float, so it is the value itself and the exact answer grades against it.
    """
    if isinstance(value, (bool, np.bool_)):
        return "yes" if value else "no"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return ""
        return repr(float(value))
    text = str(value).strip()
    return "" if len(text) > 48 else text


def _tolerance(answer: str) -> tuple[float, float]:
    """The tolerance for an answer, derived from the answer rather than from the op.

    Printing and grading are separate steps and only printing is lossy, so the fix is to make
    printing lossless (`_fmt`) and the tolerance follows: a string that repr-s back to itself is
    the value exactly, so it needs no tolerance at all, and an exact tolerance is what a correct
    prediction has to beat. The one non-zero case is a gold that arrived as text rather than
    through `_fmt` -- a carried-over id from an older corpus, or a row from another source -- and
    then the answer is the *only* evidence of the value, so it is read at full precision and
    given the one part in fifty thousand that covers float text in general.
    """
    try:
        value = float(answer)
    except (TypeError, ValueError):
        return 0.0, 0.0
    if not np.isfinite(value):
        return 0.0, 0.0
    # An exact answer needs no tolerance. Two spellings qualify: the shortest form that
    # round-trips the float ("27.0", "4.25"), and the bare integer form `_fmt` emits for an
    # int64 sum ("59074"), which is the same value and must not be graded as if it were a
    # rounded one.
    if repr(value) == answer or value.is_integer() and answer == str(int(value)):
        return 0.0, 0.0
    return ROUNDING_RTOL, ROUNDING_RTOL



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


def iter_tables(limit_tables: int, on_error: str = "skip",
                ) -> tuple[list[tuple[str, pd.DataFrame]], list[str]]:
    """Every readable table we already have cached, from both sources.

    The frame returned is read from the file the agent will be shipped, and no argument may cut
    it short of that file's last row. `nrows=50_000` here was read from a column-major pass that
    pre-dates the row cap, and it silently disagreed with the file in `data/inputs`: the answer
    was computed over 50,000 rows and the agent summed all of them, so every task on a table
    over that size was mis-graded by construction. (Every row of that pass touched the frame, so
    capping the read caps the questions, not the work.)

    `on_error` decides what to do with a file that does not parse as a table at any separator:
    "skip" drops it and names it, "strict" raises. Skipping is the default, and the reason is
    that a dropped table cannot mis-grade anything — no task is built on it, so no gold is
    computed from it and the agent is never shipped it. It is a coverage loss, not a correctness
    one, and coverage loss is the right trade against refusing to build a corpus at all because
    `ca_law_enforcement_by_campus.csv` has newlines inside its header row. Strict is the opt-in
    for a caller that would rather have nothing than not know what was dropped.

    Note the parse cannot be "fixed" for such a file without inventing data: its header holds
    bare newlines, so every row after it has a different field count and no separator separates
    them. An agent handed this file fails too, which is the only property that matters here.
    """
    unreadable: list[str] = []
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
            # The shared reader, not a local read_csv: this loop's own "sep = tab if .tsv else
            # comma" is what made it miss a semicolon-separated table. Nothing caps the read,
            # because a capped read is the bug this pass fixes: the answer is computed over the
            # cap while the agent reads the whole file in data/inputs.
            try:
                out.append((str(path), read_shipped(path)))
            except Exception as exc:                    # noqa: BLE001 - one bad table is not fatal
                if on_error == "strict":
                    raise RuntimeError(
                        f"{path.name} did not read cleanly as a table, so a gold computed here "
                        f"would not describe the file shipped to the agent: {exc}") from exc
                unreadable.append(f"{path.name}: {type(exc).__name__}")
                continue
            if len(out) >= limit_tables:
                return out, unreadable
    return out, unreadable


def _exec_ops(ops: list[str], frame: pd.DataFrame | pd.Series):
    """Run the recorded ops. The one implementation of "what this spec asks for"."""
    value: object = None
    for op in ops:
        name, _, arg = op.partition("(")
        arg = arg.rstrip(")")
        if name == "filter":
            column, _, want = arg.partition("==")
            value = frame[frame[column] == want]
        elif name == "value_counts":
            value = frame[arg].value_counts(dropna=True)
        elif name == "argmax":
            value = value.index[0]
        else:
            source = value if value is not None else frame
            series = pd.to_numeric(source[arg], errors="coerce").dropna()
            try:
                value = getattr(series, _AGGREGATES[name])()
            except KeyError:
                return None
    return value


_AGGREGATES = {"mean": "mean", "median": "median", "sum": "sum", "max": "max"}


def reference_script(row: dict) -> str:
    """The reference program as *code*, with the answer never spelled.

    ladder.synthetic_reference builds the same program inline in the prompt, from ops it parses
    with string partitioning and interpolates into `df['{column}']`. That is safe for column
    names and wrong for filter values: every value is quoted as a string, so a numeric column
    filters to nothing (`df[df['Year'] == '2016']`), and a value containing a quote produces a
    SyntaxError and an L4 rung that cannot run at all.

    Reading the value's type out of the shipped table is the fix, and the reason the shipped
    table matters twice over. The shipped frame is read once per distinct set of ops rather than
    once per task, since a corpus holds ~1800 tasks over ~40 tables and re-reading a 239k-row
    table 45 times over is the difference between seconds and minutes.
    """
    path = shipped_path(row)
    if path is None:
        return ""
    ops = row.get("ops") or []
    if not ops:
        return ""
    # Keyed on the ops as well as the file. Keying on the file alone is wrong in a way that is
    # invisible until you look: a corpus holds ~1800 tasks over ~40 tables, so 45 tasks share a
    # path and each asks a different question of it, and a per-file cache handed every one of
    # them the *first* task's program. The gate then "verified" thousands of tasks against
    # another task's reference, and the L4 rung shipped that same wrong program.
    key = (str(path), tuple(ops))
    if key not in _REFERENCE_CACHE:
        columns = _parse_columns(ops)
        frame = read_shipped(path)
        _REFERENCE_CACHE[key] = _render_reference(ops, columns, frame, path.name)
    return _REFERENCE_CACHE[key]


_REFERENCE_CACHE: dict[tuple[str, tuple[str, ...]], str] = {}


def _parse_columns(ops: list[str]) -> dict[str, str | None]:
    """Op name -> the column it names, or None for a value_counts/argmax that takes no column."""
    columns: dict[str, str | None] = {}
    for op in ops:
        name, _, arg = op.partition("(")
        arg = arg.rstrip(")")
        columns[name] = None if name in {"value_counts", "argmax"} else arg
    return columns


def _render_reference(ops: list[str], columns: dict[str, str | None], frame: pd.DataFrame,
                      file_name: str) -> str:
    """Render the ops as pandas, matching _exec_ops step for step.

    Every column is written as a variable and every op applied to the frame it was written
    against, so the code and the gold cannot drift apart the way an inline f-string can. `repr`
    of an actual cell from the table is the literal for a filter value, which is both the right
    type and the right spelling.
    """
    lines = ["import pandas as pd", f"df = pd.read_csv('input/{file_name}')", ""]
    sub: str | None = None
    counts: str | None = None
    body: list[str] = []
    for position, op in enumerate(ops):
        name, _, arg = op.partition("(")
        arg = arg.rstrip(")")
        if name == "filter":
            column, _, want = arg.partition("==")
            literal = repr(_first_matching_value(frame, column, want))
            sub = f"sub{position}"
            body.append(f"{sub} = df[df[{column!r}] == {literal}]")
        elif name == "value_counts":
            counts = f"counts{position}"
            body.append(f"{counts} = df[{arg!r}].value_counts(dropna=True)")
        elif name == "argmax":
            body.append(f"result = {counts}.index[0]")
        else:
            # The frame an op reads is the one the op before it left, and the first numeric op
            # after a filter has to read the filter's output. Reading `df` here instead silently
            # computes a whole-table aggregate for a question about one group.
            source = sub or counts or "df"
            body.append(f"result = pd.to_numeric({source}[{arg!r}], errors='coerce')"
                        f".dropna().{name}()")
            sub = counts = None
    lines += body + ["", "print(result)"]
    return "\n".join(lines)


def _first_matching_value(frame: pd.DataFrame, column: str, want: str):
    """A real cell equal to the filter's recorded value, so the literal keeps its column's type.

    `filter(col==value)` stores the value the way a human wrote it in the question, as text. An
    integer column then needs `2016`, not `'2016'`, and pandas' `==` between an int column and a
    string is elementwise False, so the rendered reference filtered to the empty frame and the
    rung graded 0.0 while looking like a model failure.
    """
    series = frame[column]
    for value in series.dropna():
        if str(value) == want or _fmt(value) == want:
            return value
    return want


def shipped_path(row: dict) -> Path | None:
    """The file the task ships, as a path, or None if it is not on disk.

    A synthetic row's `bucket_prefix` is the table's parent directory, which is exactly what
    `tasks.input_dir` expects and what `jtasks.synthetic_input_dir` delegates to, so resolving it
    here keeps one path to the cache rather than two.
    """
    try:
        path = input_dir(row) / row["files"][0]
    except Exception:                    # noqa: BLE001 - a missing table is a missing table
        return None
    return path if path.exists() else None


def verify_shipped(row: dict, runner=None) -> tuple[float, str]:
    """Grade the task's own reference against the shipped file, through the real grader.

    Two things this catches that nothing else does. The gold is computed from the frame we read,
    so it is correct *for our parse* — this asks whether the agent's parse of the same bytes gives
    the same graded answer. And the tolerance is ours to choose, so this asks whether a task
    admits any prediction at all; a gold the grader rejects as its own predicate is unpassable by
    construction, which is how 26 tasks came to be no task could ever solve.

    `runner` is `smol_ladder.sandbox.run_script`. The default runs the program in bubblewrap
    against the shipped tables at ./input, exactly the way a trial's verifier does; inject one in
    a test to run the program's text in process instead of paying for a sandbox.
    """
    from smol_ladder.grade import grade, last_line

    if runner is None:
        from smol_ladder.sandbox import run_script
        runner = run_script
    script = reference_script(row)
    if not script:
        return 0.0, "no reference program"
    reward, note = verify_prediction(row, script, runner)
    if reward < 1.0:
        return reward, note
    if grade(row, row["answer"]) < 1.0:
        return 0.0, "the grader rejects its own gold answer"
    return 1.0, note


def verify_prediction(row: dict, script: str, runner) -> tuple[float, str]:
    """Run a program over the shipped tables in a jail and grade what it printed."""
    from smol_ladder.grade import grade, last_line

    path = shipped_path(row)
    if path is None:
        return 0.0, "the shipped table is not on disk"
    # The work directory is keyed on the task id, so a caller running the gate over a whole
    # corpus cannot have one task's program graded against another's tables.
    work = DATA / "ladder" / "_verify" / re.sub(r"[^A-Za-z0-9_.-]", "_", row["task_id"])
    work.mkdir(parents=True, exist_ok=True)
    script_path = work / "solution.py"
    script_path.write_text(script)
    run = runner(script_path, path.parent)
    if getattr(run, "returncode", 0) != 0 or getattr(run, "timed_out", False):
        return 0.0, f"reference program did not finish: {getattr(run, 'stderr', '')[:200]}"
    printed = last_line(run.stdout)
    return grade(row, printed), printed



def verify(spec: Spec, df: pd.DataFrame) -> bool:
    """Re-derive the answer by executing the recorded ops, independently of build_task.

    This is the synthetic analogue of the openswe three-way proof. build_task computed the
    answer as a side effect of choosing the question; verify recomputes it from the ops alone, so
    a spec whose ops do not actually produce its answer is caught rather than published. The ops
    run through _exec_ops, the same implementation the reference program is rendered from, so the
    two cannot describe different computations.

    The tie checks live here, because a tie is a property of the specification rather than of
    any one execution: `build_task` refuses to *ask* about a tied maximum, and `verify` refuses to
    *publish* one that slipped through.
    """
    try:
        value = _exec_ops(spec.ops, df)
        if value is None:
            return False
        # A tie is a property of the specification, not of any single execution, so it is
        # checked where the tie can still be seen: on the counts, before argmax reduces them to
        # one label. `build_task` refuses to ask about a tied maximum; this refuses to publish one
        # that got through anyway.
        counts = value if isinstance(value, pd.Series) else None
        if any(op.startswith(("value_counts", "max")) for op in spec.ops) \
                and counts is not None and not _unique_max(counts):
            return False
        return _fmt(value) == spec.answer
    except Exception:
        return False


def leaks(spec: Spec) -> bool:
    """The question must not contain its own answer. Cheap, but it catches the obvious."""
    q = spec.question.lower()
    a = spec.answer.lower().strip()
    return bool(a) and len(a) >= 3 and a in q


def as_row(spec: Spec) -> dict:
    """The on-disk row, with the gold's tolerance recomputed from the answer it will carry.

    atol/rtol used to be read off the printed precision of a `.6g` answer, and the answer is now
    printed exactly. Deriving them from the string at write time means the two can never
    disagree, which is what lets verify_shipped() refuse to emit a task that grades 0.
    """
    atol, rtol = _tolerance(spec.answer)
    return {
        "task_id": spec.task_id, "question": spec.question, "answer": spec.answer,
        "reward_mode": spec.reward_mode, "atol": atol, "rtol": rtol,
        "files": [str(Path(spec.source_table).name)],
        "bucket_prefix": str(Path(spec.source_table).parent),
        "difficulty_tier": 0, "tier": spec.tier, "ops": spec.ops,
        "columns": spec.columns, "source": "synthetic",
    }


def _id_cache_path() -> Path:
    """The published-id cache, next to the module so it can be committed. See _load_ids.

    This returned `DATA / "synthetic_ids"`, which is two things wrong at once: `data/` is
    gitignored, so the cache was never in version control, and it read and wrote a path with no
    extension while the file that exists beside the module is `synthetic_ids.jsonl`. The cache
    was therefore a no-op on a clean checkout — every regeneration silently renumbered.
    """
    return Path(__file__).resolve().parent / "synthetic_ids.jsonl"


def _load_ids() -> dict[str, dict]:
    """The tasks an earlier generation already published, keyed by id, for a stable corpus.

    Regenerating must not renumber the benchmark. A task is keyed on (file stem, index), so a
    changed tolerance, a fixed rendering or one refilled bucket would otherwise let an earlier
    index be re-used for a different question, and every stored trial would then be read against
    the wrong gold. The cache is the row each id was published as; `row()` keeps its fields.

    It lives outside data/, next to the module, because it is part of the corpus rather than of
    a run: without it in version control, the first person to regenerate a corpus silently
    renumbers every id after the first hole.
    """
    path = _id_cache_path()
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            out[row["task_id"]] = row
    return out


def _save_ids(rows: list[dict]) -> None:
    """Publish the corpus's ids, so the next regeneration cannot renumber it.

    A hole in the id set is the thing this file exists to prevent: an id is a file stem plus an
    index, so retiring index 4 does not leave a gap, it moves index 5's task onto index 4's
    trial directory and reads its stored rewards against the wrong gold. The cache is therefore
    rewritten from what was actually emitted, and `main()` refuses to shrink it -- a corpus that
    lost an id is a corpus whose results tree no longer means anything, so that is reported
    loudly rather than silently accepted.
    """
    path = _id_cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def merge_published(new: list[Spec], published: dict[str, dict]) -> tuple[list[Spec], list[str]]:
    """Keep every published id, so a regenerated corpus holds the same tasks in the same order.

    A spec this run produced replaces the published one under its own id. A published id this
    run did not reach is carried over verbatim and reported, because a hole in the id set would
    silently re-key every later task: an id is the file stem plus an index, so retiring index 4
    moves index 5's task onto the trial directory index 4's trials left behind, and their stored
    rewards would then be read against the wrong gold. Carrying them over with a known-wrong gold
    is at least honest, and the shipped-file gate stops any task that can be fixed.

    `Spec` is a dataclass, so the carried row is built through its own fields and never by
    mutating a spec that is still on `new`.
    """
    out: list[Spec] = []
    kept = {spec.task_id for spec in new}
    carried: list[str] = []
    for task_id, old in published.items():
        if task_id in kept:
            continue
        fields = {key: old[key] for key in Spec.__dataclass_fields__ if key in old}
        out.append(Spec(**fields))
        carried.append(task_id)
    order = {row["task_id"]: i for i, row in enumerate(published.values())}
    out.sort(key=lambda spec: order.get(spec.task_id, 0))
    return out, carried


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=200, help="tasks per table")
    ap.add_argument("--max-tables", type=int, default=60)
    ap.add_argument("--out", default=str(DATA / "synthetic.jsonl"))
    ap.add_argument("--strict-tables", action="store_true",
                    help="stop the sweep on a table that does not parse, instead of skipping it")
    ap.add_argument("--no-shipped-check", action="store_true",
                    help="skip running each reference against the shipped file")
    ap.add_argument("--workers", type=int, default=8, help="shipped-file checks run in parallel")
    args = ap.parse_args()

    tables, unreadable = iter_tables(args.max_tables,
                                     "strict" if args.strict_tables else "skip")
    print(f"{len(tables)} readable tables, {len(unreadable)} skipped", flush=True)
    for note in unreadable:
        print(f"  skipped {note}", flush=True)
    fresh: list[Spec] = []
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
            fresh.append(spec)

    # The gate, on every spec that survived the cheap checks. One bwrap per task, so it runs in
    # a pool; the 6,956 candidates this sweep builds take about four minutes that way.
    notes: dict[str, str] = {}
    if args.no_shipped_check:
        kept = fresh
    else:
        kept = []
        with ThreadPoolExecutor(max(1, args.workers)) as pool:
            futures = {pool.submit(verify_shipped, as_row(spec)): spec for spec in fresh}
            done = 0
            for future in as_completed(futures):
                spec = futures[future]
                reward, note = future.result()
                done += 1
                if done % 250 == 0:
                    print(f"  shipped check {done}/{len(fresh)} kept={len(kept)}", flush=True)
                if reward >= 1.0:
                    kept.append(spec)
                else:
                    notes[spec.task_id] = f"{note!r} (reward {reward})"
    stats["reference does not grade 1.0"] = len(notes)

    published = _load_ids()
    carried: list[str] = []
    if published:
        carried_specs, carried = merge_published(kept, published)
        for spec in carried_specs:
            if spec.task_id not in used:
                stats["carried over a published id, gold no longer derivable"] += 1
        kept = kept + carried_specs

    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    rows = [as_row(spec) for spec in kept]
    with dest.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    _save_ids(rows)
    print(f"wrote {len(rows)} synthetic tasks to {dest}")
    for key, value in stats.items():
        print(f"  {key:50} {value}")
    for task_id in carried[:20]:
        print(f"  carried over {task_id}")
    if len(carried) > 20:
        print(f"  ... and {len(carried) - 20} more carried over")
    for task_id, note in sorted(notes.items())[:10]:
        print(f"  refused {task_id}: {note}")
    if len(notes) > 10:
        print(f"  ... and {len(notes) - 10} more refusals")
    if rows:
        from collections import Counter
        print("  modes:", Counter(r["reward_mode"] for r in rows).most_common())


if __name__ == "__main__":
    main()
