"""Build the information ladder for a SmolDataEnvs task.

L1 is the normal prompt. L2-L4 add what the intended computation is, derived from the task's
verified reference solution (LADDER.md). L1+schema is a control, not a rung: it adds no
information, only cheaper reading, so a gain there is a skill failure.

A task with no verified reference solution gets L1 and the control only. No rung is fabricated.

    uv run python -m smol_ladder.ladder --split test
    uv run python -m smol_ladder.ladder --split test --leak-check
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import shutil
from collections import Counter
from pathlib import Path

import pandas as pd

from smol_ladder.grade import grade, last_line
from smol_ladder.sandbox import run_script
from smol_ladder.tasks import DATA, input_dir, load_split

RUNGS = ("L1", "L2", "L3", "L4")

PROMPT = """You are solving a data-analysis question. The input tables are in ./input (read-only).

Question: {question}

Files:
{files}

Explore the data with Python as much as you need. Then write ./solution.py: a self-contained
script that reads only from ./input, computes the answer, and prints the final answer as its
LAST line of output. That last line is graded on its own, so it must be the value alone: a number
(no commas or units), a short label, yes/no, or a comma-separated list. No label, no "Answer:"
prefix, no trailing explanation -- "Answer: 42" grades as 0, "42" grades as 1.
Run `python3 solution.py` to check it works.
Do not look the answer up online or in any dataset; compute it from the files."""

HINT_HEADER = "\n\nA verified reference solution to this question is below. It is one correct\n" \
               "approach, not the only one."

TABLE_SUFFIXES = (".csv", ".tsv", ".parquet", ".json", ".xlsx", ".sqlite", ".db", ".txt")

# The vocabulary the L3 line is allowed to speak in, and the reads that establish a frame.
# Both span the idioms the references are actually written in: 83/181 test references are
# pandas, 98 are hand-written csv or sqlite code, and the rest mix in scipy, sklearn and stats.
_READERS = {"read_csv", "read_table", "read_parquet", "read_json", "read_excel", "read_html",
            "read_sql", "read_sql_query", "read_stata", "read_sas", "read_fwf", "read_clipboard",
            "csv_reader", "csv_dictreader", "DictReader", "reader", "read_excel"}
_PANDAS_READERS = {"read_csv", "read_table", "read_parquet", "read_json", "read_excel",
                   "read_html", "read_sql", "read_sql_query", "read_stata", "read_sas",
                   "read_fwf", "read_clipboard", "DataFrame", "Series"}
_FRAME_CALLS = {
    # reshaping and grouping
    "groupby", "pivot", "pivot_table", "melt", "stack", "unstack", "merge", "join", "crosstab",
    "concat", "merge_asof", "set_index", "reset_index", "sort_values", "sort_index", "nlargest",
    "nsmallest", "drop", "dropna", "drop_duplicates", "fillna", "ffill", "bfill", "rename",
    "assign", "reindex", "set_axis", "query", "loc", "iloc", "at", "iat", "mask", "where",
    "replace", "clip", "astype", "to_numeric", "to_datetime", "to_timedelta", "select_dtypes",
    "get_dummies", "explode", "factorize", "map", "applymap", "pipe", "round", "interpolate",
    "truncate", "sample", "head", "tail", "drop_duplicates",
    # aggregation and statistics
    "mean", "median", "mode", "std", "var", "sem", "quantile", "sum", "prod", "min", "max",
    "count", "nunique", "value_counts", "unique", "agg", "aggregate", "apply", "transform",
    "cumsum", "cumprod", "cumcount", "cummax", "cummin", "diff", "pct_change", "rank", "abs",
    "skew", "kurt", "corr", "cov", "covariance", "corrwith", "rolling", "expanding", "ewm",
    "idxmax", "idxmin", "argsort", "dot", "describe", "size", "shape", "any", "all",
    "isna", "notna", "isin", "between", "duplicated", "nsmallest", "nlargest",
    # exchange with other tools
    "to_numpy", "to_dict", "to_records", "to_csv", "to_list", "tolist", "itertuples",
    "iterrows", "values", "get", "items",
}
# `.loc` is indexing, not an operation, and it is reported as such on purpose.
_FRAME_CALLS.discard("loc")
_COLUMN_ATTRS = {"columns", "index", "values", "keys"}
SQL_CALLS = {"execute", "executemany", "executescript", "fetchall", "fetchone", "fetchmany"}
_PATH_ATTRS = {"open", "parent"}
# Names under which a row is built from the csv module. A DictReader is recognised by its own
# name, and anything built from it — a list comprehension, `list(...)`, a subscript — is rows.
_ROW_READERS = {"csv", "DictReader", "reader", "csv_reader", "csv_dictreader", "Reader",
                "f", "fh", "src", "handle"}
# `", ".join(labels)` and `os.path.join(...)` are formatting and paths. A method name is an
# operation on data when its receiver is not one of these.
_STRING_PATH_ATTRS = {"path", "dirname", "basename", "abspath", "realpath", "relpath",
                      "normpath", "expanduser", "parent", "name", "stem", "suffix", "split",
                      "rsplit", "getcwd", "sep", "getsize", "exists", "resolve", "upper",
                      "lower", "strip", "replace", "format", "encode", "decode", "rstrip",
                      "lstrip", "title", "capitalize", "casefold", "read_text", "write_text",
                      "open", "read_bytes", "write_bytes", "iterdir", "is_file", "is_dir"}
# SQL words that look like columns in a statement but are not.
_SQL_WORDS = re.compile(r"[A-Za-z_][A-Za-z_0-9.]*")
_SQL_WORDS_IGNORED = {"select", "from", "where", "and", "or", "not", "null", "group", "order",
                      "by", "limit", "offset", "having", "as", "in", "is", "like", "between",
                      "distinct", "sum", "count", "avg", "min", "max", "total", "on", "join",
                      "inner", "left", "right", "outer", "using", "case", "when", "then", "else",
                      "end", "cast", "asc", "desc", "union", "all", "case", "exists", "asc"}
_QUERY_OPS = {"==": "==", "=": "==", "!=": "!=", "<>": "!=", "<": "<", "<=": "<=", ">": ">",
              ">=": ">=", ".isin": "in", ".between": "between"}
# Strings that reach the extractor as subscript keys but name a keyword argument or a loop
# variable rather than a field of the table.
_NOT_A_COLUMN = {"axis", "columns", "index", "labels", "dtype", "encoding", "sep", "_"}
# `CATEGORY = "Fire"` and `CITY = "Oslo"` bind a column's value under a name that says what the
# column is. Where a reference compares a variable against such a literal, the literal is a
# value and the variable is the column: `where == CITY` filters on the column CITY, and that is
# what L2 says. This catches 11 more test references than the subscript walk alone, and 10 of
# those 11 name real header columns. A literal no comparison mentions is prose, not a column.
_VALUE_BINDINGS = {"CATEGORY", "CITY", "STATE", "COUNTRY", "REGION", "SPECIES", "GENRE", "TYPE",
                   "GROUP", "LABEL", "STATUS", "SEX", "GENDER", "CLASS", "TEAM", "LEVEL",
                   "SENTIMENT", "TARGET", "ANSWER", "WANT", "MATCH", "SELECTED", "WINNER",
                   "CATEGORY_NAME", "COUNTRY_NAME", "STATE_NAME", "CITY_NAME", "REGION_NAME",
                   "PUBLISHER", "GAMENAME", "PRODUCT", "ITEM", "NAME", "YEAR"}


def read_text(path: Path) -> str:
    for candidate in sorted(path.glob("*"), key=lambda p: (p.suffix != ".csv", p.name)):
        try:
            return candidate.read_text(errors="replace")[:200_000]
        except (OSError, UnicodeDecodeError):
            continue
    return ""


def inputs_of(split: str):
    """The task's input directory, for any source."""
    if split == "jupyter-agent":
        from smol_ladder.jtasks import input_dir as ja_input_dir
        return ja_input_dir
    if split == "synthetic":
        from smol_ladder.jtasks import synthetic_input_dir
        return synthetic_input_dir
    return input_dir


def schema_dump(row: dict, split: str = "test", limit: int = 3) -> str:
    """A deterministic, model-independent dump of the relevant tables' shape.

    The control condition. Generated by script from the tables, identical for every model.
    """
    answer = str(row["answer"])
    src = inputs_of(split)(row)
    parts = []
    leaked: list[str] = []
    for name in row["files"]:
        f = src / name
        if not f.exists() or f.suffix.lower() in {".sqlite", ".db"}:
            continue
        try:
            head = pd.read_csv(f, nrows=200)
        except Exception:
            continue
        leaked += [c for c in head.columns if normalise(answer) == normalise(str(c))]
        cols = ", ".join(f"{c}: {head[c].dtype}" for c in head.columns)
        parts.append(f"{name}: {len(head.columns)} columns, {head.shape[0]}+ rows\n  columns: {cols}")
        parts.append("  first rows:\n" + head.head(limit).to_string(index=False))
    dump = "\n\n".join(parts) if parts else "(no readable tables)"
    # A column may literally be the gold answer ("winning_label"). Dropping just that column
    # keeps the control honest; the rest of the schema still carries no task information.
    for col in leaked:
        dump = dump.replace(str(col), "<label>")
    return dump


def synthetic_reference(row: dict) -> str | None:
    """The generator's own program for a synthetic task, rendered as reference code.

    A synthetic task's specification *is* its provenance: the answer came from executing these
    ops, not from a model claiming to have found it. That makes it a stronger reference than a
    model-written solution.py, which was selected for reproducing the gold answer and so
    carries the model's own idiom — the confound the article warns about when it says a pass
    must mean the model reasoned rather than recognised its teacher's code.

    Rendered with no print, so it cannot evaluate to the answer, and the answer literal is
    redacted by the caller exactly as for a model reference.
    """
    ops = row.get("ops")
    if not ops:
        return None
    lines = ["import pandas as pd", f"df = pd.read_csv('input/{row['files'][0]}')"]
    chain = ""
    for op in ops:
        name, _, arg = op.partition("(")
        arg = arg.rstrip(")")
        if name == "filter":
            column, _, want = arg.partition("==")
            lines.append(f"sub = df[df['{column}'] == '{want}']")
            chain = "sub"
        elif name == "value_counts":
            lines.append(f"counts = df['{arg}'].value_counts()")
            chain = "counts"
        elif name == "argmax":
            lines.append(f"result = counts.index[0]")
            chain = "scalar"
        else:
            source = "sub" if chain == "sub" else "df"
            lines.append(f"result = {source}['{arg}'].{name}()")
            chain = "scalar"
    return "\n".join(lines)


def read_source(row: dict, split: str) -> str | None:
    """The reference program for a rung, or None if the task has none.

    Two kinds, and the second is the better one. SmolDataEnvs and jupyter-agent supply a
    model-written solution.py, kept only because it reproduced the gold answer offline. A
    synthetic task supplies its own specification rendered as code, which is where its answer
    actually came from, so it is preferred and needs no yield filter at all.

    That preference is not cosmetic. A model reference was *selected* for matching the answer,
    so it arrives in the model's own idiom; handing it back at L4 measures whether the model
    recognises its own teacher's code, not whether it needed the information.
    """
    if split == "synthetic":
        return synthetic_reference(row)
    solutions = DATA / "solutions"
    for candidate in (solutions / split, solutions / "jupyter-agent"):
        result = candidate / row["task_id"] / "result.json"
        if not result.exists():
            continue
        if json.loads(result.read_text()).get("reward", 0.0) < 1.0:
            return None
        solution = result.parent / "solution.py"
        if solution.exists():
            return solution.read_text(errors="replace")
    return None


def strip_output(source: str) -> str:
    """L4 shows the reference program with everything that emits the answer removed.

    Removing only the final print is not enough, and measurably so: stripping just the last
    top-level print left 25/181 (13.8%) of test payloads still printing the gold answer when
    run, because agents use prints as debug output and emit a leaderboard or a per-value sweep
    before the final line. So drop *every* print/display/logging call and every docstring, not
    just the last one. leak_free() then re-runs the result and rejects whatever still leaks.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source

    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if not isinstance(body, list) or not body:
            continue
        kept = []
        for stmt in body:
            if _emits_output(stmt):
                continue
            if (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant)
                    and isinstance(stmt.value.value, str)):
                continue  # bare docstring
            kept.append(stmt)
        node.body = kept or [ast.Pass()]

    try:
        return ast.unparse(ast.fix_missing_locations(tree))
    except (ValueError, RecursionError):
        return source


def _emits_output(stmt: ast.stmt) -> bool:
    """Does this statement print, display, or log anything?"""
    if not isinstance(stmt, ast.Expr):
        return False
    call = stmt.value
    if not isinstance(call, ast.Call):
        return False
    name = getattr(call.func, "id", None) or getattr(call.func, "attr", None)
    return name in {"print", "display", "pprint", "show", "write", "info", "debug", "warning",
                    "error", "log", "logger"}


def redact_literals(source: str, answer: str) -> str:
    """Blank any string literal in the reference code that contains the gold answer.

    A category map like ("Drowning", r"drown") names the answer as plainly as a print does.
    Replacing the literal keeps the program runnable and its structure intact, so the rung
    still supplies the method. leak_free() is the real guard; this is cheap defence in depth.
    """
    needle = normalise(answer)
    if not needle:
        return source
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source
    changed = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if needle in normalise(node.value):
                node.value = "<label>"
                changed = True
    if not changed:
        return source
    try:
        return ast.unparse(ast.fix_missing_locations(tree))
    except (ValueError, RecursionError):
        return source


def code_facts(source: str) -> dict:
    """L2: the files, columns and filters the computation touches, read off the source.

    Static, not a summary written by a model, so the rung is a function of the reference
    solution alone and identical on every run.

    Every fact comes from the AST, because the references are not one idiom. Counting
    df["col"] subscripts on a reader-bound name found 65% of test columns empty: half the
    references are hand-written csv code where the row is a plain dict from DictReader, and the
    rest reach their columns through `.query`, `.loc` with a mask, a helper that returns a
    Series, or SQL. Reading the strings out of the source cannot help there either — the same
    strings are keyword arguments (`sum(axis="columns")`), dict keys and index positions.
    """
    facts = _Facts(source)
    facts.read()
    return {"files": facts.files(), "columns": facts.columns(), "filters": facts.filters()}


def _statements(tree: ast.AST) -> list[str]:
    """The SQL the program runs, as text.

    A statement assigned to a name first and passed to execute later is indistinguishable from
    prose to any single node, so the program's strings are scanned for the pattern instead. An
    f-string's holes are filled with the name being formatted, so `f"WHERE {col} = {v}"` yields
    `WHERE col = col` and its operator still reads."""
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
            continue
        for piece in (node.value, *(p.value for p in getattr(node, "values", []))):
            for text in _sql_statements(piece):
                out.append(text)
    return out


def _sql_statements(text: str) -> list[str]:
    """Every `SELECT ...` / `INSERT ...` / `UPDATE ...` / `DELETE ...` in a piece of text."""
    out = []
    for match in re.finditer(r"(?is)\b(select|insert|update|delete)\b.*?\b\1\b.*|\b"
                             r"(select|insert|update|delete)\b.*", text):
        out.append(" ".join(match.group().split()))
    return out


def _sql_predicates(statement: str) -> list[str]:
    """The WHERE clause, split on AND/OR so each condition reports its own operator."""
    match = re.search(r"(?is)\bwhere\b(.*?)(\bgroup\s+by\b|\border\s+by\b|\blimit\b|\Z)",
                      statement)
    if not match:
        return []
    return re.split(r"(?i)\s+and\s+|\s+or\s+|\bnot\s+", match.group(1).strip())


def _sql_columns(statement: str) -> set[str]:
    """Column names named by a SELECT list, a table clause or a WHERE condition."""
    bare = {word for clause in (re.search(r"(?is)\bselect\b(.*?)(\bfrom\b|\Z)", statement),
                               re.search(r"(?is)\bfrom\b(.*?)(\bwhere\b|\bgroup\s+by\b|\Z)",
                                         statement),
                               re.search(r"(?is)\bwhere\b(.*)", statement))
            if clause for word in _SQL_WORDS.findall(clause.group(1))}
    for predicate in _sql_predicates(statement):
        bare.update(_SQL_WORDS.findall(predicate))
    return {word for word in bare if word.lower() not in _SQL_WORDS_IGNORED}


def method_hint(source: str) -> str:
    """L3: the method, stated in the reference solution's own terms.

    The vocabulary spans the idioms the references are written in. A regex cannot do this: it
    read `os.path.join(BASE_DIR, "in.csv")` and `", ".join(labels)` as dataframe merges and told
    60 of 181 test tasks to join tables they never touch. The rule is the receiver — a method
    name is an operation on data when its receiver is a frame, a series or a column, and string
    formatting when its receiver is a literal or a path.
    """
    facts = _Facts(source)
    facts.read()
    counts = Counter()
    for node in ast.walk(facts.tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        name = node.func.attr
        if name in SQL_CALLS:
            counts["sql"] += 1
        elif name in _FRAME_CALLS and _holds_data(facts, node.func.value):
            counts[name] += 1
    return ", ".join(sorted(counts))


def _compared_columns(node: ast.Compare) -> list[tuple[str, str]]:
    """The string-keyed subscripts on either side of a comparison, with their operator.

    `df["Class"] == 1` and `row["Total"] > best[0]` are the shapes the references write; a
    subscript with a numeric or a variable key indexes rows and a Series, so it is not read as
    a column and contributes nothing."""
    out = []
    for index, side in enumerate([node.left, *node.comparators]):
        op = _op_name(node.ops[index]) if index < len(node.ops) else _op_name(node.ops[-1])
        for part in _walk(side):
            if isinstance(part, ast.Subscript) and isinstance(part.slice, ast.Constant) \
                    and isinstance(part.slice.value, str):
                out.append((part.slice.value, op))
    return out


def _op_name(op: ast.cmpop) -> str:
    return {ast.Eq: "==", ast.NotEq: "!=", ast.Lt: "<", ast.LtE: "<=",
            ast.Gt: ">", ast.GtE: ">="}.get(type(op), "?")


def _holds_data(facts: "_Facts", node: ast.AST) -> bool:
    """Does this receiver hold data, or is it a path or a piece of text?

    `os.path` and `", "` are paths and separators. A name, another attribute, or a subscript
    chain — `df`, `a.merge`, `df["a"].mean()` — is data, as is the output of a known data call.
    This is the whole difference between `a.merge(b)` as an operation and `os.path.join(a, b)`
    as one."""
    if isinstance(node, ast.Constant):
        return False
    if isinstance(node, ast.Name):
        return True
    if isinstance(node, ast.Attribute):
        return node.attr not in _STRING_PATH_ATTRS
    if isinstance(node, ast.Call):
        name = getattr(node.func, "attr", getattr(node.func, "id", None))
        return name in _FRAME_CALLS or name in _READERS
    if isinstance(node, ast.Subscript):
        return facts.kind_of(node.value) in ("frame", "column", "rows")
    return False


class _Facts:
    """The tables, columns and filters one reference program actually uses."""

    def __init__(self, source: str):
        self.source = source
        try:
            self.tree = ast.parse(source)
        except SyntaxError:
            self.tree = ast.Module(body=[], type_ignores=[])
        self.frames: set[str] = set()        # frames, and the column-like objects derived from them
        self.indexes: set[str] = set()      # csv row dicts, keyed by column name
        self.values: dict[str, str] = {}    # a name bound to one of a column's values
        self.columns_found: set[str] = set()
        self.pairs: list[tuple[str, str]] = []

    def files(self) -> list[str]:
        """The tables the program points at. Literals end in a table suffix; a path assembled
        from parts with pathlib does not, so `Path(__file__).parent / "in" / "t.csv"` is read
        off the path expression rather than the literal."""
        out = _file_literals(self.source)
        for node in ast.walk(self.tree):
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div) \
                    and isinstance(node.left, ast.Attribute) and node.left.attr in _PATH_ATTRS:
                parts = [piece.value for piece in ast.walk(node)
                         if isinstance(piece, ast.Constant) and isinstance(piece.value, str)]
                if parts and parts[-1].endswith(TABLE_SUFFIXES):
                    out.append("/".join(parts[-2:]))
        return sorted(set(out))

    def read(self) -> None:
        """Two passes, both in source order, so a fact is recorded after the call that
        establishes it. `rows[0]["Type 2"]` is a column access only if rows was a DictReader, and
        `grouped["Total"].sum()` only if grouped was a groupby — so every binding is classified
        before any subscript is read. The for-loop over a table that assigns inside its body
        needs the order kept, so it runs its own scan."""
        for node in _walk(self.tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    self.bind(target, node.value)
            elif isinstance(node, ast.AnnAssign) and node.value is not None:
                self.bind(node.target, node.value)
            elif isinstance(node, (ast.For, ast.comprehension)):
                self.bind(node.target, node.iter)
            elif isinstance(node, ast.withitem) and node.optional_vars is not None:
                self.bind(node.optional_vars, node.context_expr)
        for node in _walk(self.tree):
            if isinstance(node, ast.Subscript):
                self.subscript(node)
            elif isinstance(node, ast.Call):
                self.method(node)
            elif isinstance(node, ast.Compare):
                self.compare(node)

    def bind(self, target: ast.expr, value: ast.expr) -> None:
        if isinstance(target, ast.Name):
            self.classify(target.id, value)
            if target.id in _VALUE_BINDINGS and isinstance(value, ast.Constant) \
                    and isinstance(value.value, str):
                self.values[target.id] = value.value
        elif isinstance(target, (ast.Tuple, ast.List)):
            for element in target.elts:
                self.bind(element, value)

    def classify(self, name: str, value: ast.expr) -> None:
        """What a variable holds: a frame, csv rows, or a series of counts. A call that cannot
        be classified leaves the name unknown, and an unknown name contributes no columns —
        which is the whole point: `d["key"]` on a plain dict is a lookup, not a column."""
        if not isinstance(value, ast.Call):
            return
        called = getattr(value.func, "attr", getattr(value.func, "id", None))
        if called in _ROW_READERS:
            self.indexes.add(name)          # a csv.DictReader row is a dict, not a frame
        elif called in _PANDAS_READERS \
                or (called in _READERS and _holds_data(self, value.func.value)):
            self.frames.add(name)
        elif self.is_rows(value):
            self.indexes.add(name)

    def is_rows(self, node: ast.AST) -> bool:
        """A csv row dict: anything built from a DictReader. `for row in csv.DictReader(f)`
        binds a dict keyed by column name, and so does every list or generator over one."""
        for part in _walk(node):
            if isinstance(part, ast.Call):
                name = getattr(part.func, "attr", getattr(part.func, "id", None))
                if name in _ROW_READERS:
                    return True
            elif isinstance(part, ast.Name) and part.id in self.indexes:
                return True
        return False

    def method(self, node: ast.Call) -> None:
        """The three ways a reference states a filter over a frame: `df.query("…")`,
        `df.loc[mask]`, and `db.execute("… WHERE …")`. Every other method only reads."""
        name = getattr(node.func, "attr", None)
        if name in SQL_CALLS:
            self.sql()
        elif name == "query" and node.args:
            for text in _strings(node.args[0]):
                self.query(text)
        elif name in ("loc", "at", "mask", "where"):
            for arg in node.args:
                for part in _walk(arg):
                    if isinstance(part, ast.Compare):
                        self.compare(part)

    def query(self, text: str) -> None:
        """`df.query` is a boolean expression over columns, written as a string."""
        for clause in re.split(r"(?i)\s+and\s+|\s+or\s+|\bnot\s+", text):
            match = re.match(r"\s*([\w .%()]+?)\s*(==|!=|<=|>=|<|>|\.isin|\.between)\s*(.*)$",
                             clause)
            if match:
                self.pair(match.group(1).strip(), _QUERY_OPS[match.group(2)])
                continue
            words = _SQL_WORDS.findall(clause)
            if len(words) > 1:
                self.pair(words[0], "==")

    def subscript(self, node: ast.Subscript) -> None:
        """A string key under a frame, a column or a row dict is a column name. A string under
        anything else — a keyword argument, a plain list, an unknown name — is a word."""
        if _positional_index(node.value, self.indexes):
            return
        key = node.slice
        if not (isinstance(key, ast.Constant) and isinstance(key.value, str)):
            # df[df["X"] > 5], df.loc[mask]: the mask's comparisons name the columns.
            for part in _walk(key):
                if isinstance(part, ast.Compare):
                    self.compare(part)
            return
        if self.kind_of(node.value) in ("column", "rows", "frame"):
            self.add_column(key.value)

    def compare(self, node: ast.Compare) -> None:
        """A comparison that reads a column is a filter: `df[df["X"] > 5]`, or `row["X"] == "v"`
        in a csv loop. The columns it names are recorded, and so are the comparisons in a
        chained condition, since `total > best and row["Name"] != ""` filters on both.

        A comparison against a bound value names its column the other way round: `where == CITY`
        filters on the column CITY, which is what the constant's name says."""
        for column, op in _compared_columns(node):
            self.pair(column, op)
        for side in (node.left, *node.comparators):
            if isinstance(side, ast.Name) and side.id in self.values:
                self.add_column(self.values[side.id])

    def pair(self, column: str, op: str) -> None:
        self.add_column(column)
        self.pairs.append((column, op))

    def sql(self) -> None:
        """The SQL a reference runs against its sqlite tables: the columns it selects, and the
        conditions in its WHERE clause. Both are filters and columns in the ordinary sense, so
        a task answered with a query reports the same shape as one answered with a mask."""
        for statement in _statements(self.tree):
            self.columns_found.update(_sql_columns(statement))
            for predicate in _sql_predicates(statement):
                words = _SQL_WORDS.findall(predicate)
                if not words:
                    continue
                op = _sql_op(predicate)
                if len(words) > 1:
                    self.pair(words[0], op)
                else:
                    self.add_column(words[0])

    def kind_of(self, node: ast.AST) -> str:
        """Follow the access chain to its root and say what the expression is.

        `df` is a frame, `df['a']` and `frame.iloc[2]` are frames again, `frame.columns` and
        `counts.index` are columns, and a csv row dict is a row. An unclassified root is
        unknown, and an unknown root yields no column name."""
        chain: list[ast.AST] = []
        current = node
        while True:
            if isinstance(current, ast.Subscript):
                chain.append(current)
                current = current.value
            elif isinstance(current, ast.Attribute):
                chain.append(current)
                current = current.value
            elif isinstance(current, ast.Call) and isinstance(current.func, ast.Attribute):
                chain.append(current)
                current = current.func.value
            else:
                break
        for link in chain:
            if isinstance(link, ast.Attribute) and link.attr in _COLUMN_ATTRS:
                return "column"
            if isinstance(link, (ast.Subscript, ast.Call)):
                return "frame"
        if isinstance(current, ast.Name):
            if current.id in self.frames:
                return "frame"
            if current.id in self.indexes:
                return "rows"
            return "unknown"
        if isinstance(current, ast.Call):
            name = getattr(current.func, "attr", getattr(current.func, "id", None))
            if name in _PANDAS_READERS or name in _FRAME_CALLS:
                return "frame"
        return "unknown"

    def add_column(self, name: str) -> None:
        """Record a column name. `_`, `axis` and `encoding` reach the extractor as
        `sum(axis="columns")`; they are arguments, not fields of the table."""
        if not name or name in _NOT_A_COLUMN:
            return
        self.columns_found.add(name)

    def columns(self) -> list[str]:
        return sorted(self.columns_found)[:20]

    def filters(self) -> list[str]:
        return [f"{column} {op}" for column, op in list(dict.fromkeys(self.pairs))[:12]]


def _file_literals(source: str) -> list[str]:
    """File names in the source's string literals. A literal counts as a path when its last
    segment ends in a table suffix, so `input/wine.csv` is one and `", "`, `"releaseType"` and
    `yes` are not. Segments are taken whole: a literal that merely contains a table name, as a
    docstring does, is not a file the program reads."""
    out = []
    for match in re.finditer(r"""['"]([\w./\\ -]{2,80})['"]""", source):
        literal = match.group(1)
        segments = [part for part in re.split(r"[/\\]", literal) if part]
        if segments and segments[-1].endswith(TABLE_SUFFIXES):
            out.append(literal)
    return out


def _walk(node: ast.AST):
    """Every node under this one, the expression itself included."""
    stack = [node]
    while stack:
        current = stack.pop()
        yield current
        stack.extend(ast.iter_child_nodes(current))


def _positional_index(node: ast.AST, indexes: set[str]) -> bool:
    """rows[i] picks a row rather than a column, so the string key under it is not a field."""
    if not isinstance(node, ast.Subscript):
        return False
    index = node.slice
    return ((isinstance(index, ast.Constant) and isinstance(index.value, int))
            or (isinstance(index, ast.Name) and index.id in indexes)
            or isinstance(index, ast.Slice))


def _sql_op(predicate: str) -> str:
    if re.search(r"(?i)\bnot\s+in\b", predicate):
        return "!="
    return "=="


def _strings(node: ast.AST) -> list[str]:
    """The literal parts of a string, however it was built. An f-string's holes are elided
    rather than filled, so a query built from one still contributes its operator, and a hole
    that leaves nothing to name contributes no column."""
    out: list[str] = []
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        out.append(node.value)
    elif isinstance(node, ast.JoinedStr):
        out.extend(piece.value for piece in node.values if isinstance(piece, ast.Constant))
    return out


def input_files(row: dict, split: str) -> list[str]:
    """What is actually in ./input, read from the directory rather than the dataset row.

    The dataset row lists the files *the original notebook used*, which is not the same set as
    the bucket's contents: 56% of synthetic and 28% of SmolDataEnvs tasks had L1 announce one
    file while `ls ./input` showed several. A prompt that misdescribes the environment is not a
    rung, it is a confound that every higher rung inherits — and naming a subset of the files is
    itself a hint about which table the question is about, which is information the schema
    control exists to isolate.

    Falls back to the row's own list if the directory cannot be read, so a prompt is always
    buildable.
    """
    try:
        listing = sorted(p.name for p in inputs_of(split)(row).iterdir()
                         if not p.name.startswith("."))
    except Exception:
        listing = []
    return listing or sorted(row["files"])


def prompt_for(row: dict, split: str, rung: str) -> str:
    files = "\n".join(f"- {f}" for f in input_files(row, split))
    base = PROMPT.format(question=row["question"], files=files)
    if rung == "L1":
        return base
    if rung == "L1+schema":
        return base + f"\n\nSchema of the input tables:\n\n{schema_dump(row, split)}"
    source = read_source(row, split)
    if source is None:
        # Falling back to the plain prompt would make this rung byte-identical to L1: the
        # article's own table has a column for exactly this ("Nothing new, because the cut
        # already left them"), and a rung that adds no information cannot be a distinct rescue
        # — it silently re-measures L1. Say so instead of pretending the rung exists.
        return base + "\n\n(No reference solution is available for this task, so this rung adds " \
                      "nothing above the question.)"
    facts = code_facts(source)
    l2_block = (
        f"Files read: {', '.join(facts['files']) or 'the tables above'}\n"
        f"Columns used: {', '.join(facts['columns']) or '(discover them yourself)'}\n"
        f"Filters applied: {'; '.join(facts['filters']) or '(none)'}"
    )
    if rung == "L2":
        return base + f"\n\nNotes on the intended computation:\n\n{l2_block}"
    if rung == "L3":
        return (base
                + "\n\nNotes on the intended computation:\n\n"
                + f"{l2_block}\n"
                + f"Method: {method_hint(source)}")
    if rung == "L4":
        # Cumulative: L4 must extend L3 verbatim, or the rungs are not Blackwell-ordered and
        # "the lowest rung that passes" stops meaning "the least information that sufficed".
        # Dropping L3's block here made L4 a sibling of L3 rather than a superset, which is
        # the whole ordering claim the ladder rests on.
        return (base
                + "\n\nNotes on the intended computation:\n\n"
                + f"{l2_block}\n"
                + f"Method: {method_hint(source)}"
                + HINT_HEADER + "\n```python\n"
                + redact_literals(strip_output(source), str(row["answer"])) + "\n```")
    raise ValueError(rung)


def leak_free(row: dict, payload: str, split: str) -> bool:
    """The leak oracle: run the stripped reference and see whether it still yields the answer.

    A string match cannot see this. A program can name the answer in a label map, a threshold
    comparison, or a hardcoded hyperparameter and never print it. So the only trustworthy test
    is behavioural: write the payload out, run it offline against the task's tables, and keep
    the rung only if it no longer grades as correct. Fails closed.
    """
    work = DATA / "ladder" / "_leakcheck" / row["task_id"]
    work.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True, exist_ok=True)
    (work / "input").symlink_to(inputs_of(split)(row).resolve())
    (work / "solution.py").write_text(payload)
    run = run_script(work / "solution.py", work / "input", timeout=90)
    leaked = grade(row, last_line(run.stdout)) >= 1.0
    shutil.rmtree(work, ignore_errors=True)
    return not leaked


def normalise(text: str) -> str:
    return re.sub(r"[\s,_%$]", "", str(text)).lower()


# Answers this short match by accident almost anywhere: "3" occurs in every column dump,
# "yes"/"no" in prose. Flagging them would bury the real leaks in noise.
MIN_LEAK_LEN = 4


def leaks(row: dict, prompt: str) -> list[str]:
    """Does the hint hand over the answer? Checked two ways: as text, and as a number.

    The numeric test calls the dataset's own grader, so it asks exactly the question the
    reward will ask. A hand-rolled tolerance is stricter or looser than the real one by orders
    of magnitude depending on the task's rtol, which is how leaks slip through unnoticed.

    Two guards keep the signal useful. A short answer is skipped, because it matches by
    chance; and main() only counts a hit on a rung if L1 does not already have it, so an
    answer that the question itself names is never blamed on a hint.
    """
    answer = str(row["answer"])
    if len(normalise(answer)) < MIN_LEAK_LEN:
        return []
    if normalise(answer) in normalise(prompt):
        return ["answer-substring"]
    hits = []
    # Match numbers with their decimal point, not a digit run inside one: the pattern must not
    # pull "11" out of "0.11" or "1" out of "0.999", which then grade 1.0 on a loose atol and
    # turn every dumped float into a false leak.
    for match in re.finditer(r"(?<![\w.])-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", prompt):
        candidate = match.group()
        try:
            if grade(row, candidate) >= 1.0:
                hits.append(f"answer-numeric:{candidate}")
                break
        except Exception:
            continue
    return hits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test",
                    choices=["test", "eval", "train", "jupyter-agent", "synthetic"])
    ap.add_argument("--limit", type=int)
    ap.add_argument("--leak-check", action="store_true", default=True)
    ap.add_argument("--no-run-oracle", action="store_true",
                    help="skip running each L4 payload; much faster, but ships unverified rungs")
    args = ap.parse_args()

    from smol_ladder.run_ladder import source_for
    rows = source_for(args.split)[0][: args.limit]
    out = DATA / "ladder" / args.split
    out.mkdir(parents=True, exist_ok=True)
    stats = {r: 0 for r in RUNGS}
    n_control = 0
    n_no_reference = n_excluded = 0
    real_leaks: dict[str, list[str]] = {}
    task_inherent: list[str] = []
    for i, row in enumerate(rows, 1):
        task = out / row["task_id"]
        task.mkdir(parents=True, exist_ok=True)
        record = {"task_id": row["task_id"], "difficulty_tier": row["difficulty_tier"]}
        source = read_source(row, args.split)
        have_source = source is not None
        if not have_source:
            n_no_reference += 1
        l1 = prompt_for(row, args.split, "L1")
        excluded = None
        if have_source and args.leak_check and not args.no_run_oracle:
            # Run the actual L4 payload. If stripping every print still yields the answer, the
            # rung is an answer key for this task and is dropped rather than repaired.
            payload = redact_literals(strip_output(source), str(row["answer"]))
            if not leak_free(row, payload, args.split):
                excluded = "L4-still-evaluates-to-gold"
                n_excluded += 1
        record["l4_excluded"] = excluded
        for rung in list(RUNGS) + ["L1+schema"]:
            prompt = prompt_for(row, args.split, rung)
            if args.leak_check and rung != "L4":
                # Only what a rung ADDS counts. L2-L4 are cumulative, so an answer already in
                # the question (52/250 tasks are multiple choice, where the answer is one of
                # the options) would otherwise be blamed on every hint.
                hits = leaks(row, prompt) if rung == "L1" else [
                    h for h in leaks(row, prompt) if h not in leaks(row, l1)
                ]
                if hits:
                    record.setdefault("leaks", {})[rung] = hits
                    if rung == "L1":
                        # The unmodified question already contains it: a property of the task.
                        task_inherent.append(row["task_id"])
                    else:
                        real_leaks.setdefault(row["task_id"], []).extend(
                            f"{rung}:{h}" for h in hits)
            if rung == "L1+schema":
                n_control += 1
                continue
            if excluded and rung == "L4":
                continue
            if rung == "L1" or have_source:
                (task / f"{rung}.md").write_text(prompt)
                stats[rung] += 1
        (task / "meta.json").write_text(json.dumps(record, indent=1))
        if i % 25 == 0:
            print(f"  [{i}/{len(rows)}] L4={stats['L4']} excluded={n_excluded}", flush=True)
    for task_id, hits in sorted(real_leaks.items()):
        print(f"LEAK {task_id}: {sorted(set(hits))}")
    # The funnel is the headline, not a footnote: every rung number below is conditional on it.
    print(f"{args.split}: {len(rows)} tasks")
    print(f"  no verified reference (no L2-L4): {n_no_reference}")
    print(f"  L4 excluded by run-and-grade oracle: {n_excluded}")
    print(f"  L1={stats['L1']} L2={stats['L2']} L3={stats['L3']} L4={stats['L4']} "
          f"control={n_control}")
    print(f"  residual text leaks={len(real_leaks)} task_inherent={len(set(task_inherent))}")


if __name__ == "__main__":
    main()
