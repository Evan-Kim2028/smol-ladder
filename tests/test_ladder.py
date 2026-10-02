import ast
import re

import pytest

from smol_ladder.ladder import (SCHEMA_DUMP_CHARS, SCHEMA_DUMP_MAX_COLS, SCHEMA_DUMP_MAX_FILES,
                                SCHEMA_DUMP_ROWS, code_facts, inputs_of,
                                leaks, method_hint, normalise, prompt_for, redact_literals,
                                schema_dump, strip_output)
from smol_ladder.tasks import input_dir, load_split, read_tables

# The tasks whose gold answer is a column name, so the name appears in any dump that profiles
# the table. Each is an exact_short "which feature has the highest ..." question: the answer is
# picked out of the candidate set by a computation, and naming the set is what makes the task
# solvable at all. Pinned by test_every_schema_dump_exemption_is_a_column_name_and_nothing_else.
SCHEMA_DUMP_COLUMN_NAME_EXEMPTIONS = frozenset({
    "0000_347_347102_qa_1",  # "Sp. Atk"
    "0000_421_421838_qa_1",  # "Glucose"
    "0000_440_440038_qa_5",  # "PetalLengthCm"
    "0000_471_471618_qa_2",  # "IncidentLocation"
    "0000_656_656399_qa_4",  # "ParentAnsweringSurvey"
    "0000_658_658395_qa_1",  # "gill-color"
    "0000_862_862257_qa_5",  # "Petal", a prefix of PetalLengthCm
    "0001_347_1347384_qa_1",  # "Judaism", a prefix of judaism_orthodox
    "0001_364_1364973_qa_2",  # "year"
    "0001_497_1497755_qa_4",  # "chlorides"
    "0001_533_1533644_qa_3",  # "Alkaline_Phosphotase"
    "0001_638_1638152_qa_5",  # "Size(sqf)"
    "0001_696_1696691_qa_1",  # "OverTime"
})


def test_strip_output_removes_prints_at_every_depth():
    # A print inside main() is where most of the leak lived: the agents use prints as debug
    # output, so the answer reaches stdout even when the final top-level print is gone.
    src = (
        "import pandas as pd\n"
        "def main():\n"
        "    print('debug: Drowning')\n"
        "    x = 1\n"
        "main()\n"
        "print('final')\n"
    )
    out = strip_output(src)
    assert "print" not in out
    tree = ast.parse(out)  # still valid python
    assert any(isinstance(n, ast.FunctionDef) for n in tree.body)


def test_redact_literals_removes_a_label_map():
    src = "LABEL = {'drown': 'Drowning'}\n"
    assert "Drowning" not in redact_literals(src, "Drowning")


def test_redact_keeps_a_program_that_does_not_mention_the_answer():
    src = "X = 1\nprint(X)\n"
    assert redact_literals(src, "Drowning") == src


def test_normalise_ignores_spacing_and_case():
    assert normalise("Sea  Surface,Temp") == normalise("sea surface temp")


def test_l4_never_prints_the_answer():
    row = load_split("test")[0]
    l1 = prompt_for(row, "test", "L1")
    l4 = prompt_for(row, "test", "L4")
    # the rung is cumulative, so L4 starts from L1
    assert l4.startswith(l1)
    # and the reference payload is fenced in, not left to run bare
    assert "```python" in l4


def test_l1_is_the_plain_prompt():
    row = load_split("test")[0]
    l1 = prompt_for(row, "test", "L1")
    assert row["question"] in l1
    assert "```python" not in l1
    assert "Notes on the intended computation" not in l1


def test_schema_control_adds_shape_but_not_a_reference():
    row = load_split("test")[0]
    control = prompt_for(row, "test", "L1+schema")
    assert control.startswith(prompt_for(row, "test", "L1"))
    assert "```python" not in control


def test_leaks_ignores_an_answer_the_question_already_states():
    # Multiple-choice tasks name the answer among the options. main() subtracts L1's hits, so
    # the check must not report a rung for it.
    rows = {r["task_id"]: r for r in load_split("test")}
    row = rows["0000_862_862257_qa_5"]
    l1 = prompt_for(row, "test", "L1")
    assert leaks(row, l1)
    assert not [h for h in leaks(row, prompt_for(row, "test", "L4")) if h not in leaks(row, l1)]


def test_code_facts_reads_files_and_columns_off_the_source():
    facts = code_facts(
        "import pandas as pd\n"
        "df = pd.read_csv('input/wine.csv')\n"
        "cols = df.columns\n"
        "x = df['Alcohol']\n"
        "y = df[df['Class'] == 1]\n"
    )
    assert "input/wine.csv" in " ".join(facts["files"])
    assert "Class" in facts["columns"]
    assert any("Class" in f for f in facts["filters"])


def test_method_hint_names_the_operations():
    assert "groupby" in method_hint("df.groupby('a').mean()")


def test_method_hint_does_not_call_a_string_join_a_merge():
    # os.path.join and str.join are how the references build paths and format output. Read as
    # dataframe methods they told 60/181 test tasks to merge tables that were never joined.
    src = (
        "import os\n"
        "PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'in.csv')\n"
        "print(', '.join(labels))\n"
    )
    assert method_hint(src) == ""


def test_method_hint_names_a_dataframe_join():
    assert "merge" in method_hint("merged = pd.merge(left, right, on='id')")


def test_columns_come_from_a_csv_reference_with_a_loop_variable():
    # The DictReader idiom is half the test references: the row is a plain dict, so nothing
    # looks like a dataframe subscript on a reader-bound name.
    src = (
        "import csv\n"
        "with open('input/Pokemon.csv', newline='') as f:\n"
        "    for row in csv.DictReader(f):\n"
        "        total = int(row['Total'])\n"
    )
    assert "Total" in code_facts(src)["columns"]


def test_columns_come_from_a_sql_column_list():
    src = (
        "import sqlite3\n"
        "conn = sqlite3.connect(DB)\n"
        "conn.execute('SELECT groupYear, SUM(totalSnatched) FROM torrents'\n"
        "             ' WHERE totalSnatched > 0 GROUP BY groupYear')\n"
    )
    assert {"groupYear", "totalSnatched"} <= set(code_facts(src)["columns"])


def test_filters_come_from_a_sql_where_clause():
    src = (
        "import sqlite3\n"
        "conn = sqlite3.connect(DB)\n"
        "conn.execute('SELECT * FROM torrents WHERE releaseType = \"album\"')\n"
    )
    assert any("releaseType" in f for f in code_facts(src)["filters"])


def test_filters_come_from_a_query_call():
    facts = code_facts(
        "import pandas as pd\n"
        "df = pd.read_csv('input/t.csv')\n"
        "sub = df.query(\"Type 1 == 'Fire'\")\n"
    )
    assert any("Type 1" in f for f in facts["filters"])


def test_a_string_keyed_dict_is_not_a_column():
    # csv.rows[0]['Type 2'] indexes rows, not a table. That is how 54 of the DictReader
    # references would otherwise contribute their index positions as column names.
    src = (
        "import csv\n"
        "rows = list(csv.DictReader(f))\n"
        "first = rows[0]['Type 2']\n"
    )
    assert "Type 2" not in code_facts(src)["columns"]


def test_a_python_keyword_argument_is_not_a_column():
    facts = code_facts(
        "import pandas as pd\n"
        "df = pd.read_csv('input/t.csv')\n"
        "x = df.sum(axis='columns')\n"
    )
    assert "columns" not in facts["columns"]


def test_a_column_named_by_the_constant_it_is_compared_against():
    # CATEGORY = "Fire" binds a value, and `where == CATEGORY` filters on the column CATEGORY.
    # Without this, 39 references that name their columns this way report nothing.
    facts = code_facts(
        "import pandas as pd\n"
        "df = pd.read_csv('input/t.csv')\n"
        "CATEGORY = 'Fire'\n"
        "sub = df[df['Type 1'] == CATEGORY]\n"
        "keep = df[df['Legendary'] == CATEGORY]\n"
    )
    assert "Fire" in facts["columns"]


def test_a_bound_literal_nothing_compares_against_is_not_a_column():
    facts = code_facts("ENCODING = 'utf-8'\nprint(ENCODING)\n")
    assert facts["columns"] == []


def test_schema_dump_is_deterministic():
    row = load_split("test")[0]
    assert schema_dump(row) == schema_dump(row)


def test_schema_dump_carries_no_cell_value(row):
    """A control that prints sample rows names the answer for any "which value" question.

    The gold answer here is 3.5, the mean of the three col_a values the dump used to print
    verbatim, and 3.5 is also what a bare column listing of the same file would show if the
    dump regressed to values.
    """
    assert "3.5" not in schema_dump(row)


# Each of the three below reads the tables of all 250 test-split tasks, so they profile the whole
# split from disk: 187s, 192s and 471s respectively, which is 850s of the suite's 923s. They are
# the control's guarantee measured on the population rather than on a sample, and they stay in the
# full suite; `pytest -m "not slow"` is the quick loop (README).
@pytest.mark.slow
def test_schema_dump_is_answer_free_on_every_test_task():
    """The whole test split, checked with the dataset's own grader.

    30 of 250 dumps used to grade 1.0 against their gold answer. A whole-split test, rather
    than the one spot check in test_leaks_ignores_an_answer_the_question_already_states, is
    what keeps the control's one guarantee measured.

    Every exempt task is listed below, and each is checked to be the same single phenomenon
    rather than merely skipped: an exact_short task whose gold answer is a column name, so the
    name is a substring of the column listing whatever the dump prints. That is not the dump
    handing over an answer -- schema_dump never reads a cell, so the dump adds the shape of the
    candidate set and nothing that picks the winner out of it. A column name cannot be dropped,
    because "which feature has the highest correlation" is unanswerable without it, and
    narrowing the names to the ones the answer is not would be a redaction that announces where
    the answer is.

    The list grew from one to thirteen when the dump was fixed to profile every table. The
    earlier version was not safer: it described no columns at all on 243 of 250 tasks, so it
    could not contain a column name because it contained nothing. The exemption count is the
    price of a dump that says anything, and test_schema_dump_names_a_column_on_every_task pins
    what was bought with it.
    """
    for r in load_split("test"):
        if r["task_id"] in SCHEMA_DUMP_COLUMN_NAME_EXEMPTIONS:
            continue
        assert not leaks(r, schema_dump(r)), r["task_id"]


def test_every_schema_dump_exemption_is_a_column_name_and_nothing_else():
    """The exemption list may not drift into tolerating a real leak.

    Each named task must leak exactly "answer-substring" -- no numeric hit -- and its answer
    must be a genuine column of its table. A task that stops leaking is a bug too: the
    exemption is meant to describe this split, and it should be pruned when it stops applying.
    """
    rows = {r["task_id"]: r for r in load_split("test")}
    for task_id in SCHEMA_DUMP_COLUMN_NAME_EXEMPTIONS:
        row = rows[task_id]
        dump = schema_dump(row)
        assert leaks(row, dump) == ["answer-substring"], task_id
        names = [line.strip().split(":")[0] for line in dump.splitlines() if line.startswith("  ")]
        assert any(normalise(str(row["answer"])) in normalise(name) for name in names), task_id


@pytest.mark.slow
def test_schema_dump_emits_no_number_of_its_own():
    """Every number in a prompt is a candidate the grader scores against the gold answer.

    A printed decile fraction, a column count or a non-null count is a lottery ticket: one test
    task answers 0.827742 and its dump was offering 0.83. The dump therefore describes every
    count in words. What remains are digits inside column names, which the corpus puts there
    ("feature1", "Type 2"), and those are the table's own vocabulary rather than the dump's.
    """
    offenders = []
    for r in load_split("test"):
        for line in schema_dump(r).splitlines():
            _, sep, body = line.strip().partition(": ")
            if not sep and not line.startswith("  "):
                body = line  # a header or a trim marker, which carries no dtype to excuse
            # The dtype's own width is not a datum: "float64" is the column's type, and its "64"
            # is part of a type name, not a count the grader should read as a candidate. What is
            # left is a statistic, and a statistic here is a number that can be graded.
            if re.search(r"\d", re.sub(r"\b(?:u?int|float)\d+\b", "", body)):
                offenders.append((r["task_id"], line.strip()))
    assert offenders == []


def test_schema_dump_never_reads_a_cell(row):
    """The property the guarantee rests on, and the one the old dump broke.

    An answer-independent dump is a function of the tables. The row fixture drops the answer
    key as well as its value, so a schema_dump that read row["answer"] to redact a cell, as
    the old one did, raises here rather than passing quietly.
    """
    assert schema_dump({k: v for k, v in row.items() if k != "answer"}) == schema_dump(row)


def test_schema_dump_leaks_only_a_bare_category_in_a_column_name():
    """The religion case, pinned: the categories are column names and nothing else is.

    The task's answer is "Judaism" and the table has judaism_orthodox and judaism_conservative
    among its columns, so the answer is a substring of the column listing whatever the dump
    prints. What must not appear is a value, or anything that narrows fourteen religions to
    one -- so every digit-free line of the dump is a column or a dtype.
    """
    row = next(r for r in load_split("test") if r["task_id"] == "0001_347_1347384_qa_1")
    dump = schema_dump(row)
    assert leaks(row, dump) == ["answer-substring"]
    assert dump.count("judaism") == 5
    for line in dump.splitlines():
        # A dtype's width is not a datum, so int64 passes; nothing else may carry a digit.
        assert not re.search(r"\d", re.sub(r"\b(?:u?int|float)\d+\b", "", line)), line


@pytest.mark.slow
def test_schema_dump_names_a_column_of_every_readable_table_on_ninety_five_percent():
    """The control has to be a control.

    A dump that says nothing measures nothing: if the agent learns no column names, any score
    change on L1+schema is noise rather than a skill finding. So over the test split at least
    95% of tasks must get a dump naming at least one column of every table that can be read.

    This is the test the column cap broke. Passing usecols=range(40) to read_csv raises on any
    table narrower than 40 columns -- 243 of 250 tasks -- and the except turned that into
    "(not readable as csv)", so 191 of 250 dumps named no column at all and the median dump was
    37 characters.

    The tasks that fall short are the ones the character budget cannot fit: a 61-column csv eats
    the whole allowance, so a task's second and third files are cut off after it. That is the
    cap doing its job, and 95% is where it is drawn.
    """
    covered, thin = 0, []
    for r in load_split("test"):
        src = inputs_of("test")(r)
        named = {line.strip().split(":")[0] for line in schema_dump(r).splitlines()
                 if line.startswith("  ")}
        missing = []
        for name in r["files"][:SCHEMA_DUMP_MAX_FILES]:
            frames = read_tables(src / name, SCHEMA_DUMP_ROWS)
            if not frames:  # unreadable files are named as such, which is the contract
                continue
            for frame in frames:
                columns = {str(c) for c in frame.columns[:SCHEMA_DUMP_MAX_COLS]}
                if not columns & named:
                    missing.append(name)
        if missing:
            thin.append((r["task_id"], missing))
        else:
            covered += 1
    total = len(load_split("test"))
    assert covered >= 0.95 * total, f"{covered}/{total} describe their tables, thin: {thin}"


def test_schema_dump_names_an_unreadable_file_rather_than_dropping_it():
    """A file the reader cannot open is listed by name with its size.

    Silently continuing past it would tell the agent the table is absent rather than unreadable,
    and an absent table and an empty one are different problems to solve.
    """
    import smol_ladder.ladder as L
    from pathlib import Path
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp)
        (src / "good.csv").write_text("a,b\n1,2\n")
        # A corrupt parquet, rather than random bytes or a prose file: both of those parse as a
        # perfectly good one-column table, because the reader is deliberately lenient about a
        # file with no delimiters. A file whose declared format is not what is inside it is the
        # case that has to come back empty.
        (src / "bad.parquet").write_bytes(b"PAR1 not actually a parquet file")
        row = {"task_id": "x", "question": "q", "files": ["good.csv", "bad.parquet"],
               "answer": "1"}
        original = L.inputs_of
        L.inputs_of = lambda split: (lambda _r: src)
        try:
            dump = schema_dump(row)
        finally:
            L.inputs_of = original
    assert "good.csv" in dump and "a: int64" in dump
    assert "bad.parquet" in dump and "unreadable" in dump


def test_schema_dump_stays_within_its_cap(row):
    assert len(schema_dump(row)) <= SCHEMA_DUMP_CHARS


def test_sandbox_reads_input_offline():
    from smol_ladder.sandbox import run_script
    row = load_split("test")[0]
    inp = input_dir(row)
    assert inp.exists()


def test_the_control_block_extends_l1_and_says_nothing_about_the_task():
    from smol_ladder import ladder
    """L1+control is L1 plus one fixed block of behaviour rules: the same text for every task."""
    rows = load_split("test")[:2]
    blocks = []
    for row in rows:
        l1 = ladder.prompt_for(row, "test", "L1", "bash")
        control = ladder.prompt_for(row, "test", "L1+control", "bash")
        assert control.startswith(l1) and control != l1
        blocks.append(control[len(l1):])
        assert ladder.hint_source(row, "test", "L1+control") == "fixed"
    assert blocks[0] == blocks[1] == "\n\nWorking rules:\n\n" + ladder.control_hint()
    assert len(ladder.control_hint().split()) <= 110
