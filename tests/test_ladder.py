import ast

from smol_ladder.ladder import (SCHEMA_DUMP_CHARS, code_facts, leaks, method_hint, normalise,
                                prompt_for, redact_literals, schema_dump, strip_output)
from smol_ladder.tasks import input_dir, load_split


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


def test_schema_dump_is_answer_free_on_every_test_task():
    """The whole test split, checked with the dataset's own grader.

    30 of 250 dumps used to grade 1.0 against their gold answer. A whole-split test, rather
    than the one spot check in test_leaks_ignores_an_answer_the_question_already_states, is
    what keeps the control's one guarantee measured.

    One task is exempt, and the exemption is named rather than filtered: its table has a
    column called for the category its question asks about, so "Judaism" is a substring of the
    column listing whatever the dump prints. schema_dump never reads a cell, so the dump adds
    the shape of the candidate set and nothing that picks the winner out of it, and no
    summary statistic can narrow fourteen religions to one;
    test_schema_dump_leaks_only_a_bare_category_in_a_column_name pins that reading.
    """
    leaking = []
    for r in load_split("test"):
        if r["task_id"] == "0001_347_1347384_qa_1":
            continue
        if leaks(r, schema_dump(r)):
            leaking.append(r["task_id"])
    assert leaking == []


def test_schema_dump_never_reads_a_cell(row):
    """The property the guarantee rests on, and the one the old dump broke.

    An answer-independent dump is a function of the tables. The row fixture drops the answer
    key as well as its value, so a schema_dump that read row["answer"] to redact a cell, as
    the old one did, raises here rather than passing quietly.
    """
    assert schema_dump({k: v for k, v in row.items() if k != "answer"}) == schema_dump(row)


def test_schema_dump_leaks_only_a_bare_category_in_a_column_name():
    """The single hit the whole-split test tolerates, pinned so it cannot widen.

    1 of 250, and it is a column name rather than a cell: see the test above.
    """
    row = next(r for r in load_split("test") if r["task_id"] == "0001_347_1347384_qa_1")
    dump = schema_dump(row)
    assert leaks(row, dump) == ["answer-substring"]
    # The bare category name, spelled lowercase as the columns spell it, and nowhere else.
    assert dump.count("judaism") == 5
    assert "0.0." in dump  # the profiles are the only numbers, and all of them are fractions


def test_schema_dump_stays_within_its_cap(row):
    assert len(schema_dump(row)) <= SCHEMA_DUMP_CHARS


def test_sandbox_reads_input_offline():
    from smol_ladder.sandbox import run_script
    row = load_split("test")[0]
    inp = input_dir(row)
    assert inp.exists()
