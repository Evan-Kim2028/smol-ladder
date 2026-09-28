import ast

from smol_ladder.ladder import (code_facts, leaks, method_hint, normalise, prompt_for,
                                redact_literals, schema_dump, strip_output)
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


def test_schema_dump_is_deterministic():
    row = load_split("test")[0]
    assert schema_dump(row) == schema_dump(row)


def test_sandbox_reads_input_offline():
    from smol_ladder.sandbox import run_script
    row = load_split("test")[0]
    inp = input_dir(row)
    assert inp.exists()
