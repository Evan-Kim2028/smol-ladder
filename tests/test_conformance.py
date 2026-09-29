"""Conformance: one ladder definition, applied identically to both task sources.

The risk is silent drift. `prompt_for` branches on the rung, and a branch that is exercised by
SmolDataEnvs but not by jupyter-agent (or the reverse) produces two subtly different ladders
that still report the same rung names. These tests pin the *text* of every rung for a synthetic
row, so any source-specific behaviour fails here rather than quietly in a 500-task run.

    uv run --with pytest pytest -q tests/test_conformance.py
"""

from __future__ import annotations

import inspect
import re

import pytest

from smol_ladder import ladder as L
from smol_ladder.run_ladder import source_for

ROW = {
    "task_id": "conformance_1",
    "question": "What is the mean of col_a for rows where flag is 1?",
    "files": ["t.csv"],
    "answer": "3.5",
    "reward_mode": "numeric",
    "atol": 1e-3,
    "rtol": 1e-3,
    "bucket_prefix": "conformance/one",
    "difficulty_tier": 1,
    "split": "test",
}
SOURCE = (
    "import pandas as pd\n"
    "df = pd.read_csv('input/t.csv')\n"
    "x = df[df['flag'] == 1]\n"
    "print(x['col_a'].mean())\n"
)
RUNGS = ["L1", "L1+schema", "L2", "L3", "L4"]


@pytest.fixture
def row(tmp_path, monkeypatch):
    """A task row with a real table on disk and a verified reference, so all five rungs build.

    prompt_for reaches the filesystem for two things: the schema control reads the tables, and
    the reference comes from read_source. Both are stubbed here so the test is about the
    ladder's shape, not about the Hub.
    """
    src = tmp_path / "src"
    src.mkdir()
    (src / "t.csv").write_text("flag,col_a\n1,3.0\n1,4.0\n0,9.0\n")
    monkeypatch.setattr(L, "inputs_of", lambda split: (lambda _r: src))
    monkeypatch.setattr(L, "read_source", lambda row, split: SOURCE)
    return dict(ROW)


def test_every_rung_is_reachable():
    missing = [r for r in RUNGS if r not in L.RUNGS and r != "L1+schema"]
    assert not missing, f"undeclared rungs: {missing}"


def test_rungs_are_cumulative(row):
    """L(k) must contain L(k-1) verbatim. That is the Blackwell ordering, and it only holds
    if no rung rewords or drops anything above it.

    Only the four rungs nest. The control is a sibling condition, not a rung: it replaces the
    hint block with a schema dump, so it is cumulative from L1, not from L2.
    """
    previous = L.prompt_for(row, "test", "L1")
    for rung in ("L2", "L3", "L4"):
        text = L.prompt_for(row, "test", rung)
        assert text.startswith(previous), f"{rung} does not extend the rung below it"
        previous = text
    control = L.prompt_for(row, "test", "L1+schema")
    assert control.startswith(L.prompt_for(row, "test", "L1"))
    assert "Schema of the input tables" in control
    assert "Notes on the intended computation" not in control


def test_l1_is_the_bare_prompt(row):
    l1 = L.prompt_for(row, "test", "L1")
    assert row["question"] in l1
    assert "- t.csv" in l1
    for forbidden in ("```python", "Notes on the intended computation", "Schema of the input"):
        assert forbidden not in l1, f"L1 leaked {forbidden!r}"


def test_l4_is_fenced_and_carries_no_final_print(row):
    l4 = L.prompt_for(row, "test", "L4")
    body = l4.split("```python", 1)[1].rsplit("```", 1)[0]
    assert "print(" not in body, "L4 still contains a print, so it evaluates to the answer"
    # the method must survive, or the rung carries no information at all
    assert "read_csv" in body


def test_every_rung_shares_one_instruction_header(row):
    """Every rung must carry the identical instruction, and every rung must carry all of it.

    The header is the text up to "Explore the data with Python"; the contract ("LAST line of
    output", the answer format) comes after it. A rung that dropped or reworded either would
    change a second variable between rungs alongside the hint.
    """
    heads, tails = {}, {}
    for rung in RUNGS:
        text = L.prompt_for(row, "test", rung)
        head, sep, tail = text.partition("Explore the data with Python")
        assert sep, f"{rung} lost the instruction body"
        heads[rung], tails[rung] = head, "Explore the data with Python" + tail
    assert len(set(heads.values())) == 1, "rungs disagree on the instruction header"
    # the contract lives in the tail: it is the text between the body and the hint
    contract = next(iter(tails.values())).split("\n\n")[0]
    for rung, tail in tails.items():
        assert tail.startswith(contract), f"{rung} altered the answer-format contract"
    assert "LAST line of output" in contract
    assert "comma-separated list" in contract


def test_hints_are_appended_after_the_shared_instruction(row):
    """Every hint lands after the instruction, never inside it. Otherwise a rung would also be
    changing where the contract text sits, which is a second uncontrolled variable."""
    for rung in RUNGS:
        text = L.prompt_for(row, "test", rung)
        contract = text.split("compute it from the files.", 1)
        assert len(contract) == 2, f"{rung} has no recognisable instruction tail"
        assert contract[0] == L.prompt_for(row, "test", "L1").split(
            "compute it from the files.", 1)[0]


def test_prompt_for_takes_no_source_specific_branches():
    """A regression guard: prompt_for must not branch on the split name.

    It dispatches input_dir through inputs_of, but the prompt text itself must be identical
    for the same row under either source, or the two ladders are not the same ladder.
    """
    src = inspect.getsource(L.prompt_for)
    assert "jupyter-agent" not in src, "prompt_for special-cases a source; keep it generic"


@pytest.mark.parametrize("split", ["test", "jupyter-agent"])
def test_both_sources_expose_the_same_row_contract(split):
    rows, _ = source_for(split)
    needed = {"task_id", "question", "files", "answer", "reward_mode", "atol", "rtol"}
    assert rows, f"no rows for {split}"
    missing = needed - set(rows[0])
    assert not missing, f"{split} rows lack {sorted(missing)}"
    for row in rows[:20]:
        assert isinstance(row["files"], list) and row["files"]
        assert isinstance(row["question"], str) and row["question"].strip()
        assert row["reward_mode"] in {"numeric", "exact_short", "exact_bool", "list_csv",
                                      "flexible"}


def test_task_ids_are_safe_as_directory_names():
    """jupyter-agent ids contain slashes; one task used to become a directory tree."""
    rows, _ = source_for("jupyter-agent")
    for row in rows:
        assert "/" not in row["task_id"] and "\\" not in row["task_id"]
        assert re.fullmatch(r"[A-Za-z0-9_.:-]+", row["task_id"]), row["task_id"]


def test_inputs_of_dispatches_by_source():
    assert L.inputs_of("test").__module__.endswith("tasks")
    assert L.inputs_of("jupyter-agent").__module__.endswith("jtasks")
