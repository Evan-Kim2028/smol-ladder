"""L1 must be the garbling of every higher rung, and must not misdescribe the task.

Two requirements come straight from the information-ladder post:

1. "Each level contains everything in the one below" — so L(k) extends L(k-1) verbatim, and L1
   is recoverable from any L(k) by dropping the added block. Without that, "the lowest rung that
   passes" does not mean "the least information that sufficed", and the Blackwell ordering the
   design rests on is not a claim about anything.
2. "The ladder holds the task fixed and changes only what the solver sees." The task includes
   the environment. So a rung may not assert something false about it: L1 listed the dataset
   row's files while ./input held a different set, in 56% of synthetic and 28% of SmolDataEnvs
   tasks. A prompt that misdescribes the environment is a confound every higher rung inherits,
   and naming a subset of the files is itself a hint about which table the question is about.

These are cheap, mechanical checks, so they are tests rather than a review convention.
"""

from __future__ import annotations

import re

import pytest

from smol_ladder import ladder as L
from smol_ladder.run_ladder import source_for


def listed_files(prompt: str) -> list[str]:
    block = re.search(r"^Files:\n((?:- .+\n)+)", prompt, re.M)
    return re.findall(r"^- (.+)$", block.group(1), re.M) if block else []


@pytest.mark.parametrize("split", ["test", "jupyter-agent", "synthetic"])
def test_l1_lists_exactly_what_is_in_input(split):
    """Requirement 2: the ladder holds the task fixed, environment included.

    L1 may not describe a file set the solver will not find.
    """
    rows, inputs_of = source_for(split)
    checked = 0
    for row in rows[:40]:
        try:
            on_disk = sorted(p.name for p in inputs_of(row).iterdir()
                             if not p.name.startswith("."))
        except Exception:
            continue
        checked += 1
        assert listed_files(L.prompt_for(row, split, "L1")) == on_disk, (
            f"{row['task_id']}: L1 misdescribes ./input")
    assert checked, f"no readable tasks for {split}"


def test_input_files_falls_back_to_the_row_when_the_dir_is_unreadable(monkeypatch):
    monkeypatch.setattr(L, "inputs_of", lambda split: (_ for _ in ()).throw(OSError("gone")))
    row = {"task_id": "t", "question": "q", "files": ["b.csv", "a.csv"],
           "answer": "1", "reward_mode": "numeric", "atol": 0.0, "rtol": 0.0}
    assert L.input_files(row, "test") == ["a.csv", "b.csv"]


def test_l1_does_not_reveal_columns_or_filters(row):
    """L1 must stay the floor: nothing about the intended computation leaks into it."""
    l1 = L.prompt_for(row, "test", "L1")
    for forbidden in ("Columns used", "Filters applied", "Method:", "Files read",
                      "Schema of the input", "Notes on the intended"):
        assert forbidden not in l1


def test_the_first_rung_adding_information_is_l2(row):
    """L2 must be the lowest rung that says anything about the computation.

    L1 already names the files, so L2 cannot be "which files" — it has to be the columns and
    filters, or the rungs are not a partition of increasing information and the labels lie.
    """
    l1 = L.prompt_for(row, "test", "L1")
    l2 = L.prompt_for(row, "test", "L2")
    assert "Columns used" in l2
    assert "Filters applied" in l2
    # L1 is contained in L2, not absent from it: containment is the ordering requirement.
    assert l2.startswith(l1.rstrip())


def test_dropping_the_hint_recovers_l1_from_every_rung(row):
    """Requirement 1, stated as an operation: garble L(k) and you should get L1 back.

    This is the Blackwell relation made mechanical. A rung that carries information not
    derivable from the one below it cannot be a garbling of it, so the ordering claim is void.
    """
    l1 = L.prompt_for(row, "test", "L1").rstrip()
    for rung in ("L2", "L3", "L4"):
        higher = L.prompt_for(row, "test", rung)
        assert higher.startswith(l1), f"{rung} does not contain L1 verbatim"
        # the extra information is a suffix, so removing it leaves L1 exactly
        assert higher[: len(l1)] == l1


def test_no_rung_is_byte_identical_to_the_one_below(row):
    """The article's own table has a column for this: "Nothing new, because the cut already
    left them."

    A rung that adds no information is not a rung — it re-measures the rung below and spends a
    trial to learn the same thing twice. It must not happen by accident, and when a task truly
    has no reference the prompt must say so rather than silently repeat L1.
    """
    for rung, below in (("L2", "L1"), ("L3", "L2"), ("L4", "L3")):
        assert L.prompt_for(row, "test", rung) != L.prompt_for(row, "test", below), \
            f"{rung} is identical to {below}"


def test_a_task_without_a_reference_says_so(monkeypatch):
    """SmolDataEnvs has no reference for 69/250 tasks. Their L2-L4 must be marked, not faked."""
    monkeypatch.setattr(L, "read_source", lambda row, split: None)
    row = {"task_id": "t", "question": "Q?", "files": ["t.csv"], "answer": "1",
           "reward_mode": "numeric", "atol": 0.0, "rtol": 0.0, "difficulty_tier": 1,
           "bucket_prefix": "x", "split": "test"}
    l2 = L.prompt_for(row, "test", "L2")
    assert "No reference solution" in l2
    assert l2 != L.prompt_for(row, "test", "L1")


# Builds 20 synthetic references and reads all four rung prompts for each, so it walks the
# synthetic split's own solution programs from disk: 7.6s. Marked slow because it is a
# split-wide read rather than a unit check, which is what `pytest -m "not slow"` skips.
@pytest.mark.slow
def test_synthetic_rungs_are_live():
    """A synthetic task's spec is its reference, so every rung must add something.

    Before this, all 275 synthetic tasks had a read_source of None, prompt_for fell back to the
    bare L1 text, and L2/L3/L4 were byte-identical to L1 — 80/80 dead rungs, so the control run
    was measuring the same prompt four times.
    """
    rows, _ = source_for("synthetic")
    for row in rows[:20]:
        assert L.read_source(row, "synthetic") is not None, row["task_id"]
        prompts = [L.prompt_for(row, "synthetic", r) for r in ("L1", "L2", "L3", "L4")]
        assert len(set(prompts)) == 4, f"{row['task_id']}: rungs collapsed"


def test_synthetic_reference_has_no_print_and_omits_the_answer():
    """The L4 payload must not evaluate to the answer, and must not name it either.

    The reference now ends in `print(result)` rather than leaving `result` bound: the same
    program has to be runnable as a script, because it is what the shipped-file gate grades and
    what the table's real dtype and NA inference produce. What must not survive into the payload
    is any statement that emits the answer, which is what strip_output drops and this checks.
    """
    rows, _ = source_for("synthetic")
    for row in rows[:20]:
        source = L.read_source(row, "synthetic")
        assert source.endswith("print(result)"), row["task_id"]
        payload = L.redact_literals(L.strip_output(source), str(row["answer"]))
        assert "print(" not in payload, row["task_id"]
        assert L.normalise(row["answer"]) not in L.normalise(payload), row["task_id"]
