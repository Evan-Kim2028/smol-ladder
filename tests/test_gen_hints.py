"""Plain-language hints: the cache contract, the validation, and the ladder's use of them.

The AST extractor was replaced by a model writing the same facts in plain language, because
reading columns and filters off the AST is empty on most references (filters empty on 71% of the
181 test references, and an L3 that says "get, items, values" is not a method). So these tests
pin the properties the new source has to hold: the hint is only used if it survives the same
hallucination and leak checks the AST version was implicitly trusted on, the cache is
deterministic and resumable, and the ladder still falls back to the AST when there is no usable
hint. The Blackwell ordering is a ladder.py test and must keep passing.
"""

from __future__ import annotations

import json

import pytest

from smol_ladder import gen_hints as G
from smol_ladder import ladder as L

REFERENCE = (
    "import pandas as pd\n"
    "df = pd.read_csv('input/t.csv')\n"
    "x = df[df['flag'] == 1]\n"
    "print(x['col_a'].mean())\n"
)

ROW = {
    "task_id": "nlhint_1",
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


def test_reference_hash_changes_with_the_reference():
    a = G.reference_hash(REFERENCE)
    b = G.reference_hash(REFERENCE + "\nprint(1)\n")
    assert a != b
    assert a == G.reference_hash(REFERENCE)


def test_hint_rejects_a_hallucinated_column(tmp_path, monkeypatch):
    """A column that is not in the table header is a fabrication. The generator must drop it.

    The whole point of asking a model for columns is that they get checked against the actual
    headers; otherwise L2 confidently names a column the solver cannot find, which is worse than
    the AST saying nothing.
    """
    tables = tmp_path / "in"
    tables.mkdir()
    (tables / "t.csv").write_text("flag,col_a\n1,3.0\n")
    row = dict(ROW)
    hdrs = {"t.csv": ["flag", "col_a"]}
    monkeypatch.setattr(G, "headers_of", lambda row, split: hdrs)
    hint = {"files": ["t.csv"], "columns": ["col_a", "imaginary"], "filters": "none"}
    bad = [c for c in hint["columns"] if not G.column_is_real(c, hdrs, hint["files"])]
    assert bad == ["imaginary"]


def test_nested_wrapper_output_is_unwrapped():
    """The model sometimes nests the two fields under a wrapper key.

    Rejecting that cost the first real task's hint three attempts for output that was already
    correct, so the shape is coerced rather than treated as a validation failure.
    """
    wrapped = {"hint": {"files": ["t.csv"], "columns": ["Gender"], "filters": "none"},
               "method": "count values of Gender and take the largest"}
    out = G.normalise_output(wrapped)
    assert out["l2"]["columns"] == ["Gender"]
    assert "largest" in out["l3"]

    flat = {"l2": {"files": ["t.csv"], "columns": ["Gender"], "filters": "none"},
            "l3": "count values of Gender"}
    assert G.normalise_output(flat)["l2"]["columns"] == ["Gender"]
    assert G.normalise_output({}) == {"l2": {}, "l3": ""}


def test_leak_check_flags_a_hint_that_states_the_answer(row):
    """The same leak rule the AST path is held to: normalised substring, then the grader.

    The answer has to clear MIN_LEAK_LEN (4 characters) — a shorter one matches by chance and is
    skipped by ladder.leaks on purpose.
    """
    row = dict(row, answer="3.4567")
    hint = "take the mean of col_a where flag is 1; the result is 3.4567"
    assert G._leak_hits(row, hint, "test") == ["answer-substring"]


def test_a_short_answer_is_not_treated_as_a_leak(row):
    """A 3-character answer matches everywhere by accident; flagging it would bury real leaks."""
    row = dict(row, answer="3.5")
    assert G._leak_hits(row, "the mean is 3.5 over 12 rows", "test") == []


def test_validate_rejects_a_leaking_hint(row):
    row = dict(row, answer="3.4567")
    record = {"l2": {"files": ["t.csv"], "columns": ["col_a"], "filters": "none"},
              "l3": "average col_a and report the result 3.4567"}
    ok, reason = G._validate(row, record, "test", {"t.csv": ["flag", "col_a"]}, ["t.csv"])
    assert not ok
    assert reason.startswith("leak:")


def test_l3_must_add_information_beyond_l2():
    """If the method sentence is empty or identical to the L2 facts, L3 is not a rung.

    The ladder forbids a rung that adds nothing above the one below it, because it re-measures
    the rung below and spends a trial to learn the same thing twice.
    """
    l2_text = "columns: col_a; filters: flag == 1"
    assert not G.l3_adds_information(l2_text, l2_text)
    assert not G.l3_adds_information(l2_text, "")
    assert G.l3_adds_information(l2_text, "group by category and take the mode")


def test_generate_one_writes_a_resumable_cache(tmp_path, monkeypatch):
    """One JSON call, cached with model + prompt version + reference hash, and reused verbatim.

    The hash is what makes a rerun deterministic: if the reference changes, the cached hint is
    stale and must be regenerated; if it is unchanged, the cache hit is reused.
    """
    calls = {"n": 0}

    def fake_call(row, source, files, headers, model):
        calls["n"] += 1
        return {
            "l2": {"files": ["t.csv"], "columns": ["col_a"], "filters": "rows where flag == 1"},
            "l3": "keep rows where flag is 1, average col_a, report the mean",
        }

    monkeypatch.setattr(G, "call_model", fake_call)
    monkeypatch.setattr(G, "headers_of", lambda row, split: {"t.csv": ["flag", "col_a"]})
    monkeypatch.setattr(G, "output_file", lambda split, task_id: tmp_path / f"{task_id}.json")

    record = G.generate_one(dict(ROW), "test", REFERENCE, model="m", attempts=3)
    assert record["l2"]["columns"] == ["col_a"]
    assert record["l3"]
    assert record["model"] == "m"
    assert record["reference_hash"] == G.reference_hash(REFERENCE)
    assert record["prompt_version"] == G.PROMPT_VERSION
    assert not record.get("failed")
    assert calls["n"] == 1

    # Second call is served from the cache: the fake is not invoked again.
    again = G.generate_one(dict(ROW), "test", REFERENCE, model="m", attempts=3)
    assert again == record
    assert calls["n"] == 1


def test_generate_one_retries_then_marks_failed(tmp_path, monkeypatch):
    """A hint that keeps leaking or hallucinating is retried, then marked failed, not shipped."""
    calls = {"n": 0}

    def always_leaks(row, source, files, headers, model):
        calls["n"] += 1
        return {
            "l2": {"files": ["t.csv"], "columns": ["col_a"], "filters": "none"},
            "l3": "the answer is 3.4567",
        }

    monkeypatch.setattr(G, "call_model", always_leaks)
    monkeypatch.setattr(G, "headers_of", lambda row, split: {"t.csv": ["flag", "col_a"]})
    monkeypatch.setattr(G, "output_file", lambda split, task_id: tmp_path / f"{task_id}.json")

    record = G.generate_one(dict(ROW, answer="3.4567"), "test", REFERENCE,
                            model="m", attempts=3)
    assert record["failed"]
    assert calls["n"] == 3


def test_a_hallucinated_file_is_dropped(tmp_path, monkeypatch):
    def names_a_file_that_is_not_there(row, source, files, headers, model):
        return {
            "l2": {"files": ["t.csv", "ghost.csv"], "columns": ["col_a"], "filters": "none"},
            "l3": "average col_a over all rows",
        }

    monkeypatch.setattr(G, "call_model", names_a_file_that_is_not_there)
    monkeypatch.setattr(G, "headers_of", lambda row, split: {"t.csv": ["flag", "col_a"]})
    monkeypatch.setattr(G, "output_file", lambda split, task_id: tmp_path / f"{task_id}.json")

    record = G.generate_one(dict(ROW), "test", REFERENCE, model="m", attempts=3)
    assert "ghost.csv" not in record["l2"]["files"]
    assert record["l2"]["files"] == ["t.csv"]


def test_ladder_prefers_a_cached_plain_language_hint(row, tmp_path, monkeypatch):
    """With a valid cached hint present, L2/L3 use it verbatim; without one, the AST path."""
    monkeypatch.setattr(G, "headers_of", lambda r, s: {"t.csv": ["flag", "col_a"]})
    cache = {
        "l2": {"files": ["t.csv"], "columns": ["col_a"], "filters": "rows where flag is 1"},
        "l3": "keep rows where flag is 1, then average col_a",
    }
    monkeypatch.setattr(L, "load_hint", lambda r, s: cache)
    l2 = L.prompt_for(row, "test", "L2")
    l3 = L.prompt_for(row, "test", "L3")
    assert "average col_a" in l3
    assert "Files read: t.csv" in l2
    assert "Columns used: col_a" in l2
    assert "Filters applied: rows where flag is 1" in l2

    # No hint cached -> the AST extraction, which is what ships today.
    monkeypatch.setattr(L, "load_hint", lambda r, s: None)
    ast_l2 = L.prompt_for(row, "test", "L2")
    assert "Columns used" in ast_l2
    assert "average col_a" not in ast_l2


def test_a_failed_hint_falls_back_to_the_ast(row, monkeypatch):
    """With no usable hint, L2/L3 come from the AST extraction, which is what ships today."""
    monkeypatch.setattr(L, "load_hint", lambda r, s: None)
    ast_l2 = L.prompt_for(row, "test", "L2")
    ast_l3 = L.prompt_for(row, "test", "L3")
    # The fixture reference filters on flag, so the AST names that column and the operator.
    assert "Columns used: flag" in ast_l2
    assert "Filters applied: flag ==" in ast_l2
    # The AST method is empty here — `x['col_a'].mean()` names no grouped or frame-level op —
    # which is the "get, items, values" thinness the plain-language hints replace. What matters
    # for this test is only that the fallback is reached and produces the AST's rung.
    assert ast_l3.startswith(ast_l2)


def test_cached_hint_keeps_the_blackwell_prefix(row, monkeypatch):
    """A cached hint must not break the cumulative ordering: L2 still starts with L1, L3 with
    L2's block verbatim."""
    monkeypatch.setattr(L, "load_hint", lambda r, s: {
        "l2": {"files": ["t.csv"], "columns": ["col_a"], "filters": "rows where flag is 1"},
        "l3": "keep rows where flag is 1, then average col_a",
    })
    l1 = L.prompt_for(row, "test", "L1").rstrip()
    l2 = L.prompt_for(row, "test", "L2")
    l3 = L.prompt_for(row, "test", "L3")
    l4 = L.prompt_for(row, "test", "L4")
    assert l2.startswith(l1)
    assert l3.startswith(l1)
    assert l4.startswith(l1)
    # L3 and L4 carry L2's block identically.
    block = L.l2_block(row, "test")
    assert block in l3 and block in l4


def test_hint_cache_file_is_a_plain_dict_on_disk(tmp_path, monkeypatch):
    def fake_call(row, source, files, headers, model):
        return {
            "l2": {"files": ["t.csv"], "columns": ["col_a"], "filters": "none"},
            "l3": "average col_a over all rows",
        }

    monkeypatch.setattr(G, "call_model", fake_call)
    monkeypatch.setattr(G, "headers_of", lambda row, split: {"t.csv": ["flag", "col_a"]})
    monkeypatch.setattr(G, "output_file", lambda split, task_id: tmp_path / "hints" / f"{task_id}.json")
    G.generate_one(dict(ROW), "test", REFERENCE, model="m", attempts=1)
    on_disk = json.loads((tmp_path / "hints" / "nlhint_1.json").read_text())
    assert on_disk["l2"]["columns"] == ["col_a"]
    assert on_disk["prompt_version"] == G.PROMPT_VERSION