import pandas as pd
import pytest

from smol_ladder.synthetic import (Spec, _fmt, _tolerance, _unique_max, build_task,
                                   leaks, usable_columns, verify)


def test_columns_are_split_into_measures_and_keys(frame):
    numeric, categorical = usable_columns(frame)
    assert [n for _, n in numeric] == ["score"]
    assert [n for _, n in categorical] == ["team"]


def test_id_and_junk_columns_are_excluded(frame):
    names = {n for _, n in usable_columns(frame)[0]} | {n for _, n in usable_columns(frame)[1]}
    assert "id" not in names
    assert "#" not in names


def test_text_columns_are_found_under_pandas_str_dtype():
    """pandas 3 gives text a dedicated str dtype, not object.

    An `object` check matched nothing, so every table produced numeric-only tasks and the
    categorical family silently never ran.
    """
    df = pd.DataFrame({"label": [f"v{i%5}" for i in range(50)]})
    _, categorical = usable_columns(df)
    assert categorical and categorical[0][1] == "label"


def test_a_tied_maximum_is_rejected():
    """A tie means the answer depends on a tie-break we never stated.

    This is the ARBITRARY case from the openswe pipeline: the task would measure our choice
    rather than the analyst's, so it is dropped instead of published.
    """
    assert _unique_max(pd.Series([1, 2, 3])) is True
    assert _unique_max(pd.Series([3, 3, 1])) is False


def test_verify_recomputes_the_answer_from_the_ops(frame):
    spec = Spec(task_id="t", source_table="t.csv", question="q", answer="5.5",
                reward_mode="numeric", atol=1e-6, rtol=1e-6, files=["t.csv"],
                ops=["mean(score)"], columns=["score"])
    assert verify(spec, frame) is True
    # a spec whose ops do not produce its answer must not verify
    assert verify(Spec(**{**spec.__dict__, "answer": "99"}), frame) is False


def test_verify_handles_a_grouped_spec(frame):
    spec = Spec(task_id="t", source_table="t.csv", question="q", answer="2",
                reward_mode="numeric", atol=0.0, rtol=0.0, files=["t.csv"],
                ops=["filter(team==a)", "mean(score)"], columns=["team", "score"])
    # rows where team == a are scores 1,3,5 repeated; mean is 3
    assert verify(Spec(**{**spec.__dict__, "answer": "3"}), frame) is True
    assert verify(spec, frame) is False


def test_verify_handles_a_mode_spec(frame):
    spec = Spec(task_id="t", source_table="t.csv", question="q", answer="c",
                reward_mode="exact_short", atol=0.0, rtol=0.0, files=["t.csv"],
                ops=["value_counts(team)", "argmax"], columns=["team"])
    assert verify(spec, frame) is True


def test_build_task_produces_a_verifiable_spec(frame):
    for index in range(8):
        spec = build_task(frame, "/tmp/ds/table.csv", "stem", index)
        if spec is None:
            continue
        assert verify(spec, frame), f"{spec.task_id} does not verify"
        assert spec.question.strip()
        assert spec.answer != ""


def test_no_task_states_its_own_answer(frame):
    for index in range(8):
        spec = build_task(frame, "/tmp/ds/table.csv", "stem", index)
        if spec is not None:
            assert leaks(spec) is False, f"{spec.task_id} leaks its answer"


def test_tolerance_is_tighter_than_the_precision_printed():
    assert _tolerance("42") == (0.0, 0.0)        # an exact integer
    assert _tolerance("4.25") == (1e-6, 1e-6)    # exact at the printed precision
    assert _fmt(3.0) == "3"
    assert _fmt(True) == "yes"


def test_task_ids_are_safe_as_directory_names(frame):
    for index in range(8):
        spec = build_task(frame, "/tmp/ds/table.csv", "stem", index)
        if spec is None:
            continue
        assert "/" not in spec.task_id
        assert spec.task_id.startswith("syn_")


def test_the_id_keys_on_the_file_not_its_directory():
    """SmolDataEnvs keeps several tables under one bucket_prefix.

    Keying the id on the parent directory gave four different CSVs the same id, so 275
    generated tasks collapsed to 201 distinct ones and a quarter of the corpus was silently
    unreachable: two tasks with one id write to the same result file.
    """
    from pathlib import Path
    path = "/ds/prefix/2016.csv"
    assert Path(path).stem == "2016"                 # what the id now uses
    assert Path(path).parent.name == "prefix"        # what it used to use
    frame = pd.DataFrame({"score": [float(i) for i in range(30)],
                          "team": ["a"] * 10 + ["b"] * 10 + ["c"] * 10})
    spec = build_task(frame, path, "2016", 0)
    if spec is not None:
        assert "2016" in spec.task_id
        assert "prefix" not in spec.task_id


def test_generated_ids_are_unique_across_a_real_corpus():
    """The end-to-end property: every emitted id is distinct, so no trial overwrites another."""
    from smol_ladder.synthetic import build_task as build
    from smol_ladder.synthetic import verify as check
    seen = set()
    for i in range(4):
        df = pd.DataFrame({
            "score": [float((i + j) % 17) for j in range(60)],
            "team": [f"g{j % (3 + i)}" for j in range(60)],
        })
        for index in range(8):
            spec = build(df, f"/ds/prefix/table{i}.csv", f"table{i}", index)
            if spec is None or not check(spec, df):
                continue
            assert spec.task_id not in seen, f"duplicate id {spec.task_id}"
            seen.add(spec.task_id)
    assert seen, "no tasks generated; the fixture is not exercising the generator"
