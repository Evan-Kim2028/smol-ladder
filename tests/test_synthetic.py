import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from smol_ladder import synthetic as syn
from smol_ladder.grade import grade
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
    spec = Spec(task_id="t", source_table="t.csv", question="q", answer="2.0",
                reward_mode="numeric", atol=0.0, rtol=0.0, files=["t.csv"],
                ops=["filter(team==a)", "mean(score)"], columns=["team", "score"])
    # rows where team == a are scores 1,3,5 repeated; mean is exactly 3
    assert verify(Spec(**{**spec.__dict__, "answer": "3.0"}), frame) is True
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


def test_the_gold_is_printed_exactly_and_gets_an_exact_tolerance():
    """Printing and grading are separate steps, and only printing is lossy.

    A sum of 200k rows printed at `.6g` differs from the sum by more than 1e-6 relative, and the
    grader has no absolute floor, so the exact answer graded 0.0. An exact answer with an exact
    tolerance is the only combination that cannot reject itself.
    """
    assert _fmt(3.0) == "3.0" and _fmt(True) == "yes"
    assert _fmt(42) == "42"
    for value in (4.25, 3.0, 31535.63530029, 0.1 + 0.2):
        assert _tolerance(_fmt(value)) == (0.0, 0.0), value


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


# --- the gold is computed on the shipped file, and the shipped file is what the agent reads ----

def local_runner(script, inputs):
    """Run a reference program's text in process, standing in for the jail.

    Sandboxing every case would make this file take minutes and pay a bwrap fork per assertion.
    The sandbox itself is exercised by test_harness.test_sandbox_reads_input_offline; what these
    tests are about is which bytes reach the computation. The work directory is where the jail
    puts it, with the tables at ./input, so a program that reads `input/t.csv` resolves the way
    it does in a real trial and the print goes where run_script looks for it.
    """
    here = os.getcwd()
    work = tempfile.mkdtemp()
    try:
        (Path(work) / "input").symlink_to(Path(inputs).resolve(), target_is_directory=True)
        os.chdir(work)
        return subprocess.run([sys.executable, str(script)], capture_output=True, text=True,
                              timeout=120, cwd=work)
    finally:
        os.chdir(here)
        shutil.rmtree(work, ignore_errors=True)


@pytest.fixture
def shipped(tmp_path, monkeypatch):
    """A table on disk and a row pointing at it, with input_dir stubbed to the local directory.

    `shipped_path` resolves through `input_dir` — the one path to the cache — so stubbing that
    one function leaves the reference renderer and the verify step on the real code path.
    """
    src = tmp_path / "inputs"
    src.mkdir()
    (src / "t.csv").write_text("team,score,year\n"
                               "a,1,2016\na,3,2016\nb,9,2015\nc,5,2015\n"
                               "c,7,2015\nc,2,2015\n")
    monkeypatch.setattr(syn, "input_dir", lambda row: src)
    return src


def row_for(ops, answer, mode="numeric", name="t.csv"):
    return {"task_id": "syn_t", "question": "q", "answer": answer, "reward_mode": mode,
            "atol": _tolerance(answer)[0], "rtol": _tolerance(answer)[1],
            "files": [name], "ops": ops, "columns": [], "source": "synthetic"}


def test_the_gold_is_computed_on_every_row_of_the_shipped_file(shipped):
    """The bug: iter_tables read nrows=50_000 and shipped the whole file.

    The gold is whatever build_task computed over the frame it was handed, so the only way a
    table over 50k rows can grade correctly is if the frame it was handed *is* the file. This
    asserts the two are the same length, which is what `nrows` broke.
    """
    path = shipped / "t.csv"
    assert len(syn.read_shipped(path)) == 6


def test_the_reference_program_runs_the_specification(shipped):
    """One implementation of the ops, rendered as code: the rung and the gold cannot drift."""
    row = row_for(["mean(score)"], "4.5")
    assert "to_numeric" in syn.reference_script(row)
    reward, printed = syn.verify_shipped(row, runner=local_runner)
    assert (reward, printed) == (1.0, "4.5")


def shipped_reward(row, runner=local_runner):
    return syn.verify_shipped(row, runner=runner)[0]


def shipped_broken(row, script):
    """Grade a deliberately mis-rendered reference, to show the gate would have caught it."""
    from smol_ladder.grade import grade as _grade
    return _grade(row, syn.verify_prediction(row, script, local_runner)[1])


def test_a_task_whose_reference_would_not_grade_one_is_refused(shipped):
    """The gate. A task that cannot grade 1.0 from its own reference on the shipped file is not
    emitted: the agent cannot pass it whatever it writes."""
    assert shipped_reward(row_for(["mean(score)"], "4.5")) == 1.0
    assert shipped_reward(row_for(["mean(score)"], "4.5001")) == 0.0    # 2.2e-5 relative off


def test_a_gold_the_grader_rejects_as_its_own_predicate_is_refused(shipped):
    """Printing and grading are separate steps, and only printing is lossy.

    A sum printed at `.6g` on a 15k answer differs from the sum by 3.5e-5 relative, which a
    1e-6 tolerance rejects. The exact answer graded against that gold is 0.0, so the task is
    unpassable by construction — this is what happened to 26 tasks whose tables were small.
    """
    # A gold printed at `.6g` on a big answer: the exact sum is right and grades 0.0 against it.
    # 27,000 is the smallest sum whose `.6g` printing falls outside a 1e-6 relative tolerance,
    # which is the scale at which 26 of the corpus's own answers mis-graded.
    lossy = {"answer": f"{27_000.25:.6g}", "reward_mode": "numeric", "atol": 1e-6, "rtol": 1e-6}
    assert grade(lossy, "27000.25") == 0.0               # printing lost 9e-6 relative
    row = row_for(["sum(score)"], "27000.25")
    row["atol"] = row["rtol"] = 1e-6
    assert shipped_reward(row) == 0.0                    # and the real table sums to 27


def test_the_tolerance_is_wide_enough_for_the_precision_the_gold_is_printed_at():
    """A gold is unpassable unless the grader accepts it as a prediction of itself.

    Hence _tolerance derives atol/rtol from the answer string rather than from the op that
    produced it, and an exact answer gets an exact tolerance.
    """
    for exact in ("27.0", "4.5", "31535.63530029", f"{0.1 + 0.2!r}"):
        assert _tolerance(exact) == (0.0, 0.0), exact
        assert repr(float(exact)) == exact, exact


def test_a_filter_value_keeps_the_type_of_its_column(shipped):
    """`filter(col==value)` stores the value as text, and the rendered reference used to quote
    it: `df[df['year'] == '2016']` on an int column is elementwise False, so the rung computed a
    mean over nothing and graded 0.0 looking like a model failure."""
    row = row_for(["filter(year==2016)", "mean(score)"], "2.0")
    script = syn.reference_script(row)
    assert "== 2016]" in script, script
    assert shipped_reward(row) == 1.0
    # a quoted int filters to nothing: pandas' == against a string is elementwise False, and a
    # mean over an empty frame is NaN, so the rung graded 0.0 while looking like a model failure
    quoted = row_for(["filter(year==2016)", "mean(score)"], "2.0")
    broken = syn.reference_script(quoted).replace("== 2016]", "== '2016']")
    assert grade(quoted, "2.0") == 1.0 and shipped_broken(quoted, broken) == 0.0


def test_a_filter_on_a_text_column_keeps_its_quotes(shipped):
    row = row_for(["filter(team==a)", "mean(score)"], "2.0")
    assert "== 'a']" in syn.reference_script(row)
    assert shipped_reward(row) == 1.0


def test_a_column_name_with_a_quote_does_not_break_the_reference(shipped):
    """The inline f-string renderer interpolated column names into `df['{column}']`, so a name
    containing a quote produced a SyntaxError and an L4 rung that could not run."""
    (shipped / "t.csv").write_text('the "best" score,team\n1,a\n3,a\n5,b\n')
    row = row_for(['mean(the "best" score)'], "3.0")
    compile(syn.reference_script(row), "<reference>", "exec")
    assert shipped_reward(row) == 1.0


def test_a_column_name_with_a_newline_survives_the_reference(shipped):
    """`df['Total\ncivilians']` in the old renderer was a line continuation, so the rung did not
    even parse. With a real cell behind the name, the op names it and the reference compiles."""
    (shipped / "t.csv").write_text('"Total\ncivilians",team\n1,a\n3,a\n5,b\n')
    row = row_for(["mean(Total\ncivilians)"], "3.0")
    compile(syn.reference_script(row), "<reference>", "exec")
    assert shipped_reward(row) == 1.0


def test_the_reference_reads_the_shipped_file_by_its_own_name(shipped):
    """The reference the L4 rung shows must read the same file the agent is shipped, from the
    directory it is placed in. That is what makes the rung's program and the gold agree."""
    row = row_for(["mean(score)"], "4.5")
    assert "input/t.csv" in syn.reference_script(row)


def test_a_ragged_table_is_a_hard_error_not_a_silently_short_frame(tmp_path):
    """pandas drops an over-long row and warns. Skipping that row makes the gold a sum over fewer
    rows than the agent sums over, which is the truncation bug in a smaller package."""
    path = tmp_path / "bad.csv"
    path.write_text("a,b\n1,2\n3,4,5\n")
    with pytest.raises(pd.errors.ParserError):
        syn.read_shipped(path)


def test_a_tsv_separator_is_inferred_from_the_bytes_not_the_suffix(tmp_path):
    """A `.csv` that is really semicolon-separated, or a `.tsv` that is not: `read_csv` without
    `sep` infers what pandas can read, so a mis-suffixed file becomes one garbage column."""
    path = tmp_path / "actually.tsv"
    path.write_text("a\tb\n1\t2\n3\t4\n")
    assert syn.detect_separator(path) == "\t"
    assert list(syn.read_shipped(path).columns) == ["a", "b"]
    commas = tmp_path / "commas.tsv"
    commas.write_text("a,b\n1,2\n3,4\n")
    assert syn.detect_separator(commas) == ","


def test_iter_tables_never_truncates_a_table(shipped, monkeypatch):
    """iter_tables is the other half of the root fix: the frame it hands build_task must be the
    whole file, whatever the file's size."""
    monkeypatch.setattr(syn, "load_split", lambda split: [
        {"bucket_prefix": "p", "files": ["t.csv"]}])
    monkeypatch.setattr(syn, "input_dir", lambda row: shipped)
    (shipped / "big.csv").write_text("a\n" + "1\n" * 60_000)
    frames, unreadable = syn.iter_tables(10)
    assert unreadable == []
    assert len(frames[0][1]) == 60_000


def test_iter_tables_drops_a_table_it_cannot_read_and_names_it(shipped, monkeypatch):
    """A file that is not a table cannot carry a gold, so it is dropped rather than half-read.

    `ca_law_enforcement_by_campus.csv` has bare newlines inside its header row, so every data row
    has a different field count and no separator separates them. The old loop swallowed the
    ParserError and kept the file, and the agent was then free to build a task on a parse the
    gold was never computed from. Skipping is the default because a dropped table cannot
    mis-grade anything -- no gold, no task -- and it is named so the coverage loss is on record.
    """
    monkeypatch.setattr(syn, "load_split", lambda split: [
        {"bucket_prefix": "p", "files": ["t.csv"]}])
    monkeypatch.setattr(syn, "input_dir", lambda row: shipped)
    (shipped / "ragged.csv").write_text('University,Enrollment\nAllan Hancock,"11,047"\n'
                                        'Two Words College,Pomona,"23,966",extra\n')
    frames, unreadable = syn.iter_tables(10)          # skip is the default
    assert not any(name[0].endswith("ragged.csv") for name in frames)
    assert any("ragged.csv" in note for note in unreadable)
    # the good table is still built from, so one bad file costs coverage rather than the corpus
    assert any(name[0].endswith("t.csv") for name in frames)
    with pytest.raises(RuntimeError, match="did not read cleanly"):
        syn.iter_tables(10, "strict")


def test_two_tasks_on_one_table_get_different_reference_programs(shipped):
    """A per-file cache is a wrong-answer generator wearing a performance hat.

    A corpus holds ~1800 tasks over ~40 tables, so dozens of tasks share a shipped file and each
    asks a different question of it. Caching the rendered program on the path alone hands every
    task after the first the *first* task's program, so the gate verifies thousands of tasks
    against another task's reference and the L4 rung ships it.
    """
    syn._REFERENCE_CACHE.clear()
    first = syn.reference_script(row_for(["mean(score)"], "4.5"))
    second = syn.reference_script(row_for(["sum(score)"], "27.0"))
    assert "mean()" in first and "sum()" in second
    assert first != second
    # and each still grades its own task, which is what the cache broke
    assert shipped_reward(row_for(["mean(score)"], "4.5")) == 1.0
    assert shipped_reward(row_for(["sum(score)"], "27.0")) == 1.0


def test_a_task_whose_table_is_not_on_disk_is_refused(tmp_path, monkeypatch):
    """No shipped file means no answer anyone can check, and gold by construction is this
    source's whole claim, so a task we cannot verify is not emitted."""
    monkeypatch.setattr(syn, "input_dir", lambda row: tmp_path)
    assert syn.verify_shipped(row_for(["mean(score)"], "4.5"))[0] < 1.0
