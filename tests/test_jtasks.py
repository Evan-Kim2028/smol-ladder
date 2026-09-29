from smol_ladder.jtasks import classify, grade_params


def test_classify_accepts_the_three_gradable_shapes():
    assert classify("453") == ("numeric", "453")
    assert classify("8.89663104713") == ("numeric", "8.89663104713")
    assert classify("88.52%") == ("numeric", "88.52")
    assert classify("$1,234.5") == ("numeric", "1234.5")
    assert classify("Wii Sports") == ("exact_short", "Wii Sports")
    assert classify("Yes") == ("exact_bool", "yes")
    assert classify("  no  ") == ("exact_bool", "no")


def test_classify_rejects_answers_that_look_gradable_but_are_not():
    # The dataset's way of saying there is no answer. Matches a label pattern, so it has to be
    # excluded by name or it becomes a task with gold "Not explicitly stated...".
    assert classify("Not explicitly stated in the notebook outputs") is None
    # Units and derivations: the value is present but the intended answer is a sentence.
    assert classify("61,257 USD (70,187 for failed minus 8,930 for successful)") is None
    assert classify("area_se (5.447186)") is None
    assert classify("Ca (10.76)") is None
    assert classify("1 time per week (maximum grade of 20)") is None
    # The question restated back with a count attached.
    assert classify("Y=3 with 95,293 instances") is None
    assert classify("Fraudulent: $2660.80, Non-Fraudulent: $669.02") is None


def test_classify_rejects_empty_and_absurdly_long():
    assert classify("") is None
    assert classify("   ") is None
    assert classify("x" * 80) is None


def test_grade_params_tolerate_the_printed_precision_only():
    # 8.89663104713 stored to 11 dp. A 1e-12 tolerance would call a model that printed
    # 8.896631 wrong, so the tolerance floors at 1e-4 relative.
    assert grade_params("numeric", "8.89663104713") == (1e-4, 1e-4)
    # A 2dp answer gets a 1e-3 tolerance, so the metric is about the computation and not
    # about the last printed digit.
    assert grade_params("numeric", "88.52") == (1e-3, 1e-3)
    # Exact modes stay exact.
    assert grade_params("exact_short", "Wii Sports") == (0.0, 0.0)
    assert grade_params("exact_bool", "yes") == (0.0, 0.0)
    # An integer gets a small tolerance, not a whole unit: "453" must not admit 453.4.
    assert grade_params("numeric", "453") == (0.05, 0.05)


def test_a_rounded_but_correct_answer_still_grades():
    from smol_ladder.grade import grade
    value = "8.89663104713"
    atol, rtol = grade_params("numeric", value)
    row = {"answer": value, "reward_mode": "numeric", "atol": atol, "rtol": rtol}
    assert grade(row, "8.896631") == 1.0     # right, printed to 6 dp
    assert grade(row, "8.89663") == 1.0      # right, printed to 5 dp
    # 1e-4 relative on a value near 8.9 is about 9e-4, so the first wrong answer is ~1e-3 out.
    assert grade(row, "8.8956") == 0.0
    assert grade(row, "8.9") == 0.0


def test_tolerance_never_admits_a_different_value():
    # A tolerance in proportion to precision must not turn 6 into 6.5.
    from smol_ladder.grade import grade
    row = {"answer": "6.5", "reward_mode": "numeric", "atol": 1e-3, "rtol": 1e-3}
    assert grade(row, "6.5") == 1.0
    assert grade(row, "6") == 0.0
    assert grade(row, "7") == 0.0
