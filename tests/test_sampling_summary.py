"""The summary over repeated samples: pass probability, its CI, and monotonicity.

At k=1 the summary reported a bare fraction and a first-passing-rung bucket. Four reruns of L1
disagreed on 18.9% of tasks, so both are noisy: the fraction hides its own error bar and the
bucket is decided by a coin flip on the tasks that straddle the boundary. These tests pin the
sample-aware summary -- mean pass probability with a bootstrap CI, the monotonicity test that
only a non-climbing design can measure, and the refusal to pool two ladder versions.

    uv run --with pytest pytest -q tests/test_sampling_summary.py
"""

import hashlib
import json
from pathlib import Path

import pytest

from smol_ladder import summarize as S

HASH_A = "a" * 64
HASH_B = "b" * 64


def trial(reward=0.0, status="exit 0", hash_=HASH_A):
    return {"reward": reward, "agent_status": status, "prediction": str(reward),
            "prompt_sha256": hash_}


def tree(**rungs):
    """runs dict: task_id -> rung -> list of trials (sample 0 first)."""
    return {task: {rung: list(ts) for rung, ts in per_rung.items()} for task, per_rung in
            rungs.items()}


# --- collection: the sample layout ----------------------------------------------------------

def test_collect_reads_both_the_flat_and_the_suffixed_layout(tmp_path, monkeypatch):
    """One sample reads the legacy <task>/<rung>/result.json; s1.. sit beside it."""
    monkeypatch.setattr(S, "DATA", tmp_path / "data")
    root = tmp_path / "data" / "runs" / "test"

    def put(relative, result):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result))

    put("t1/L1/result.json", trial(1.0))
    put("t1/L1/s1/result.json", trial(0.0))
    put("t1/L1/s2/result.json", trial(1.0))
    put("t1/L2/result.json", trial(1.0))
    put("t2/L1_schema/result.json", trial(1.0, hash_=HASH_B))

    runs = S.collect("test")
    assert [t["reward"] for t in runs["t1"]["L1"]] == [1.0, 0.0, 1.0]
    assert [t["reward"] for t in runs["t1"]["L2"]] == [1.0]
    assert S.rung_name("L1_schema") == "L1+schema"


def test_collect_orders_samples_by_index_not_by_directory_walk(tmp_path, monkeypatch):
    """s10 must not sort before s2. The order is the recorded sample index, then the fallback."""
    monkeypatch.setattr(S, "DATA", tmp_path / "data")
    root = tmp_path / "data" / "runs" / "test"
    for name, reward in [("s2", 0.2), ("s10", 0.3), ("s1", 0.1)]:
        path = root / "t1" / "L1" / name / "result.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"reward": reward, "agent_status": "exit 0",
                                    "sample": int(name[1:])}))

    runs = S.collect("test")
    assert [t["reward"] for t in runs["t1"]["L1"]] == [0.1, 0.2, 0.3]


# --- pass probability and its interval ------------------------------------------------------

def test_the_mean_is_over_tasks_not_over_trials(tmp_path):
    """A task with 4 samples must not outweigh a task with 1. This is the whole reason to
    average per task: a per-trial mean silently reweights tasks by how many trials they got."""
    runs = tree(a={"L1": [trial(1.0), trial(1.0), trial(1.0), trial(1.0)]},
                b={"L1": [trial(0.0)]})
    block = S.rung_stats(runs)["L1"]
    assert block["mean_pass_probability"] == pytest.approx(0.5)
    assert block["tasks"] == 2 and block["trials"] == 5


def test_a_task_with_no_finished_trials_is_counted_separately_not_as_a_failure():
    """agent_status != 'exit 0' is a harness failure. Scoring it as a model failure is how a
    crash rate turns into a pass-rate drop, and the two were conflated for months."""
    runs = tree(a={"L1": [trial(1.0)]}, b={"L1": [trial(0.0, status="timeout")]},
                c={"L1": [trial(0.0), trial(1.0)]},
                d={"L1": [trial(0.0, status="error: TimeoutError"), trial(1.0)]})
    block = S.rung_stats(runs)["L1"]
    assert block["tasks"] == 4
    # b crashed outright and d lost one trial to a crash; both are counted, neither is scored.
    assert block["harness_failures"] == 2
    assert block["trials"] == 6 and block["trials_scored"] == 4
    # b contributes no fraction at all, so it is not in the mean's denominator...
    assert block["scored_tasks"] == 3
    # ...and d is scored on the one trial that finished, which is the only honest reading.
    assert S.pass_fraction(runs["d"]["L1"]) == pytest.approx(1.0)
    # a=1.0, c=0.5, d=1.0 over the three tasks that have a fraction
    assert block["mean_pass_probability"] == pytest.approx(2.5 / 3)


def test_a_fully_crashed_task_does_not_drag_the_mean_down():
    """The regression this guards: b above contributes nothing, so a sweep whose harness died
    on half the tasks does not report a collapsed pass rate."""
    runs = tree(a={"L1": [trial(1.0)]}, b={"L1": [trial(1.0, status="timeout")]},
                c={"L1": [trial(1.0, status="exit 1")]})
    assert S.rung_stats(runs)["L1"]["mean_pass_probability"] == pytest.approx(1.0)


def test_the_bootstrap_interval_brackets_the_estimate_and_narrows_with_more_tasks():
    """A CI that does not contain the estimate is worse than no CI.

    Both subsets are drawn from the same interleaved pool so the point estimate is identical,
    and the only thing that changes is how many tasks the resampling has to work with."""
    pool = [v for pair in zip([0.0] * 40, [1.0] * 60) for v in pair]
    few = S.rung_stats(tree(**{f"a{i}": {"L1": [trial(v)]} for i, v in enumerate(pool[:20])}))
    many = S.rung_stats(tree(**{f"b{i}": {"L1": [trial(v)]} for i, v in enumerate(pool)}))
    assert few["L1"]["mean_pass_probability"] == pytest.approx(many["L1"]["mean_pass_probability"])
    assert (many["L1"]["ci95"][1] - many["L1"]["ci95"][0]) < \
        (few["L1"]["ci95"][1] - few["L1"]["ci95"][0])
    for block in (few["L1"], many["L1"]):
        assert block["ci95"][0] <= block["mean_pass_probability"] <= block["ci95"][1]


def test_the_interval_is_deterministic_for_a_given_seed():
    """Two runs of the summary must not disagree, or a diff between two reports is noise."""
    runs = tree(**{f"t{i}": {"L1": [trial(i % 2)]} for i in range(20)})
    first, second = S.rung_stats(runs)["L1"], S.rung_stats(runs)["L1"]
    # Not a whole-dict equality: an unrun rung carries NaN, and NaN != NaN.
    assert first["mean_pass_probability"] == second["mean_pass_probability"]
    assert first["ci95"] == second["ci95"]


def test_a_single_task_gets_a_degenerate_interval():
    """Bootstrap resampling one task can only ever return that task. Report it as [p, p]."""
    block = S.rung_stats(tree(a={"L1": [trial(1.0), trial(0.0)]}))["L1"]
    assert block["ci95"] == [0.5, 0.5]


def test_per_task_pass_fraction_is_reported_for_every_task():
    runs = tree(a={"L1": [trial(1.0), trial(1.0), trial(0.0)]}, b={"L1": [trial(0.0)]})
    fractions = S.pass_fractions(runs)["L1"]
    assert fractions["a"] == pytest.approx(2 / 3)
    assert fractions["b"] == pytest.approx(0.0)


# --- monotonicity ----------------------------------------------------------------------------

def test_a_monotone_run_has_no_violations():
    """Monotone here means pass probability never decreases as information is added."""
    runs = tree(**{f"t{i}": {"L1": [trial(1.0)], "L2": [trial(1.0)], "L3": [trial(1.0)]}
                   for i in range(20)})
    mono = S.monotonicity(runs)
    assert mono["L1->L2"]["violations"] == 0
    assert mono["L1->L2"]["improved"] == 0
    assert mono["L1->L2"]["p_value"] is None


def test_a_decreasing_pair_is_flagged_with_a_significant_exact_test():
    """L2 adds information over L1. A drop is not noise to be averaged away, it is the finding.
    The test is exact and paired on tasks, so it does not assume a normal approximation at the
    n where these sweeps live."""
    runs = tree(**{f"t{i}": {"L1": [trial(1.0)], "L2": [trial(0.0)]} for i in range(10)})
    pair = S.monotonicity(runs)["L1->L2"]
    assert pair["discordant"] == 10
    assert pair["violations"] == 10
    assert pair["p_value"] == pytest.approx(2 * 0.5 ** 10)   # two-sided exact sign test


def test_violations_are_counted_on_tasks_measured_at_both_rungs():
    """Climbing left a task unmeasured at L2, and a pair needs both. Such a task is excluded
    from the pair's denominator and reported, not silently dropped."""
    runs = tree(a={"L1": [trial(1.0)], "L2": [trial(0.0)]},
                b={"L1": [trial(1.0)]})
    pair = S.monotonicity(runs)["L1->L2"]
    assert pair["paired"] == 1
    assert pair["excluded_unpaired"] == 1


def test_the_sign_test_is_omitted_when_no_task_is_discordant():
    """A pair where nothing disagrees carries no information about ordering, and a p-value
    computed from zero discordant pairs is meaningless."""
    runs = tree(**{f"t{i}": {"L1": [trial(1.0)], "L2": [trial(1.0)]} for i in range(5)})
    pair = S.monotonicity(runs)["L1->L2"]
    assert pair["violations"] == 0 and pair["discordant"] == 0
    assert pair["p_value"] is None


# --- the partition, from majority pass -------------------------------------------------------

def test_the_partition_uses_the_majority_pass_not_any_pass():
    """The old partition took 'reward >= 1 once'. With samples, a task that passes 1 of 4 is not
    an L1 task; the majority verdict is the one that is stable to a rerun, and the bucket's
    disagreement rate is reported next to it so a marginal bucket cannot hide."""
    runs = tree(flaky={"L1": [trial(1.0), trial(0.0), trial(0.0), trial(0.0)]},
                solid={"L1": [trial(1.0), trial(1.0), trial(1.0)]})
    hist = S.partition(runs, lambda _t: True)
    assert hist["L1"] == 1 and hist["never"] == 1
    marginal = S.marginality(runs)
    assert marginal["flaky"] == pytest.approx(0.25)


def test_a_tie_is_not_a_pass():
    """1 of 2 samples is 50/50. Calling that a pass reproduces the coin-flip bucket that A7 is
    about, so the tie goes to the failure side and scores as the least decisive task there is."""
    runs = tree(t={"L1": [trial(1.0), trial(0.0)]})
    assert S.first_passing_rung(runs["t"], has_reference=True) == "never"
    assert S.marginality(runs)["t"] == pytest.approx(0.0)


def test_the_partition_still_sums_to_the_task_count_with_samples():
    runs = tree(
        passes={"L1": [trial(1.0)], "L2": [trial(1.0)]},
        climbs={"L1": [trial(0.0), trial(0.0)], "L2": [trial(1.0), trial(1.0)]},
        hard={"L1": [trial(0.0)], "L2": [trial(0.0)], "L3": [trial(1.0), trial(1.0)]},
        never={"L1": [trial(0.0)], "L2": [trial(0.0)]},
        unattempted={},
    )
    refs = {"passes", "climbs", "hard", "never"}
    hist = S.partition(runs, lambda t: t in refs)
    assert sum(hist.values()) == len(runs)
    assert hist["L1"] == 1 and hist["L2"] == 1 and hist["L3"] == 1
    assert hist["never"] == 1 and hist["not attempted"] == 1


def test_the_control_is_still_not_a_bucket():
    runs = tree(t={"L1": [trial(0.0)], S.CONTROL: [trial(1.0)]})
    assert S.first_passing_rung(runs["t"], has_reference=True) == "never"
    assert S.CONTROL not in S.BUCKETS


# --- refusing to pool two ladder versions -----------------------------------------------------

def test_different_tasks_with_different_hashes_are_not_a_mixed_rung():
    """Every task's prompt embeds its own question, so every task's hash differs BY CONSTRUCTION.

    Collecting the hashes of a whole rung and calling a set of size >1 "mixed" made a clean
    250-task sweep un-summarisable: 250 tasks at L1 is 250 hashes, which is the opposite of
    evidence that two ladders were pooled. It is what made the v2 run need --allow-mixed.
    """
    runs = tree(**{f"t{i}": {"L1": [trial(i % 2, hash_=f"{i:064d}")]} for i in range(250)})
    assert S.summarise("test", runs, lambda _t: True)["mixed_prompts"] == {}


def test_one_task_measured_on_two_prompts_is_refused():
    """This is the real defect the hash exists to catch: a resumed sweep or a mid-run prompt
    reword changed the text for the SAME (task, rung), so its samples are not replicates."""
    runs = tree(a={"L1": [trial(1.0, hash_=HASH_A), trial(0.0, hash_=HASH_B)]})
    with pytest.raises(S.MixedPrompts):
        S.summarise("test", runs, lambda _t: True)


def test_the_error_names_the_task_the_rung_and_both_hashes():
    runs = tree(a={"L1": [trial(1.0, hash_=HASH_A), trial(0.0, hash_=HASH_B)]})
    with pytest.raises(S.MixedPrompts) as excinfo:
        S.summarise("test", runs, lambda _t: True)
    message = str(excinfo.value)
    assert "a" in message and "L1" in message
    assert HASH_A[:12] in message and HASH_B[:12] in message


def test_allow_mixed_pools_them_and_records_that_it_did():
    runs = tree(a={"L1": [trial(1.0, hash_=HASH_A), trial(0.0, hash_=HASH_B)]})
    report = S.summarise("test", runs, lambda _t: True, allow_mixed=True)
    assert report["mixed_prompts"] == {"L1": [HASH_A, HASH_B]}
    assert report["rungs"]["L1"]["mean_pass_probability"] == pytest.approx(0.5)


def test_results_with_no_prompt_hash_are_never_pooled_with_hashed_ones():
    """A legacy result predates provenance. Its prompt is unknown, not known-equal, so pairing it
    with a hashed one would average two ladder versions while claiming they matched."""
    runs = tree(a={"L1": [trial(1.0), trial(0.0, hash_=HASH_B)]})
    with pytest.raises(S.MixedPrompts):
        S.summarise("test", runs, lambda _t: True)


def test_a_whole_run_of_distinct_tasks_is_accepted():
    """The regression in one assertion: 20 real tasks, each measured on its own prompt, all of
    which the old check refused to report."""
    runs = tree(**{f"t{i}": {"L1": [trial(i % 2, hash_=f"{i:064x}"),
                                    trial(i % 2, hash_=f"{i:064x}")]} for i in range(20)})
    report = S.summarise("test", runs, lambda _t: True)
    assert report["mixed_prompts"] == {}
    assert report["rungs"]["L1"]["tasks"] == 20


def test_one_consistent_hash_across_a_rung_is_fine():
    runs = tree(**{f"t{i}": {"L1": [trial(i % 2)]} for i in range(10)})
    assert S.summarise("test", runs, lambda _t: True)["mixed_prompts"] == {}


def test_a_different_hash_at_a_different_rung_is_not_a_conflict():
    """L1 and L2 have different prompts by construction. The check is within a rung."""
    runs = tree(a={"L1": [trial(1.0, hash_=HASH_A)], "L2": [trial(0.0, hash_=HASH_B)]})
    assert S.summarise("test", runs, lambda _t: True)["mixed_prompts"] == {}


# --- a ladder-version mismatch is mixing too ----------------------------------------------------

def test_a_task_at_the_same_rung_on_two_ladder_fingerprints_is_refused():
    """Two different ladder definitions measured the same cell. The per-task prompt hash cannot
    always see it, because the prompt text can be identical while what built or graded it
    changed underneath."""
    runs = tree(a={"L1": [dict(trial(1.0), ladder_sha256="c" * 64),
                          dict(trial(0.0), ladder_sha256="d" * 64)]})
    with pytest.raises(S.MixedPrompts):
        S.summarise("test", runs, lambda _t: True)


def test_two_commits_with_one_ladder_definition_are_not_a_conflict():
    """This is v2's own situation, and the reason the fingerprint exists rather than the commit.

    v2 was resumed on a commit whose only change to the ladder was bookkeeping: run_ladder.py
    started recording `verify_status` and summarize stopped trusting a bare "exit 0". Every file
    that defines a rung's text or grades it -- ladder.py, gen_hints.py, gen_refs.py, or_agent.py,
    upstream.py, grade.py -- is byte-identical between the two commits, so both launches measured
    the same ladder. Refusing on the commit alone would have sent a clean run to --allow-mixed
    again, which is how a real mixing gets waved through next time.
    """
    runs = tree(a={"L1": [dict(trial(1.0), ladder_sha256="c" * 64, git_commit="a" * 40),
                          dict(trial(0.0), ladder_sha256="c" * 64, git_commit="b" * 40)]},
                b={"L1": [dict(trial(1.0), ladder_sha256="c" * 64, git_commit="b" * 40)]})
    assert S.summarise("test", runs, lambda _t: True)["mixed_prompts"] == {}


def test_a_run_record_fingerprint_that_no_result_carries_is_refused():
    """The mismatch the per-result check cannot make: RUN.json says the run was launched by a
    ladder version that produced none of the results on disk -- a tree assembled by hand, or a
    sweep that resumed under new code without re-running anything."""
    runs = tree(a={"L1": [dict(trial(1.0), ladder_sha256="c" * 64)]})
    record = {"launches": [{"ladder_sha256": "e" * 64}]}
    with pytest.raises(S.MixedPrompts):
        S.summarise("test", runs, lambda _t: True, record=record)


def test_a_run_record_fingerprint_the_results_agree_with_is_fine():
    runs = tree(a={"L1": [dict(trial(1.0), ladder_sha256="c" * 64)]})
    record = {"launches": [{"ladder_sha256": "c" * 64}]}
    assert S.summarise("test", runs, lambda _t: True, record=record)["mixed_prompts"] == {}


def test_a_run_record_with_no_fingerprint_is_not_invented():
    """A run tagged before fingerprints were recorded has nothing to compare. Refusing on an
    absent key would retire every existing tree at once."""
    runs = tree(a={"L1": [dict(trial(1.0), ladder_sha256="c" * 64)]})
    record = {"launches": [{"git_commit": "c" * 40}]}
    assert S.summarise("test", runs, lambda _t: True, record=record)["mixed_prompts"] == {}


def test_the_ladder_fingerprint_covers_every_file_that_defines_a_rung():
    """The fingerprint is the whole point, so it has to actually cover the ladder. A rung's text
    comes from ladder.py and the hint it interpolates from gen_hints.py, gen_refs.py and
    or_agent.py, under the protocols in upstream.py, and grade.py decides pass or fail. Change any
    of them and the hash must move, or a new ladder would be silently pooled with the old one."""
    sources = Path(S.__file__).parent
    baseline = S.ladder_fingerprint()
    for name in ("ladder.py", "gen_hints.py", "gen_refs.py", "or_agent.py", "upstream.py",
                 "grade.py"):
        assert (sources / name).exists(), name
        changed = hashlib.sha256(
            (sources / name).read_bytes() + b"smol-ladder-probe").hexdigest()
        assert changed != baseline, f"{name} is not covered by the fingerprint"


def test_a_hint_prompt_version_mismatch_is_refused():
    """L2-L4 text is built by gen_hints and its own PROMPT_VERSION stamps each hint. A rung built
    from a hint of another version is a different ladder, and prompt_sha256 cannot see it because
    the hint text itself changed."""
    runs = tree(a={"L2": [dict(trial(1.0), hint_prompt_version="nl-hints-v1"),
                          dict(trial(0.0), hint_prompt_version="nl-hints-v2")]})
    with pytest.raises(S.MixedPrompts):
        S.summarise("test", runs, lambda _t: True)


def test_one_hint_version_behind_l2_l3_l4_of_a_task_is_fine():
    """L2, L3 and L4 read the same cached hint, so a task measured on all three at one hint
    version is the normal case and must not be refused."""
    runs = tree(a={"L2": [dict(trial(1.0), hint_prompt_version="nl-hints-v1")],
                   "L3": [dict(trial(1.0), hint_prompt_version="nl-hints-v1")],
                   "L4": [dict(trial(0.0), hint_prompt_version="nl-hints-v1")]})
    assert S.summarise("test", runs, lambda _t: True)["mixed_prompts"] == {}


def test_l3_built_from_another_hint_version_than_l2_is_refused():
    """L4 must extend L3 verbatim or the rungs are not cumulative. If a task's L2 and L3 were
    built from different hint versions they are not the same ladder, and the ordering claim the
    "lowest rung that passes" rests on is void for that task."""
    runs = tree(a={"L2": [dict(trial(1.0), hint_prompt_version="nl-hints-v1")],
                   "L3": [dict(trial(0.0), hint_prompt_version="nl-hints-v2")]})
    with pytest.raises(S.MixedPrompts):
        S.summarise("test", runs, lambda _t: True)


def test_the_check_reports_a_tree_mixed_on_several_axes_at_once():
    runs = tree(a={"L1": [trial(1.0, hash_=HASH_A), trial(0.0, hash_=HASH_B)]},
                b={"L2": [dict(trial(1.0), hint_prompt_version="nl-hints-v1"),
                          dict(trial(0.0), hint_prompt_version="nl-hints-v2")]})
    with pytest.raises(S.MixedPrompts) as excinfo:
        S.summarise("test", runs, lambda _t: True)
    assert "L1" in str(excinfo.value) and "L2" in str(excinfo.value)


# --- the reference-gathering consumer must see the samples too -------------------------------

def test_failed_task_ids_finds_a_task_that_only_failed_on_sample_zero(tmp_path, monkeypatch):
    """refs_for_failures globs the results tree to decide which tasks need a reference.

    With --samples K the extra trials sit one level deeper, so a pattern that stopped at
    */L1/result.json would read only sample 0 -- and would then file a task that failed sample 0
    but passed on sample 2 as a failure. That is the same coin-flip bucket A7 is about, one
    level up: references would be built for tasks the model can already solve.
    """
    from smol_ladder import refs_for_failures as R
    monkeypatch.setattr(R, "DATA", tmp_path)
    root = tmp_path / "runs" / "test"

    def put(relative, reward, status="exit 0"):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"reward": reward, "agent_status": status}))

    put("always_fails/L1/result.json", 0.0)
    put("always_fails/L1/s1/result.json", 0.0)
    put("flaky/L1/result.json", 0.0)        # failed sample 0
    put("flaky/L1/s1/result.json", 1.0)     # passed on a rerun: solved, needs no reference
    put("solved/L1/result.json", 1.0)
    put("crashed/L1/result.json", 0.0, status="timeout")
    put("solved/L2/result.json", 0.0)       # a different rung must not leak in

    assert R.failed_task_ids("test") == ["always_fails"]


def test_failed_task_ids_is_empty_when_nothing_failed(tmp_path, monkeypatch):
    from smol_ladder import refs_for_failures as R
    monkeypatch.setattr(R, "DATA", tmp_path)
    root = tmp_path / "runs" / "test"
    (root / "t" / "L1").mkdir(parents=True)
    (root / "t" / "L1" / "result.json").write_text(json.dumps(
        {"reward": 1.0, "agent_status": "exit 0"}))
    assert R.failed_task_ids("test") == []


# --- the report -------------------------------------------------------------------------------

def test_the_report_carries_the_per_rung_block_the_headline_needs():
    runs = tree(**{f"t{i}": {"L1": [trial(i % 2)]} for i in range(10)})
    report = S.summarise("test", runs, lambda _t: True)
    block = report["rungs"]["L1"]
    assert set(block) >= {"tasks", "trials", "trials_scored", "harness_failures",
                          "mean_pass_probability", "ci95"}
    assert report["monotonicity"]["L1->L2"]["paired"] == 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))