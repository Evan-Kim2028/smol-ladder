"""report_runs: common task set, per-task pass rates over attempts, and the paired comparison."""
import json

from smol_ladder import report_runs as R


def trial(root, tag, task, rung_dir, reward, sample=None, stop="model_stopped", pred="1", cmds=("a", "b")):
    d = root / tag / "test" / task / rung_dir / (f"s{sample}" if sample else "")
    d.mkdir(parents=True, exist_ok=True)
    (d / "result.json").write_text(json.dumps({"task_id": task, "rung": rung_dir.replace("_", "+"),
                                                "reward": reward, "stop_reason": stop, "prediction": pred}))
    (d / "transcript.json").write_text(json.dumps([{"role": "assistant", "tool_calls": [
        {"function": {"name": "bash", "arguments": {"command": c}}}]} for c in cmds]))


def test_common_set_attempt_means_and_pairing(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "RUNS", tmp_path)
    monkeypatch.setattr(R, "load_split", lambda split: [{"task_id": f"t{i}", "difficulty_tier": "easy"} for i in range(4)])
    for i in range(4):                                   # base: one attempt, passes t0 and t1
        trial(tmp_path, "b", f"t{i}", "L1", 1.0 if i < 2 else 0.0)
    for i in range(3):                                   # A: 2 attempts, only 3 tasks scored; t2 passes once
        trial(tmp_path, "a", f"t{i}", "L1", 1.0 if i < 2 else 0.0)
        trial(tmp_path, "a", f"t{i}", "L1", 1.0 if i == 2 else 0.0, sample=1,
              stop="max_turns", pred="", cmds=("x", "x", "x", "x"))
    trial(tmp_path, "a", "t3", "L1", 0.0, stop="error")  # a harness error is not a scored trial
    rep = R.report({"base": "b", "A": "a"}, "L1", "base")
    assert rep["common_tasks"] == 3
    base, a = rep["models"]["base"], rep["models"]["A"]
    assert base["pass_rate"] == round(2 / 3, 4) and base["attempts_per_task"] == 1.0
    assert a["attempts_per_task"] == 2.0 and a["pass_rate"] == round((0.5 + 0.5 + 0.5) / 3, 4)
    assert a["vs_base"] == {"gained": 1, "lost": 2, "sign_test_p": 1.0, "mean_difference": round(-1 / 6, 4),
                            "difference_ci95": a["vs_base"]["difference_ci95"]}
    assert a["ended_in_repeat_loop"] == 0.5 and a["wrote_any_answer"] == 0.5
    assert R.sign_test(15, 4) < 0.05 and R.sign_test(10, 10) == 1.0
