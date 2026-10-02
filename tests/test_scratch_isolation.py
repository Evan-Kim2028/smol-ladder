"""Two concurrent trials of the same task and rung must not share a scratch directory.

The per-trial scratch used to be `$SMOL_LADDER_SCRATCH/trials/<task_id>/<rung_label>`, and once()
rmtree'd it on entry. Two models evaluated at once under different run tags (or a sweep and a
retry) run the same task and rung together, so each one's rmtree deleted the other's working
directory mid-trial: `FileNotFoundError: 'transcript.json'`, or one trial reading the other's file.
"""

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path

from smol_ladder import run_ladder as runner

ROW = {"task_id": "t1", "question": "Q?", "files": ["t.csv"], "answer": "42",
       "reward_mode": "numeric", "atol": 0.0, "rtol": 0.0}


def test_concurrent_trials_of_one_task_and_rung_do_not_interfere(tmp_path, monkeypatch):
    inputs = tmp_path / "in"
    inputs.mkdir()
    (inputs / "t.csv").write_text("a\n1\n")
    scratch = tmp_path / "scratch"
    monkeypatch.setenv("SMOL_LADDER_SCRATCH", str(scratch))
    both_in_flight = threading.Barrier(2, timeout=20)
    cwds: list[Path] = []

    def fake_run(cmd, cwd, env, timeout):
        # The trial's prompt is the last argv element; it says whose trial this is.
        who = cmd[-1]
        cwd = Path(cwd)
        cwds.append(cwd)
        (cwd / "transcript.json").write_text(json.dumps([{"who": who}]))
        (cwd / "answer.txt").write_text(who)
        both_in_flight.wait()            # both trials now hold their scratch at the same time
        assert (cwd / "transcript.json").exists(), "another trial deleted this trial's scratch"
        assert json.loads((cwd / "transcript.json").read_text()) == [{"who": who}]

        class P:
            returncode, stdout, stderr = 0, b"", b""
        return P()

    results, errors = {}, []

    def trial(tag):
        try:
            work = tmp_path / "runs" / tag / "t1" / "L1"
            results[tag] = runner.once(ROW, f"prompt-{tag}", work, Path(sys.prefix), "m", 2,
                                       inputs_of=lambda r: inputs, rung_label="L1", agent="bash",
                                       provenance={"rung": "L1", "sample": 0, "run_tag": tag})
        except BaseException as e:  # noqa: BLE001 - the test reports it below
            errors.append(e)

    monkeypatch.setattr(runner, "_run_jailed", fake_run)
    threads = [threading.Thread(target=trial, args=(t,)) for t in ("tagA", "tagB")]
    [t.start() for t in threads]
    [t.join(30) for t in threads]
    assert not errors, errors
    assert len(set(cwds)) == 2
    for tag in ("tagA", "tagB"):
        work = tmp_path / "runs" / tag / "t1" / "L1"
        assert json.loads((work / "transcript.json").read_text()) == [{"who": f"prompt-{tag}"}]
        assert (work / "answer.txt").read_text() == f"prompt-{tag}"
        assert results[tag]["prediction"] == f"prompt-{tag}"
    for c in cwds:                       # and each cleaned up after itself
        assert not c.exists()


def test_the_scratch_name_carries_rung_tag_sample_and_is_unique_per_call():
    a = runner.scratch_label("L1", "amd1-a", 2)
    b = runner.scratch_label("L1", "amd1-a", 2)
    assert a != b and a.startswith("L1.amd1-a.s2.p")
