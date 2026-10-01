"""GRPO on our jail and our offline grader, with the hint curriculum.

    uv run --extra train python -m train.grpo --steps 200 --rung-schedule ladder

The reward is our own: the value is whatever `solution.py` prints when re-run **sealed** (no
network, its own working directory), graded by SmolDataEnvs' `grader.py`. That is
`smol_ladder.run_ladder.once`'s grading path, reused rather than reimplemented -- a second grader
here would be a second definition of correctness in a study whose entire question is whether a
number transfers.

## The shaping reward, and the guard against repeating upstream's mistake

Upstream paid `+0.1` for "the program runs" and the policy found the loophole twice. Their README:
the first version paid it for *no traceback*, which an empty program also satisfies, and by step ~70
the policy emitted an empty `<think></think>`, collected 0.1 per rollout, and every completion in a
group scored identically -- so the advantage was zero and training had quietly stopped. Requiring
the program to have **printed** something closed that; requiring it to have *run* then turned the
bonus into a verbosity reward, and completion length climbed 300 -> 900 tokens.

So: **the shaping term is logged at weight 0, exactly as upstream now ships it** (`reward_weights=
[1.0, 0.0]`), and `SHAPING_WEIGHT` defaults to 0. That is the guard. Correctness alone is a usable
signal -- the first step of an upstream run scores around 0.3 -- so there is no need for the
shaping term to earn its keep, and a bonus that can be collected without being right is a bonus the
optimiser will find.

If someone turns it on anyway, `shape_bonus()` is the only place it is defined and its output is
also recorded per rollout in the training log, so a run that used it has the number its rollouts
actually collected and cannot be quietly compared against a run that did not.

## The rung schedule

Each task starts at a high rung (L3, or L2) and is **demoted** toward L1 as its rolling pass rate
rises: a task the model has learned gets less information, a task it cannot solve keeps the hints
that make its group non-degenerate. That is the adaptive curriculum the information-ladder
post suggests, and it targets a real GRPO failure -- at a ~28% pass rate many groups score all
zeros and contribute no gradient at all.

`--rung-schedule off` trains every task at L1, which is upstream's setting and the control.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict, deque
from pathlib import Path

# The ladder's rungs, most information first. Order matters: it is the demotion order.
RUNGS = ("L1", "L2", "L3", "L4")
# Logged, never optimised. See the module docstring.
SHAPING_WEIGHT = 0.0


class RungScheduler:
    """Per-task rung, demoted as the task's rolling pass rate rises.

    The whole point of the schedule is that the *same* task gets progressively less help, so the
    state is per task: a deque of that task's recent rewards, and an index into RUNGS.

    - `promote`/demote thresholds are deliberately far apart (`promote_below=0.25`,
      `demote_above=0.75`). A task hovering near 50% would otherwise oscillate rung every step,
      and a rung that flickers is a rung that is not a rung.
    - `window` is the number of recent attempts counted. Small on purpose: this is a *rolling*
      rate over recent attempts, not a lifetime pass rate, so a task that started failing after it
      was learned still loses its hints.
    - A task with no history returns its starting rung, which is what makes the first step of a run
      well-defined rather than a special case at every call site.
    """

    def __init__(self, start: str = "L3", window: int = 8, promote_below: float = 0.25,
                 demote_above: float = 0.75, floor: str = "L1", enabled: bool = True):
        if start not in RUNGS or floor not in RUNGS:
            raise ValueError(f"start/floor must be one of {RUNGS}")
        if RUNGS.index(start) < RUNGS.index(floor):
            raise ValueError(f"start {start} carries less information than floor {floor}")
        self.start, self.floor = start, floor
        # `start` is where a task *begins*, not the most it can be given. A task that starts at the
        # floor and keeps failing must be able to climb above it -- that is the whole reason for a
        # hint curriculum, since an all-zero group gives no gradient and the hints are what make
        # the group non-degenerate. The ceiling is therefore the most informative rung.
        self.ceiling = RUNGS[-1]
        self.window = window
        self.promote_below, self.demote_above = promote_below, demote_above
        self.enabled = enabled
        self.history: dict[str, deque] = defaultdict(lambda: deque(maxlen=window))
        self.current: dict[str, str] = {}

    def rung(self, task_id: str) -> str:
        if not self.enabled:
            return self.floor
        if task_id in self.current:
            return self.current[task_id]
        self.current[task_id] = self.start
        return self.start

    def rate(self, task_id: str) -> float:
        seen = self.history.get(task_id)
        return sum(seen) / len(seen) if seen else 0.0

    def update(self, task_id: str, rewards: list[float]) -> str:
        """Record one task's group of rollout rewards and return its rung for next time.

        The group, not the mean, is what is recorded: a GRPO group is the unit the advantage is
        computed over, and a task that passed once out of eight is not "half learned".
        """
        if not self.enabled:
            return self.floor
        seen = self.history[task_id]
        seen.extend(1.0 if r >= 1.0 else 0.0 for r in rewards)
        rung = self.rung(task_id)
        rate = self.rate(task_id)
        if rate >= self.demote_above and RUNGS.index(rung) > RUNGS.index(self.floor):
            rung = RUNGS[RUNGS.index(rung) - 1]
        elif rate <= self.promote_below and RUNGS.index(rung) < RUNGS.index(self.ceiling):
            rung = RUNGS[RUNGS.index(rung) + 1]
        self.current[task_id] = rung
        return rung


def shape_bonus(prediction: str) -> float:
    """Upstream's "it printed something", as a diagnostic. Weighted 0.

    Requires *output*, not merely a clean exit: an empty program satisfies "no traceback" and that
    is the exact loophole upstream's policy found. Whitespace does not count either -- a program
    whose last line is blank printed something only in the sense that stdout existed.
    """
    return 1.0 if (prediction or "").strip() else 0.0


def reward_for(row: dict, prediction: str, offline_stdout: str = "") -> dict:
    """Correctness from our grader, plus the logged diagnostic.

    `prediction` is the last line of the *sealed offline run*, which is the same evidence the eval
    harness grades. A rollout whose program printed nothing scores 0 and reports why, so an
    all-zero group is distinguishable from a dead sandbox.
    """
    from smol_ladder.grade import grade, last_line

    printed = last_line(offline_stdout) if offline_stdout else ""
    correct = grade(row, prediction or printed)
    return {"reward": correct, "shaping": shape_bonus(printed or prediction),
            "prediction": prediction or printed}


def build_reward_funcs(sandbox=None, shaping_weight: float = SHAPING_WEIGHT):
    """TRL reward functions over our grader.

    `sandbox` is a callable `program(row, code) -> stdout`, injected so the wiring is testable
    without a GPU or a sandbox. The default runs the program through the same sealed bubblewrap
    pass `run_ladder.once` uses.
    """
    run = sandbox or _sealed_run

    def reward_correct(completions, **columns):
        rewards, shaping = [], []
        for i, completion in enumerate(completions):
            row = _row_for(columns, i)
            code = _extract(completion)
            result = reward_for(row, "", run(row, code))
            rewards.append(result["reward"])
            shaping.append(result["shaping"])
        # The shaping values are logged rather than returned, so a run that has them can still be
        # compared honestly: they say how often the policy could have collected the bonus it was
        # not paid for, which is what the guard exists to make visible.
        shaping_log.extend(shaping)
        return rewards

    def reward_shaping(completions, **columns):
        return [0.0] * len(completions)

    shaping_log: list[float] = []
    return [reward_correct, reward_shaping], shaping_log


def _row_for(columns: dict, index: int) -> dict:
    """The task row for rollout `index`, from whichever column names TRL happened to pass through."""
    for key in ("row", "task", "task_row"):
        rows = columns.get(key)
        if isinstance(rows, list) and index < len(rows):
            return rows[index]
    return {k: v[index] for k, v in columns.items()
            if isinstance(v, list) and len(v) > index and k in
            {"answer", "reward_mode", "atol", "rtol", "bucket_prefix", "task_id"}}


def _extract(completion) -> str:
    from smol_ladder.upstream import extract_code

    return extract_code(completion)


def _sealed_run(row: dict, code: str, timeout: int = 180) -> str:
    """Run a generated program offline, sealed, against the task's tables. Returns stdout.

    The grading pass, not a reward hack's escape hatch: `--unshare-net`, its own work directory,
    the tables copied in. A program that tries to reach the network gets nothing, and a program
    that reads the gold answers cannot, because they are not on this machine inside the jail.
    """
    import subprocess
    import sys
    import tempfile

    from smol_ladder.sandbox import run_script

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        (work / "solution.py").write_text(code)
        inputs = None
        try:
            from smol_ladder.tasks import input_dir

            inputs = input_dir(row)
        except Exception:
            return ""
        run = run_script(work / "solution.py", inputs)
        return run.stdout if not run.timed_out else ""


def load_tasks(split: str = "train", limit: int = 256) -> list[dict]:
    """Train tasks, firewalled against the held-out splits.

    The firewall here is not inherited from the exporter -- GRPO trains online and picks its own
    tasks -- so it is applied again at load time. Training on a held-out task would put the answer
    into the policy's own gradient, which is worse than training on it offline.
    """
    from smol_ladder.tasks import load_split
    from train.format import heldout_keys, is_heldout

    keys = heldout_keys({"test": load_split("test"), "eval": load_split("eval")})
    rows = [r for r in load_split(split) if not is_heldout(r, keys)]
    return rows[:limit]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=os.environ.get("MODEL", "Qwen/Qwen3.5-2B"))
    ap.add_argument("--adapter", default=os.environ.get("ADAPTER", ""),
                    help="LoRA adapter to start from (the better SFT arm); empty = full fine-tune")
    ap.add_argument("--out", type=Path, default=Path("runs/grpo"))
    ap.add_argument("--split", default="train")
    ap.add_argument("--num-tasks", type=int, default=256)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--num-generations", type=int, default=8)
    ap.add_argument("--max-completion-length", type=int, default=1024)
    ap.add_argument("--learning-rate", type=float, default=3e-6)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--per-device-batch", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--shaping-weight", type=float, default=SHAPING_WEIGHT,
                    help="weight on the 'printed something' bonus. 0 (upstream's shipped value) "
                         "unless you have read the module docstring; any non-zero value is logged "
                         "in the run record so the run cannot be compared to one without it.")
    ap.add_argument("--rung-schedule", default="off", choices=["off", "ladder"])
    ap.add_argument("--rung-start", default="L3", choices=list(RUNGS))
    ap.add_argument("--rung-floor", default="L1", choices=list(RUNGS))
    ap.add_argument("--rung-window", type=int, default=8)
    ap.add_argument("--hub-model-id", default=os.environ.get("HUB_MODEL_ID", ""))
    args = ap.parse_args()

    scheduler = RungScheduler(start=args.rung_start, floor=args.rung_floor,
                              window=args.rung_window,
                              enabled=args.rung_schedule == "ladder")
    tasks = load_tasks(args.split, args.num_tasks)
    print(f"{len(tasks)} train tasks after the firewall; rung schedule={args.rung_schedule} "
          f"(start={args.rung_start}, floor={args.rung_floor})")

    record = {"model": args.model, "adapter": args.adapter, "steps": args.steps,
              "rung_schedule": args.rung_schedule, "rung_start": args.rung_start,
              "rung_floor": args.rung_floor, "shaping_weight": args.shaping_weight,
              "tasks": len(tasks), "num_generations": args.num_generations}
    print(json.dumps(record, indent=1))
    _train(args, tasks, scheduler, record)


def _train(args, tasks, scheduler: RungScheduler, record: dict) -> None:
    """The actual TRL loop. Imports here so the scheduler and reward are testable without torch."""
    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer

    from train.rungs import rung_row

    def build(row):
        rung = scheduler.rung(row["task_id"])
        prompt = _prompt_for(row, rung)
        entry = rung_row(row["question"], row.get("files") or [], prompt, "print()", "x")
        return {"prompt": [{"role": m["role"], "content": m["content"]} for m in entry["messages"]],
                "answer": row["answer"], "reward_mode": row["reward_mode"],
                "atol": row["atol"], "rtol": row["rtol"], "row": row, "rung": rung}

    dataset = Dataset.from_list([build(row) for row in tasks])
    reward_funcs, shaping_log = build_reward_funcs()
    config = GRPOConfig(
        output_dir=str(args.out),
        chat_template_kwargs={"enable_thinking": False},
        reward_weights=[1.0, args.shaping_weight],
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        max_steps=args.steps,
        learning_rate=args.learning_rate,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=1.05,
        mask_truncated_completions=True,
        per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        bf16=True,
        logging_steps=1,
        save_steps=50,
        log_completions=True,
        push_to_hub=bool(args.hub_model_id),
        hub_model_id=args.hub_model_id or None,
    )
    trainer = GRPOTrainer(model=args.model, reward_funcs=reward_funcs,
                          train_dataset=dataset, args=config)
    trainer.train()
    trainer.save_model(str(args.out))
    record["mean_printing_bonus"] = (sum(shaping_log) / len(shaping_log)) if shaping_log else None
    (args.out / "grpo_run.json").write_text(json.dumps(record, indent=1))


def _prompt_for(row: dict, rung: str) -> str:
    """The rung's prompt text, from the same function the eval sweep uses."""
    from smol_ladder.ladder import prompt_for

    return prompt_for(row, "train", rung)


if __name__ == "__main__":
    main()