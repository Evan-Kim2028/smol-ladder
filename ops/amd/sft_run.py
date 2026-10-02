"""Run `train.sft_lora` with a checkpoint cadence it has no flag for.

    python -m ops.amd.sft_run --save-steps 50 --data ... --out ... (every train.sft_lora flag)

`train/sft_lora.py` writes a checkpoint every `max(50, max_steps // 2)` steps, which with no
`--max-steps` is 100: at the measured 5,446 tokens/s that is a long time to lose to a GPU reset.
The trainer is not ours to change from here, so this wrapper takes the one flag the trainer lacks,
forces it into the `SFTConfig` the trainer builds (it imports the class inside `build()`, at call
time, so replacing the name on the `trl` module is enough), and hands everything else through
untouched. A checkpoint folder is pushed to the arm's private Hub repo as `last-checkpoint/` each
time it is saved (hub_strategy="checkpoint"; see tests/test_hub_layout.py).
"""

from __future__ import annotations

import argparse
import sys

RESUMABLE = ("checkpoint",)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--save-steps", type=int, default=50)
    ns, rest = ap.parse_known_args(sys.argv[1:] if argv is None else argv)
    if ns.save_steps < 1:
        raise SystemExit("--save-steps must be at least 1")

    import trl

    real = trl.SFTConfig

    def sft_config(*args, **kwargs):
        kwargs["save_steps"] = ns.save_steps
        # `last-checkpoint/` is what a fresh droplet resumes from (ops/amd/resume.py), and the
        # Trainer writes it only for this strategy. "every_save" pushed the adapter files
        # alone, so every arm restarted from step 0 after a reclaim: refuse that config here too.
        if kwargs.get("push_to_hub") and kwargs.get("hub_strategy") not in RESUMABLE:
            raise SystemExit(f"hub_strategy={kwargs.get('hub_strategy')!r} does not push "
                             "last-checkpoint/, so a fresh droplet could not resume: use "
                             "'checkpoint' (train/sft_lora.py --hub-strategy)")
        return real(*args, **kwargs)

    trl.SFTConfig = sft_config
    from train import sft_lora

    sys.argv = ["train.sft_lora", *rest]
    sft_lora.main()


if __name__ == "__main__":
    main()
