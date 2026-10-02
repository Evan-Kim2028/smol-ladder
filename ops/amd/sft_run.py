"""Run `train.sft_lora` with a checkpoint cadence it has no flag for.

    python -m ops.amd.sft_run --save-steps 50 --data ... --out ... (every train.sft_lora flag)

`train/sft_lora.py` writes a checkpoint every `max(50, max_steps // 2)` steps, which with no
`--max-steps` is 100: at the measured 5,446 tokens/s that is a long time to lose to a GPU reset.
The trainer is not ours to change from here, so this wrapper takes the one flag the trainer lacks,
forces it into the `SFTConfig` the trainer builds (it imports the class inside `build()`, at call
time, so replacing the name on the `trl` module is enough), and hands everything else through
untouched. A checkpoint is pushed to the arm's private Hub repo each time it is saved.
"""

from __future__ import annotations

import argparse
import sys


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
        return real(*args, **kwargs)

    trl.SFTConfig = sft_config
    from train import sft_lora

    sys.argv = ["train.sft_lora", *rest]
    sft_lora.main()


if __name__ == "__main__":
    main()
