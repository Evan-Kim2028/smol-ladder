"""Measure SFT throughput on the droplet with the REAL trainer, and assemble the smoke's report.

    python ops/amd/bench.py run --out bench.json --seconds 120
    python ops/amd/bench.py finalize --checks checks.tsv --bench bench.json --resume resume.json --out measurements.json

Why the real trainer: the number that decides whether three arms fit the budget is tokens per
second of `python -m train.sft_lora` on the real base, real rows, real collator, gradient
checkpointing and bf16 included. A hand-rolled loop measures a different program. So each
configuration is a subprocess of the actual trainer with `--logging-steps 1`; every optimizer step
logs one line, the lines are timestamped as they arrive, and throughput is steps per second over
a steady window times the real tokens per step.

Real tokens, not padded tokens: rows are cut at --max-length and batches are padded to their
longest row, so tokens/s is steps/s x effective batch x mean trained tokens per row (from the
staged tokens.json), which is exactly what the budget arithmetic divides by.

Configurations: per-device batch 2, 4, 8 with gradient accumulation chosen to keep the effective
batch at 8 (never batch 1: the card has 288 GB). `train/sft_lora.py` exposes no packing flag, so
packing is not benchmarked; that is stated in the runbook rather than hidden here.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from ops.amd.plan import BENCH_BATCHES, EFFECTIVE_BATCH  # noqa: E402

LOSS_LINE = re.compile(r"""['"]loss['"]""")
OOM = re.compile(r"out of memory|OutOfMemoryError|HIP out of memory", re.I)
WARMUP_STEPS = 3


def steps_per_second(stamps: list[float], warmup: int = WARMUP_STEPS) -> float | None:
    """Steady-state steps/s from the arrival times of the per-step log lines. The first
    `warmup` steps (compilation, allocator growth) are dropped; fewer than 3 steps after that
    is not a measurement."""
    usable = stamps[warmup:]
    if len(usable) < 3:
        return None
    span = usable[-1] - usable[0]
    return (len(usable) - 1) / span if span > 0 else None


def tokens_per_second(sps: float, effective_batch: int, mean_tokens_per_row: float) -> float:
    return sps * effective_batch * mean_tokens_per_row


def choose_best(configs: list[dict]) -> dict | None:
    """The fastest configuration that did not run out of memory."""
    ok = [c for c in configs if c.get("ok") and c.get("tokens_per_s")]
    return max(ok, key=lambda c: c["tokens_per_s"]) if ok else None


def run_config(cmd: list[str], seconds: float, mean_tokens: float, bs: int, accum: int,
               env: dict | None = None, hard_timeout: float | None = None) -> dict:
    """Run one configuration for about `seconds` after its first step, then stop it.

    `hard_timeout` (default: the window plus 15 minutes for model load) kills a run that hangs
    before or between steps: a benchmark that waits forever on a stuck process is a billed hang.
    """
    hard = hard_timeout if hard_timeout is not None else seconds + 900.0
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            bufsize=1, env=dict(os.environ, PYTHONUNBUFFERED="1", **(env or {})),
                            start_new_session=True)
    stamps: list[float] = []
    tail: list[str] = []
    first = None
    oom = False
    timed_out = threading.Event()

    def kill_hung() -> None:
        timed_out.set()
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    timer = threading.Timer(hard, kill_hung)
    timer.daemon = True
    timer.start()
    assert proc.stdout is not None
    for line in proc.stdout:
        now = time.monotonic()
        tail = (tail + [line.rstrip()])[-15:]
        if OOM.search(line):
            oom = True
        if LOSS_LINE.search(line):
            stamps.append(now)
            first = first or now
        if first is not None and now - first >= seconds:
            break
    timer.cancel()
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
    sps = steps_per_second(stamps)
    result = {"per_device_batch_size": bs, "grad_accum": accum, "steps_logged": len(stamps),
              "oom": oom, "timed_out": timed_out.is_set(), "ok": sps is not None and not oom}
    if sps is not None:
        result["steps_per_s"] = round(sps, 4)
        result["tokens_per_s"] = round(tokens_per_second(sps, bs * accum, mean_tokens), 1)
    else:
        result["tail"] = tail[-6:]
    return result


def make_bench_data(a_train: Path, b_file: Path, out: Path, rows: int, seed: int = 0) -> int:
    """A reproducible mixed sample of A and B: the padding the real arms see, not a synthetic one."""
    pool = [l for f in (a_train, b_file) for l in f.read_text().splitlines() if l.strip()]
    random.Random(seed).shuffle(pool)
    out.write_text("\n".join(pool[:rows]) + "\n")
    return min(rows, len(pool))


def cmd_run(args: argparse.Namespace) -> int:
    tokens = json.loads((ROOT / "tokens.json").read_text())["sets"]["AB"]
    mean_tokens = tokens["trained_tokens"] / tokens["rows"]
    data = Path("/tmp/bench_rows.jsonl")
    n = make_bench_data(ROOT / "data/train/sft_upstream/train.jsonl",
                        ROOT / "data/train/ja3_sft.jsonl", data, args.rows)
    py = sys.executable
    results = []
    for bs in args.batches:
        accum = max(1, EFFECTIVE_BATCH // bs)
        out = Path(f"/tmp/bench_bs{bs}")
        cmd = [py, "-m", "train.sft_lora", "--data", str(data), "--out", str(out),
               "--protocol", "bash", "--max-length", str(args.max_length), "--precision", "bf16",
               "--batch-size", str(bs), "--grad-accum", str(accum), "--logging-steps", "1",
               "--max-steps", "100000", "--seed", "42"]
        print(f"  batch {bs} x accum {accum}: {n}-row sample, {args.seconds:.0f}s", flush=True)
        r = run_config(cmd, args.seconds, mean_tokens, bs, accum)
        print(f"    -> {json.dumps({k: v for k, v in r.items() if k != 'tail'})}", flush=True)
        if r.get("tail") and not r["ok"]:
            print("    last lines:\n      " + "\n      ".join(r["tail"]), flush=True)
        results.append(r)
    best = choose_best(results)
    Path(args.out).write_text(json.dumps(
        {"max_length": args.max_length, "mean_tokens_per_row": round(mean_tokens, 1),
         "configs": results, "best": best}, indent=2))
    if best is None:
        print("no configuration trained: refusing to project from nothing", file=sys.stderr)
        return 2
    print(f"BEST batch {best['per_device_batch_size']} x accum {best['grad_accum']}: "
          f"{best['tokens_per_s']:,.0f} real tokens/s")
    return 0


def read_checks(path: Path) -> list[dict]:
    """checks.tsv: `PASS|FAIL<TAB>name` per line, written by smoke.sh."""
    out = []
    for line in path.read_text().splitlines():
        status, _, name = line.partition("\t")
        if status in ("PASS", "FAIL"):
            out.append({"ok": status == "PASS", "name": name})
    return out


def finalize(checks: list[dict], bench: dict | None, resume: dict | None) -> dict:
    """The smoke's report. `checks_ok` is False on any failed item; the laptop makes the go/no-go."""
    best = (bench or {}).get("best")
    return {"checks": checks, "checks_ok": bool(checks) and all(c["ok"] for c in checks),
            "bench": (bench or {}).get("configs", []), "best": best,
            "resume": resume or {"ok": False, "note": "not run"},
            "max_length": (bench or {}).get("max_length"),
            "measured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def cmd_finalize(args: argparse.Namespace) -> int:
    load = lambda p: json.loads(Path(p).read_text()) if p and Path(p).exists() else None  # noqa: E731
    report = finalize(read_checks(Path(args.checks)), load(args.bench), load(args.resume))
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"measurements -> {args.out}: checks_ok={report['checks_ok']} "
          f"resume_ok={report['resume'].get('ok')} best={report['best']}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out", required=True)
    r.add_argument("--seconds", type=float, default=120.0)
    r.add_argument("--max-length", type=int, default=8192)
    r.add_argument("--rows", type=int, default=640)
    r.add_argument("--batches", type=int, nargs="+", default=list(BENCH_BATCHES))
    f = sub.add_parser("finalize")
    f.add_argument("--checks", required=True)
    f.add_argument("--bench")
    f.add_argument("--resume")
    f.add_argument("--out", required=True)
    args = ap.parse_args()
    return cmd_run(args) if args.cmd == "run" else cmd_finalize(args)


if __name__ == "__main__":
    sys.exit(main())
