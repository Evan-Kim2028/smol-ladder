# SFT checkpoint: GPU session 2 (2026-10-02)

This is the stop point agreed before any GRPO run: what the three SFT arms did to Qwen3.5-2B, measured
at L1 (question only) on the 250 held-out SmolDataEnvs test tasks. Session 1's adapters and numbers
are void (see `TRAINING.md` §0); nothing here uses them.

## The short version

- SFT on upstream's trajectories (A) or on A plus ours (A+B) is **not detectably different from the
  untrained base model**, at temperature 0 or sampled.
- SFT on our own trajectories alone (B) **makes the model much worse**, and a mechanical lean
  rewrite of those trajectories (B3) does not fix it.
- The adapter upstream released (R) scores **below** the base model on this harness.

## What was trained

LoRA r=16, alpha=32, bf16, max length 8,192, effective batch 8, one pass over the data.

| Arm | Data | Rows | Steps | Loss at the end |
|---|---|---|---|---|
| A | `sft_upstream` (SmolDataEnvs-sft through the firewall) | 4,439 | 555 | validation 0.378 |
| B | `ja3_sft_v2` (space-bunny on jupyter-agent tasks, verified) | 1,122 | 141 | training ~0.49 (no validation set) |
| A+B | both | 5,561 | 696 | validation 0.369 |
| B3 | `ja3_sft_v3` (B rewritten leaner, `train/export_ja3_v3.py`) | 989 | 124 | training ~0.50 |

Adapters: `evandekim/smol-ladder-sft-{a,b,ab,b3}-s2` (private), each with `final.done`.
R is `AdithyaSK/smoldataenvs-sft-2b-v0` at the pinned revision.

## L1, temperature 0, 250 test tasks, one attempt

| Model | Correct | Accuracy | Without the 27 notebook-overlap tasks | vs base, task by task (gained / lost) |
|---|---|---|---|---|
| base | 60 / 250 | 24.0% | 57 / 223 (25.6%) | |
| A | 64 / 250 | 25.6% | 60 / 223 (26.9%) | 28 / 24 (sign test p = 0.68) |
| A+B | 62 / 250 | 24.8% | 60 / 223 (26.9%) | 33 / 31 (p = 0.90) |
| R | 48 / 250 | 19.2% | 46 / 223 (20.6%) | 19 / 31 (p = 0.12) |
| B | 24 / 249 | 9.6% | 24 / 222 (10.8%) | 9 / 45 (p < 0.001) |
| B3 | 27 / 249 | 10.8% | | 8 / 41; against B: 12 / 9 |

Zero harness failures. B and B3 each have one task that never returned (a different one each).
With 250 tasks near 25%, a difference under about 7 points is noise.

How the episodes ended:

| Model | Stopped by itself | Hit 16 turns | Ran out of context |
|---|---|---|---|
| base | 111 | 118 | 21 |
| A | 129 | 100 | 21 |
| A+B | 116 | 113 | 21 |
| R | 88 | 142 | 20 |
| B | 36 | 157 | 56 |
| B3 | 42 | 166 | 41 |

## L1, sampled (temperature 0.7, top-p 0.8, top-k 20), one attempt

The 60 gate tasks, all five models: base 9, A 20, A+B 15, R 11, B 6. A against base was 15 gained,
4 lost (p = 0.02), so the setting was extended to all 250 tasks for base, A and A+B. The servers hung
repeatedly under sampled decoding and the extension stopped at the 174 tasks all three had finished
(the easier ones finish first, so the rates are above the full-set figures):

| Model | Sampled | Temperature 0, same 174 | vs base (gained / lost) |
|---|---|---|---|
| base | 51 (29.3%) | 57 | |
| A | 58 (33.3%) | 58 | 26 / 19 (p = 0.37) |
| A+B | 62 (35.6%) | 57 | 28 / 17 (p = 0.14) |

The 60-task lead did not hold. Single-attempt sampled scores on 60 tasks are too noisy to rank arms.

## Why B fails

B's commands are cut off mid-script. The model starts a long Python script, falls into repeating
near-identical lines, reaches the per-turn output limit, and the shell rejects the command because a
quote or heredoc was never closed. It then retries the same command until the turn limit.

| Over 250 tasks | base | A | B | B3 |
|---|---|---|---|---|
| "unexpected EOF" shell errors | 18 | 135 | 783 | 437 |
| Episodes that wrote an answer at all | 118 | 128 | 36 | 43 |
| Episodes ending in four identical commands | 93 | 98 | 121 | 124 |

The training rows are not malformed (3 of 2,600 `python3 -c` commands in v2 have unbalanced quotes).
What differs is the style: v2 rows run 7.2 commands against A's 3.7 and carry about twice the command
text, because our sweep required every answer to come with a `solution.py`. About 230 steps were also
a failed `cd /app`, provoked by our write tool reporting the script at `/app/solution.py`.

B3 removes the failed and redundant steps (5.4 commands a row) but keeps the script, and the result
is unchanged. The untested explanation that remains is the script itself: a 2B model trained for
~130 steps learns to start a long script and cannot finish one.

## What this does and does not show

- It shows that one pass of LoRA SFT on these datasets does not move L1 accuracy for this model on
  this harness, and that R does not reproduce an improvement here.
- It does not show SFT cannot help. Not tried: more than one pass, a higher rank, several sampled
  attempts per task, B without the script (inline work only), or rungs above L1.
- All figures are one attempt per task. Temperature-0 runs are deterministic per model; the sampled
  ones are not, and were not repeated.

## Operations

- Serving: vLLM 0.17.1 on ROCm, one merged model per engine. All engines stop generating at once,
  roughly every ten minutes, under sampled decoding with 18 to 40 concurrent requests; the
  temperature-0 pass ran 65 minutes without it. Partial results from the first hung attempt are in
  `data/runs/_void/`.
- A KV-cache budget does not exempt an engine from vLLM's startup free-memory check, so each engine
  now passes `--gpu-memory-utilization 0.3` (commit `fdb0989`).
- Cost: session 1 $14.87 (nothing usable), session 2 $16.09; **total $30.96** of the $100 credit.
  The instance was destroyed and the account verified empty at 14:57.

Run trees: `data/runs/amd2-{base,a,b,ab,r,b3}` (temperature 0) and `data/runs/amd2s-*` (sampled).
