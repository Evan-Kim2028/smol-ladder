# SFT and ladder results on Qwen3.5-2B (2026-10-02)

The checkpoint agreed before any GRPO run. Three models on the 250 held-out SmolDataEnvs test
tasks, then the information ladder on two of them. Every comparison is paired on the common task
set (`smol_ladder.report_runs`); intervals are 95% bootstrap over tasks; "vs base" is the number of
tasks gained and lost against the base model with a sign test.

## The models

LoRA r=16, alpha=32, bf16, max length 8,192, effective batch 8, one pass over the data.

| Model | Trained on | Rows | Steps | Adapter |
|---|---|---|---|---|
| base | nothing (Qwen3.5-2B as released) | | | |
| A | SmolDataEnvs-sft, upstream's verified trajectories, through the firewall (`sft_upstream`) | 4,439 | 555 | `evandekim/smol-ladder-sft-a-s2` |
| B | our trajectories: space-bunny solving jupyter-agent tasks inside the same shell harness the student runs in, kept when the grader passed (`ja4_sft`) | 1,897 | 238 | seed 1 `…-sft-b4-s2`, seed 2 `…-sft-b5-s2` |

A's validation loss flattened (0.489 → 0.378). B has no validation split; its training loss sat
near 0.5 for the last third of the run. B's rows look like A's: median command 150 characters
against A's 126, the same 90th percentile (~540), half shell and half multi-line Python, no
`solution.py`, every row replays byte-identically through the harness.

## L1 (question only), temperature 0, 250 tasks, one attempt

| Model | Pass | 95% CI | Wrote any answer | Ended in a repeat loop | vs base |
|---|---|---|---|---|---|
| base | 24.0% | 18.8 – 29.2 | 47% | 37% | |
| A | 25.6% | 20.4 – 31.2 | 51% | 39% | +28 / −24, p = 0.68 |
| B seed 1 | 10.8% | 7.2 – 14.8 | 23% | 42% | +10 / −43, p < 0.001 |
| B seed 2 | 13.6% | 9.6 – 18.0 | 24% | 39% | +12 / −38, p < 0.001 |

Without the 27 tasks whose notebook overlaps A's training data: base 25.6%, A 26.9%, same picture.
By tier (base / A / B seed 1): easy 64 / 64 / 33%, medium 29 / 27 / 10%, hard 5 / 11 / 4%.

## L1, sampled: 4 attempts per task, 60 tasks stratified by tier

Temperature 0.7, top-p 0.8, top-k 20. Score = mean over tasks of each task's pass rate.

| Model | Pass | 95% CI | Wrote any answer | Ended in a repeat loop | vs base |
|---|---|---|---|---|---|
| base | 20.0% | 12.5 – 27.9 | 43% | 18% | |
| A | 22.5% | 15.0 – 31.7 | 45% | 39% | +14 / −11, p = 0.69 |
| B seed 1 | 18.8% | 11.7 – 26.2 | 37% | 14% | +12 / −15, p = 0.70 |

## Reading

- **A does not change L1 accuracy** under either decoding mode. Its one consistent effect is
  behavioural: it writes an answer a little more often (51% vs 47%).
- **B is harmful at greedy decoding and neutral when sampled.** Both seeds land 10 to 13 points
  below base at temperature 0, so that is the data and not one run. With sampling the gap closes
  entirely and B loops least of the three. At temperature 0, B's generations run away: median
  command 686 characters against 150 in its training rows, 64% of lines are `print`, a quarter of
  its multi-line commands repeat themselves, 1,050 commands over 250 tasks were cut off with an
  unclosed quote. The training rows have none of this (0% repetitive commands); it is what greedy
  decoding does to a 2B model trained on a somewhat print-heavy exploratory style (43% `print`
  lines in B's rows against 30% in A's). An aside, not a headline.
- **The upstream released adapter** (`AdithyaSK/smoldataenvs-sft-2b-v0`, 100 steps) scored 19.2%
  on the same 250 tasks at temperature 0 (+19 / −31 vs base, p = 0.12).

## The ladder: base and A, temperature 0, one attempt

L1+control appends seven behaviour rules (no task information) to the L1 prompt; L1+schema is the
tables' schema. L2 adds the files, columns and filters the reference used; L3 adds the method in
words; L4 adds the verified reference program (without its final print). L2–L4 exist for the 213
tasks with a verified reference.

| Rung | Tasks | base | A | A vs base | base: wrote any answer | base: loop |
|---|---|---|---|---|---|---|
| L1 | 250 | 24.0% [18.8, 29.2] | 25.6% [20.4, 31.2] | +28 / −24, p = 0.68 | 47% | 37% |
| L1+control | 250 | 16.8% [12.4, 21.6] | 24.8% [20.0, 30.4] | +30 / −10, p = 0.002 | 37% | 47% |
| L2 | 213 | 23.5% [17.8, 29.6] | 31.0% [24.9, 37.6] | +32 / −16, p = 0.03 | 49% | 31% |
| L3 | 213 | 34.7% [28.6, 41.3] | 36.1% [29.6, 42.7] | +31 / −28, p = 0.79 | 49% | 25% |
| L4 | 213 | 68.5% [62.0, 75.1] | 72.8% [66.7, 78.9] | +31 / −22, p = 0.27 | 84% | 8% |

By tier at L4 (base): easy 73%, medium 69%, hard 65%: with the program in hand, difficulty
nearly stops mattering.

- **Information lifts the base model, and more lifts it more**: +11 points for the method in
  words, +44 for the program. Execution is not the wall; knowing what to compute is.
- **Telling it how to behave hurts.** The control rules cost base 7 points and raised its loop
  rate from 37% to 47%; A was unaffected. Control is not promptable at this size; it has to be
  trained.
- **A uses information slightly better than base** at every rung, but only the L1+control and L2
  differences clear the noise, and both are single-attempt.
- **The L4 ceiling is a control ceiling.** Given working code the model still fails a third of
  tasks, by not running it cleanly or not committing to the answer, and at L4 it stops by itself
  in 84% of episodes against 47% at L1. Given a plan, it behaves better. That ceiling should move
  with training that fixes control, which is the case for RL and the reason GRPO is pinned, not
  dropped.

## How the base model fails at L1 (250 episodes, temperature 0)

| Outcome | Episodes |
|---|---|
| correct | 60 |
| never wrote an answer | 132 |
| – repeating a command that worked | 59 |
| – repeating a command that errored | 26 |
| – 16 turns, no loop | 25 |
| – ran out of context (printed whole tables) | 21 |
| – stopped without answering | 1 |
| wrote a wrong answer | 58 |
| – wrong text answer (often the wrong kind: a number for a column name, "Not Applicable") | 30 |
| – wrong number (2 within 5% of the reference) | 28 |

In about 36 of the 132 unanswered episodes, one of the model's own commands had already printed
the correct value (crude string match). Easy 21/33 correct, medium 34/118, hard 5/99.

## Pre-registered reading rules (written 19:10, before the robustness runs) and what happened

1. **Second seed for B**: the claim "B is below base" only if both seeds are below base on the
   paired count. Outcome: seed 1 +10/−43, seed 2 +12/−38. The claim stands, for temperature 0.
2. **Sampled repeats**: 60 gate tasks x 4 attempts, three models, paired per task, a difference
   counts only if its 95% interval excludes 0. Outcome: no difference excludes 0 (A +2.5
   [−5.4, +11.7]; B −1.2 [−8.8, +6.7]).
3. Harness-error attempts were retried; after retries every set is complete except B seed 2 at
   L1 (249/250; one episode never returned) and sampled B (237/240).

## What this does not show

One recipe (one pass, rank 16). Not tried: more passes, a higher rank, A+B mixes with the native
B, rungs above L1 for B, several seeds for A, SFT as a warm start for GRPO (the role upstream gives
it; their pages publish no SFT-alone number). All ladder figures are single-attempt.

## Operations and cost

- Serving: vLLM 0.17.1 on ROCm, one merged model per engine (LoRA mode does not work for this
  model). Engines freeze together under sampled decoding at 18–40 concurrent requests, roughly
  every ten minutes; 12 concurrent ran for hours. Temperature-0 runs never froze. Each engine
  needs `--gpu-memory-utilization 0.3` beside its KV budget or the third one fails to start.
- A spot instance was pre-empted mid-run (powered off by the provider; the dead-man destroyed it);
  one tunnel/server outage produced error episodes that were retried. Watch results by
  `stop_reason`, not by count: an unreachable server fills the tree with `error` records fast.
- Cost: four sessions, $44.02 by the ledger ($14.87 of it a first session that produced nothing
  usable; about $4 to hangs, the pre-emption and the outage). The provider billed **$46.65** for the
  same usage (read 2026-10-03), $2.63 above the ledger; its figure is the one that counts against the
  $100 credit, which leaves about $53.

Run trees: `data/runs/amd2-{base,a}` (L1 temperature 0), `data/runs/amd3-{base,a}` (the ladder),
`data/runs/amd3-b4`, `amd3-b5` (B, two seeds), `data/runs/amd3s-*` (sampled repeats). Teacher
trajectories: `data/runs/ja4` (2,111 tasks; 1,897 exported) and `ja4r` (the 1,769 tasks the
teacher had failed before: 136 graded correct in the shell harness, not exported).
