# What a 2B data-analysis agent is missing, measured one hint at a time

*Draft, 2026-10-02. Numbers from `docs/SFT_RESULTS.md`; every figure is paired on common tasks
with a bootstrap interval, and the reading rules were written down before the last runs.*

## The question

SmolDataEnvs is a set of 5,000 data-analysis tasks: a question about one or more CSV files, a
deterministic grader, and a held-out test split of 250 harder tasks. The intended use is RL on a
small model, with supervised fine-tuning on 4,677 verified agent trajectories as the warm start.

We wanted to know something more basic first. When a 2B model fails one of these tasks, what is it
missing? Does it not know *what* to compute, or does it know and fail to *carry it out*? The answer
decides what kind of training can help. Imitation teaches a style of working; a hint curriculum
teaches what to compute; RL rewards finishing. They are not interchangeable.

So we did two things to Qwen3.5-2B. We fine-tuned it on trajectories (the obvious move), and we
built an **information ladder**: the same question asked with progressively more of the solution
revealed, so that the gap between rungs says what the model lacked.

- **L1**: the question and the file names.
- **L2**: plus the files, columns and filters the reference solution used.
- **L3**: plus the method, in plain words (which split, which model, which aggregation).
- **L4**: plus the verified reference program itself, without its final print.

Each rung is a prefix of the next, and the model still has to do all the work: run the commands,
read the output, write the answer to a file. L4 is the diagnostic ceiling: if the model fails with
working code in front of it, the problem is not knowledge.

## The setup, briefly

The agent has one tool, `bash`, in a sandbox with the data under `/home/user/input`, up to 16
turns, and must finish by writing only the answer to `/workdir/answer.txt`. The prompt is
byte-identical to the one in the SFT trajectories, checked by replaying every training row
through the harness. Evaluation is at temperature 0 unless stated; a 60-task stratified subset
was also run with 4 sampled attempts per task to see how much the greedy numbers depend on the
decoding.

Two SFT arms, LoRA r=16, one pass each:

- **A**: upstream's 4,439 verified trajectories (after removing four that leak test questions).
- **B**: 1,897 trajectories we collected ourselves, by running a strong model through the *same*
  shell harness on jupyter-agent tasks and keeping the episodes the grader passed. B's rows look
  like A's: about five commands, half shell and half Python, the same command lengths.

All of this ran on one AMD MI350X spot instance for a total of $44, including a first session
that produced nothing usable.

## Result 1: fine-tuning on trajectories does not move accuracy

| L1, 250 tasks, temperature 0 | Pass | 95% CI | vs base |
|---|---|---|---|
| base | 24.0% | 18.8 – 29.2 | |
| A | 25.6% | 20.4 – 31.2 | +28 tasks / −24, p = 0.68 |
| B (seed 1 / seed 2) | 10.8% / 13.6% | | −43 / +10; −38 / +12, p < 0.001 |

A is indistinguishable from the untrained model. Its validation loss had flattened (0.49 → 0.38),
so this is not an under-trained run; it learned to imitate the trajectories and that did not
translate into more correct answers. Upstream's own released SFT adapter scores 19.2% on the same
tasks, also no better than base.

B is worse than base, on both seeds. Not because the data is wrong: every row is a verified
solution, and in style the rows are as short as A's. The 2B model trained on them degenerates at
greedy decoding into long runs of `print` lines that hit the output limit and get cut off with an
unclosed quote (1,050 such commands over 250 tasks), then loop. With sampling the effect
disappears entirely:

| L1, 60 tasks x 4 sampled attempts | Pass | 95% CI | vs base |
|---|---|---|---|
| base | 20.0% | 12.5 – 27.9 | |
| A | 22.5% | 15.0 – 31.7 | p = 0.69 |
| B | 18.8% | 11.7 – 26.2 | p = 0.70 |

So the three models are equivalent when sampled, and B is fragile at temperature 0. That fragility
is a real cost (greedy is the cheap, deterministic way to evaluate), but it is a decoding
interaction, not evidence that the trajectories taught anything wrong. We leave it as an aside.

## Result 2: the ladder says the model mostly does not know what to compute

| Rung | base | A |
|---|---|---|
| L1: question | 24.0% | 25.6% |
| L2: + files, columns, filters | 23.5% | 31.0% |
| L3: + method in words | 34.7% | 36.1% |
| L4: + reference program | 68.5% | 72.8% |

(213 tasks with a verified reference for L2–L4; intervals are about ±6 points. Task by task against
L1, the base model gained 19 and lost 28 at L2 (churn), gained 34 and lost 19 at L3 (p = 0.05), and
gained 95 and lost 8 at L4. The large step is beyond doubt; the method step is real but its size is
uncertain by about its own magnitude at one attempt.)

Three things stand out.

**Columns alone are indistinguishable from nothing for the base model.** It already finds the right columns; L2 is flat.
The method in words is worth roughly 5 to 15 points, and the program is worth about 44. The failures are concentrated
in *deciding what to do*, then in *writing the code for it*, and only last in running it.

**With the program in hand, difficulty stops mattering.** At L4 the base model passes 73% of
easy, 69% of medium and 65% of hard tasks. At L1 the same model passes 64%, 29% and 5%. "Hard" on
this benchmark means "hard to know what to compute", not "hard to execute".

**A third of tasks fail even with the code.** Of the 67 L4 failures, 32 are wrong answers and 25 of
those are the wrong *kind* of answer (a number where a label was asked for, or "Not Applicable");
16 are loops; 11 ran the context out by printing whole tables. None of that is knowledge. It is
control: reading the question's answer format, not re-printing the data, stopping once the value
is on screen.

### Where the failures go as information is added

The same classifier over the base model's failed episodes at each rung (213 tasks with a reference
for L3 and L4):

| How the failed episodes ended | L1 (190 failures) | L3 (139) | L4 (67) |
|---|---|---|---|
| wrong answer: a different number | 18 | 8 | 2 |
| wrong answer: different text | 10 | 7 | 4 |
| wrong answer: wrong *kind* (a number for a label, or "Not Applicable") | 28 | 14 | 25 |
| no answer: repeat loop | 85 | 49 | 16 |
| no answer: wandered 16 turns | 26 | 29 | 8 |
| no answer: ran out of context | 21 | 31 | 11 |
| unanswered, but the correct value was already printed | 36 of 132 | 39 of 109 | 13 of 35 |

**L3 fixes knowledge and leaves control alone.** The method hint halves the wrong answers
("different number" falls from 18 to 8), but 109 of its 139 failures still end with no answer, about
the same as at L1, and in 39 of them the right value had been printed. Context exhaustion rises,
because a model that knows the method writes longer scripts and prints more.

**L4 leaves only control.** With the program in hand the computation is essentially right (2 wrong
numbers in 213 tasks). What remains is reporting the wrong kind of value after running the code, and
not committing at all.

## Result 3: control is the wall, and it is not promptable

We classified every L1 episode of the base model by how it ended:

| Outcome | Episodes |
|---|---|
| correct | 60 |
| never wrote an answer | 132 |
| – repeating one command, which worked | 59 |
| – repeating one command, which errored | 26 |
| – 16 turns of exploring | 25 |
| – filled the context with a table | 21 |
| wrote a wrong answer | 58 |

More than half of all episodes end with no answer at all, and in about 36 of them the correct value
had already been printed by one of the model's own commands. The model is not short of ability to
compute; it is short of the habit of committing.

The obvious fix is to tell it. We appended seven behaviour rules to the L1 prompt (stop repeating
commands, never print whole tables, write the answer once you have it). The base model got
**worse**: 24.0% → 16.8%, with the loop rate rising from 37% to 47%. The fine-tuned A was
unaffected (25.6% → 24.8%). Control, at this size, is not a thing you can ask for.

That is also what the L4 ceiling is made of. Given a plan, the base model stops by itself in 84% of
episodes against 47% at L1; the behaviour improves when the uncertainty is removed, which says the
behaviour is trainable. The ceiling should move with training that rewards finishing, which is
exactly what RL does and imitation does not.

## What we think it means

- **Imitation of trajectories is the wrong tool for this gap.** The 2B model already works in the
  right style; it fails on decisions and on commitment, and one pass of SFT changed neither.
- **The hint curriculum has a case.** L3 raises the pass rate from 24% to 35% while leaving all the
  work to the model. For GRPO, that is the difference between reward groups that are mostly all-zero
  and groups with signal, on exactly the behaviour (finish, commit) that limits every rung.
- **Measure before you train.** The ladder cost a few dollars of GPU and told us more about what to
  train than the SFT runs did. It also gave us a ceiling to aim at: when a trained model's L1
  approaches its own L4, the control problem is solved and the remaining gap is knowledge.

## On robustness, since small benchmarks lie

Three things bit us and are worth stating as rules.

1. **Subsets drift.** Easy tasks finish first, so every interim number flattered the trained
   models. Compare only on completed common sets, paired per task.
2. **A 60-task result is a coin.** A beat base 20 to 9 on a 60-task sampled run (p = 0.02); on
   174 tasks the gap was 7 points and insignificant; on 240 attempts it was 2.5 points.
3. **Greedy determinism is not stability.** Of the tasks the base model solved at L1, only 68% were
   solved again at L3 with a strictly more informative prompt. One attempt per task at temperature
   0 is a measurement of the model *and* of the prompt's exact wording; several sampled attempts
   per task are the honest unit.

We wrote the reading rules for the last runs before running them (two seeds for B; a sampled
difference counts only if its interval excludes zero), and we report what happened against them.

## What's next

GRPO with the L3 rung as a curriculum, withdrawn as per-task pass rates rise, starting from the
base model. The thing the ladder says is missing is a reward for finishing, and that is what RL
supplies.

*Code, prompts, run trees and the per-episode classifier are in the repository; the two trained
adapters and our native trajectories are on the Hub.*
