# smol-ladder

Does RL training of small data-analysis agents transfer, and when a model fails, is it missing
skill or information? Built on [SmolDataEnvs](https://huggingface.co/datasets/FineEnvs/SmolDataEnvs)
and the [information ladder](https://evan-kim2028.github.io/evan_writings/writings/difficulty-is-an-information-gap/).

## Docs

- `docs/PLAN.md`: research questions, arms, milestones
- `docs/LADDER.md`: rung definitions and the Blackwell ordering
- `docs/LOCAL_MODELS.md`: what the released 2B models were trained under, and serving them locally
- `docs/TRAINING.md`: exporting data, LoRA SFT and GRPO
- `docs/audits/`: read-only audits of the results tree, with the findings later work overturned

## Modules

| module | what it does |
|---|---|
| `tasks.py`, `jtasks.py`, `jtasks_v2.py`, `synthetic.py` | task loaders. `jtasks_v2` builds the tagged jupyter-agent pools and writes `data/jtasks_v2.jsonl`; `--out data/jtasks_v3.jsonl` writes the corrected pool, with the overlap firewall firing on SmolDataEnvs `test`/`eval` and matching the bare dataset name instead of the full `owner/name` slug. `--v1-compatible` restores the shipped rule. The rest are the v1 and SmolDataEnvs sources. `tasks.read_tables` is the one table reader the schema dump and the synthetic tables share, so the two cannot drift apart; `tasks.read_shipped` is the uncapped reader the synthetic gold is computed from, so no row cap can sit between a gold and the file the agent is shipped |
| `pool.py` | which jupyter-agent pool a split name means. `--split jupyter-agent` is v1 (2,000 tasks, what every published jupyter-agent number was measured over); `--split jupyter-agent-v3` is `data/jtasks_v3.jsonl` filtered to its 4,217 ladder-grade tasks. It never writes a pool — `data/jtasks.jsonl` is an input to other checkouts' live sweeps — and naming the split is also what keeps the two sweeps' results and references apart |
| `ladder.py` | the rungs: prompts, the schema-dump control, and the hint blocks — a validated model hint when one is cached, the AST extraction otherwise |
| `run_ladder.py` | the runner: one trial per (task, rung, sample), in a jail, graded offline. Saves each trial's full assistant/tool conversation to `<task>/<rung>/transcript.json` by default — `--no-transcript` opts out — and removes the verifier's copy of the task tables once grading is done |
| `sandbox.py`, `or_agent.py`, `upstream.py` | the offline grading pass; the solver agent; the two upstream 2B protocols |
| `grade.py` | the SmolDataEnvs grader |
| `summarize.py` | per-rung pass rates with a bootstrap CI, and a first-passing-rung partition |
| `regrade.py` | re-scores stored predictions strict and prefix-normalised, offline. Writes nothing |
| `regrade_gold.py` | re-scores stored predictions against a *corrected* gold, writing a separate `data/runs/regrade_gold_<tag>.jsonl` with `reward` beside `reward_old`. Never writes into the results tree, and grades trials whose id the corrected corpus dropped against the superseded gold rather than dropping them |
| `gen_refs.py`, `gen_solutions.py` | generate and verify reference solutions |
| `gen_hints.py` | writes the plain-language L2/L3 hints from each verified reference, validates them, and caches them per task |
| `hint_report.py`, `hint_audit.py` | coverage, leak and cost numbers for the hints; a seeded side-by-side audit against the references |
| `refs_for_failures.py` | build references for the tasks that need one |
| `fetch_inputs.py`, `fetch_shards.py` | task tables and jupyter-agent shards |
| `reclaim.py` | reclaim disk from a results tree without deleting a result. Also collects the verifier's leftover `verify/input` copies of the task tables, guarded on the trial having a `result.json` |

`train/` is the post-training pipeline, a separate package so the eval harness needs none of it
and the existing tests run in seconds without torch: `traces.py` converts our verified trials into
upstream's SFT format, `format.py` owns that format and the held-out firewall, `rungs.py` emits
the L1–L4 curriculum in the `program` protocol, `export_sft.py` writes the rows, and `sft_lora.py`
/ `grpo.py` train. The training extras are optional and heavy: `uv run --extra train ...`. See
`docs/TRAINING.md`.

`data/` (gitignored) holds cached tables and results.

```sh
uv sync
uv run --with pytest pytest -m "not slow" -q tests   # quick loop: ~2 min
uv run --with pytest pytest -q tests                 # full suite: ~15 min
```

The four `slow` tests read the tables of a whole split from disk and are ~850 s of the suite on
their own — three of them profile all 250 `test` tasks to check the schema control's guarantee on
the population rather than a sample. They are not marked for being fragile, only for being wide,
and the full suite still runs them.

## Running a ladder

```sh
# K trials per (task, rung), every rung on every task so each pass rate has the same denominator
uv run python -m smol_ladder.run_ladder --split test --rungs L1,L1+schema,L2,L3,L4 \
  --samples 4 --no-climb --workers 20

# --run-tag gives the sweep its own results tree, data/runs/TAG/<split>/, plus a RUN.json
# recording the code, command line, model, protocol, rungs, samples, climb setting and the
# reference denominator at launch, so one ladder version's results are never silently pooled
# with another's.
uv run python -m smol_ladder.run_ladder --split test --rungs L1 --samples 2 \
  --no-climb --run-tag v2 --workers 20

uv run python -m smol_ladder.summarize --split test                 # legacy data/runs/test tree
uv run python -m smol_ladder.summarize --split test --run-tag v2    # that run's own tree
uv run python -m smol_ladder.regrade --split test                   # what the ANSWER: prefix costs
uv run python -m smol_ladder.regrade_gold --split synthetic          # what the old synthetic gold cost
```

Sample 0 is the existing `<task>/<rung>/result.json`, so a tree written before `--samples` was
added reads as one sample and a later `--samples 3` only adds `s1` and `s2`. A cached result is
reused, never re-graded, and never restamped with this run's commit.

Reference solutions, and a task pool:

```sh
# --attempts N gives every task without a verified reference up to N fresh tries. Only
# reward >= 1.0 counts as final, so a solution that runs cleanly and prints the wrong answer
# is retried rather than accepted; each attempt keeps its own attempt_<i>/ directory and is
# never overwritten.
uv run python -m smol_ladder.gen_refs --split test --attempts 3 --workers 8

uv run python -m smol_ladder.jtasks_v2                              # data/jtasks_v2.jsonl
uv run python -m smol_ladder.jtasks_v2 --out data/jtasks_v3.jsonl   # corrected pool
```

## Pointing at a different model server

The solver talks to any OpenAI-compatible endpoint. A loopback URL needs no API key.

```sh
export SMOL_LADDER_BASE_URL=http://127.0.0.1:8000/v1

uv run python -m smol_ladder.run_ladder --split test --rungs L1 --agent program \
  --model <the server's --served-model-name>
```

`--agent` picks the protocol: `program` is upstream's one-turn no-tools rollout (the one
`eval_pass1.py` scores), `bash` is the SmolDataEnvs-sft agent, `tools` is ours. A model has to
run under the protocol it was trained in or the number is about the protocol. Each result
records the model, the endpoint and the protocol beside the prompt hash, so a summary never
pools two different measurements. See `docs/LOCAL_MODELS.md`.
