# smol-ladder

Does RL training of small data-analysis agents transfer, and when a model fails, is it missing
skill or information? Built on [SmolDataEnvs](https://huggingface.co/datasets/FineEnvs/SmolDataEnvs)
and the [information ladder](https://evan-kim2028.github.io/evan_writings/writings/difficulty-is-an-information-gap/).

## Docs

- `docs/PLAN.md`: research questions, arms, milestones
- `docs/LADDER.md`: rung definitions and the Blackwell ordering
- `docs/LOCAL_MODELS.md`: what the released 2B models were trained under, and serving them locally
- `docs/audits/`: read-only audits of the results tree, with the findings later work overturned

## Modules

| module | what it does |
|---|---|
| `tasks.py`, `jtasks.py`, `jtasks_v2.py`, `synthetic.py` | task loaders. `jtasks_v2` builds the larger tagged jupyter-agent pool; the rest are the v1 and SmolDataEnvs sources |
| `ladder.py` | the rungs: prompts, the schema-dump control, and the AST extractors that read L2/L3 hints off a reference |
| `run_ladder.py` | the runner: one trial per (task, rung, sample), in a jail, graded offline |
| `sandbox.py`, `or_agent.py`, `upstream.py` | the offline grading pass; the solver agent; the two upstream 2B protocols |
| `grade.py` | the SmolDataEnvs grader |
| `summarize.py` | per-rung pass rates with a bootstrap CI, and a first-passing-rung partition |
| `regrade.py` | re-scores stored predictions strict and prefix-normalised, offline. Writes nothing |
| `gen_refs.py`, `gen_solutions.py` | generate and verify reference solutions |
| `refs_for_failures.py` | build references for the tasks that need one |
| `fetch_inputs.py`, `fetch_shards.py` | task tables and jupyter-agent shards |
| `reclaim.py` | reclaim disk from a results tree without deleting a result |

`data/` (gitignored) holds cached tables and results.

```sh
uv sync
uv run --with pytest pytest -q tests
```

## Running a ladder

```sh
# K trials per (task, rung), every rung on every task so each pass rate has the same denominator
uv run python -m smol_ladder.run_ladder --split test --rungs L1,L1+schema,L2,L3,L4 \
  --samples 4 --no-climb --workers 20

uv run python -m smol_ladder.summarize --split test        # writes data/runs/summary_test.json
uv run python -m smol_ladder.regrade --split test          # what the ANSWER: prefix costs
```

Sample 0 is the existing `<task>/<rung>/result.json`, so a tree written before `--samples` was
added reads as one sample and a later `--samples 3` only adds `s1` and `s2`. A cached result is
reused, never re-graded, and never restamped with this run's commit.

Reference solutions, and a task pool:

```sh
uv run python -m smol_ladder.gen_refs --split test --workers 8
uv run python -m smol_ladder.jtasks_v2                      # data/jtasks_v2.jsonl
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
