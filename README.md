# smol-ladder

Does RL training of small data-analysis agents transfer, and when a model fails, is it missing
skill or information? Built on [SmolDataEnvs](https://huggingface.co/datasets/FineEnvs/SmolDataEnvs)
and the [information ladder](https://evan-kim2028.github.io/evan_writings/writings/difficulty-is-an-information-gap/).

- `docs/PLAN.md`: research questions, arms, milestones
- `docs/LADDER.md`: rung definitions and the Blackwell ordering
- `docs/LOCAL_MODELS.md`: what the released 2B models were trained under, and serving them locally
- `smol_ladder/`: task loader, offline sandbox, grader, reference-solution generator
- `data/` (gitignored): cached tables and generated solutions

```sh
uv sync
uv run --with pytest pytest -q tests
uv run python -m smol_ladder.gen_solutions --split test --workers 4
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
run under the protocol it was trained in or the number is about the protocol. See
`docs/LOCAL_MODELS.md`.
