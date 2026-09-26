# smol_data_transfer (working name)

Does RL training of small data-analysis agents transfer, and when a model fails, is it missing
skill or information? Built on [SmolDataEnvs](https://huggingface.co/datasets/FineEnvs/SmolDataEnvs)
and the [information ladder](https://evan-kim2028.github.io/evan_writings/writings/difficulty-is-an-information-gap/).

- `docs/PLAN.md`: research questions, arms, milestones
- `docs/LADDER.md`: rung definitions and the Blackwell ordering
- `sdt/`: task loader, offline sandbox, grader, reference-solution generator
- `data/` (gitignored): cached tables and generated solutions

```sh
uv sync
uv run --with pytest pytest -q tests
uv run python -m sdt.gen_solutions --split test --workers 4
```
