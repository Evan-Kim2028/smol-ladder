"""Hint-rung variants of our tasks, for the curriculum arm.

Each rung is the same task with more information: L1 is the question and the file names, L2 adds
which files/columns/filters the computation uses, L3 adds the method, L4 adds the reference code
with its final print removed. The rung text comes from `smol_ladder.ladder.prompt_for` verbatim,
so a rung exported here is byte-identical to the rung the eval sweep would have sent -- which is
the entire point. A curriculum built on a reworded rung is a curriculum over a different prompt.

Unlike the other two sources this one emits a **single-turn program trajectory** in upstream's
GRPO/eval protocol (one fenced program, no tools) rather than a bash agent, because a rung is
prompt-level information for a one-shot program and a bash trajectory would need invented
exploration turns to carry it. That is a deliberate format split and it is documented in
`docs/TRAINING.md`: rung data trains the `program` protocol, `traces` trains the `bash` protocol,
and mixing them in one export would make the resulting model a third protocol that is neither.

The firewall here is stricter than for the other sources, and not out of caution. **All 1,830
synthetic tasks are built on tables that appear in a held-out split** -- verified, not guessed --
so on the synthetic split the `bucket_prefix` firewall removes the entire split and leaves nothing.
That is the firewall working, and it is why `collect_rungs` reports the count it refused instead
of quietly returning an empty training set.
"""

from __future__ import annotations

import json

from smol_ladder.ladder import hint_source, prompt_for, read_source
from smol_ladder.tasks import DATA, load_split
from smol_ladder.upstream import PROGRAM_SYSTEM

from train.format import heldout_keys, is_heldout

# Where the rungs for a split come from. jupyter-agent and synthetic are the only splits with
# verified references on this machine (`solutions/train` is empty), so L2+ is unavailable for
# everything else until gen_solutions is run against them.
SPLITS = ("jupyter-agent", "synthetic", "train")


def program_turns(prompt: str, code: str, prediction: str) -> list[dict]:
    """One assistant turn holding a fenced program, in the shape the `program` protocol expects.

    `bash_row` is deliberately not used: this is the no-tools protocol. It is built here rather
    than reusing upstream's `extract_code` in reverse because the round trip is lossy (the extractor
    takes the *last* fence, and a trajectory may legitimately contain several).
    """
    body = code.rstrip("\n")
    return [{"content": f"```python\n{body}\n```\n\nThis prints {prediction}."}]


def rung_row(question: str, files: list[str], prompt: str, code: str, prediction: str) -> dict:
    """A rung variant as one `messages` + no-tools row.

    `tools` is `[]`, not absent: an empty list is what says "this protocol has no tools", and the
    chat template's `{% if tools %}` branch is false either way. Upstream's GRPO protocol sends no
    `tools` key at all, and `smol_ladder.or_agent.call_model` treats `tools=None` as that exact
    request -- so the exported row carries `[]` and the trainer drops the key, in
    `sft_lora.py --protocol program`.
    """
    return {"messages": [{"role": "system", "content": PROGRAM_SYSTEM},
                         {"role": "user", "content": prompt},
                         *program_turns(prompt, code, prediction)],
            "tools": [], "rung_prompt": prompt}


def rows_for_split(split: str, rung: str) -> list[dict]:
    """Every task of a split that can run this rung, as a row. Empty when the rung needs a
    reference the split does not have."""
    if split == "jupyter-agent":
        from smol_ladder.jtasks import load_rows

        tasks = load_rows()
    elif split == "synthetic":
        from smol_ladder.jtasks import load_synthetic

        tasks = load_synthetic()
    else:
        tasks = load_split(split)
    out = []
    for row in tasks:
        if rung in {"L2", "L3", "L4"} and read_source(row, split) is None:
            continue
        prompt = prompt_for(row, split, rung)
        source = read_source(row, split)
        if source is None:
            # L1 needs no reference; for it the "program" is the question's own task, and there is
            # no verified program to teach. An L1 variant would have to invent one.
            continue
        out.append((row, prompt, source))
    return out


def collect_rungs(rung: str, data=None, limit: int | None = None,
                  keys: dict[str, set[str]] | None = None) -> tuple[list[dict], dict]:
    """Every trainable (task, rung) pair, firewall applied.

    `data` is accepted for a signature-compatible call site and deliberately unused: the rungs live
    in `smol_ladder.ladder`, which resolves its own inputs, and passing a different root would make
    the exported text disagree with the text an eval sweep sends.
    """
    if keys is None:
        keys = heldout_keys({"test": load_split("test"), "eval": load_split("eval")})
    dropped: dict[str, int] = {}
    rows: list[dict] = []
    for split in SPLITS:
        for row, prompt, source in rows_for_split(split, rung):
            column = is_heldout(row, keys)
            if column:
                dropped[column] = dropped.get(column, 0) + 1
                continue
            prediction = _prediction(row, split)
            if not prediction:
                continue
            entry = rung_row(row["question"], row.get("files") or [], prompt, source, prediction)
            entry["task_id"] = row["task_id"]
            entry["rung"] = rung
            entry["hint_source"] = hint_source(row, split, rung)
            rows.append(entry)
            if limit and len(rows) >= limit:
                return rows, dropped
    return rows, dropped


def _prediction(row: dict, split: str) -> str:
    """The value this task's verified reference prints, read off the solution's own result.

    Read from disk rather than recomputed: the value in `result.json` came from a sealed offline
    run of that exact program, which is the same evidence the grader gets, and recomputing here
    would mean running code during an export.
    """
    directory = DATA / "solutions" / split / row["task_id"]
    path = directory / "result.json"
    if not path.exists():
        return ""
    try:
        return json.loads(path.read_text()).get("prediction") or ""
    except (OSError, ValueError):
        return ""


def counts(rungs=("L1", "L2", "L3", "L4")) -> dict:
    """What is available right now, per rung, per source. The number that decides whether the
    curriculum arm is even buildable today."""
    keys = heldout_keys({"test": load_split("test"), "eval": load_split("eval")})
    out: dict[str, dict[str, int]] = {}
    for rung in rungs:
        out[rung] = {}
        for split in SPLITS:
            kept = dropped = 0
            for row, _prompt, _source in rows_for_split(split, rung):
                if is_heldout(row, keys):
                    dropped += 1
                elif _prediction(row, split):
                    kept += 1
            out[rung][split] = kept
            if dropped:
                out[rung][f"{split}_dropped_by_firewall"] = dropped
    return out


def main() -> None:
    import argparse
    import json as _json

    ap = argparse.ArgumentParser()
    ap.add_argument("--counts", action="store_true", help="print availability and exit")
    args = ap.parse_args()
    if args.counts:
        print(_json.dumps(counts(), indent=1))


if __name__ == "__main__":
    main()