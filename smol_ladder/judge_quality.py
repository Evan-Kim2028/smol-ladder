"""A blind LLM rubric over sampled questions from both collections.

The overlap counts in `compare_sets` say how much of B is the same *provenance* as A. They say
nothing about whether either collection's questions are actually answerable, which is the axis the
project's own audit found to matter: the failures it read were mostly not reasoning failures but
questions whose gold is not recoverable from the shipped table. So this module measures that
directly, on a sample from each set, with the judge blind to which collection a question came
from.

**Blind, and shuffled together.** The two samples are interleaved with a seeded PRNG before any
judge call, so the judge sees one stream with no set labels, and the order carries no information
about set membership. A judge that is systematically harsher on the second half of a stream would
otherwise show up as a difference between A and B. The seed is reported, so the exact sample is
reproducible.

**Four criteria, each yes/no, plus a reason.** The criteria are the four properties a data-analysis
task needs to be trainable and gradeable:

- `answerable`      the tables alone determine the answer
- `unambiguous`     one specific computation is intended
- `determinate`     the answer's surface form is pinned by the question
- `requires_compute` the answer needs a computation over the tables, not a lookup

**The judge is the same model the ladder measures.** `stealth/space-bunny-alpha`, called through
`or_agent.endpoint`. So a question this judge calls unambiguous is unambiguous to the model under
test, and the judge cannot be accused of applying a standard the model does not share. It is a
weaker judge than a frontier model would be, and the report says so.

**Agreement is measured, not assumed.** A 30-question repeat runs the same questions a second
time, and raw per-criterion agreement is reported next to Cohen's kappa. A criterion whose kappa
is near zero is a criterion whose rate is not a property of the data; the report labels those.

    uv run python -m smol_ladder.judge_quality --n 150 --repeat 30 --seed 17
    uv run --with pytest pytest -q tests/test_judge_quality.py
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from smol_ladder.or_agent import MODEL, call_model, endpoint

CRITERIA = ("answerable", "unambiguous", "determinate", "requires_compute")

#: The judge is told what a task looks like and nothing about which collection it is from. Naming
#: the source, or the split, would let the judge compare instead of judge.
SYSTEM = """You audit data-analysis tasks for trainability. You are given one question that a
data-analysis agent will be asked, plus a description of the files it can see. Decide four things
about that question. Answer with JSON only, no prose, no code fence.

{
  "answerable": true|false,
  "unambiguous": true|false,
  "determinate": true|false,
  "requires_compute": true|false,
  "reason": "<one sentence>"
}

answerable: the supplied tables alone contain enough information to determine the answer. False
if the question needs a column, file, or external fact that is not in the files.

unambiguous: one specific computation is clearly intended. False if the question defers to an
unstated criterion ("based on correlation analysis"), an undefined threshold, or a method it never
pins, so two careful readers would compute different things.

determinate: the expected final answer has a determinate form. False if the surface form is not
pinned (is it "88.52" or "88.52%" or "0.8852"?) or the answer is inherently one draw from a
distribution the question does not fix, such as a fitted model's score.

requires_compute: answering needs a computation over the tables (filter, aggregate, join, fit).
False if the answer is a value you could read off by looking, or a matter of general knowledge.

Be strict. If a careful reader could reasonably get a different answer, unambiguous is false."""


#: `space-bunny-alpha` emits a hidden reasoning block before its answer, and 400 completion tokens
#: is not enough for it: measured on 2026-10-02, `finish_reason` came back `length` with
#: `content: None` on two thirds of calls, which would have silently dropped most of the sample and
#: biased the remainder toward the short questions the model can decide quickly. 1,200 leaves room
#: for the reasoning plus the four booleans, and a truncated answer is retried rather than scored.
MAX_TOKENS = 1200


@dataclass
class Judgement:
    task_id: str
    set_name: str
    answerable: bool | None
    unambiguous: bool | None
    determinate: bool | None
    requires_compute: bool | None
    reason: str = ""
    error: str = ""

    def as_dict(self) -> dict:
        return {"task_id": self.task_id, "set": self.set_name,
                "answerable": self.answerable, "unambiguous": self.unambiguous,
                "determinate": self.determinate,
                "requires_compute": self.requires_compute, "reason": self.reason,
                "error": self.error}


# --- parsing ---------------------------------------------------------------------------------

def parse_verdict(text: str) -> dict:
    """The judge's JSON, or every criterion `None` if it did not give any.

    Deliberately forgiving about the envelope and strict about the values: a fenced block, a
    leading sentence, or a missing key must not turn a judgement into a guess. A criterion the
    judge did not answer stays `None` and is dropped from that criterion's denominator rather
    than counted as a failure -- silently scoring an unanswered question as `false` would bias
    every rate downward by the model's formatting failures.
    """
    match = re.search(r"\{.*\}", text or "", re.S)
    if not match:
        return {c: None for c in CRITERIA} | {"reason": "", "error": "no JSON object"}
    try:
        payload = json.loads(match.group(0))
    except ValueError as error:
        return {c: None for c in CRITERIA} | {"reason": "", "error": f"invalid JSON: {error}"}
    out = {}
    for criterion in CRITERIA:
        value = payload.get(criterion)
        if isinstance(value, bool):
            out[criterion] = value
        elif isinstance(value, str) and value.strip().lower() in {"true", "false"}:
            out[criterion] = value.strip().lower() == "true"
        else:
            out[criterion] = None
    out["reason"] = str(payload.get("reason") or "")[:400]
    out["error"] = ""
    return out


# --- statistics ------------------------------------------------------------------------------

def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """A Wilson score interval, which is the right one here: the rates are near 0.8-0.95 with
    n = 150, where the normal approximation runs past 1 and the "rule of three" reads as though
    the interval were a point."""
    if total == 0:
        return 0.0, 0.0
    p = successes / total
    denominator = 1 + z * z / total
    centre = p + z * z / (2 * total)
    spread = z * ((p * (1 - p) / total + z * z / (4 * total * total)) ** 0.5)
    return ((centre - spread) / denominator, (centre + spread) / denominator)


def cohens_kappa(pairs: list[tuple[bool, bool]]) -> float:
    """Chance-corrected agreement between two runs on the same items.

    Reported next to raw agreement because raw agreement on a criterion with a 90% base rate is
    0.90 even for a judge that is flipping a coin on 20% of items, which is exactly the case where
    a rate should not be trusted.

    A degenerate run -- one that answered "yes" to all 100 items and "no" to none -- is the case
    that breaks the formula: chance agreement is then 1.0, the denominator is 0, and the ratio
    runs away to a large negative number that would be reported as an implausible kappa. That
    happens whenever a criterion is near-saturated and the judge never exercised the other
    response, so the degenerate case returns 0.0: no correction can be computed, and no
    correction is better than a fabricated one.
    """
    if not pairs:
        return 0.0
    n = len(pairs)
    observed = sum(1 for a, b in pairs if a == b) / n
    rate_a = sum(1 for a, _ in pairs if a) / n
    rate_b = sum(1 for _, b in pairs if b) / n
    expected = rate_a * rate_b + (1 - rate_a) * (1 - rate_b)
    if expected >= 1.0:
        # One run (or both) never used a response class, so chance already explains everything.
        return 0.0
    return (observed - expected) / (1 - expected)


def criterion_rate(judgements: list[Judgement], criterion: str) -> dict:
    answered = [j for j in judgements if getattr(j, criterion) is not None]
    successes = sum(1 for j in answered if getattr(j, criterion))
    low, high = wilson_interval(successes, len(answered))
    return {"criterion": criterion, "yes": successes, "no": len(answered) - successes,
            "n": len(answered), "rate": round(successes / len(answered), 4) if answered else 0.0,
            "ci95": [round(low, 4), round(high, 4)]}


def agreement(first: list[Judgement], second: list[Judgement], criterion: str) -> dict:
    by_id = {j.task_id: j for j in second}
    pairs = [(getattr(j, criterion), getattr(by_id[j.task_id], criterion))
             for j in first if j.task_id in by_id
             and getattr(j, criterion) is not None
             and getattr(by_id[j.task_id], criterion) is not None]
    if not pairs:
        return {"criterion": criterion, "n": 0, "raw_agreement": 0.0, "cohens_kappa": 0.0}
    raw = sum(1 for a, b in pairs if a == b) / len(pairs)
    return {"criterion": criterion, "n": len(pairs),
            "raw_agreement": round(raw, 4), "cohens_kappa": round(cohens_kappa(pairs), 4)}


# --- sampling ------------------------------------------------------------------------------

def sample_blind(sets: dict[str, list[dict]], n: int, seed: int,
                 shuffle: bool = True) -> list[tuple[str, dict]]:
    """`n` rows per set, interleaved into one stream with no set label exposed to the judge.

    A seeded `random.Random` rather than `random.sample`, so the draw does not move when the rest
    of the module imports something else, and `shuffle=False` gives the caller the grouped order
    the repeat pass needs to select the same items.
    """
    rng = random.Random(seed)
    picks: list[tuple[str, dict]] = []
    for set_name, rows in sets.items():
        if not rows:
            continue
        chosen = rows if len(rows) <= n else rng.sample(sorted(rows, key=lambda r: r["task_id"]), n)
        picks.extend((set_name, row) for row in chosen)
    if shuffle:
        rng.shuffle(picks)
    return picks


def files_description(row: dict, limit: int = 6) -> str:
    files = row.get("files") or []
    shown = ", ".join(files[:limit])
    if len(files) > limit:
        shown += f", ... ({len(files)} files total)"
    return shown or "no files listed"


def user_prompt(row: dict) -> str:
    return (f"Files available to the agent:\n{files_description(row)}\n\n"
            f"Question:\n{row['question']}")


# --- judging ---------------------------------------------------------------------------------

_JUDGE_LOCK = threading.Lock()
_CALLS: Counter = Counter()


def judge_one(row: dict, set_name: str, model: str = MODEL, ep=None,
              attempts: int = 3) -> Judgement:
    """One question through the judge. Never raises: a failed call is a row with an `error`, so a
    transient OpenRouter 429 costs one sample rather than the whole run.

    A truncated answer is retried rather than recorded as a failure. `space-bunny-alpha` spends
    most of its budget on a hidden reasoning block and returns `content: None` with
    `finish_reason: "length"` when it runs out, which looks exactly like an unparseable answer but
    is really a too-small cap -- and left unretried it silently drops the questions it spent longest
    thinking about, which are the long and hard ones. The retry doubles the cap, so a genuinely
    unanswerable verdict still costs one call and a verbose one costs two.
    """
    parsed, last = None, ""
    for attempt in range(attempts):
        cap = MAX_TOKENS * (2 ** attempt)
        try:
            completion = call_model(
                [{"role": "system", "content": SYSTEM},
                 {"role": "user", "content": user_prompt(row)}],
                model, None, ep, max_tokens=cap)
            choice = completion["choices"][0]
            text = choice["message"].get("content") or ""
            parsed = parse_verdict(text)
            truncated = choice.get("finish_reason") == "length"
            last = str(choice.get("finish_reason"))
            if not parsed["error"] or not truncated:
                break
        except Exception as error:  # noqa: BLE001 - one lost sample must not stop the sweep
            parsed = {c: None for c in CRITERIA} | {"reason": "", "error": str(error)[:200]}
            last = str(error)[:80]
    with _JUDGE_LOCK:
        _CALLS["calls"] += 1
        if parsed and parsed.get("error"):
            _CALLS["errors"] += 1
    return Judgement(task_id=row["task_id"], set_name=set_name, reason=parsed["reason"],
                     error=parsed.get("error", ""), **{c: parsed[c] for c in CRITERIA})


def judge(picks: list[tuple[str, dict]], model: str = MODEL, workers: int = 8,
          ep=None) -> list[Judgement]:
    """Judge a stream of `(set_name, row)` pairs. The stream is already blind, so this does not
    re-shuffle: the caller's order is the order the judge saw."""
    if not picks:
        return []
    ep = ep or endpoint()
    with ThreadPoolExecutor(workers) as pool:
        return list(pool.map(lambda pick: judge_one(pick[1], pick[0], model, ep), picks))


# --- the whole measurement ---------------------------------------------------------------------

def run(sets: dict[str, list[dict]], n: int = 150, repeat: int = 30, seed: int = 17,
        model: str = MODEL, workers: int = 8) -> dict:
    """Blind rubric over `n` rows per set, plus a `repeat`-row second run for agreement.

    The repeat draws from the *same* first sample, so it measures the judge, not the sampling: a
    different draw would confound "the judge disagreed" with "it was a different question".
    """
    blind = sample_blind(sets, n, seed, shuffle=True)
    judgements = judge(blind, model, workers)
    by_set: dict[str, list[Judgement]] = defaultdict(list)
    for j in judgements:
        by_set[j.set_name].append(j)

    repeats: list[Judgement] = []
    if repeat:
        rng = random.Random(seed + 1)
        ids = sorted({j.task_id for j in judgements})
        chosen = set(rng.sample(ids, min(repeat, len(ids))))
        second = judge([(j.set_name, row) for j in judgements if j.task_id in chosen
                        for row in [next(s for s in blind if s[1]["task_id"] == j.task_id)[1]]],
                       model, workers)
        repeats = second

    report = {
        "model": model,
        "seed": seed,
        "max_tokens": MAX_TOKENS,
        "sampled_per_set": {name: len(rows) for name, rows in by_set.items()},
        "sampled_total": len(judgements),
        "criteria": list(CRITERIA),
        "errors": sum(1 for j in judgements if j.error),
        # Reported per set and in total, because an unparsed question is dropped from every
        # criterion's denominator. If one set answers 150/150 and the other 90/150, the comparison
        # is between two different populations and the rates are not comparable.
        "answered_per_set": {name: sum(1 for j in rows
                                       if getattr(j, CRITERIA[0]) is not None)
                             for name, rows in by_set.items()},
        "unparsed": sum(1 for j in judgements if not j.error
                        and getattr(j, CRITERIA[0]) is None),
        "repeat_n": len(repeats),
        "repeat_answered": sum(1 for j in repeats if getattr(j, CRITERIA[0]) is not None),
        "calls": dict(_CALLS),
        "by_set": {},
        "agreement": [agreement(judgements, repeats, c) for c in CRITERIA],
    }
    for name, rows in by_set.items():
        report["by_set"][name] = {
            "n": len(rows),
            "criteria": {c: criterion_rate(rows, c) for c in CRITERIA},
        }
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=150, help="questions per set")
    ap.add_argument("--repeat", type=int, default=30, help="questions re-judged for agreement")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="data/compare/judge.json")
    args = ap.parse_args()

    from smol_ladder.compare_sets import load_pool_rows, tag_pool
    from smol_ladder.tasks import load_split
    from smol_ladder.pool import V3_SPLIT, ladder_grade

    pool_rows = load_pool_rows()
    ladder = ladder_grade(pool_rows)
    sets = {
        "A_sde_train": [{"task_id": r["task_id"], "question": r["question"],
                         "files": r.get("files") or []} for r in load_split("train")],
        f"B_{V3_SPLIT}_ladder_grade": [{"task_id": r["task_id"], "question": r["question"],
                                         "files": r.get("files") or []} for r in ladder],
    }
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise SystemExit("OPENROUTER_API_KEY is not set")
    report = run(sets, args.n, args.repeat, args.seed, args.model, args.workers)
    from pathlib import Path

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()