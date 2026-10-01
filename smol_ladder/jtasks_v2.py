"""A larger, tagged jupyter-agent pool: data/jtasks_v2.jsonl.

The v1 extract took 8 of 103 shards and stopped at 2,000 tasks, and it shipped rows with no
properties beyond the question, the answer and the grader mode. Three things are wrong with
that, and this file fixes all three.

**It is small.** All 103 shards hold 51,389 rows; the e2b rows that survive a gradable-answer
filter and the SmolDataEnvs overlap exclusion number 9,187, so 4.6x more tasks were sitting on
local disk and unused.

**Its quality problems are invisible in the row.** An audit of the ladder runs (Part B/C of
docs/_audit_ladder.md) read every L1 failure and found that only about a third were failures of
reasoning: `stat_test` questions whose answer is an artefact of a method choice, ML-fit
questions where the gold is one draw from a distribution the question does not pin, label
answers graded on an exact surface form the question never specifies, and questions that defer
to a criterion the reference invented. None of that is visible in a row, so it could not be
filtered on, balanced, or reported — it could only be discovered by spending trials on it.

**It cannot answer "how much of this pool do I have tables for?"** The Kaggle cache is 48 GB and
Kaggle was returning 403s on the day this was built, so bulk fetching is not available and the
cost of filling the pool has to be estimated instead of paid.

So: tags, not deletions. Every task survives, keeps its `task_id` (all 2,000 v1 ids are stable
under the same `ja_<slug>` rule), and gains a family, an answer type, a nondeterminism flag with
reasons, an ambiguity flag with reasons, and its file count and input size. The subset a ladder
run should use is a *selection* over those tags, computed on demand by `is_ladder_grade`, so a
reviewer can change the definition without rebuilding the pool.

    uv run python -m smol_ladder.jtasks_v2 --report-only      # sizes only, no pool written
    uv run python -m smol_ladder.jtasks_v2                    # writes data/jtasks_v2.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from smol_ladder.fetch_shards import SHARDS, read_shard, shard_path
from smol_ladder.jtasks import DATASET, classify, grade_params, smoldataenvs_datasets
from smol_ladder.tasks import DATA

#: Measured Kaggle dataset sizes, so a rebuild does not re-hit a rate-limited endpoint and
#: silently under-report the download cost. Not task data; safe to keep next to the pool.
SIZE_CACHE = DATA / "jl_dataset_sizes.json"

# Operation family: an ordered regex list, first match wins, so the order encodes the
# precedence and is worth stating rather than leaving to whoever reorders the list next.
#
# `ml_fit` is first because "what accuracy did the random forest reach" matches both it and
# `stat_test`, and the fit is the reason the number is not reproducible. `stat_test` is kept
# for the tests with no fitted model (a p-value, a correlation, an ANOVA), which are
# reproducible once the method is pinned. `join` precedes the aggregates because when a
# question names a join verb, the join is the work; "how many" in the same sentence is
# bookkeeping. `agg` precedes `argmax` because a superlative over a numeric column is an
# aggregate ("the highest total sales"), while `argmax` is a superlative over category
# membership ("the most common gender"), whose answer is a label rather than a number. A bare
# "which" cannot separate the two, so `argmax` does not claim it: it matches an explicit
# category noun or an explicit superlative over membership. That precision does double duty —
# it is also what keeps "among" from stealing every "most common X among Y" from `argmax`.
FAMILIES: tuple[tuple[str, re.Pattern], ...] = (
    ("ml_fit", re.compile(
        r"\b(?:random forest|decision tree|gradient boost\w*|xgboost|lightgbm|adaboost|svm|"
        r"support vector|knn|k-nearest|neural net\w*|cnn|logistic regression|linear regression|"
        r"polynomial regression|\w*regression|regressor|regression model|ridge|lasso|"
        r"elastic ?net|naive bayes|k-means|kmeans|cluster\w*|pca|classifier|classification model|"
        r"ensemble|bagging\w*|boosting\w*|stacking\w*|predict\w*|sarima|arima|"
        r"train(?:ing)? (?:the |a )?(?:model|classifier)|cross[- ]validation|cross_val\w*|"
        r"grid ?search|parameter tuning|tuning (?:the )?(?:hyper)?parameters?|hyperparameter|"
        r"model training|fitting the model|fit the model|epoch|learning rate|"
        r"feature importance|feature selection|confusion matrix|f1[- ]score|auc[- ]?roc|"
        r"roc[- ]auc|accuracy(?: score| of)?|precision|recall|r2\b|r[- ]?squared|rmse|mae\b|"
        r"mean (?:absolute|squared) error|residual\w*|baseline model|test set|overfitting|"
        r"underfitting|whiten\w*|label encoding)\b", re.I)),
    ("stat_test", re.compile(
        r"\bp-?value\b|significan\w*|chi[- ]?square|t[- ]test|anova|wilcoxon|mann[- ]whitney|"
        r"kruskal|pearson|spearman|kendall|correlat\w*|regression coefficient|hypothesis test|"
        r"normality|shapiro|kolmogorov|confidence interval|statistical (?:test|significance)|"
        r"statistical propert\w+|white noise|stationar\w+|skew\w*|kurtos\w*|"
        r"distribution\w* shape|dispers\w+|p\s*[-=]\s*value", re.I)),
    ("join", re.compile(
        r"\bjoin(?:ed|ing|s)?\b|\bmerge[ds]?\b|\bcombine[sd]? (?:with|into)\b|"
        r"\b(?:inner|left|right|outer|full)[- ]join\b", re.I)),
    ("lookup", re.compile(
        r"\blook ?up\b|\bvalue of\b[^?.]{0,60}\b(?:where|for|at|when)\b|"
        r"\bcorresponding (?:value|entry|row|record)\b|\bretrieve\b|\bfind the (?:entry|record|"
        r"row)\b|\bfor the (?:row|record) where\b", re.I)),
    ("groupby", re.compile(
        r"\bper\b|\bby (?:country|region|category|genre|team|company|author|artist|state|"
        r"year|month|day|group|type|class|status|species|gender|brand|department|product|"
        r"channel)\b|breakdown|\beach \w+|\bgrouped?\b|for every|\bby each\b", re.I)),
    ("count", re.compile(
        r"\bhow many\b|\bnumber of\b|\bcount\b|\bhow often\b|\bhow much\b|\btotal (?:number|"
        r"count)\b|\bhow many times\b", re.I)),
    ("argmax", re.compile(
        r"\bmost common\b|\bmost frequent\b|\btop \d+\b|\bwhich (?:one|country|region|"
        r"category|genre|team|company|author|artist|state|gender|brand|type|class|status|"
        r"species|department|product|channel)\b|\bname the\b|\blist the\b|\bidentify\b", re.I)),
    ("filter", re.compile(
        r"\bonly\b|\bwhere\b|\bwith\b|\bthat (?:is|are|was|were)\b|\bconsidering\b|"
        r"\bexcluding\b|\bexcept\b|\bamong\b|\bfilt\w*|\bsatisf\w+|\bgreater than\b|"
        r"\bless than\b|\babove\b|\bbelow\b|\bbetween\b|\bat least\b|\bif\b", re.I)),
    ("string", re.compile(
        r"\bpercentage\b|\bfraction\b|\bproportion\b|\bshare\b|\bratio\b|\brate\b|\bpercent\b|"
        r"\bcontains\b|\bstarts? with\b|\bends? with\b|\bmatches\b|\bregex\b|\bpattern\b|"
        r"\bword\b|\bcharacter\b|\blength of\b", re.I)),
    ("agg", re.compile(
        r"\bmean\b|\bmedian\b|\bsum\b|\btotal\b|\baverage\b|\bstd\b|\bstandard deviation\b|"
        r"\bvariance\b|\bmax\w*|\bmin\w*|\blargest\b|\bsmallest\b|\bhighest\b|\blowest\b|"
        r"\bfastest\b|\bslowest\b|\bweighted\b|\bpercentile\b|\bquantile\b|\bagg\w*|"
        r"\bsumma\w*|\btop\b|\bmost\b|\bfrequency\b|\bmode\b", re.I)),
)

#: Families that need a method choice the question does not pin. These are where the ladder's
#: information rungs have something to disambiguate, and where the audit found L1 at 21.6%.
HARD_FAMILIES = ("ml_fit", "stat_test", "groupby", "lookup", "join")

#: Phrases that mean the answer is one draw from a distribution the question does not fix.
NONDETERMINISM: tuple[tuple[str, re.Pattern], ...] = (
    ("model fit", re.compile(
        r"\bfit\b|\bfitted\b|\bfitting\b|\btrain(?:ed|ing)? (?:a|the|an)? ?model|"
        r"\bclassifier\b|\bregress\w+|\brandom forest\b|\bgradient boost\w*|\bknn\b|"
        r"\bk-nearest\b|\bsvm\b|\bneural\b|\bdeep learning\b|\bml\b|\bmachine learning\b|"
        r"\bcross[- ]validation\b|\bcross_val\w*\b|\bgrid ?search\b|\btuning\b|\btuned\b|"
        r"\btrain(?:ing)?[- ]?test\b|\btrain_test_split\b|\bepoch\w*\b|\bdropout\b|"
        r"\blearning rate\b|\bhyperparameter\w*\b|\bpredict\w*\b|\bfeature importance\b|"
        r"\bconfusion matrix\b|\bf1[- ]score\b|\bauc\b|\broc\b|\baccuracy\b|\bprecision\b|"
        r"\brecall\b|\bsensitivity\b|\bspecificity\b|\bclassification\b", re.I)),
    ("random sampling", re.compile(
        r"\brandom\b|\bshuffle\b|\bsampl\w*|\bbootstrap\b|\bpermut\w*|\bresampl\w*|\bdraw\w*|"
        r"\bsimulat\w*\b", re.I)),
)

#: Questions that defer to a criterion the row never states. From the audit's "arbitrary task"
#: and "ambiguous question" categories: the reference invented a constant and the agent cannot
#: recover it from the question.
AMBIGUOUS_PHRASES: tuple[tuple[str, re.Pattern], ...] = (
    ("unstated threshold", re.compile(
        r"\bthreshold\b|\bcut[- ]?off\b|\bcriterion\b|\bcriteria\b|\bbased on .{0,20}analysis\b|"
        r"\bseparat\w+ .{0,20}(?:from|into)\b|\bdefinition of\b|\bconsidered\b|"
        r"\bqualif\w+ (?:as|by)\b|\bredundant\b|\bcompetitive\b|\bdominant\b|\bsignificant\b|"
        r"\bmeaningful\b|\bappropriat\w+\b", re.I)),
    ("unjudged comparison", re.compile(
        r"\bbetter\b|\bbest\b|\bworse\b|\bworst\b|\boptimal\b|\bideal\b|\bstrongest\b|"
        r"\bimprovement\b|\bmuch more\b|\bmost (?:efficient|effective|popular|influential)\b",
        re.I)),
)

#: answer -> the surface form the grader compares against.
ANSWER_TYPES = {"numeric": "numeric", "exact_short": "label", "exact_bool": "bool"}


def _slug(task_id: str) -> str:
    """The v1 directory-safe id. Raw jupyter-agent ids contain slashes, so one task would
    become a directory tree under data/runs; this rule has not changed since v1, which is
    what keeps every v1 task_id stable in v2."""
    return re.sub(r"[^A-Za-z0-9_.-]", "_", str(task_id))


def task_slug(row: dict) -> str:
    return _slug(row["id"] if "id" in row else row["task_id"])


def op_family(question: str) -> str:
    """The first family whose pattern matches, else `other`."""
    for name, pattern in FAMILIES:
        if pattern.search(question or ""):
            return name
    return "other"


def nondeterminism_reasons(question: str) -> list[str]:
    return [name for name, pattern in NONDETERMINISM if pattern.search(question or "")]


def ambiguity_reasons(question: str, answer: str, mode: str) -> list[str]:
    """Why a task's gold answer is not recoverable from the question alone.

    Two independent sources, because they fail differently:

    - A *label* answer graded on exact match, when the answer's own words never appear in the
      question. The grader cannot tell `North America` from `NA`, and the agent picked the
      right one. This is a third of the audited failures.
    - A question that defers to a criterion it never states ("based on correlation analysis",
      "the threshold that separates ..."). The reference invented a constant; no reader can
      recover it.
    """
    reasons = [name for name, pattern in AMBIGUOUS_PHRASES if pattern.search(question or "")]
    if mode == "exact_short":
        words = [w for w in re.split(r"[^A-Za-z0-9]+", answer or "") if len(w) > 3]
        q = (question or "").lower()
        missing = [w for w in words if w.lower() not in q]
        if missing:
            reasons.append("label vocabulary not in question")
    return reasons


def answer_type(mode: str) -> str:
    return ANSWER_TYPES.get(mode, mode or "unknown")


def cache_index(root: Path | str | None = None) -> dict[str, list[Path]]:
    """basename -> every cached file with that name, under the Kaggle cache.

    One index for the whole 48 GB cache so sizing the pool costs one walk instead of one per
    task. Dotfiles are skipped: a `.<something>.complete` marker is not an input table.
    """
    import os

    base = Path(root) if root else Path(
        os.environ.get("SMOL_LADDER_CACHE", "/var/tmp/smol-ladder")) / "kaggle" / "datasets"
    index: dict[str, list[Path]] = defaultdict(list)
    if not base.exists():
        return {}
    for path in base.rglob("*"):
        if path.is_file() and not path.name.startswith("."):
            index[path.name].append(path)
    return dict(index)


def input_bytes(row: dict, index: dict[str, list[Path]]) -> int | None:
    """Bytes of the files this task names, or None if any of them is not cached.

    None is the honest answer when a table is missing: it separates "this pool needs no
    download" from "this pool needs 12 GB we have not paid for yet", which is the whole
    question part (3) asks.
    """
    total = 0
    for name in row.get("files") or []:
        hits = index.get(Path(name).name)
        if not hits:
            return None
        total += hits[0].stat().st_size
    return total


def tag(row: dict, index: dict[str, list[Path]] | None = None) -> dict:
    """The v1 row plus its tags. Every field v1 carried is preserved unchanged."""
    out = dict(row)
    question = row.get("question") or ""
    mode = row.get("reward_mode") or ""
    out["op_family"] = op_family(question)
    out["answer_type"] = answer_type(mode)
    out["nondeterminism_reasons"] = nondeterminism_reasons(question)
    out["nondeterministic"] = bool(out["nondeterminism_reasons"])
    out["ambiguity_reasons"] = ambiguity_reasons(question, row.get("answer", ""), mode)
    out["ambiguous"] = bool(out["ambiguity_reasons"])
    out["n_files"] = len(row.get("files") or [])
    size = input_bytes(row, index) if index else None
    out["input_bytes"] = size
    out["inputs_cached"] = size is not None
    return out


def tags_for_rows(rows: list[dict], index: dict[str, list[Path]] | None = None) -> list[dict]:
    return [tag(row, index) for row in rows]


def is_ladder_grade(tags: dict) -> bool:
    """The recommended subset: a task whose answer a careful reader can actually derive.

    Excluded, and only these three reasons:

    - ``nondeterministic``: the gold is one draw from a distribution the question does not
      pin, so a correct agent scores 0 for a reason no information rung can fix.
    - ``ambiguous``: the gold is an exact surface form or an invented constant the question
      does not specify, so a correct agent scores 0 for a wording reason.
    - no input files: nothing to explore.

    Kept on purpose: the underspecified families (`ml_fit`, `stat_test`, `groupby`,
    `lookup`, `join`) and large tables. Those are the tasks where the ladder's information
    rungs have something to disambiguate, which is the population the ladder exists to
    measure. Excluding them would hand back the count/agg pool that is already too easy.
    """
    if tags.get("nondeterministic") or tags.get("ambiguous"):
        return False
    if not tags.get("n_files"):
        return False
    return True


# --- the pool ---------------------------------------------------------------------------

def pool_rows(shards: int, limit: int | None, exclude_overlap: bool = True):
    """Every gradable task in the local shards, plus the funnel that got there.

    Same filters as v1, same order, so a v1 id is a v2 id: e2b only, no SmolDataEnvs dataset
    overlap, at least one named file, a gradable answer, unique slug.
    """
    out: list[dict] = []
    stats: Counter = Counter()
    seen: set[str] = set()
    banned = smoldataenvs_datasets() if exclude_overlap else set()
    for index in range(shards):
        for row in read_shard(index):
            stats["rows"] += 1
            if row.get("executor_type") != "e2b":
                stats["not e2b"] += 1
                continue
            stats["e2b"] += 1
            if exclude_overlap and row.get("kaggle_dataset_name") in banned:
                stats["overlaps SmolDataEnvs"] += 1
                continue
            files = row.get("files_used") or []
            if not files:
                stats["no files"] += 1
                continue
            graded = classify(row.get("answer", ""))
            if graded is None:
                stats["ungradable answer"] += 1
                continue
            mode, value = graded
            slug = _slug(row["id"])
            if slug in seen:
                stats["duplicate id"] += 1
                continue
            seen.add(slug)
            stats[f"gradable {mode}"] += 1
            out.append({
                "task_id": f"ja_{slug}",
                "question": (row.get("question") or "").strip(),
                "answer": value,
                "reward_mode": mode,
                "atol": grade_params(mode, value)[0],
                "rtol": grade_params(mode, value)[1],
                "files": [Path(f).name for f in files],
                "source": DATASET,
                "kaggle_dataset_name": row.get("kaggle_dataset_name"),
                "edu_score": row.get("edu_score"),
            })
            if limit and len(out) >= limit:
                return out, stats
    stats["final"] = len(out)
    return out, stats


def existing_ids(path: Path | str | None = None) -> set[str]:
    """task_ids already in v1, so the report can prove they all survived."""
    src = Path(path or DATA / "jtasks.jsonl")
    if not src.exists():
        return set()
    return {json.loads(line)["task_id"] for line in src.read_text().splitlines() if line.strip()}


def load_size_cache(path: Path | str) -> dict[str, int]:
    """Previously-measured dataset sizes, slug -> bytes.

    Kept on disk because Kaggle's metadata endpoint rate-limits: a sweep of ~640 requests gets
    a 429 partway through, and without a cache every rebuild silently reports a *smaller*
    download cost each time as more lookups fail. An under-reported number reads as progress.
    """
    src = Path(path)
    if not src.exists():
        return {}
    try:
        return {k: int(v) for k, v in json.loads(src.read_text()).items() if v is not None}
    except (json.JSONDecodeError, TypeError, ValueError):
        return {}


def save_size_cache(path: Path | str, sizes: dict[str, int | None]) -> None:
    """Merge new measurements into the size cache, keeping the old ones for missing slugs."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    merged = load_size_cache(out)
    merged.update({k: int(v) for k, v in sizes.items() if v is not None})
    out.write_text(json.dumps(merged, indent=0, sort_keys=True))


def download_estimate(rows: list[dict], index: dict[str, list[Path]],
                      dataset_bytes: dict[str, int] | None = None) -> dict:
    """How much of the pool is already on disk, and what filling the rest would cost.

    A dataset counts as covered when every file any of its tasks names is in the cache. The
    cost of the rest comes from `dataset_bytes`, a slug -> total size map from
    `kaggle_dataset_sizes()`. When a dataset has no measured size it is counted in
    `datasets_unpriced` and excluded from `download_gb`, so the GB figure is always a floor
    over the priced subset and never a guess dressed as a measurement.
    """
    by_dataset: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        if row.get("kaggle_dataset_name"):
            by_dataset[row["kaggle_dataset_name"]].update(
                Path(f).name for f in row.get("files") or [])
    datasets = set(by_dataset)
    uncovered = {ds for ds in datasets
                 if not all(Path(f).name in index for f in by_dataset[ds])}
    measured = dataset_bytes or {}
    unpriced = sorted(ds for ds in uncovered if ds not in measured)

    cached_tasks = sum(1 for r in rows if all(Path(f).name in index for f in r.get("files") or []))
    return {
        "tasks": len(rows),
        "datasets": len(datasets),
        "datasets_covered_by_cache": len(datasets) - len(uncovered),
        "datasets_needing_download": len(uncovered),
        "tasks_inputs_cached": cached_tasks,
        "tasks_needing_download": len(rows) - cached_tasks,
        "download_gb": round(sum(measured[ds] for ds in uncovered if ds in measured) / 1e9, 1),
        "datasets_priced": len(uncovered) - len(unpriced),
        "datasets_unpriced": len(unpriced),
        "cache_gb_named_files": round(
            sum(index[f][0].stat().st_size
                for f in {f for r in rows for f in r.get("files") or []} if f in index) / 1e9, 1),
    }


def kaggle_dataset_sizes(slugs: list[str], workers: int = 8) -> dict[str, int | None]:
    """slug -> total bytes, from Kaggle's dataset-view endpoint. Metadata only.

    One ~4 KB request per dataset and no archive is fetched, which is the point: the cache is
    48 GB and the archive fetches were returning 403s, but the *size* of a dataset is the only
    thing needed to price filling the pool. A dataset the account cannot view comes back None
    rather than zero, so an unpriceable dataset is visible instead of free.
    """
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor

    import requests
    from kagglehub.clients import get_kaggle_credentials

    creds = get_kaggle_credentials()
    if not creds:
        return {slug: None for slug in slugs}
    out: dict[str, int | None] = {}
    lock = threading.Lock()

    def one(slug: str) -> None:
        session = requests.Session()
        session.auth = (creds.username, creds.key)
        for _ in range(2):
            try:
                resp = session.get(
                    f"https://www.kaggle.com/api/v1/datasets/view/{slug}", timeout=40)
                with lock:
                    out[slug] = resp.json().get("totalBytes") if resp.status_code == 200 else None
                return
            except Exception:  # noqa: BLE001 - a flaky request must not stop the estimate
                time.sleep(2)
        with lock:
            out[slug] = None

    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(one, slugs))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shards", type=int, default=SHARDS)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out", default=str(DATA / "jtasks_v2.jsonl"))
    ap.add_argument("--allow-overlap", action="store_true")
    ap.add_argument("--no-kaggle", action="store_true",
                    help="skip the metadata size lookups; download_gb is then omitted")
    ap.add_argument("--report-only", action="store_true",
                    help="print the funnel and distributions without writing the pool")
    args = ap.parse_args()

    rows, stats = pool_rows(args.shards, args.limit, not args.allow_overlap)
    index = cache_index()
    tagged = tags_for_rows(rows, index)

    print(f"funnel over {args.shards} shards")
    for key, value in stats.most_common():
        print(f"  {key:24} {value}")
    v1 = existing_ids()
    ids = {r["task_id"] for r in tagged}
    print(f"  v1 ids still present       {len(v1 & ids)}/{len(v1)}")

    print("\nop_family        " + json.dumps(Counter(r["op_family"] for r in tagged).most_common()))
    print("answer_type      " + json.dumps(
        Counter(r["answer_type"] for r in tagged).most_common()))
    print(f"nondeterministic {sum(r['nondeterministic'] for r in tagged)}")
    print(f"ambiguous        {sum(r['ambiguous'] for r in tagged)}")
    print(f"ladder-grade     {sum(map(is_ladder_grade, tagged))}")
    print(f"n_files          {json.dumps(Counter(r['n_files'] for r in tagged).most_common())}")

    sizes = load_size_cache(SIZE_CACHE)
    missing = [s for s in
               sorted({r["kaggle_dataset_name"] for r in tagged if r["kaggle_dataset_name"]})
               if s not in sizes]
    if not args.no_kaggle and missing:
        fresh = kaggle_dataset_sizes(missing)
        save_size_cache(SIZE_CACHE, fresh)
        sizes.update({k: v for k, v in fresh.items() if v is not None})
    est = download_estimate(tagged, index, sizes)
    print("\ninputs\n" + json.dumps(est, indent=2))
    hard = [r for r in tagged if r["op_family"] in HARD_FAMILIES]
    print(f"hard families    {len(hard)}; ladder-grade among them "
          f"{sum(map(is_ladder_grade, hard))}")

    if args.report_only:
        return
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        for row in tagged:
            fh.write(json.dumps(row) + "\n")
    print(f"\nwrote {len(tagged)} tasks to {out}")


if __name__ == "__main__":
    main()