"""Build every input the droplet needs, on the laptop, before the droplet exists.

    uv run --extra train python ops/amd/stage.py --out /tmp/smol-ladder-stage --commit HEAD

The droplet bills from the instant it is created, so everything that can be prepared beforehand is.
The output directory is what `driver.py bootstrap` copies over; it contains

    code.tar.gz       `git archive` of the pinned commit: no git, no GitHub, no clone on the droplet
    sft_a.tar.gz      arm A: upstream's SmolDataEnvs-sft export (train.jsonl + val.jsonl)
    sft_b.tar.gz      arm B: our ja3 traces, v2 (the harness's conversation) + manifest + index
    tokens.json       trained-token counts per arm at this --max-length: the chat template's
                      rendering (tool calls and results included) through the Qwen3.5 tokenizer
                      when `transformers` and the cached tokenizer exist (`--extra train`), else a
                      labelled estimate
    repo.txt          the pinned commit sha
    SHA256SUMS        checksums of the three tarballs, verified on the droplet before unpacking
    entrypoint.sh     the ONE remote entry script, taken from the pinned commit
    remote.env        HF_TOKEN and the Hub namespace (mode 600; never in an argv, never in the sums)

Task tables and the sandbox are NOT staged: the ladder runs on the laptop.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ops.amd.doapi import load_dotenv  # noqa: E402

REQUIRED_IN_COMMIT = ("train/sft_lora.py", "train/format.py", "smol_ladder/upstream.py",
                      "ops/amd/entrypoint.sh", "ops/amd/smoke.sh", "ops/amd/run_sft.sh",
                      "ops/amd/serve.sh", "ops/amd/common.sh")
MSG_OVERHEAD_TOKENS = 8      # role markers and separators per message in the chat template
FALLBACK_BYTES_PER_TOKEN = 3.0
TOKENIZER_GLOB = "models--Qwen--Qwen3.5-2B/snapshots/*/tokenizer.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_sums(out: Path, names: list[str]) -> Path:
    sums = out / "SHA256SUMS"
    sums.write_text("".join(f"{sha256_file(out / n)}  {n}\n" for n in names))
    return sums


def verify_sums(out: Path) -> list[str]:
    """Names whose bytes do not match SHA256SUMS (empty list = intact)."""
    bad = []
    for line in (out / "SHA256SUMS").read_text().splitlines():
        digest, _, name = line.partition("  ")
        if not (out / name).exists() or sha256_file(out / name) != digest:
            bad.append(name)
    return bad


# ── tokens ────────────────────────────────────────────────────────────────────────

def row_tokens(row: dict, encode) -> int:
    """Tokens of one SFT row: tool schemas once, then each message plus any tool-call JSON."""
    n = 0
    if row.get("tools"):
        n += encode(json.dumps(row["tools"]))
    for m in row.get("messages", []):
        content = m.get("content") or ""
        if not isinstance(content, str):
            content = json.dumps(content)
        n += encode(content) + MSG_OVERHEAD_TOKENS
        if m.get("tool_calls"):
            n += encode(json.dumps(m["tool_calls"]))
    return n


def count_file(path: Path, max_length: int, encode, render=None) -> dict:
    """Rows, raw tokens, and tokens actually trained on (each row cut at `max_length`).

    With `render` (row -> the text the trainer will see) the count is `encode(render(row))`: the
    chat template's own output, tool calls and tool results included. Without it, `row_tokens`'s
    per-message estimate.
    """
    rows = raw = trained = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        n = encode(render(row)) if render is not None else row_tokens(row, encode)
        rows += 1
        raw += n
        trained += min(n, max_length)
    return {"rows": rows, "raw_tokens": raw, "trained_tokens": trained}


def find_tokenizer() -> object | None:
    """A callable str -> token count, from the local HF cache; None if unavailable."""
    try:
        from tokenizers import Tokenizer
    except ImportError:
        return None
    cache = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    for candidate in sorted(cache.glob(TOKENIZER_GLOB)):
        tok = Tokenizer.from_file(str(candidate))
        return lambda text: len(tok.encode(text).ids)
    return None


def find_renderer() -> tuple | None:
    """(render, encode) that count the text SFT really trains on, or None.

    `render(row)` is `train.sft_lora.prepare`'s row through the Qwen3.5 chat template with tools
    and `enable_thinking=False`; `encode(text)` is the tokenizer's token count of that text. Needs
    `transformers` and the cached Qwen/Qwen3.5-2B tokenizer; without them the caller falls back to
    the labelled estimate. Counting the *rendering* matters: before 2026-10-02 the trainer fed the
    template messages with no tool calls, and any count of "the rows" says nothing about the text
    that is trained on.
    """
    try:
        from train.render import load_tokenizer, render_row, tokenizer_renderer
        from train.sft_lora import prepare
        tokenizer = load_tokenizer()
    except Exception:  # noqa: BLE001 - no transformers or no cached tokenizer: estimate instead
        return None
    template = tokenizer_renderer(tokenizer)

    def render(row: dict) -> str:
        return render_row(prepare([row], "bash")[0], template)

    def encode(text: str) -> int:
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])

    return render, encode


def build_token_counts(a_files: list[Path], b_file: Path, max_length: int, encode=None,
                       render=None) -> dict:
    """tokens.json content. `encode=None` falls back to bytes/3.0, and says so."""
    def one(files: list[Path]) -> dict:
        if encode is not None:
            parts = [count_file(f, max_length, encode, render) for f in files]
            method = ("chat template rendered (tools, enable_thinking=False) + Qwen3.5-2B "
                      "tokenizer, truncated at max_length" if render is not None else
                      "tokenizers (Qwen3.5-2B) + 8/message overhead, truncated at max_length")
            return {"rows": sum(p["rows"] for p in parts),
                    "raw_tokens": sum(p["raw_tokens"] for p in parts),
                    "trained_tokens": sum(p["trained_tokens"] for p in parts),
                    "method": method}
        rows = sum(1 for f in files for line in f.read_text().splitlines() if line.strip())
        est = int(sum(f.stat().st_size for f in files) / FALLBACK_BYTES_PER_TOKEN)
        return {"rows": rows, "raw_tokens": est, "trained_tokens": est,
                "method": "bytes/3.0 heuristic, no truncation (pessimistic)"}

    a, b = one(a_files), one([b_file])
    ab = {"rows": a["rows"] + b["rows"], "raw_tokens": a["raw_tokens"] + b["raw_tokens"],
          "trained_tokens": a["trained_tokens"] + b["trained_tokens"], "method": "A + B"}
    return {"max_length": max_length, "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "sets": {"A": a, "B": b, "AB": ab}}


# ── tarballs ──────────────────────────────────────────────────────────────────────

def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def check_commit(repo: Path, commit: str) -> str:
    """Resolve `commit` and confirm it holds everything the droplet will run."""
    sha = git(repo, "rev-parse", "--verify", f"{commit}^{{commit}}")
    files = set(git(repo, "ls-tree", "-r", "--name-only", sha).splitlines())
    missing = [f for f in REQUIRED_IN_COMMIT if f not in files]
    if missing:
        raise SystemExit(
            f"commit {sha[:10]} lacks {missing}. The droplet runs code from one pinned commit; "
            "merge `main` (train/, smol_ladder/) into the AMD branch and commit before staging.")
    return sha


def dirty_paths(repo: Path) -> list[str]:
    """Tracked files under the code the droplet runs that differ from HEAD."""
    out = git(repo, "status", "--porcelain", "--", "ops/amd", "train", "smol_ladder", "pyproject.toml")
    return [line for line in out.splitlines() if line and not line.startswith("??")]


def make_code_tarball(repo: Path, sha: str, out: Path) -> None:
    with out.open("wb") as fh:
        subprocess.check_call(["git", "-C", str(repo), "archive", "--format=tar.gz", sha], stdout=fh)


def make_tarball(out: Path, base: Path, members: list[str]) -> None:
    with tarfile.open(out, "w:gz") as tar:
        for m in members:
            if not (base / m).exists():
                raise SystemExit(f"missing {base / m}")
            tar.add(base / m, arcname=m)


def resolve_namespace(explicit: str = "") -> str:
    if explicit or os.environ.get("AMD_HUB_NAMESPACE"):
        return explicit or os.environ["AMD_HUB_NAMESPACE"]
    try:
        from huggingface_hub import HfApi
        return HfApi().whoami()["name"]
    except Exception as exc:  # noqa: BLE001 - say what to do instead of guessing a namespace
        raise SystemExit("set AMD_HUB_NAMESPACE (your HF user or org) in .env: could not look it "
                         f"up from HF_TOKEN ({exc})")


def write_remote_env(out: Path, namespace: str) -> None:
    token = os.environ.get("HF_TOKEN", "")
    if not token:
        raise SystemExit("HF_TOKEN is not set (repo .env). Without it nothing can be pushed, and "
                         "the Hub is the only copy of an adapter that survives the droplet.")
    path = out / "remote.env"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as fh:
        fh.write(f"HF_TOKEN={token}\nAMD_HUB_NAMESPACE={namespace}\n")
    os.chmod(path, 0o600)


def stage(out: Path, commit: str, max_length: int, data: Path, repo: Path = REPO_ROOT,
          allow_dirty: bool = False, namespace: str = "", encode=None, render=None) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    sha = check_commit(repo, commit)
    dirty = dirty_paths(repo)
    if dirty and not allow_dirty:
        raise SystemExit("uncommitted changes in code the droplet will run:\n  "
                         + "\n  ".join(dirty) + "\ncommit them (the pin is the commit), or pass "
                         "--allow-dirty to stage the commit without them.")
    a_dir = data / "train" / "sft_upstream"
    b_file = data / "train" / "ja3_sft_v2.jsonl"
    for needed in (a_dir / "train.jsonl", a_dir / "val.jsonl", b_file):
        if not needed.exists():
            hint = ("; arm B is ja3_sft_v2 (the harness's conversation, `python -m "
                    "train.export_ja3_v2`). ja3_sft.jsonl is v1 and adapters trained on it are "
                    "invalid" if needed == b_file else "")
            raise SystemExit(f"missing {needed}{hint}")
    b_manifest = data / "train" / "ja3_sft_v2.manifest.json"
    if b_manifest.exists():
        replay = (json.loads(b_manifest.read_text()).get("replay") or {}).get("rate")
        if replay != 1.0:
            raise SystemExit(f"{b_manifest} reports a replay rate of {replay}: every arm-B row must "
                             "replay byte-identically through the harness (train/replay.py)")

    make_code_tarball(repo, sha, out / "code.tar.gz")
    make_tarball(out / "sft_a.tar.gz", data / "train",
                 ["sft_upstream/train.jsonl", "sft_upstream/val.jsonl"])
    members_b = ["ja3_sft_v2.jsonl"] + [
        name for name in ("ja3_sft_v2.manifest.json", "ja3_sft_v2.index.jsonl")
        if (data / "train" / name).exists()]
    make_tarball(out / "sft_b.tar.gz", data / "train", members_b)

    tokens = build_token_counts([a_dir / "train.jsonl"], b_file, max_length, encode, render)
    (out / "tokens.json").write_text(json.dumps(tokens, indent=2))
    (out / "repo.txt").write_text(sha + "\n")
    names = ["code.tar.gz", "sft_a.tar.gz", "sft_b.tar.gz", "tokens.json", "repo.txt"]
    write_sums(out, names)
    entry = git(repo, "show", f"{sha}:ops/amd/entrypoint.sh")
    (out / "entrypoint.sh").write_text(entry + "\n")
    os.chmod(out / "entrypoint.sh", 0o755)
    write_remote_env(out, resolve_namespace(namespace))
    return {"commit": sha, "tokens": tokens["sets"],
            "sizes": {n: (out / n).stat().st_size for n in names}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--commit", default="HEAD")
    ap.add_argument("--max-length", type=int, default=8192)
    ap.add_argument("--data", type=Path, default=REPO_ROOT / "data")
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--hub-namespace", default="")
    args = ap.parse_args()
    load_dotenv(REPO_ROOT / ".env")
    found = find_renderer()
    render = None
    if found is not None:
        render, encode = found
    else:
        encode = find_tokenizer()
    if encode is None:
        print("note: no Qwen3.5 tokenizer in the HF cache (or `tokenizers` not installed): token "
              "counts fall back to a pessimistic estimate. `uv run --extra train` and a "
              "cached Qwen/Qwen3.5-2B give exact counts.", file=sys.stderr)
    info = stage(args.out, args.commit, args.max_length, args.data,
                 allow_dirty=args.allow_dirty, namespace=args.hub_namespace, encode=encode,
                 render=render)
    print(json.dumps(info, indent=2))
    print(f"staged into {args.out}; next: python ops/amd/driver.py create --yes ...")


if __name__ == "__main__":
    main()
