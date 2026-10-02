#!/usr/bin/env bash
# Make an AMD Developer Cloud MI300X droplet run this repo. Idempotent: safe to re-run.
#
#   bootstrap.sh --commit <sha> [--with-data] [--with-eval]
#
# What it does, in order, each step skipped if already satisfied:
#   1. sanity: this really is a GPU droplet with ROCm on it
#   2. apt packages the ROCm docker images and the harness both want (rsync, bubblewrap, git)
#   3. the repo, at a pinned commit, into /opt/smol-ladder
#   4. a uv venv with the `train` extra (ROCm torch comes from the container, not from PyPI)
#   5. HF login from HF_TOKEN in the environment; never echoed
#   6. optional: the exported SFT sets and the eval task tables rsynced in
#
# It runs on the droplet as root (AMD's own docs say `ssh root@<ip>`).

set -euo pipefail
# shellcheck source=common.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

COMMIT=""
WITH_DATA=0
WITH_EVAL=0
DATA_SOURCE="${AMD_DATA_SOURCE:-/opt/smol-ladder-data}"
INPUT_SOURCE="${AMD_INPUT_SOURCE:-/opt/smol-ladder-data/inputs}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --commit)      COMMIT="$2"; shift 2 ;;
    --with-data)   WITH_DATA=1; shift ;;
    --with-eval)   WITH_EVAL=1; shift ;;
    *) amd_die "unknown argument '$1'" ;;
  esac
done
[[ -n "$COMMIT" ]] || COMMIT="$AMD_COMMIT"
amd_load_env /opt/smol-ladder/.env

# ── 1. what am I on ────────────────────────────────────────────────────────────
amd_log "host: $(uname -n)  kernel: $(uname -r)"
if [[ -e /dev/kfd ]]; then
  amd_log "ROCm: /dev/kfd present"
else
  amd_log "WARNING: /dev/kfd is absent. This is not a GPU droplet, or ROCm is not installed."
  amd_log "         The vendor GPU image has it. If this is a bare-OS droplet, install ROCm first."
fi
# rocm-smi on MI300X, or amdsmi on newer ROCm. Neither is nvidia-smi; do not go looking for one.
if command -v rocm-smi >/dev/null; then
  rocm-smi --showproductname 2>/dev/null | head -5 || true
elif command -v amdsmi >/dev/null; then
  amdsmi static 2>/dev/null || true
else
  amd_log "WARNING: neither rocm-smi nor amdsmi found; cannot confirm the GPU is visible."
fi

# ── 2. apt ─────────────────────────────────────────────────────────────────────
# Bubblewrap is not optional: the ladder grades every trial inside a bwrap jail, and a grader that
# cannot jail is a grader that has silently stopped isolating the model's file access.
NEEDED=(git rsync bubblewrap bzip2 curl ca-certificates jq)
MISSING=()
for pkg in "${NEEDED[@]}"; do
  dpkg -s "$pkg" >/dev/null 2>&1 || MISSING+=("$pkg")
done
if (( ${#MISSING[@]} )); then
  amd_log "apt-get install: ${MISSING[*]}"
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq
  apt-get install -y -qq "${MISSING[@]}"
else
  amd_log "apt: all of ${NEEDED[*]} already present"
fi

# ── 3. the repo, at a pinned commit ─────────────────────────────────────────────
if [[ ! -d "$AMD_REMOTE_ROOT/.git" ]]; then
  amd_log "cloning $AMD_REPO_URL into $AMD_REMOTE_ROOT"
  git clone "$AMD_REPO_URL" "$AMD_REMOTE_ROOT"
else
  amd_log "repo already present at $AMD_REMOTE_ROOT"
fi
if [[ "$(git -C "$AMD_REMOTE_ROOT" rev-parse HEAD)" != "$COMMIT" ]]; then
  amd_log "checking out $COMMIT"
  git -C "$AMD_REMOTE_ROOT" fetch --all --tags --quiet
  git -C "$AMD_REMOTE_ROOT" checkout --quiet "$COMMIT"
fi
git -C "$AMD_REMOTE_ROOT" submodule update --init --recursive || true
printf '%s\n' "$COMMIT" > "$AMD_REMOTE_ROOT/.amd-commit"

# The worktree's `data` is a symlink to a laptop path. On the instance the real tree is
# $AMD_DATA_ROOT and must be a directory, or every loader in the harness follows a dangling link.
if [[ -L "$AMD_REMOTE_ROOT/data" && ! -e "$AMD_REMOTE_ROOT/data" ]]; then
  amd_log "replacing dangling data symlink with $AMD_DATA_ROOT"
  rm -f "$AMD_REMOTE_ROOT/data"
  mkdir -p "$AMD_DATA_ROOT"
  ln -s "$AMD_DATA_ROOT" "$AMD_REMOTE_ROOT/data"
fi

# ── 4. the python environment ──────────────────────────────────────────────────
cd "$AMD_REMOTE_ROOT"
if [[ ! -d .venv ]]; then
  amd_log "creating .venv with uv and installing the train extra"
  # The ROCm torch wheel cannot come from PyPI (PyPI's `torch` is CUDA), so the venv is built
  # inside the ROCm container, where torch is already installed. On a non-container droplet this
  # step installs a CUDA torch that cannot see the GPU, which is why bootstrap says so loudly.
  if [[ -n "${AMD_ROCM_IMAGE:-}" ]]; then
    amd_log "note: torch must come from the ROCm image, not PyPI. See run_sft.sh --inside-rocm."
  fi
  uv venv --python 3.12 .venv
  VIRTUAL_ENV="$AMD_REMOTE_ROOT/.venv" uv pip install -e '.[train]' huggingface_hub
else
  amd_log ".venv already present"
fi

# ── 5. HF login ────────────────────────────────────────────────────────────────
# The token is read from the environment, written to the standard credentials file, and never
# echoed. If it is absent the run cannot push an adapter, and an adapter that cannot be pushed is
# an adapter that dies with the droplet, so this is a hard failure rather than a warning.
if [[ -n "${HF_TOKEN:-}" ]]; then
  umask 077
  mkdir -p /root/.cache/huggingface
  printf '%s' "$HF_TOKEN" > /root/.cache/huggingface/token
  huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential >/dev/null 2>&1 \
    || hf auth login --token "$HF_TOKEN" >/dev/null 2>&1 \
    || amd_log "WARNING: hf CLI login failed; relying on the token file at ~/.cache/huggingface/token"
  amd_log "HF login done (token not printed)"
else
  amd_die "HF_TOKEN is not set. The Hub is the only copy of an adapter that outlives the droplet."
fi

mkdir -p "$AMD_REMOTE_LOG" /opt/smol-ladder-data

# ── 6. data ────────────────────────────────────────────────────────────────────
if (( WITH_DATA )); then
  for f in "$DATA_SOURCE"/train/sft_upstream/train.jsonl \
           "$DATA_SOURCE"/train/sft_upstream/val.jsonl \
           "$DATA_SOURCE"/train/ja3_sft.jsonl; do
    [[ -f "$f" ]] || amd_die "missing $f; sync_back.sh must have pushed the SFT sets first"
  done
  amd_log "SFT sets present under $DATA_SOURCE/train"
fi

if (( WITH_EVAL )); then
  # The ladder's task tables. It skips tasks whose tables are not cached (--skip-uncached is the
  # default), so the cached set is what decides the denominator of every pass rate.
  N_INPUTS=$(find "$INPUT_SOURCE" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
  amd_log "eval input tables under $INPUT_SOURCE: $N_INPUTS directories"
  (( N_INPUTS > 0 )) || amd_die "no eval input tables at $INPUT_SOURCE"
  # The pool rows themselves.
  for f in jtasks.jsonl jtasks_v2.jsonl jtasks_v3.jsonl; do
    [[ -f "$DATA_SOURCE/$f" ]] || amd_log "WARNING: $DATA_SOURCE/$f is absent"
  done
fi

amd_log "bootstrap complete at commit $COMMIT"
