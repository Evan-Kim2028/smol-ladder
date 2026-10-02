#!/usr/bin/env bash
# Shared environment for every script in ops/amd/. Sourced, never executed.
#
# Everything here is a default that the owner can override from the environment or from .env on
# the instance, because the numbers that matter (the price, the credit, the cap) are the ones most
# likely to be wrong, and a hard-coded price is how a runbook quietly spends money it does not have.

set -euo pipefail

# The commit the instance checks out. Pinned rather than "main": the laptop's results and the
# instance's results must be produced by the same code, and an unpinned branch is a run whose
# provenance cannot be reconstructed after the branch moves.
AMD_COMMIT="${AMD_COMMIT:-$(git -C "$(dirname "${BASH_SOURCE[0]}")/../.." rev-parse HEAD)}"

# Price per GPU-hour. $2.59 is DigitalOcean's published MI300X rate
# (https://docs.digitalocean.com/products/amd/details/pricing/). $1.99 came from a third-party blog
# and could not be confirmed against AMD or DigitalOcean; it is kept only so the owner can override
# it if their console shows something else. Everything is computed from this one variable.
AMD_PRICE_PER_GPU_HOUR="${AMD_PRICE_PER_GPU_HOUR:-2.59}"

# Hard budget. The credit is $100, so the cap is set below it: the run must stop with credit left,
# not with a card charge. The watchdog enforces it by wall clock, not by reading a balance, because
# the only balance API is the console and the instance has no idea what it has spent.
AMD_BUDGET_USD="${AMD_BUDGET_USD:-80}"
AMD_WALLCLOCK_LIMIT_MIN="${AMD_WALLCLOCK_LIMIT_MIN:-$(python3 -c "print(int(float('$AMD_BUDGET_USD')/float('$AMD_PRICE_PER_GPU_HOUR')*60))")}"

# Dead-man switch: no ssh connection and no running job for this long -> power off, then destroy.
AMD_IDLE_LIMIT_MIN="${AMD_IDLE_LIMIT_MIN:-45}"

# The arms. Keys are the arm names used in run tags, Hub repo suffixes and run directories.
AMD_ARMS_DEFAULT="${AMD_ARMS_DEFAULT:-A B AB}"
AMD_BASE_MODEL="${AMD_BASE_MODEL:-Qwen/Qwen3.5-2B}"
AMD_SPLIT="${AMD_SPLIT:-test}"
AMD_RUN_TAG_PREFIX="${AMD_RUN_TAG_PREFIX:-amd1}"

# The HF repos the arms push to. Must be private and must exist (or be creatable) before a run:
# the Hub is the only copy of an adapter that survives a destroyed instance.
AMD_HUB_NAMESPACE="${AMD_HUB_NAMESPACE:-}"
AMD_HUB_ADAPTER_A="${AMD_HUB_ADAPTER_A:-smol-ladder-sft-a}"
AMD_HUB_ADAPTER_B="${AMD_HUB_ADAPTER_B:-smol-ladder-sft-b}"
AMD_HUB_ADAPTER_AB="${AMD_HUB_ADAPTER_AB:-smol-ladder-sft-ab}"
AMD_HUB_ARTIFACTS="${AMD_HUB_ARTIFACTS:-smol-ladder-runs}"

# On-instance paths.
AMD_REMOTE_ROOT="${AMD_REMOTE_ROOT:-/opt/smol-ladder}"
AMD_REMOTE_LOG="${AMD_REMOTE_LOG:-/var/log/smol-ladder}"

amd_log() { printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*" >&2; }

amd_die() { amd_log "FATAL: $*"; exit 1; }

# Load .env from the repo root if present. It is git-ignored and holds HF_TOKEN and, in API mode,
# AMD_CLOUD_API_TOKEN. It must never be committed and never printed.
amd_load_env() {
  local env_file="${1:-/opt/smol-ladder/.env}"
  if [[ -f "$env_file" ]]; then
    set -a
    # shellcheck disable=SC1090  # path is a runtime argument, not a literal
    source "$env_file"
    set +a
  fi
}

# Arm -> its exported dataset directory, on the instance. `data` is a symlink on the laptop that
# points at the shared results tree; on the instance it is a real directory that sync_back filled.
amd_data_dir() {
  case "$1" in
    A)  printf '%s\n' "${AMD_DATA_ROOT:-/opt/smol-ladder/data}/train/sft_upstream" ;;
    B)  printf '%s\n' "${AMD_DATA_ROOT:-/opt/smol-ladder/data}/train/sft_ab" ;;
    AB) printf '%s\n' "${AMD_DATA_ROOT:-/opt/smol-ladder/data}/train/sft_ab" ;;
    *)  amd_die "unknown arm '$1' (want one of: A, B, AB)" ;;
  esac
}

amd_arm_hub_repo() {
  local ns="$AMD_HUB_NAMESPACE"
  case "$1" in
    A)  printf '%s\n' "${ns:+$ns/}$AMD_HUB_ADAPTER_A" ;;
    B)  printf '%s\n' "${ns:+$ns/}$AMD_HUB_ADAPTER_B" ;;
    AB) printf '%s\n' "${ns:+$ns/}$AMD_HUB_ADAPTER_AB" ;;
    *)  amd_die "unknown arm '$1'" ;;
  esac
}

amd_run_tag() { printf '%s-%s\n' "$AMD_RUN_TAG_PREFIX" "$1"; }

amd_eval_model_name() { printf 'amd-%s-2b\n' "$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')"; }

# Does the arm need its dataset exported before training, or is it already on disk?
amd_needs_export() {
  local arm="$1" data
  data="$(amd_data_dir "$arm")"
  [[ -f "$data/train.jsonl" ]]
}
