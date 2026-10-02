#!/usr/bin/env bash
# shellcheck source-path=SCRIPTDIR
# Get everything that matters off the droplet. Runs ON the droplet; the laptop then pulls the same
# directory and verifies it (driver.py verify-sync).
#
#   sync_back.sh --push-hub [--arms A,B,AB]
#
# The evaluation ran on the laptop and wrote straight into its results tree, so the only things
# that must outlive the droplet are the adapters, the logs and the measurements. Order matters:
#   1. copy each finished adapter next to the logs and write SHA256SUMS.artifacts, so the laptop
#      can pull ONE directory and prove byte-for-byte that it arrived
#   2. push logs to the private dataset repo (the trainer already pushed checkpoints, and
#      run_sft.sh the final adapter, to each arm's own private repo)
#   3. verify every finished arm's adapter is readable on the Hub
# Safe to re-run, and safe to run while racing a shutdown: nothing here deletes anything.

set -euo pipefail
# shellcheck source=common.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
amd_load_env "$AMD_REMOTE_ROOT/.env"

PUSH=0
ARMS="A,B,AB"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --push-hub) PUSH=1; shift ;;
    --arms)     ARMS="$2"; shift 2 ;;
    *) amd_die "unknown argument '$1'" ;;
  esac
done

PY="$AMD_VENV/bin/python"
# No arm may have finished (a sync after a reclaim, or a failed smoke): the directory must still
# exist or `find` fails, and under pipefail that aborts the script before it pushes the logs.
mkdir -p "$AMD_REMOTE_LOG/adapters"
FINISHED=()
for arm in ${ARMS//,/ }; do
  dir="$(amd_arm_dir "$arm")"
  if [[ -f "$dir/.done" ]]; then
    mkdir -p "$AMD_REMOTE_LOG/adapters/$arm"
    cp "$dir/adapter_config.json" "$dir/adapter_model.safetensors" "$AMD_REMOTE_LOG/adapters/$arm/"
    FINISHED+=("$arm")
  fi
done
cp "$AMD_REMOTE_ROOT/tokens.json" "$AMD_REMOTE_LOG/" 2>/dev/null || true

( cd "$AMD_REMOTE_LOG" && find adapters -type f -print0 | sort -z | xargs -0 -r sha256sum > SHA256SUMS.artifacts.tmp \
  && mv SHA256SUMS.artifacts.tmp SHA256SUMS.artifacts )
amd_log "finished arms copied: ${FINISHED[*]:-none}; checksums in $AMD_REMOTE_LOG/SHA256SUMS.artifacts"

if (( PUSH )); then
  "$PY" "$AMD_REMOTE_ROOT/ops/amd/push_artifacts.py" \
    --repo "$AMD_HUB_NAMESPACE/$(amd_hub_name artifacts)" --log-dir "$AMD_REMOTE_LOG" \
    || amd_log "WARNING: the log push failed; re-run before destroying"
  if (( ${#FINISHED[@]} )); then
    REPOS=()
    for arm in "${FINISHED[@]}"; do REPOS+=("$(amd_arm_hub_repo "$arm")=$(amd_arm_dir "$arm")/adapter_model.safetensors"); done
    "$PY" "$AMD_REMOTE_ROOT/ops/amd/push_artifacts.py" --verify "${REPOS[@]}" \
      || { amd_log "FAIL: an adapter is not readable on the Hub; do NOT destroy yet"; exit 1; }
  fi
fi
amd_log "sync complete"
