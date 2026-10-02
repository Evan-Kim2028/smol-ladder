#!/usr/bin/env bash
# Get everything off the droplet. Runs on the droplet; pulls to the laptop over ssh.
#
#   sync_back.sh --to-laptop [dest]     rsync adapters, logs and sweep results back
#   sync_back.sh --push-hub             push the whole run tree to a private Hub dataset repo
#
# The two orders matter. The Hub copy goes first: it is the one that survives if the droplet dies
# during the rsync, and it is a single idempotent API call per file. The laptop rsync comes second,
# because it is the only one that needs the laptop to be reachable.
#
# What is copied back: every adapter directory (runs/sft_*), every log, and the whole results tree
# for the run tags this driver owns (data/runs/<tag>/). The results tree is the actual measurement;
# an adapter without its results tree is a model nobody evaluated.

set -euo pipefail
# shellcheck source=common.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

TO_LAPTOP=0
PUSH_HUB=0
DEST="${AMD_LAPTOP_DEST:-}"
REMOTE_USER="${AMD_LAPTOP_USER:-}"
TAGS="${AMD_SYNC_TAGS:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --to-laptop) TO_LAPTOP=1; shift ;;
    --push-hub)  PUSH_HUB=1; shift ;;
    --dest)      DEST="$2"; shift 2 ;;
    --tags)      TAGS="$2"; shift 2 ;;
    *) amd_die "unknown argument '$1'" ;;
  esac
done
(( TO_LAPTOP || PUSH_HUB )) || amd_die "choose --to-laptop and/or --push-hub"
amd_load_env

if (( PUSH_HUB )); then
  if [[ -z "$AMD_HUB_NAMESPACE" ]]; then
    amd_log "WARNING: AMD_HUB_NAMESPACE is unset; nothing to push to."
  else
    REPO="${AMD_HUB_NAMESPACE}/${AMD_HUB_ARTIFACTS}"
    amd_log "pushing adapters and logs to the private repo $REPO"
    HF="$AMD_REMOTE_ROOT/.venv/bin/python" "$AMD_REMOTE_ROOT/ops/amd/push_artifacts.py" \
      --repo "$REPO" \
      --run-tag-prefix "$AMD_RUN_TAG_PREFIX" \
      || amd_log "WARNING: the Hub push failed"
  fi
fi

if (( TO_LAPTOP )); then
  [[ -n "$REMOTE_USER" ]] || amd_die "AMD_LAPTOP_USER (the laptop's user@host) is required"
  [[ -n "$DEST" ]] || DEST="$REMOTE_USER:smol-ladder-out"

  # -a: archive (the whole point). --partial: a dropped connection leaves a resumable file rather
  # than a truncated one. -z: compressed, worth it on jsonl. No --delete: this droplet's tree is a
  # strict subset of the laptop's, and a --delete here would remove the results of every other
  # agent's runs from the shared tree.
  RSYNC=(rsync -az --partial --info=stats2)
  amd_log "rsyncing adapters and logs to $DEST"
  "${RSYNC[@]}" "$AMD_REMOTE_ROOT/runs/" "$DEST/runs/"
  "${RSYNC[@]}" "$AMD_REMOTE_LOG/" "$DEST/logs/"

  for tag in ${TAGS:-}; do
    SRC="$AMD_REMOTE_ROOT/data/runs/$tag"
    [[ -d "$SRC" ]] || { amd_log "no results at $SRC"; continue; }
    amd_log "rsyncing the sweep results for tag $tag"
    "${RSYNC[@]}" "$SRC/" "$DEST/data/runs/$tag/"
  done
fi

amd_log "sync complete"
