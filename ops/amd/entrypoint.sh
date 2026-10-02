#!/usr/bin/env bash
# shellcheck source-path=SCRIPTDIR
# THE remote entry script: one non-interactive command, no prompt anywhere, safe to re-run.
#
#   ssh root@<ip> -- bash /var/tmp/smol-ladder-stage/entrypoint.sh
#
# Everything it needs was staged on the laptop before the droplet existed, so on the droplet it
# only verifies, unpacks and installs:
#
#   1. refuse early if this is not an AMD GPU image (/dev/kfd)
#   2. verify the staged tarballs against SHA256SUMS, unpack code and both SFT sets
#   3. stop the image's jupyter container, start ONE long-lived container (`smol`) with the GPU
#   4. container_setup.sh inside it (docker exec): jq/procps, a venv over the image's torch with
#      the training deps (torch/vllm/triton pinned by constraint), HF login, private repos, model
#   5. arm the on-droplet watchdog (host side: it needs poweroff)
#
# torch and vLLM live in the image, not on the host python. There is no clone and no GitHub: the code is the pinned commit's `git archive`. The `train`
# extra in pyproject.toml is NOT installed: it pins CUDA torch wheels and bitsandbytes, which on
# this hardware would replace the image's ROCm torch with one that cannot see the GPU.

set -euo pipefail

STAGE="${AMD_STAGE_DIR:-/var/tmp/smol-ladder-stage}"
ROOT="${AMD_REMOTE_ROOT:-/opt/smol-ladder}"
LOGDIR="${AMD_REMOTE_LOG:-/var/log/smol-ladder}"
log() { printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*" >&2; }
die() { log "FATAL: $*"; exit 1; }
export DEBIAN_FRONTEND=noninteractive PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_INPUT=1
mkdir -p "$LOGDIR" "$ROOT"

[[ -d "$STAGE" ]] || die "no staged inputs at $STAGE (driver.py bootstrap uploads them)"
[[ -e /dev/kfd ]] || die "/dev/kfd is absent: no ROCm driver, so nothing below can work. Wrong image."
log "host $(uname -n), kernel $(uname -r)"

# ── 2. verify, then unpack ───────────────────────────────────────────────────────
( cd "$STAGE" && sha256sum --quiet -c SHA256SUMS ) || die "a staged file is corrupt or truncated: re-upload"
COMMIT="$(tr -d '[:space:]' < "$STAGE/repo.txt")"
if [[ "$(cat "$ROOT/.amd-commit" 2>/dev/null || true)" != "$COMMIT" ]]; then
  log "unpacking code at $COMMIT"
  tar -xzf "$STAGE/code.tar.gz" -C "$ROOT"
  printf '%s\n' "$COMMIT" > "$ROOT/.amd-commit"
fi
mkdir -p "$ROOT/data/train" "$ROOT/runs"
[[ -s "$ROOT/data/train/sft_upstream/train.jsonl" ]] || tar -xzf "$STAGE/sft_a.tar.gz" -C "$ROOT/data/train"
[[ -s "$ROOT/data/train/ja3_sft_v2.jsonl" ]] || tar -xzf "$STAGE/sft_b.tar.gz" -C "$ROOT/data/train"
for f in sft_upstream/train.jsonl sft_upstream/val.jsonl ja3_sft_v2.jsonl; do
  [[ -s "$ROOT/data/train/$f" ]] || die "missing $ROOT/data/train/$f after unpacking"
done
cp "$STAGE/tokens.json" "$ROOT/tokens.json"
install -m 600 "$STAGE/remote.env" "$ROOT/.env"
log "data: A $(wc -l < "$ROOT/data/train/sft_upstream/train.jsonl") rows, B $(wc -l < "$ROOT/data/train/ja3_sft_v2.jsonl") rows"

# shellcheck source=common.sh
source "$ROOT/ops/amd/common.sh"
amd_load_env "$ROOT/.env"
[[ -n "${HF_TOKEN:-}" ]] || die "HF_TOKEN missing from remote.env: the Hub is the only copy that outlives the droplet"
[[ -n "${AMD_HUB_NAMESPACE:-}" ]] || die "AMD_HUB_NAMESPACE missing from remote.env"

# ── 3. the container ─────────────────────────────────────────────────────────────
amd_container_up
amd_in_container bash "$ROOT/ops/amd/container_setup.sh"

# ── 5. the on-droplet watchdog ───────────────────────────────────────────────────
if ! pgrep -f 'ops/amd/watchdog.sh' >/dev/null; then
  # setsid + </dev/null: the ssh session that runs this script ends, and a watchdog that is still
  # attached to its terminal gets SIGHUP and dies with it, or holds the ssh channel open.
  setsid nohup bash "$ROOT/ops/amd/watchdog.sh" --arm >>"$LOGDIR/watchdog.log" 2>&1 </dev/null &
  log "watchdog armed (pid $!)"
fi
log "entrypoint complete at $COMMIT. Next: bash $ROOT/ops/amd/smoke.sh"
