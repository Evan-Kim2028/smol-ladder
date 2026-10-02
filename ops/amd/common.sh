#!/usr/bin/env bash
# Shared settings for every script that runs ON THE DROPLET. Sourced, never executed.
# Everything is an overridable default; none of it is read on the laptop (the laptop's numbers
# live in plan.py, and a test keeps the two in step).

set -euo pipefail

# ── paths ────────────────────────────────────────────────────────────────────────
AMD_REMOTE_ROOT="${AMD_REMOTE_ROOT:-/opt/smol-ladder}"
AMD_REMOTE_LOG="${AMD_REMOTE_LOG:-/var/log/smol-ladder}"
AMD_STAGE_DIR="${AMD_STAGE_DIR:-/var/tmp/smol-ladder-stage}"
AMD_DATA_ROOT="${AMD_DATA_ROOT:-$AMD_REMOTE_ROOT/data}"
AMD_VENV="${AMD_VENV:-$AMD_REMOTE_ROOT/.venv}"

# ── models, data, training ───────────────────────────────────────────────────────
AMD_BASE_MODEL="${AMD_BASE_MODEL:-Qwen/Qwen3.5-2B}"
AMD_MAX_LENGTH="${AMD_MAX_LENGTH:-8192}"
AMD_SEED="${AMD_SEED:-42}"
AMD_LORA_R="${AMD_LORA_R:-16}"
# Effective batch is held at upstream's 8 sequences per optimizer step so arm A stays comparable
# to the published recipe; the smoke picks how that 8 is split into batch x accumulation.
AMD_EFFECTIVE_BATCH="${AMD_EFFECTIVE_BATCH:-8}"
# train/sft_lora.py saves every max(50, max_steps // 2) steps and, with no --max-steps, every
# 100: at 8 sequences a step that is a checkpoint every few minutes, each pushed to the Hub by
# hub_strategy="every_save". It has no flag for this; the bound is stated in the runbook.

# ── the Hub: the only copy of anything that outlives the droplet ─────────────────
AMD_HUB_NAMESPACE="${AMD_HUB_NAMESPACE:-}"
AMD_HUB_ADAPTER_A="${AMD_HUB_ADAPTER_A:-smol-ladder-sft-a}"
AMD_HUB_ADAPTER_B="${AMD_HUB_ADAPTER_B:-smol-ladder-sft-b}"
AMD_HUB_ADAPTER_AB="${AMD_HUB_ADAPTER_AB:-smol-ladder-sft-ab}"
AMD_HUB_ARTIFACTS="${AMD_HUB_ARTIFACTS:-smol-ladder-runs}"

# ── serving ──────────────────────────────────────────────────────────────────────
AMD_VLLM_PORT="${AMD_VLLM_PORT:-8000}"
AMD_VLLM_WAIT_S="${AMD_VLLM_WAIT_S:-900}"
AMD_MAX_MODEL_LEN="${AMD_MAX_MODEL_LEN:-16384}"
# One server for four models: the base is 4.6 GB and an adapter is ~90 MB, so the rest of the
# card is KV cache, which is what makes four concurrent sweeps fast.
# That 85% is why the probe server MUST be stopped before any arm trains (the plan has an explicit
# stop step and run_sft.sh stops it again and then asserts the memory is free).
AMD_GPU_UTIL="${AMD_GPU_UTIL:-0.85}"
# qwen3_coder is what the vLLM Qwen3.5 recipe names. If the smoke probe reports raw <tool_call> text,
# switch the parser here (for example `hermes`) without editing the script.
AMD_TOOL_PARSER="${AMD_TOOL_PARSER:-qwen3_coder}"

amd_log() { printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*" >&2; }
amd_die() { amd_log "FATAL: $*"; exit 1; }

# Load the secrets file the stage step wrote (HF_TOKEN, AMD_HUB_NAMESPACE). Never printed.
amd_load_env() {
  local env_file="${1:-$AMD_REMOTE_ROOT/.env}"
  if [[ -f "$env_file" ]]; then
    set -a
    # shellcheck disable=SC1090  # runtime path, not a literal
    source "$env_file"
    set +a
  fi
}

# The python that owns torch and vLLM (the image's). Recorded by the entrypoint.
amd_syspy() {
  if [[ -f "$AMD_REMOTE_ROOT/.syspy" ]]; then cat "$AMD_REMOTE_ROOT/.syspy"; else command -v python3; fi
}

amd_lower() { printf '%s' "$1" | tr '[:upper:]' '[:lower:]'; }

amd_arm_dir() { printf '%s/runs/sft_%s\n' "$AMD_REMOTE_ROOT" "$(amd_lower "$1")"; }

amd_arm_hub_repo() {
  local ns="${AMD_HUB_NAMESPACE:?AMD_HUB_NAMESPACE is not set (remote.env)}"
  case "$1" in
    A)  printf '%s/%s\n' "$ns" "$AMD_HUB_ADAPTER_A" ;;
    B)  printf '%s/%s\n' "$ns" "$AMD_HUB_ADAPTER_B" ;;
    AB) printf '%s/%s\n' "$ns" "$AMD_HUB_ADAPTER_AB" ;;
    *)  amd_die "unknown arm '$1' (want A, B or AB)" ;;
  esac
}

# The name the server exposes each model under; plan.py's served_name() must agree.
amd_served_name() {
  case "$1" in
    base) printf 'amd-base-2b\n' ;;
    probe) printf 'amd-probe-2b\n' ;;
    A|B|AB) printf 'amd-%s-2b\n' "$(amd_lower "$1")" ;;
    *) amd_die "unknown model '$1'" ;;
  esac
}

# ── GPU memory must be free before an arm trains ────────────────────────────────
# The smoke's probe server holds ~85% of the card (AMD_GPU_UTIL) and the benchmark picked its batch
# size on an EMPTY card: training beside a live server would run on ~15% of the memory with a batch
# size nobody measured, and a live vLLM also counts as "work" for the idle watchdog.
AMD_GPU_FREE_MIN="${AMD_GPU_FREE_MIN:-0.90}"      # fraction of the card that must be free

amd_gpu_free_fraction() {
  "$AMD_VENV/bin/python" -c 'import torch; free, total = torch.cuda.mem_get_info(); print(free / total)'
}

amd_assert_gpu_free() {
  local tries="${AMD_GPU_FREE_TRIES:-12}" i frac=""
  for ((i = 1; i <= tries; i++)); do
    frac="$(amd_gpu_free_fraction 2>/dev/null || true)"
    if [[ -n "$frac" ]] && awk -v f="$frac" -v m="$AMD_GPU_FREE_MIN" 'BEGIN { exit !(f >= m) }'; then
      amd_log "GPU memory is free (${frac} of the card >= ${AMD_GPU_FREE_MIN})"
      return 0
    fi
    amd_log "GPU memory not free yet (free fraction '${frac:-unreadable}', need ${AMD_GPU_FREE_MIN}); try $i/$tries"
    sleep "${AMD_GPU_FREE_SLEEP:-5}"
  done
  pgrep -af 'vllm|sft_lora' >&2 || true
  amd_die "GPU memory is not free (free fraction '${frac:-unreadable}', need ${AMD_GPU_FREE_MIN}): something still holds the card. Refusing to train beside it."
}

# apt-get with a lock timeout and a bounded retry. A fresh image runs unattended-upgrades and
# cloud-init in the first minutes, and under `set -e` a plain apt-get dies on the dpkg lock.
amd_apt() {
  local n tries="${AMD_APT_TRIES:-5}"
  for ((n = 1; n <= tries; n++)); do
    if apt-get -o DPkg::Lock::Timeout="${AMD_APT_LOCK_S:-180}" "$@"; then return 0; fi
    amd_log "apt-get $* failed (attempt $n/$tries)"
    sleep "${AMD_APT_SLEEP:-10}"
  done
  return 1
}
