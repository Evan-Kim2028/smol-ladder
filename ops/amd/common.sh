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

# ── the container: torch and vLLM live in an image, not on the host ──────────────
# ONE long-lived container runs every droplet-side step, with the same paths bind-mounted, so a
# script behaves the same inside it as the layout in this file says. Host-only: the watchdog,
# rocm-smi, poweroff. The image's jupyter container is stopped so nothing else holds the GPU.
AMD_CONTAINER="${AMD_CONTAINER:-smol}"
AMD_IMAGE="${AMD_IMAGE:-vllm/vllm-openai-rocm:v0.17.1}"   # already on disk; never pulled
AMD_HF_CACHE="${AMD_HF_CACHE:-/var/cache/smol-hf}"         # host disk, mounted as the container's HF cache
AMD_STOP_CONTAINERS="${AMD_STOP_CONTAINERS:-rocm}"

# ── models, data, training ───────────────────────────────────────────────────────
AMD_BASE_MODEL="${AMD_BASE_MODEL:-Qwen/Qwen3.5-2B}"
AMD_MAX_LENGTH="${AMD_MAX_LENGTH:-8192}"
AMD_SEED="${AMD_SEED:-42}"
AMD_LORA_R="${AMD_LORA_R:-16}"
# Effective batch is held at upstream's 8 sequences per optimizer step so arm A stays comparable
# to the published recipe; the smoke picks how that 8 is split into batch x accumulation.
AMD_EFFECTIVE_BATCH="${AMD_EFFECTIVE_BATCH:-8}"
# Checkpoints: run_sft.sh --save-steps (default 50) through ops/amd/sft_run.py, because
# train/sft_lora.py has no flag for it. Each save is pushed to the Hub by hub_strategy="checkpoint" (as `last-checkpoint/`).

# ── the Hub: the only copy of anything that outlives the droplet ─────────────────
AMD_HUB_NAMESPACE="${AMD_HUB_NAMESPACE:-}"
AMD_HUB_ADAPTER_A="${AMD_HUB_ADAPTER_A:-smol-ladder-sft-a}"
AMD_HUB_ADAPTER_B="${AMD_HUB_ADAPTER_B:-smol-ladder-sft-b}"
AMD_HUB_ADAPTER_AB="${AMD_HUB_ADAPTER_AB:-smol-ladder-sft-ab}"
AMD_HUB_ARTIFACTS="${AMD_HUB_ARTIFACTS:-smol-ladder-runs}"

# ── serving ──────────────────────────────────────────────────────────────────────
# One vLLM process per model, each serving a MERGED checkpoint (LoRA serving does not work for this
# model on vLLM 0.17.1: it crashes at cuda-graph warmup, and with --enforce-eager it cannot load a
# peft all-linear adapter). Ports are fixed by model, so the laptop's tunnel and harness and these
# scripts cannot disagree (plan.py's port_order is the other copy; a test keeps the two in step):
# base 8000, A 8001, B 8002, AB 8003, then the Hub adapters in the order they are given (R 8004).
AMD_VLLM_PORT="${AMD_VLLM_PORT:-8000}"
AMD_VLLM_WAIT_S="${AMD_VLLM_WAIT_S:-900}"
AMD_MAX_MODEL_LEN="${AMD_MAX_MODEL_LEN:-16384}"
# Five 2B engines share one card (288 GB): each gets the same explicit KV-cache budget, and they
# start one at a time (serve.sh explains why: session 1's per-engine caches came out 3 to 35 GiB).
# 24 GiB of KV + about 5 GiB of weights and 10 of graphs/activations is ~40 GB an engine, ~200 GB
# for five, which leaves ~85 GB of headroom; a 16k-token context is a few hundred MiB of KV for
# this model, so 24 GiB is far more than the 8 workers per model can use. 0 selects the older
# equal --gpu-memory-utilization shares (AMD_SERVER_UTIL, 0.17 each: measured 0.15-0.2 works).
AMD_KV_CACHE_GIB="${AMD_KV_CACHE_GIB:-24}"
AMD_SERVER_UTIL="${AMD_SERVER_UTIL:-0.17}"
AMD_PREFIX_ARGS="${AMD_PREFIX_ARGS---enable-prefix-caching}"
# An engine can fail to start ("Engine core initialization failed", a cuda-graph capture assertion)
# when the servers that were just killed have not given the card back yet. The start waits for the
# memory first and retries a failed engine once after waiting again.
AMD_START_TRIES="${AMD_START_TRIES:-2}"
AMD_GPU_ROOM_WAIT_S="${AMD_GPU_ROOM_WAIT_S:-180}"
# qwen3_coder is what the vLLM Qwen3.5 recipe names. If the probe reports raw <tool_call> text,
# switch the parser here (for example `hermes`) without editing the script.
AMD_TOOL_PARSER="${AMD_TOOL_PARSER:-qwen3_coder}"
# The released upstream adapter: evaluated, never trained, and the gate's subject.
AMD_HUB_R="${AMD_HUB_R:-AdithyaSK/smoldataenvs-sft-2b-v0}"

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
    [A-Z0-9]*) printf 'amd-%s-2b\n' "$(amd_lower "$1")" ;;
    *) amd_die "unknown model '$1'" ;;
  esac
}

# The fixed port of a model trained here (Hub adapters are given theirs explicitly).
amd_port() {
  case "$1" in
    base) printf '%s\n' "$AMD_VLLM_PORT" ;;
    A)    printf '%s\n' "$((AMD_VLLM_PORT + 1))" ;;
    B)    printf '%s\n' "$((AMD_VLLM_PORT + 2))" ;;
    AB)   printf '%s\n' "$((AMD_VLLM_PORT + 3))" ;;
    *)    amd_die "no fixed port for '$1'" ;;
  esac
}

# ── GPU memory must be free before an arm trains or a server starts ─────────────
# The servers hold most of the card and the benchmark picked its batch size on an EMPTY card:
# training beside a live server would run on a sliver of the memory with a batch size nobody
# measured, and a live vLLM also counts as "work" for the idle watchdog.
AMD_GPU_FREE_MIN="${AMD_GPU_FREE_MIN:-0.90}"      # fraction of the card that must be free

amd_gpu_free_fraction() {
  "$AMD_VENV/bin/python" -c 'import torch; free, total = torch.cuda.mem_get_info(); print(free / total)'
}

# amd_wait_gpu_room <fraction> [seconds]: poll until at least that fraction of the card is free.
# Used after servers are killed and before the next one starts: a killed engine returns its memory
# a few seconds late, and starting into the gap is what made the base server fail once.
amd_wait_gpu_room() {
  local need="$1" limit="${2:-$AMD_GPU_ROOM_WAIT_S}" t0 frac=""
  t0=$(date +%s)
  while :; do
    frac="$(amd_gpu_free_fraction 2>/dev/null || true)"
    if [[ -n "$frac" ]] && awk -v f="$frac" -v m="$need" 'BEGIN { exit !(f >= m) }'; then
      amd_log "GPU room: ${frac} of the card is free (need ${need})"
      return 0
    fi
    if (( $(date +%s) - t0 >= limit )); then
      pgrep -af 'vllm|sft_lora|sft_run' >&2 || true
      amd_die "GPU memory not released after ${limit}s (free fraction '${frac:-unreadable}', need ${need})"
    fi
    amd_log "waiting for GPU memory to be released (free '${frac:-unreadable}', need ${need})"
    sleep "${AMD_GPU_FREE_SLEEP:-5}"
  done
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
  pgrep -af 'vllm|sft_lora|sft_run' >&2 || true
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

# ── host side: the container ─────────────────────────────────────────────────────
# Idempotent: a running container on the right image is kept; anything else is replaced.
amd_container_up() {
  local c
  for c in $AMD_STOP_CONTAINERS; do docker stop "$c" >/dev/null 2>&1 || true; done
  if [[ "$(docker inspect -f '{{.State.Running}} {{.Config.Image}}' "$AMD_CONTAINER" 2>/dev/null || true)" == "true $AMD_IMAGE" ]]; then
    amd_log "container $AMD_CONTAINER already running"; return 0
  fi
  docker rm -f "$AMD_CONTAINER" >/dev/null 2>&1 || true
  mkdir -p "$AMD_HF_CACHE" "$AMD_REMOTE_LOG" "$AMD_REMOTE_ROOT"
  docker image inspect "$AMD_IMAGE" >/dev/null 2>&1 || amd_die "image $AMD_IMAGE is not on this host (no pulls here)"
  # --network host: vLLM on 127.0.0.1:8000 is what the laptop's ssh -L reaches. --init reaps the
  # detached servers. seccomp=unconfined + SYS_PTRACE + video/render are what ROCm wants.
  docker run -d --name "$AMD_CONTAINER" --restart no --init --network host --ipc host \
    --device /dev/kfd --device /dev/dri --group-add "$(getent group video | cut -d: -f3)" --group-add "$(getent group render | cut -d: -f3)" \
    --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
    -v "$AMD_REMOTE_ROOT:$AMD_REMOTE_ROOT" -v "$AMD_STAGE_DIR:$AMD_STAGE_DIR" \
    -v "$AMD_REMOTE_LOG:$AMD_REMOTE_LOG" -v "$AMD_HF_CACHE:/root/.cache/huggingface" \
    -w "$AMD_REMOTE_ROOT" --entrypoint sleep "$AMD_IMAGE" infinity >/dev/null
  amd_log "container $AMD_CONTAINER started from $AMD_IMAGE"
}

# Run a command in the container. Secrets are never passed here: scripts read $AMD_REMOTE_ROOT/.env.
amd_in_container() { docker exec "$AMD_CONTAINER" "$@"; }
