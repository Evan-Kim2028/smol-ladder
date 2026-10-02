#!/usr/bin/env bash
# Serve the base model, or the base model with one LoRA adapter attached, inside the ROCm
# container. Kept separate from run_eval.sh so `--serve-only` can start a server the driver
# points the laptop's harness at, and so the serve command has one definition rather than two.
#
#   serve.sh --arm A|B|AB|base [--port 8000]
#
# The LoRA flags are the reason this exists. An adapter repo id is not a model: pointing vLLM at
# `smol-ladder-sft-a` alone serves an adapter directory and fails. The base is always loaded and
# the adapter is attached to it with --enable-lora/--lora-modules, and the arm is addressed by
# --served-model-name, never by the repo id.

set -euo pipefail
# shellcheck source=common.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

ARM=""
PORT="${AMD_VLLM_PORT:-8000}"
MAX_MODEL_LEN="${AMD_MAX_MODEL_LEN:-16384}"
GPU_UTIL="${AMD_GPU_UTIL:-0.90}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --arm)   ARM="$2"; shift 2 ;;
    --port)  PORT="$2"; shift 2 ;;
    *) amd_die "unknown argument '$1'" ;;
  esac
done
[[ -n "$ARM" ]] || amd_die "--arm is required"

amd_load_env
cd "$AMD_REMOTE_ROOT"

# vLLM on ROCm is built against a specific ROCm; the container is what pins it. HIP_VISIBLE_DEVICES
# is the ROCm spelling -- vLLM v0.30 removed the CUDA_VISIBLE_DEVICES fallback on ROCm, so the CUDA
# name is silently ignored there.
RUN=(docker exec -e HF_TOKEN="${HF_TOKEN:-}"
     -e HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
     -e VLLM_USE_TRITON_FLASH_ATTN="${VLLM_USE_TRITON_FLASH_ATTN:-0}"
     -p "${PORT}:${PORT}"
     -w "$AMD_REMOTE_ROOT"
     "${AMD_ROCM_CONTAINER:-smol-rocm}"
     vllm serve "$AMD_BASE_MODEL")

if [[ "$ARM" == "base" ]]; then
  SERVED="amd-base-2b"
else
  HUB_REPO="$(amd_arm_hub_repo "$ARM")"
  LOCAL="$AMD_REMOTE_ROOT/runs/sft_$(printf '%s' "$ARM" | tr '[:upper:]' '[:lower:]')"
  SRC="$LOCAL"
  [[ -d "$LOCAL" ]] || SRC="$HUB_REPO"
  [[ -e "$SRC/adapter_config.json" ]] \
    || amd_die "no adapter for arm $ARM at $LOCAL or in $HUB_REPO. Train it first."
  SERVED="$(amd_eval_model_name "$ARM")"
  RUN+=(--served-model-name "$SERVED"
        --enable-lora
        --max-lora-rank 16
        --lora-modules "${SERVED}=${SRC}")
fi

RUN+=(--served-model-name "${SERVED}"
      --host 0.0.0.0
      --port "$PORT"
      --max-model-len "$MAX_MODEL_LEN"
      --gpu-memory-utilization "$GPU_UTIL"
      --dtype bfloat16
      # bash-protocol arms submit through a tool call, so auto tool choice and the Qwen parser
      # are what turn the model's raw output into a tool_call the harness can run.
      --enable-auto-tool-choice
      --tool-call-parser qwen3_coder
      --default-chat-template-kwargs '{"enable_thinking": false}')

exec "${RUN[@]}"
