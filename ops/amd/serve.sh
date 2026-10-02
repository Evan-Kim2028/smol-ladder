#!/usr/bin/env bash
# shellcheck source-path=SCRIPTDIR
# ONE vLLM server carrying the base model and every adapter, so four models are evaluated
# concurrently against a single process.
#
#   serve.sh --all --wait         base + adapters A, B, AB   (the real evaluation)
#   serve.sh --probe --wait       base + the smoke's adapter  (proves LoRA + tool calls first)
#   serve.sh --all --merged --wait   fallback: merge each adapter, one server per model
#   serve.sh --stop
#
# This is the single largest saving in the plan: four server starts would pay model load, kernel
# autotune and KV-cache allocation four times over. vLLM serves LoRA adapters as named modules on
# one loaded base; the four harness processes on the laptop differ only in the `model` string.
#
# Flags, checked against the vLLM 0.17 CLI (`vllm serve --help`):
#   --enable-lora --lora-modules name=path   adapters attached to the base, which stays addressable
#   --max-loras N                            adapters in ONE batch; N = number of adapters, so the
#                                            concurrent sweeps are not serialised
#   --max-lora-rank 16                       the rank they were trained at
#   --enable-auto-tool-choice --tool-call-parser qwen3_coder   (AMD_TOOL_PARSER overrides)
#                                            what turns Qwen3.5's output into a `tool_call`; the
#                                            smoke probe checks it before any arm is trained
#   --default-chat-template-kwargs '{"enable_thinking": false}'
#                                            the template the SFT rows were rendered with
#   --host 127.0.0.1                         NOT 0.0.0.0: an unauthenticated model server on a
#                                            public IP is an open GPU; the laptop reaches it
#                                            through ssh -L
# `HIP_VISIBLE_DEVICES`, not `CUDA_VISIBLE_DEVICES`, is the ROCm spelling, and one GPU needs neither.
#
# The server is started detached (setsid + nohup), so this script returns once it is up and the
# ssh session that ran it can end without killing it. With --wait it prints
# `READY_AFTER_S=<seconds>` and, for --probe, the result of probe_tools.py.

set -euo pipefail
# shellcheck source=common.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
amd_load_env "$AMD_REMOTE_ROOT/.env"

PORT="$AMD_VLLM_PORT"
MODE=""        # all | probe
WAIT=0
STOP=0
MERGED=0
ARMS="A,B,AB"
PIDFILE="$AMD_REMOTE_LOG/vllm.pids"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --all)    MODE=all; shift ;;
    --probe)  MODE=probe; shift ;;
    --wait)   WAIT=1; shift ;;
    --merged) MERGED=1; shift ;;
    --stop)   STOP=1; shift ;;
    --arms)   ARMS="$2"; shift 2 ;;
    --port)   PORT="$2"; shift 2 ;;
    *) amd_die "unknown argument '$1'" ;;
  esac
done
mkdir -p "$AMD_REMOTE_LOG"

stop_servers() {
  if [[ -f "$PIDFILE" ]]; then
    while read -r pid; do
      if [[ -n "$pid" ]]; then kill "$pid" 2>/dev/null || true; fi
    done < "$PIDFILE"
    sleep 3
    while read -r pid; do
      if [[ -n "$pid" ]]; then kill -9 "$pid" 2>/dev/null || true; fi
    done < "$PIDFILE"
    rm -f "$PIDFILE"
  fi
  # Stragglers: a server whose pidfile is gone (an earlier run, a lost log directory) still holds
  # the card. Sweep by command line, then wait until none is left.
  local i
  pkill -f 'vllm\.entrypoints\.openai\.api_server' 2>/dev/null || true
  for ((i = 0; i < 20; i++)); do
    pgrep -f 'vllm\.entrypoints\.openai\.api_server' >/dev/null 2>&1 || return 0
    if (( i == 10 )); then pkill -9 -f 'vllm\.entrypoints\.openai\.api_server' 2>/dev/null || true; fi
    sleep 2
  done
  amd_log "WARNING: a vLLM process is still alive after the stop"
}
if (( STOP )); then stop_servers; amd_log "servers stopped"; exit 0; fi
[[ -n "$MODE" ]] || amd_die "give --all or --probe (or --stop)"
stop_servers   # one server at a time: the probe's must be gone before the real one starts

SYSPY="$(amd_syspy)"
cd "$AMD_REMOTE_ROOT"

# adapter_dir <arm>: where its adapter is, fetching it from the Hub if the disk has none (the
# reclaim case: the adapter survived, the disk did not).
adapter_dir() {
  local arm="$1" dir
  dir="$(amd_arm_dir "$arm")"
  if [[ -f "$dir/adapter_config.json" && -f "$dir/adapter_model.safetensors" ]]; then
    printf '%s\n' "$dir"; return 0
  fi
  "$AMD_VENV/bin/python" - "$(amd_arm_hub_repo "$arm")" <<'PYDL'
import sys
from huggingface_hub import snapshot_download
print(snapshot_download(sys.argv[1], allow_patterns=["adapter_config.json", "adapter_model.safetensors"]))
PYDL
}

NAMES=(); PATHS=()
if [[ "$MODE" == "probe" ]]; then
  [[ -f "$AMD_REMOTE_ROOT/runs/resume_check/adapter_model.safetensors" ]] \
    || amd_die "no probe adapter: the smoke's kill-and-resume leaves it at runs/resume_check"
  NAMES+=("$(amd_served_name probe)"); PATHS+=("$AMD_REMOTE_ROOT/runs/resume_check")
else
  for arm in ${ARMS//,/ }; do
    NAMES+=("$(amd_served_name "$arm")"); PATHS+=("$(adapter_dir "$arm")") \
      || amd_die "arm $arm has no adapter on disk or on the Hub"
  done
fi

start_server() { # start_server <port> <model path> <served name> <gpu util> [lora args...]
  local port="$1" model="$2" name="$3" util="$4"; shift 4
  setsid nohup "$SYSPY" -m vllm.entrypoints.openai.api_server \
    --model "$model" --served-model-name "$name" --host 127.0.0.1 --port "$port" \
    --dtype bfloat16 --max-model-len "$AMD_MAX_MODEL_LEN" --gpu-memory-utilization "$util" \
    --enable-auto-tool-choice --tool-call-parser "$AMD_TOOL_PARSER" \
    --default-chat-template-kwargs '{"enable_thinking": false}' "$@" \
    >"$AMD_REMOTE_LOG/vllm_$port.log" 2>&1 < /dev/null &
  echo $! >> "$PIDFILE"
  amd_log "vllm pid $! on :$port as '$name' (log $AMD_REMOTE_LOG/vllm_$port.log)"
}

wait_ready() { # wait_ready <port> <expected name>...
  local port="$1"; shift
  local t0 now name ok
  t0=$(date +%s)
  while :; do
    ok=1
    for name in "$@"; do
      curl -sf "http://127.0.0.1:$port/v1/models" 2>/dev/null | jq -e --arg n "$name" '.data[] | select(.id == $n)' >/dev/null 2>&1 || ok=0
    done
    (( ok )) && break
    now=$(date +%s)
    if ! pgrep -f "vllm.entrypoints.openai.api_server.*--port $port" >/dev/null; then
      tail -40 "$AMD_REMOTE_LOG/vllm_$port.log" >&2
      amd_die "vLLM on :$port exited during startup (last lines above). Common causes are in the runbook failure playbook."
    fi
    if (( now - t0 > AMD_VLLM_WAIT_S )); then
      tail -40 "$AMD_REMOTE_LOG/vllm_$port.log" >&2
      stop_servers; amd_die "vLLM on :$port not ready after ${AMD_VLLM_WAIT_S}s; stopped it"
    fi
    sleep 5
  done
  echo "READY_AFTER_S=$(( $(date +%s) - t0 ))"
}

BASE="$(amd_served_name base)"
if (( MERGED )); then
  i=0; ports=("$PORT")
  start_server "$PORT" "$AMD_BASE_MODEL" "$BASE" 0.2
  for k in "${!NAMES[@]}"; do
    i=$((i + 1)); p=$((PORT + i)); ports+=("$p")
    merged="$AMD_REMOTE_ROOT/runs/merged_${NAMES[$k]}"
    [[ -f "$merged/config.json" ]] || "$AMD_VENV/bin/python" ops/amd/merge_adapter.py \
      --base "$AMD_BASE_MODEL" --adapter "${PATHS[$k]}" --out "$merged"
    start_server "$p" "$merged" "${NAMES[$k]}" 0.2
  done
  (( WAIT )) && { wait_ready "$PORT" "$BASE"; for k in "${!NAMES[@]}"; do wait_ready "${ports[$((k + 1))]}" "${NAMES[$k]}"; done; }
else
  LORA=()
  for k in "${!NAMES[@]}"; do LORA+=("${NAMES[$k]}=${PATHS[$k]}"); done
  start_server "$PORT" "$AMD_BASE_MODEL" "$BASE" "$AMD_GPU_UTIL" \
    --enable-lora --max-loras "${#NAMES[@]}" --max-lora-rank "$AMD_LORA_R" --lora-modules "${LORA[@]}"
  (( WAIT )) && wait_ready "$PORT" "$BASE" "${NAMES[@]}"
fi

if (( WAIT )) && [[ "$MODE" == "probe" ]]; then
  if (( MERGED )); then   # the adapter's server first: the driver reads the first TOOL_CALLS_OK line
    "$AMD_VENV/bin/python" -m ops.amd.probe_tools --port "$((PORT + 1))" --models "${NAMES[0]}"
    "$AMD_VENV/bin/python" -m ops.amd.probe_tools --port "$PORT" --models "$BASE"
  else
    "$AMD_VENV/bin/python" -m ops.amd.probe_tools --port "$PORT" --models "${NAMES[0]}" "$BASE"
  fi
fi
