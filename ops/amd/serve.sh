#!/usr/bin/env bash
# shellcheck source-path=SCRIPTDIR
# One vLLM server per model, every adapter MERGED into its own copy of the base, each on its own
# port, all on the one GPU.
#
#   serve.sh --wait --verify --arms A,B,AB --hub R=AdithyaSK/smoldataenvs-sft-2b-v0:8004
#   serve.sh --wait --verify --arms "" --hub R=<repo>:8004      the gate: base + the released adapter
#   serve.sh --stop
#
# Why merged, one process per model: LoRA serving does not work for Qwen3.5 on the image's vLLM
# 0.17.1 (it crashes at cuda-graph warmup, and with --enforce-eager it cannot load a peft
# all-linear adapter), so every adapter is merged (merge_adapter.py) and the merged model is served
# as an ordinary one. Five 2B engines fit one card (see AMD_KV_CACHE_GIB). They share the GPU,
# so each is slower than it would be alone, which the plan's trials-per-minute accounts for.
#
# Models and ports (the laptop's tunnel and harness use the same table, plan.py's port_order):
#   base 8000; --arms A 8001, B 8002, AB 8003; --hub NAME=owner/repo:PORT for a Hub adapter that is
#   evaluated but not trained (the released adapter, R, is 8004). A Hub adapter is downloaded first.
#
# Safety, in the order it happens:
#   1. any server still running is stopped, then the card must GIVE ITS MEMORY BACK (a killed
#      engine returns it late; starting into the gap made the base server fail once with "Engine
#      core initialization failed")
#   2. every adapter is merged and its merge_report.json checked BEFORE any server starts. A
#      directory without a passing report is re-merged, and a merge that is a no-op (it once copied
#      the base's own weights over the merged ones, so three adapters were evaluated as the base)
#      refuses to serve. `MERGE_OK model=... modules_applied=... tensors_changed=...` is printed
#      per adapter; the driver wires it into the evaluation guard
#   3. servers are started ONE AT A TIME (with --wait): each is READY before the next starts, each
#      has the same explicit KV-cache budget (AMD_KV_CACHE_GIB) and prefix caching on; one that
#      DIES during startup is restarted once after waiting for the GPU memory again (AMD_START_TRIES)
#   4. with --verify, each adapter's tool calls and its temperature-0 output on a fixed training
#      prompt against the base's are checked (probe_tools.py): identical output means the adapter
#      is not applied. `ADAPTER_CHECK model=... differs=... tool_calls_ok=...` is printed and the
#      script exits non-zero if any adapter is identical to the base or returns no tool call
#
# Flags per server, checked against the vLLM 0.17 CLI: --enable-auto-tool-choice
# --tool-call-parser qwen3_coder (AMD_TOOL_PARSER overrides), --default-chat-template-kwargs
# '{"enable_thinking": false}' (the template the SFT rows were rendered with), --host 127.0.0.1
# (NOT 0.0.0.0: an unauthenticated model server on a public IP is an open GPU; the laptop reaches
# it through ssh -L). `--tokenizer` is always the base's: a merged model is saved by the training
# venv's transformers 5, whose tokenizer files vLLM's transformers 4.x cannot read.
#
# The servers are started detached (setsid + nohup), so this script returns once they are up and
# the ssh session that ran it can end without killing them. With --wait it prints
# `READY_AFTER_S=<seconds>` per server.

set -euo pipefail
# shellcheck source=common.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
amd_load_env "$AMD_REMOTE_ROOT/.env"

PORT="$AMD_VLLM_PORT"
WAIT=0
STOP=0
VERIFY=0
ARMS=""
HUBS=()
PIDFILE="$AMD_REMOTE_LOG/vllm.pids"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --wait)   WAIT=1; shift ;;
    --verify) VERIFY=1; shift ;;
    --stop)   STOP=1; shift ;;
    --arms)   ARMS="$2"; shift 2 ;;
    --hub)    HUBS+=("$2"); shift 2 ;;
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
stop_servers   # a clean slate: nothing may be left holding the card
amd_wait_gpu_room "$AMD_GPU_FREE_MIN"

SYSPY="$(amd_syspy)"
cd "$AMD_REMOTE_ROOT"

# adapter_dir <arm>: where a trained arm's adapter is, fetching it from the Hub if the disk has none
# (the reclaim case: the adapter survived, the disk did not).
hub_download() { # hub_download <repo> [revision]: the adapter's files and the final marker, from the Hub
  "$AMD_VENV/bin/python" - "$1" "${2:-}" <<'PYDL'
import sys
from huggingface_hub import snapshot_download
print(snapshot_download(sys.argv[1], revision=sys.argv[2] or None,
                        allow_patterns=["adapter_config.json", "adapter_model.safetensors", "final.done"]))
PYDL
}
# A trained arm's adapter is served only when it is FINISHED. While an arm trains, the Trainer copies
# every checkpoint's adapter files into the arm's output directory (and pushes them to the Hub
# root), so the two adapter files alone prove nothing: the local marker `.done` (written by
# run_sft.sh after the Hub upload was verified) or, from the Hub, `final.done` (pushed last).
adapter_dir() {
  local arm="$1" dir hub
  dir="$(amd_arm_dir "$arm")"
  if [[ -f "$dir/.done" && -f "$dir/adapter_config.json" && -f "$dir/adapter_model.safetensors" ]]; then
    printf '%s\n' "$dir"; return 0
  fi
  if [[ -f "$dir/adapter_model.safetensors" ]]; then
    amd_log "arm $arm: $dir has adapter files but no .done: an unfinished run's checkpoint, not served from disk"
  fi
  hub="$(hub_download "$(amd_arm_hub_repo "$arm")")" || return 1
  [[ -f "$hub/final.done" && -f "$hub/adapter_model.safetensors" ]] || {
    amd_log "arm $arm: the Hub repo has no final.done: its adapter is a checkpoint or stale, not served"; return 1; }
  printf '%s\n' "$hub"
}

BASE="$(amd_served_name base)"
NAMES=(); SRCS=(); PORTS=()
for arm in ${ARMS//,/ }; do
  src="$(adapter_dir "$arm")" || amd_die "arm $arm has no adapter on disk or on the Hub"
  NAMES+=("$(amd_served_name "$arm")"); PORTS+=("$(amd_port "$arm")"); SRCS+=("$src")
done
for spec in "${HUBS[@]}"; do   # NAME=owner/repo:PORT
  name="${spec%%=*}"; rest="${spec#*=}"; repo="${rest%:*}"; port="${rest##*:}"
  [[ "$spec" == *=*:* && "$port" =~ ^[0-9]+$ ]] || amd_die "--hub wants NAME=owner/repo:PORT, got '$spec'"
  src="$(hub_download "$repo")" || amd_die "could not download $repo from the Hub"
  NAMES+=("$(amd_served_name "$name")"); PORTS+=("$port"); SRCS+=("$src")
done

# Every merge is made and checked BEFORE any server starts, so a refused one leaves no server.
MERGED=()
for k in "${!NAMES[@]}"; do
  merged="$AMD_REMOTE_ROOT/runs/merged_${NAMES[$k]}"
  MERGED+=("$merged")
  # A merged directory is served only with a passing merge_report.json for the weight files it
  # holds now. One that fails the check (no report, a no-op merge, files changed since) is
  # re-merged, and the merge exits non-zero if it still is not a real one.
  # ...and only a merge made FROM this adapter: the report's adapter sha256 must equal the adapter's.
  if ! "$AMD_VENV/bin/python" ops/amd/merge_adapter.py --check "$merged" --adapter "${SRCS[$k]}" >/dev/null 2>&1; then
    "$AMD_VENV/bin/python" ops/amd/merge_adapter.py \
      --base "$AMD_BASE_MODEL" --adapter "${SRCS[$k]}" --out "$merged" \
      || amd_die "merging ${NAMES[$k]} failed its checks (see merge_report.json in $merged); not serving it"
  fi
  "$AMD_VENV/bin/python" ops/amd/merge_adapter.py --check "$merged" --adapter "${SRCS[$k]}" --label "${NAMES[$k]}" \
    || amd_die "refusing to serve $merged: no passing merge_report.json"
done

# Models: index 0 is the base, then the adapters in the order given.
M_NAMES=("$BASE" "${NAMES[@]}")
M_PORTS=("$PORT" "${PORTS[@]}")
M_PATHS=("$AMD_BASE_MODEL" "${MERGED[@]}")
declare -A PIDS=()

# Memory per engine. Session 1 started five engines at once with --gpu-memory-utilization 0.17 and the
# KV caches came out at 3 to 35 GiB, because each engine sized its cache from what the others had
# taken by then. Engines now start ONE AT A TIME with an explicit, equal KV budget
# (--kv-cache-memory-bytes, which vLLM 0.17 documents as ignoring gpu_memory_utilization):
# AMD_KV_CACHE_GIB per engine; 0 falls back to equal --gpu-memory-utilization shares.
kv_args() {
  if (( AMD_KV_CACHE_GIB > 0 )); then
    printf '%s\n' "--kv-cache-memory-bytes" "$((AMD_KV_CACHE_GIB * 1024 * 1024 * 1024))"
  else
    printf '%s\n' "--gpu-memory-utilization" "$AMD_SERVER_UTIL"
  fi
}
# Prefix caching: Qwen3.5 is a hybrid (gated delta net) model, which vLLM's recipe serves with
# --enable-prefix-caching (its 'align' mamba mode, marked experimental). Session 1's hit rate was
# 0%, so every turn re-prefilled the conversation. AMD_PREFIX_ARGS="" turns it off; an engine that
# fails to start with it, and whose log names prefix caching or the mamba cache, is retried without.
declare -A NO_PREFIX=()

start_model() { # start_model <index>
  local i="$1" port="${M_PORTS[$1]}" kv=() prefix=()
  mapfile -t kv < <(kv_args)
  if [[ -z "${NO_PREFIX[$i]:-}" && -n "$AMD_PREFIX_ARGS" ]]; then read -r -a prefix <<< "$AMD_PREFIX_ARGS"; fi
  setsid nohup "$SYSPY" -m vllm.entrypoints.openai.api_server \
    --model "${M_PATHS[$i]}" --tokenizer "$AMD_BASE_MODEL" --served-model-name "${M_NAMES[$i]}" \
    --host 127.0.0.1 --port "$port" \
    --dtype bfloat16 --max-model-len "$AMD_MAX_MODEL_LEN" "${kv[@]}" "${prefix[@]}" \
    --enable-auto-tool-choice --tool-call-parser "$AMD_TOOL_PARSER" \
    --default-chat-template-kwargs '{"enable_thinking": false}' \
    >"$AMD_REMOTE_LOG/vllm_$port.log" 2>&1 < /dev/null &
  PIDS[$i]=$!
  echo "${PIDS[$i]}" >> "$PIDFILE"
  amd_log "vllm pid ${PIDS[$i]} on :$port as '${M_NAMES[$i]}' (log $AMD_REMOTE_LOG/vllm_$port.log)"
}

wait_model() { # wait_model <index>: 0 ready, 1 the process died during startup, 2 timed out
  local i="$1" port="${M_PORTS[$1]}" name="${M_NAMES[$1]}" t0 now
  t0=$(date +%s)
  while :; do
    if curl -sf "http://127.0.0.1:$port/v1/models" 2>/dev/null \
        | jq -e --arg n "$name" '.data[] | select(.id == $n)' >/dev/null 2>&1; then
      echo "READY_AFTER_S=$(( $(date +%s) - t0 )) port=$port model=$name"
      return 0
    fi
    if ! kill -0 "${PIDS[$i]}" 2>/dev/null; then
      tail -40 "$AMD_REMOTE_LOG/vllm_$port.log" >&2
      amd_log "vLLM on :$port exited during startup (last lines above)"
      return 1
    fi
    now=$(date +%s)
    if (( now - t0 > AMD_VLLM_WAIT_S )); then
      tail -40 "$AMD_REMOTE_LOG/vllm_$port.log" >&2
      kill "${PIDS[$i]}" 2>/dev/null || true
      amd_log "vLLM on :$port not ready after ${AMD_VLLM_WAIT_S}s; stopped it"
      return 2
    fi
    sleep "${AMD_READY_SLEEP:-5}"
  done
}

bring_up() { # bring_up <index>: wait for it; if it died or hung, wait for the GPU and start it again
  local i="$1" attempt rc
  for ((attempt = 1; attempt <= AMD_START_TRIES; attempt++)); do
    rc=0; wait_model "$i" || rc=$?
    (( rc == 0 )) && return 0
    if (( attempt >= AMD_START_TRIES )); then
      amd_die "${M_NAMES[$i]} on :${M_PORTS[$i]} did not start after $attempt attempts (see $AMD_REMOTE_LOG/vllm_${M_PORTS[$i]}.log)"
    fi
    kill -9 "${PIDS[$i]}" 2>/dev/null || true
    if [[ -n "$AMD_PREFIX_ARGS" && -z "${NO_PREFIX[$i]:-}" ]] \
        && grep -qiE 'prefix.cach|mamba.cache' "$AMD_REMOTE_LOG/vllm_${M_PORTS[$i]}.log" 2>/dev/null; then
      NO_PREFIX[$i]=1
      amd_log "WARNING: ${M_NAMES[$i]} failed with prefix caching named in its log: restarting WITHOUT it (every turn will re-prefill)"
    fi
    amd_log "restarting ${M_NAMES[$i]} (attempt $((attempt + 1)) of $AMD_START_TRIES) once the card has room"
    amd_wait_gpu_room "$(awk -v u="$AMD_SERVER_UTIL" 'BEGIN { printf "%.2f", u + 0.03 }')"
    start_model "$i"
  done
}

# One at a time: with --wait each engine is READY before the next starts. Without --wait there is
# nothing to wait on, so they are launched back to back.
for i in "${!M_NAMES[@]}"; do
  start_model "$i"
  if (( WAIT )); then bring_up "$i"; fi
done

if (( WAIT && VERIFY )); then
  # Each adapter's tool calls, then its temperature-0 output on a fixed training prompt against the
  # base's: identical output means the adapter is not applied (probe_tools prints
  # ADAPTER_CHECK ... differs=0). Every adapter is checked, and the script fails if any is not.
  ROWS="$AMD_DATA_ROOT/train/sft_upstream/train.jsonl"
  bad=0
  for k in "${!NAMES[@]}"; do
    i=$((k + 1))
    out="$("$AMD_VENV/bin/python" -m ops.amd.probe_tools --port "${M_PORTS[$i]}" --models "${M_NAMES[$i]}" \
      --base-port "$PORT" --base-model "$BASE" --train-rows "$ROWS")" || true
    printf '%s\n' "$out"
    grep -q "^ADAPTER_CHECK model=${M_NAMES[$i]} differs=1 tool_calls_ok=1" <<<"$out" || bad=$((bad + 1))
  done
  (( bad == 0 )) || amd_die "$bad adapter(s) failed the check (output identical to the base's, or no tool call): do not evaluate"
fi
