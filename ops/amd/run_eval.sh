#!/usr/bin/env bash
# Serve a model (base, or base + one LoRA adapter) and run the ladder over it with a run tag.
#
#   run_eval.sh --arm A|B|AB|base [--limit N] [--rungs L1,L2,L3,L4] [--serve-only]
#
# Arm 0 (the base control) is `--arm base`: Qwen3.5-2B is a *base* model with no instruction
# tuning for either protocol, and its score is the floor every other arm is measured against. It is
# evaluated under `--agent program`, the one protocol upstream ever reports a number for.
#
# The adapter arms are evaluated under `--agent bash`, because that is the protocol they were
# trained in. Measuring an SFT arm under a protocol it was not trained for measures the protocol
# (see docs/LOCAL_MODELS.md), so the agent is chosen per arm here rather than by the caller.

set -euo pipefail
# shellcheck source=common.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

ARM=""
LIMIT="${AMD_EVAL_LIMIT:-}"
RUNGS="${AMD_EVAL_RUNGS:-L1,L1+schema,L2,L3,L4}"
WORKERS="${AMD_EVAL_WORKERS:-32}"
MAX_TURNS="${AMD_EVAL_MAX_TURNS:-16}"
SERVE_ONLY=0
PORT="${AMD_VLLM_PORT:-8000}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --arm)        ARM="$2"; shift 2 ;;
    --limit)      LIMIT="$2"; shift 2 ;;
    --rungs)      RUNGS="$2"; shift 2 ;;
    --workers)    WORKERS="$2"; shift 2 ;;
    --serve-only) SERVE_ONLY=1; shift ;;
    *) amd_die "unknown argument '$1'" ;;
  esac
done
[[ -n "$ARM" ]] || amd_die "--arm is required (base, A, B or AB)"

if (( SERVE_ONLY )); then
  exec "$AMD_REMOTE_ROOT/ops/amd/serve.sh" --arm "$ARM" --port "$PORT"
fi

amd_load_env
TAG="$(amd_run_tag "$ARM")"
BASE_URL="http://127.0.0.1:${PORT}/v1"
SERVED="$(amd_eval_model_name "$ARM")"
SERVE_LOG="$AMD_REMOTE_LOG/vllm_${ARM}.log"

if [[ "$ARM" == "base" ]]; then
  AGENT="program"
  MAX_TURNS=1
else
  AGENT="bash"
fi

# The server is started, given a bounded time to come up, and killed afterwards whether the sweep
# passed or failed. A leaked vLLM holding 192 GB is the one failure that makes the *next* step of a
# multi-arm run impossible, so the trap is unconditional.
"$AMD_REMOTE_ROOT/ops/amd/serve.sh" --arm "$ARM" --port "$PORT" >"$SERVE_LOG" 2>&1 &
SERVER_PID=$!
cleanup() {
  if kill -0 "$SERVER_PID" 2>/dev/null; then
    amd_log "stopping vLLM (pid $SERVER_PID)"
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

amd_log "waiting for vLLM on $BASE_URL"
READY=0
for _ in $(seq 1 "${AMD_VLLM_WAIT_S:-180}"); do
  if curl -sf "$BASE_URL/v1/models" >/dev/null 2>&1; then READY=1; break; fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    amd_log "vLLM exited during startup; last lines of $SERVE_LOG:"
    tail -20 "$SERVE_LOG" >&2
    exit 1
  fi
  sleep 5
done
(( READY )) || { amd_log "vLLM did not become ready; see $SERVE_LOG"; exit 1; }
amd_log "vLLM ready, serving '$SERVED'"

cd "$AMD_REMOTE_ROOT"
# The endpoint is a loopback URL, so per docs/LOCAL_MODELS.md it needs no API key, and the
# non-thinking template is mandatory: Qwen3.5-2B will not stop generating without it.
export SMOL_LADDER_BASE_URL="$BASE_URL"
export SMOL_LADDER_API_KEY_ENV="OPENROUTER_API_KEY"
export SMOL_LADDER_CHAT_TEMPLATE_KWARGS='{"enable_thinking": false}'

CMD=("$AMD_REMOTE_ROOT/.venv/bin/python" -m smol_ladder.run_ladder
     --split "$AMD_SPLIT"
     --run-tag "$TAG"
     --model "$SERVED"
     --agent "$AGENT"
     --rungs "$RUNGS"
     --workers "$WORKERS"
     --max-turns "$MAX_TURNS")
[[ -n "$LIMIT" ]] && CMD+=(--limit "$LIMIT")

SWEEP_LOG="$AMD_REMOTE_LOG/eval_${ARM}.log"
amd_log "${CMD[*]}"
set +e
"${CMD[@]}" 2>&1 | tee -a "$SWEEP_LOG"
STATUS=${PIPESTATUS[0]}
set -e
amd_log "ladder exit $STATUS; log $SWEEP_LOG"
exit "$STATUS"
