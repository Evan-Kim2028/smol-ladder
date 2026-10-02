#!/usr/bin/env bash
# shellcheck source-path=SCRIPTDIR
# Prove that resume works by actually interrupting a run, once, before any arm is trained.
#
#   resume_check.sh [--out resume.json]
#
# "A spot reclaim costs minutes, not hours" needs two things nobody has observed yet: that a
# checkpoint lands on disk within a few steps, and that the next start picks it up instead of
# starting over. So: train a few dozen steps, SIGKILL the process the moment the first checkpoint
# exists (no handler, no flush: the failure being modelled), then run the SAME decision logic the
# real arms use (`ops.amd.resume status`: prune partial checkpoints, then resume) and the SAME
# trainer with --resume, and check where it restarted.
#
# train/sft_lora.py saves every max(50, max_steps // 2) steps, so a 60-step run is the smallest
# one that checkpoints at step 50 and still has something left to do. (The real arms checkpoint
# every --save-steps through ops/amd/sft_run.py; this check calls the trainer directly, which is
# enough to prove the restart logic.)

set -euo pipefail
# shellcheck source=common.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
amd_load_env "$AMD_REMOTE_ROOT/.env"

OUT="$AMD_REMOTE_LOG/resume.json"
STEPS="${AMD_RESUME_CHECK_STEPS:-60}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --out)   OUT="$2"; shift 2 ;;
    --steps) STEPS="$2"; shift 2 ;;
    *) amd_die "unknown argument '$1'" ;;
  esac
done

cd "$AMD_REMOTE_ROOT"
PY="$AMD_VENV/bin/python"
RUN="$AMD_REMOTE_ROOT/runs/resume_check"
DATA="$AMD_DATA_ROOT/train/sft_upstream/train.jsonl"
[[ -f "$DATA" ]] || amd_die "no arm A rows at $DATA; run entrypoint.sh first"
rm -rf "$RUN"; mkdir -p "$RUN"

CMD=("$PY" -m train.sft_lora --data "$DATA" --model "$AMD_BASE_MODEL" --out "$RUN"
     --protocol bash --max-length "$AMD_MAX_LENGTH" --seed "$AMD_SEED" --precision bf16
     --batch-size 2 --grad-accum 1 --max-steps "$STEPS" --logging-steps 2)

amd_log "resume check: $STEPS steps, checkpoint at 50, then SIGKILL"
PYTHONUNBUFFERED=1 "${CMD[@]}" >"$AMD_REMOTE_LOG/resume_train.log" 2>&1 &
PID=$!
DEADLINE=$(( $(date +%s) + 900 ))
until compgen -G "$RUN/checkpoint-*/trainer_state.json" >/dev/null; do
  kill -0 "$PID" 2>/dev/null || { tail -20 "$AMD_REMOTE_LOG/resume_train.log" >&2; amd_die "training died before its first checkpoint"; }
  (( $(date +%s) < DEADLINE )) || amd_die "no checkpoint after 15 minutes"
  sleep 3
done
kill -9 "$PID" 2>/dev/null || true
wait "$PID" 2>/dev/null || true
KILLED_AT="$(find "$RUN" -maxdepth 1 -type d -name 'checkpoint-*' | sed 's/.*checkpoint-//' | sort -n | tail -1)"
amd_log "killed with checkpoint-$KILLED_AT on disk"

# The decision the real arms make on restart. It must say resume-local, at that step: if the
# completeness rules were wrong for this transformers version it would say "fresh" and a real
# reclaim would silently restart from zero.
STATUS="$("$PY" -m ops.amd.resume status --out "$RUN")"
amd_log "resume status: $STATUS"
STATE="$(printf '%s' "$STATUS" | jq -r .state)"
if [[ "$STATE" != "resume-local" ]]; then
  jq -n --arg s "$STATE" --argjson k "$KILLED_AT" '{ok:false, killed_at:$k, note:("status said "+$s)}' > "$OUT"
  amd_log "FAIL resume: the restart decision was '$STATE', not resume-local"
  exit 1
fi

PYTHONUNBUFFERED=1 timeout 900 "${CMD[@]}" --resume >"$AMD_REMOTE_LOG/resume_resume.log" 2>&1 || {
  tail -20 "$AMD_REMOTE_LOG/resume_resume.log" >&2; amd_die "the resumed run failed"; }
if "$PY" -m ops.amd.resume verdict --killed-at "$KILLED_AT" --log "$AMD_REMOTE_LOG/resume_resume.log" --out "$OUT"; then
  [[ -f "$RUN/adapter_model.safetensors" ]] || amd_die "resumed run left no adapter"
  amd_log "PASS resume: killed at $KILLED_AT, resumed from $KILLED_AT, adapter at $RUN"
  exit 0
fi
amd_log "FAIL resume: see $OUT and $AMD_REMOTE_LOG/resume_resume.log"
exit 1
