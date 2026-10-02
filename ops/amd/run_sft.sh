#!/usr/bin/env bash
# shellcheck source-path=SCRIPTDIR
# One SFT arm on the droplet. Resumable, checkpoint-pushing, and a no-op if the arm is finished.
#
#   run_sft.sh --arm A|B|AB [--max-length 8192] [--batch-size N --grad-accum M] [--max-steps N]
#
#   A   upstream's SmolDataEnvs-sft export     data/train/sft_upstream  (train.jsonl + val.jsonl)
#   B   our exported ja3 traces                 data/train/ja3_sft.jsonl
#   AB  the union, built here from the two staged sets (A's val set is kept for eval-loss)
#
# What makes a spot reclaim cost minutes:
#   * train/sft_lora.py writes a checkpoint every 100 steps (it has no flag for the cadence) and
#     hub_strategy="every_save" pushes the newest to the private Hub repo as `last-checkpoint/`;
#   * on start, `ops.amd.resume status` (a) skips a finished arm, (b) moves any half-written
#     checkpoint aside, and (c) on a fresh droplet restores `last-checkpoint/` from the Hub;
#   * the trainer is then run with --resume, which picks the newest complete checkpoint.
# The Hub repos were created PRIVATE by the entrypoint. The trainer would otherwise create a missing
# repo with the account default, and arm B is our own traces.
#
# No QLoRA: the card has 288 GB, 4-bit buys nothing, and bitsandbytes on ROCm is alpha. bf16 LoRA is
# what upstream's recipe used. Batch size comes from the smoke's benchmark; the effective batch
# stays at 8 so the arm remains comparable to upstream's batch-1 x accum-8 recipe.

set -euo pipefail
# shellcheck source=common.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
amd_load_env "$AMD_REMOTE_ROOT/.env"

ARM=""; MAX_STEPS=0; BATCH=""; ACCUM=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --arm)         ARM="$2"; shift 2 ;;
    --max-length)  AMD_MAX_LENGTH="$2"; shift 2 ;;
    --batch-size)  BATCH="$2"; shift 2 ;;
    --grad-accum)  ACCUM="$2"; shift 2 ;;
    --max-steps)   MAX_STEPS="$2"; shift 2 ;;
    *) amd_die "unknown argument '$1'" ;;
  esac
done
[[ "$ARM" == "A" || "$ARM" == "B" || "$ARM" == "AB" ]] || amd_die "--arm must be A, B or AB"

cd "$AMD_REMOTE_ROOT"
PY="$AMD_VENV/bin/python"
# The smoke's probe server (and any server left by an earlier run) holds ~85% of the GPU. Stop it,
# defensively, whatever the plan did before this: an arm must train on an empty card.
bash "$(dirname "${BASH_SOURCE[0]}")/serve.sh" --stop
OUT="$(amd_arm_dir "$ARM")"
LOG="$AMD_REMOTE_LOG/sft_$(amd_lower "$ARM").log"
HUB_REPO="$(amd_arm_hub_repo "$ARM")"
MEAS="$AMD_REMOTE_LOG/measurements.json"
mkdir -p "$OUT" "$AMD_REMOTE_LOG"

# ── is there anything to do, and from where? ─────────────────────────────────────
STATUS="$("$PY" -m ops.amd.resume status --out "$OUT" --repo "$HUB_REPO")"
STATE="$(printf '%s' "$STATUS" | jq -r .state)"
amd_log "arm $ARM: $STATUS"
if [[ "$STATE" == "done" ]]; then amd_log "arm $ARM already finished; nothing to do"; exit 0; fi
amd_assert_gpu_free

# ── data ─────────────────────────────────────────────────────────────────────────
case "$ARM" in
  A)  DATA="$AMD_DATA_ROOT/train/sft_upstream" ;;
  B)  DATA="$AMD_DATA_ROOT/train/ja3_sft.jsonl" ;;
  AB) DATA="$AMD_DATA_ROOT/train/sft_ab"
      mkdir -p "$DATA"
      cat "$AMD_DATA_ROOT/train/sft_upstream/train.jsonl" "$AMD_DATA_ROOT/train/ja3_sft.jsonl" > "$DATA/train.jsonl"
      cp "$AMD_DATA_ROOT/train/sft_upstream/val.jsonl" "$DATA/val.jsonl" ;;
esac
[[ -e "$DATA" ]] || amd_die "no data at $DATA; run entrypoint.sh first"

# ── batch size: the smoke's choice unless given ──────────────────────────────────
if [[ -z "$BATCH" && -f "$MEAS" ]]; then
  BATCH="$(jq -r '.best.per_device_batch_size // empty' "$MEAS")"
  ACCUM="$(jq -r '.best.grad_accum // empty' "$MEAS")"
fi
BATCH="${BATCH:-4}"
ACCUM="${ACCUM:-$(( AMD_EFFECTIVE_BATCH / BATCH ))}"
(( ACCUM >= 1 )) || ACCUM=1

CMD=("$PY" -m train.sft_lora --data "$DATA" --model "$AMD_BASE_MODEL" --out "$OUT"
     --protocol bash --max-length "$AMD_MAX_LENGTH" --seed "$AMD_SEED" --precision bf16
     --batch-size "$BATCH" --grad-accum "$ACCUM" --hub-model-id "$HUB_REPO" --resume)
(( MAX_STEPS > 0 )) && CMD+=(--max-steps "$MAX_STEPS")

amd_log "arm=$ARM state=$STATE batch=$BATCH accum=$ACCUM max_length=$AMD_MAX_LENGTH hub=$HUB_REPO"
amd_log "  ${CMD[*]}"
set +e
PYTHONUNBUFFERED=1 "${CMD[@]}" 2>&1 | tee -a "$LOG"
STATUS_CODE=${PIPESTATUS[0]}
set -e
if (( STATUS_CODE != 0 )); then
  amd_log "training exited $STATUS_CODE. Checkpoints on disk and on the Hub are intact; re-run this command to resume."
  exit "$STATUS_CODE"
fi

# ── finish: prove the adapter is whole, mark it, push it ─────────────────────────
"$PY" - "$OUT" "$HUB_REPO" <<'PYFIN'
import sys
from pathlib import Path
from huggingface_hub import HfApi
from ops.amd.resume import ADAPTER, DONE, HUB_DONE, safetensors_ok

out, repo = Path(sys.argv[1]), sys.argv[2]
assert (out / "adapter_config.json").exists() and safetensors_ok(out / ADAPTER), "adapter missing or truncated"
(out / DONE).write_text("done\n")
api = HfApi()
api.upload_file(path_or_fileobj=str(out / ADAPTER), path_in_repo=ADAPTER, repo_id=repo)
api.upload_file(path_or_fileobj=str(out / "adapter_config.json"), path_in_repo="adapter_config.json", repo_id=repo)
# Last, and only after both files: the marker that says THIS is the final adapter and not an
# intermediate checkpoint the trainer pushed. A fresh droplet skips the arm on seeing it.
api.upload_file(path_or_fileobj=b"done\n", path_in_repo=HUB_DONE, repo_id=repo)
print("final adapter pushed to", repo)
PYFIN
amd_log "arm $ARM done: adapter at $OUT, log $LOG"
