#!/usr/bin/env bash
# One SFT arm on the droplet. Resumable: re-running after a kill picks up the newest checkpoint.
#
#   run_sft.sh --arm A|B|AB [--max-steps N] [--resume] [--inside-rocm] [--smoke]
#
# Arms:
#   A   upstream's SmolDataEnvs-sft, through the firewall (data/train/sft_upstream)
#   B   our exported ja3 trajectories (data/train/ja3_sft.jsonl)
#   AB  the union, exported first if it is not already on disk
#
# No QLoRA. MI300X has 192 GB, so 4-bit weights buy nothing and bitsandbytes on ROCm is a
# preview-alpha backend. bf16 LoRA is the plan's instruction and the reason for it.

set -euo pipefail
# shellcheck source=common.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

ARM=""
MAX_STEPS="${AMD_MAX_STEPS:-0}"
RESUME=1
INSIDE_ROCM=0
SEED="${AMD_SEED:-42}"
MAX_LENGTH="${AMD_MAX_LENGTH:-8192}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --arm)         ARM="$2"; shift 2 ;;
    --max-steps)   MAX_STEPS="$2"; shift 2 ;;
    --resume)      RESUME=1; shift ;;
    --no-resume)   RESUME=0; shift ;;
    --inside-rocm) INSIDE_ROCM=1; shift ;;
    # A smoke run is a step-limited run; the flag exists only so the plan reads as what it does.
    --smoke)       MAX_STEPS="${AMD_SMOKE_STEPS:-20}"; shift ;;
    --seed)        SEED="$2"; shift 2 ;;
    --max-length)  MAX_LENGTH="$2"; shift 2 ;;
    *) amd_die "unknown argument '$1'" ;;
  esac
done
[[ -n "$ARM" ]] || amd_die "--arm is required (A, B or AB)"

# Inside the ROCm container the whole path and python move in, and only once.
if (( INSIDE_ROCM )); then
  INNER=(--arm "$ARM" --max-steps "$MAX_STEPS" --seed "$SEED" --max-length "$MAX_LENGTH")
  (( RESUME )) && INNER+=(--resume)
  exec docker exec \
    -e HF_TOKEN="${HF_TOKEN:-}" \
    -e HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}" \
    -e PYTORCH_TUNABLEOP_ENABLED="${PYTORCH_TUNABLEOP_ENABLED:-1}" \
    -w "$AMD_REMOTE_ROOT" \
    "${AMD_ROCM_CONTAINER:-smol-rocm}" \
    "$AMD_REMOTE_ROOT/ops/amd/run_sft.sh" "${INNER[@]}"
fi

amd_load_env
cd "$AMD_REMOTE_ROOT"
DATA="$(amd_data_dir "$ARM")"
[[ -f "$DATA/train.jsonl" || -f "$DATA/ja3_sft.jsonl" ]] || amd_die "no training rows at $DATA"
OUT="$AMD_REMOTE_ROOT/runs/sft_$(printf '%s' "$ARM" | tr '[:upper:]' '[:lower:]')"
LOG="$AMD_REMOTE_LOG/sft_$(printf '%s' "$ARM" | tr '[:upper:]' '[:lower:]').log"
HUB_REPO="$(amd_arm_hub_repo "$ARM")"
mkdir -p "$AMD_REMOTE_LOG" "$OUT"

# The `sft_ab` export is a derived dataset, so it is built here rather than synced: it is a
# concatenation of two files that are already on the instance and it must be rebuilt whenever
# either changes. The same 5% deterministic split and the same seed, or the two arms would not be
# comparable on the same rows.
if [[ "$ARM" == "AB" && ! -f "$DATA/train.jsonl" ]]; then
  amd_log "building the A+B export at $DATA"
  mkdir -p "$DATA"
  python3 - <<'PY' "$DATA/train.jsonl" "$DATA/val.jsonl"
import json, sys
out_train, out_val = sys.argv[1], sys.argv[2]
base = f"{'/opt/smol-ladder/data'}/train"
train_rows, val_rows = [], []
for name in ("sft_upstream/train.jsonl", "sft_upstream/val.jsonl"):
    with open(f"{base}/{name}") as fh:
        (train_rows if name.endswith('train.jsonl') else val_rows).extend(json.loads(l) for l in fh if l.strip())
for name, bucket in ((("ja3_sft.jsonl"), train_rows),):
    with open(f"{base}/{name}") as fh:
        bucket.extend(json.loads(l) for l in fh if l.strip())
for path, rows in ((out_train, train_rows), (out_val, val_rows)):
    with open(path, "w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
print(f"A+B: {len(train_rows)} train, {len(val_rows)} val")
PY
fi

# Without a namespace there is no repo id to push to, and the Hub is the only copy of an adapter
# that outlives the droplet. Say so rather than training for hours and finding out at the end.
if [[ -n "$AMD_HUB_NAMESPACE" ]]; then
  # hub_model_id turns on TRL's push_to_hub with hub_strategy="every_save", so a checkpoint is on
  # the Hub at every save rather than only at the end. That is what makes a killed run resumable
  # on a *fresh* droplet, where the local disk is gone.
  CMD_HUB=(--hub-model-id "$HUB_REPO")
else
  amd_log "WARNING: AMD_HUB_NAMESPACE is unset, so this adapter cannot be pushed to the Hub."
  amd_log "         Set it to a private namespace, or accept that the result dies with the droplet."
  CMD_HUB=()
fi

CMD=("$AMD_REMOTE_ROOT/.venv/bin/python" -m train.sft_lora
     --data "$DATA"
     --model "$AMD_BASE_MODEL"
     --out "$OUT"
     --protocol bash
     --max-length "$MAX_LENGTH"
     --seed "$SEED"
     --precision bf16 "${CMD_HUB[@]}")
(( MAX_STEPS > 0 )) && CMD+=(--max-steps "$MAX_STEPS")
(( RESUME )) && CMD+=(--resume)

amd_log "arm=$ARM steps=$MAX_STEPS resume=$RESUME"
amd_log "${CMD[*]}"

# teed, because a killed run's log is the only record of how far it got.
set +e
"${CMD[@]}" 2>&1 | tee -a "$LOG"
STATUS=${PIPESTATUS[0]}
set -e

if (( STATUS != 0 )); then
  amd_log "training exited $STATUS; log at $LOG"
  exit "$STATUS"
fi

# Push the finished adapter to a private repo, then rsync it back beside the log so a droplet
# death before the Hub push is recoverable from the laptop copy.
if [[ -n "$AMD_HUB_NAMESPACE" ]]; then
  amd_log "pushing the adapter to $HUB_REPO"
  "$AMD_REMOTE_ROOT/.venv/bin/python" -m huggingface_hub.commands.huggingface_cli \
    upload "$HUB_REPO" "$OUT" --repo-type model --private \
    || amd_log "WARNING: adapter upload failed; the rsync copy is the only remaining copy"
fi

amd_log "done. adapter at $OUT, log at $LOG"
