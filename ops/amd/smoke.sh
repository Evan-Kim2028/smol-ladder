#!/usr/bin/env bash
# shellcheck source-path=SCRIPTDIR
# Measure before spending. Runs on the droplet, once, before any arm is trained.
#
#   smoke.sh [--max-length 8192] [--bench-seconds 120] [--bench-batches 4]
#
#   A. the ROCm checklist, scripted, one PASS/FAIL per item; any critical FAIL stops here
#   B. SFT throughput with the real trainer on the real base and data (batch 4 by default: that is
#      what session 1 trained at; `--bench-batches 2 4 8` widens it at about 3.5 billed minutes each)
#   C. one real kill-and-resume (resume_check.sh)
#
# It writes $AMD_REMOTE_LOG/measurements.json. The LAPTOP turns that into a costed projection and
# a go/no-go (driver.py project): token counts and prices live there, so the droplet only
# measures. The gate (the released adapter served and evaluated, driver.py gate) is the driver's
# next step and is what proves serving, tool calls and the harness before anything is trained.

set -euo pipefail
# shellcheck source=common.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
amd_load_env "$AMD_REMOTE_ROOT/.env"

BENCH_SECONDS=120
BENCH_BATCHES=(4)
while [[ $# -gt 0 ]]; do
  case "$1" in
    --max-length)    AMD_MAX_LENGTH="$2"; shift 2 ;;
    --bench-seconds) BENCH_SECONDS="$2"; shift 2 ;;
    --bench-batches) shift; BENCH_BATCHES=()
                     while [[ $# -gt 0 && "$1" =~ ^[0-9]+$ ]]; do BENCH_BATCHES+=("$1"); shift; done ;;
    *) amd_die "unknown argument '$1'" ;;
  esac
done

cd "$AMD_REMOTE_ROOT"
SYSPY="$(amd_syspy)"
PY="$AMD_VENV/bin/python"
CHECKS="$AMD_REMOTE_LOG/checks.tsv"
mkdir -p "$AMD_REMOTE_LOG"
: > "$CHECKS"
CRITICAL_FAILS=0

pass() { printf 'PASS\t%s\n' "$1" >> "$CHECKS"; printf '  \033[32mPASS\033[0m  %s\n' "$1"; }
fail() { printf 'FAIL\t%s\n' "$1" >> "$CHECKS"; printf '  \033[31mFAIL\033[0m  %s\n' "$1"; CRITICAL_FAILS=$((CRITICAL_FAILS + 1)); }
check() { # check "name" cmd...  : PASS if the command succeeds, else FAIL plus its last two lines
  local name="$1" out; shift
  if out="$("$@" 2>&1)"; then pass "$name"; else fail "$name"; printf '          %s\n' "$(printf '%s' "$out" | tail -2)"; fi
}

amd_log "=== A. ROCm checklist ==="
check "/dev/kfd present (ROCm driver loaded)" test -e /dev/kfd
if command -v amd-smi >/dev/null 2>&1 || command -v amdsmi >/dev/null 2>&1; then
  check "amd-smi lists a GPU" bash -c 'amd-smi list 2>/dev/null | grep -qi gpu || amdsmi list 2>/dev/null | grep -qi gpu'
else
  check "rocm-smi lists a GPU" bash -c 'rocm-smi --showproductname 2>/dev/null | grep -qi "card\|gfx"'
fi
check "image python: torch sees the GPU" "$SYSPY" -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)'
check "venv python: torch is the image's ROCm torch" "$PY" -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() and torch.version.hip else 1)'
check "bf16 matmul matches fp32 (relative error < 1%)" "$PY" -c '
import torch
a = torch.randn(4096, 4096, dtype=torch.bfloat16, device="cuda")
got, want = (a @ a).float(), a.float() @ a.float()
assert ((got - want).norm() / want.norm()).item() < 1e-2'
check "transformers/peft/trl/accelerate/datasets import" "$PY" -c 'import transformers, peft, trl, accelerate, datasets'
check "vllm >= 0.16.2 on the image python (Qwen3.5 support)" "$SYSPY" -c '
import vllm
from packaging.version import Version
assert Version(vllm.__version__) >= Version("0.16.2"), vllm.__version__'
check "GPU memory >= 200 GiB visible" "$PY" -c '
import torch
assert torch.cuda.mem_get_info()[1] / 2**30 >= 200'
check "free disk >= 100 GB under $AMD_REMOTE_ROOT" bash -c "[[ \$(df --output=avail -BG '$AMD_REMOTE_ROOT' | tail -1 | tr -dc 0-9) -ge 100 ]]"
check "HF token valid" "$PY" -c 'from huggingface_hub import HfApi; HfApi().whoami()'
check "adapter Hub repos are private" "$PY" -c "
import os
from huggingface_hub import HfApi
ns = os.environ['AMD_HUB_NAMESPACE']
for n in ('$AMD_HUB_ADAPTER_A', '$AMD_HUB_ADAPTER_B', '$AMD_HUB_ADAPTER_AB'):
    assert HfApi().model_info(f'{ns}/{n}').private"
if (( CRITICAL_FAILS > 0 )); then
  amd_log "checklist FAILED ($CRITICAL_FAILS): stopping before any GPU time is spent on the benchmark"
  "$PY" ops/amd/bench.py finalize --checks "$CHECKS" --out "$AMD_REMOTE_LOG/measurements.json"
  exit 3
fi

amd_log "=== B. SFT throughput on the real base, ${BENCH_SECONDS}s at batch ${BENCH_BATCHES[*]} ==="
"$PY" ops/amd/bench.py run --out "$AMD_REMOTE_LOG/bench.json" --seconds "$BENCH_SECONDS" \
  --max-length "$AMD_MAX_LENGTH" --batches "${BENCH_BATCHES[@]}" || { amd_log "benchmark produced nothing trainable"; \
  "$PY" ops/amd/bench.py finalize --checks "$CHECKS" --bench "$AMD_REMOTE_LOG/bench.json" --out "$AMD_REMOTE_LOG/measurements.json"; exit 3; }

amd_log "=== C. kill and resume ==="
if bash "$AMD_REMOTE_ROOT/ops/amd/resume_check.sh" --out "$AMD_REMOTE_LOG/resume.json"; then
  pass "kill -9 mid-run, then resume from the checkpoint"
else
  fail "kill -9 mid-run, then resume from the checkpoint"
fi

"$PY" ops/amd/bench.py finalize --checks "$CHECKS" --bench "$AMD_REMOTE_LOG/bench.json" \
  --resume "$AMD_REMOTE_LOG/resume.json" --out "$AMD_REMOTE_LOG/measurements.json"
if (( CRITICAL_FAILS > 0 )); then amd_log "smoke finished with $CRITICAL_FAILS failure(s)"; exit 3; fi
amd_log "smoke passed; measurements at $AMD_REMOTE_LOG/measurements.json"
