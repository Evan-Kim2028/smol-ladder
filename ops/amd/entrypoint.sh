#!/usr/bin/env bash
# shellcheck source-path=SCRIPTDIR
# THE remote entry script: one non-interactive command, no prompt anywhere, safe to re-run.
#
#   ssh root@<ip> -- bash /var/tmp/smol-ladder-stage/entrypoint.sh
#
# Everything it needs was staged on the laptop before the droplet existed, so on the droplet it
# only verifies, unpacks and installs:
#
#   1. refuse early if this is not an AMD GPU image (/dev/kfd)
#   2. verify the staged tarballs against SHA256SUMS, unpack code and both SFT sets
#   3. find the image's python (the one that has torch AND vLLM) and check it sees the GPU
#   4. a venv layered over it with ONLY the pure-python training deps; torch is never touched
#   5. HF login, private Hub repos created, the base model downloaded once
#   6. arm the on-droplet watchdog
#
# There is no clone and no GitHub: the code is the pinned commit's `git archive`. The `train`
# extra in pyproject.toml is NOT installed: it pins CUDA torch wheels and bitsandbytes, which on
# this hardware would replace the image's ROCm torch with one that cannot see the GPU.

set -euo pipefail

STAGE="${AMD_STAGE_DIR:-/var/tmp/smol-ladder-stage}"
ROOT="${AMD_REMOTE_ROOT:-/opt/smol-ladder}"
LOGDIR="${AMD_REMOTE_LOG:-/var/log/smol-ladder}"
VENV="$ROOT/.venv"
log() { printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*" >&2; }
die() { log "FATAL: $*"; exit 1; }
export DEBIAN_FRONTEND=noninteractive PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_INPUT=1
mkdir -p "$LOGDIR" "$ROOT"

[[ -d "$STAGE" ]] || die "no staged inputs at $STAGE (driver.py bootstrap uploads them)"
[[ -e /dev/kfd ]] || die "/dev/kfd is absent: no ROCm driver, so nothing below can work. Wrong image."
log "host $(uname -n), kernel $(uname -r)"

# ── 2. verify, then unpack ───────────────────────────────────────────────────────
( cd "$STAGE" && sha256sum --quiet -c SHA256SUMS ) || die "a staged file is corrupt or truncated: re-upload"
COMMIT="$(tr -d '[:space:]' < "$STAGE/repo.txt")"
if [[ "$(cat "$ROOT/.amd-commit" 2>/dev/null || true)" != "$COMMIT" ]]; then
  log "unpacking code at $COMMIT"
  tar -xzf "$STAGE/code.tar.gz" -C "$ROOT"
  printf '%s\n' "$COMMIT" > "$ROOT/.amd-commit"
fi
mkdir -p "$ROOT/data/train" "$ROOT/runs"
[[ -s "$ROOT/data/train/sft_upstream/train.jsonl" ]] || tar -xzf "$STAGE/sft_a.tar.gz" -C "$ROOT/data/train"
[[ -s "$ROOT/data/train/ja3_sft.jsonl" ]] || tar -xzf "$STAGE/sft_b.tar.gz" -C "$ROOT/data/train"
for f in sft_upstream/train.jsonl sft_upstream/val.jsonl ja3_sft.jsonl; do
  [[ -s "$ROOT/data/train/$f" ]] || die "missing $ROOT/data/train/$f after unpacking"
done
cp "$STAGE/tokens.json" "$ROOT/tokens.json"
install -m 600 "$STAGE/remote.env" "$ROOT/.env"
log "data: A $(wc -l < "$ROOT/data/train/sft_upstream/train.jsonl") rows, B $(wc -l < "$ROOT/data/train/ja3_sft.jsonl") rows"

# shellcheck source=common.sh
source "$ROOT/ops/amd/common.sh"
amd_load_env "$ROOT/.env"
[[ -n "${HF_TOKEN:-}" ]] || die "HF_TOKEN missing from remote.env: the Hub is the only copy that outlives the droplet"
[[ -n "${AMD_HUB_NAMESPACE:-}" ]] || die "AMD_HUB_NAMESPACE missing from remote.env"

# ── 1b. apt: only what is missing ────────────────────────────────────────────────
MISSING=()
for pkg in curl jq ca-certificates; do dpkg -s "$pkg" >/dev/null 2>&1 || MISSING+=("$pkg"); done
if (( ${#MISSING[@]} )); then
  log "apt-get install ${MISSING[*]}"
  apt-get update -qq && apt-get install -y -qq "${MISSING[@]}"
fi

# ── 3. the image's python: the one that has torch AND vllm ───────────────────────
SYSPY=""
for cand in python3 python /opt/venv/bin/python /opt/conda/bin/python /usr/local/bin/python3 \
            /root/venv/bin/python /opt/rocm/venv/bin/python; do
  command -v "$cand" >/dev/null 2>&1 || continue
  if "$cand" -c 'import torch, vllm' >/dev/null 2>&1; then SYSPY="$(command -v "$cand")"; break; fi
done
if [[ -z "$SYSPY" ]]; then
  log "no host python imports both torch and vllm. Docker containers on this host:"
  docker ps -a --format '  {{.Names}}  {{.Image}}  {{.Status}}' 2>&1 | head -5 >&2 || true
  die "vLLM is not on the host python (it may live in a container). See docs/AMD_RUNBOOK.md, 'vLLM image layout'."
fi
printf '%s\n' "$SYSPY" > "$ROOT/.syspy"
"$SYSPY" - <<'PYCHK' || die "the image's torch cannot see the GPU"
import sys, torch, vllm
from packaging.version import Version
print("torch", torch.__version__, "hip", getattr(torch.version, "hip", None), "gpu", torch.cuda.is_available())
print("vllm", vllm.__version__)
if not torch.cuda.is_available():
    sys.exit(1)
print("device", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
sys.exit(0 if Version(vllm.__version__) >= Version("0.16.2") else 2)
PYCHK
log "image python: $SYSPY"

# ── 4. the training venv, layered over the image ─────────────────────────────────
# A venv made from a venv python does not inherit the parent's packages, so the image's
# site-packages are added by a .pth file instead. It comes AFTER the venv's own directory, so a
# newer transformers/peft/trl in the venv wins and vLLM, which runs on the image python, is
# never affected.
if [[ ! -x "$VENV/bin/python" ]]; then
  "$SYSPY" -m venv "$VENV" 2>/dev/null || { apt-get install -y -qq python3-venv && "$SYSPY" -m venv "$VENV"; }
fi
SITE_PARENT="$("$SYSPY" -c 'import site; print(site.getsitepackages()[0])')"
SITE_VENV="$("$VENV/bin/python" -c 'import site; print(site.getsitepackages()[0])')"
printf '%s\n' "$SITE_PARENT" > "$SITE_VENV/zz-image-site.pth"
TORCH_V="$("$SYSPY" -c 'import torch; print(torch.__version__.split("+")[0])')"
printf 'torch==%s\n' "$TORCH_V" > "$LOGDIR/constraints.txt"
if ! "$VENV/bin/python" -c 'import trl, peft, transformers, accelerate, datasets' >/dev/null 2>&1; then
  log "installing the training stack (torch pinned at $TORCH_V by constraint)"
  "$VENV/bin/python" -m pip install -q -c "$LOGDIR/constraints.txt" \
    "transformers>=5.17" "trl>=1.13" "peft>=0.21" "accelerate>=1.0" "datasets>=5.0" huggingface_hub \
    || die "pip could not install the training stack without replacing the image's torch"
fi
"$VENV/bin/python" - <<'PYV' | tee -a "$LOGDIR/versions.log"
import torch, transformers, peft, trl, accelerate, datasets
print("venv sees torch", torch.__version__, "transformers", transformers.__version__,
      "peft", peft.__version__, "trl", trl.__version__)
assert torch.cuda.is_available(), "the venv's torch is not the image's ROCm torch"
PYV

# ── 5. HF: login, private repos, the base model ──────────────────────────────────
# The repos are created private HERE, before training: the Trainer would create a missing repo
# with the account default, which can be public, and arm B is our own traces.
"$VENV/bin/python" - <<'PYHF'
import os
from huggingface_hub import HfApi, snapshot_download, login
login(token=os.environ["HF_TOKEN"], add_to_git_credential=False)
api = HfApi()
ns = os.environ["AMD_HUB_NAMESPACE"]
print("HF user", api.whoami()["name"])
for name in (os.environ.get("AMD_HUB_ADAPTER_A", "smol-ladder-sft-a"),
             os.environ.get("AMD_HUB_ADAPTER_B", "smol-ladder-sft-b"),
             os.environ.get("AMD_HUB_ADAPTER_AB", "smol-ladder-sft-ab")):
    api.create_repo(f"{ns}/{name}", repo_type="model", private=True, exist_ok=True)
    assert api.model_info(f"{ns}/{name}").private, f"{ns}/{name} is not private"
api.create_repo(f"{ns}/{os.environ.get('AMD_HUB_ARTIFACTS', 'smol-ladder-runs')}",
                repo_type="dataset", private=True, exist_ok=True)
path = snapshot_download(os.environ.get("AMD_BASE_MODEL", "Qwen/Qwen3.5-2B"))
print("base model at", path)
PYHF

# ── 6. the on-droplet watchdog ───────────────────────────────────────────────────
if ! pgrep -f 'ops/amd/watchdog.sh' >/dev/null; then
  nohup bash "$ROOT/ops/amd/watchdog.sh" --arm >>"$LOGDIR/watchdog.log" 2>&1 &
  log "watchdog armed (pid $!)"
fi
log "entrypoint complete at $COMMIT. Next: bash $ROOT/ops/amd/smoke.sh"
