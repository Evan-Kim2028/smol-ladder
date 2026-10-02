#!/usr/bin/env bash
# shellcheck source-path=SCRIPTDIR
# Container half of the bootstrap, run by entrypoint.sh via `docker exec`: tools, the image's
# python check, the training venv, HF login, private Hub repos, the base model. Safe to re-run.
set -euo pipefail
ROOT="${AMD_REMOTE_ROOT:-/opt/smol-ladder}"
LOGDIR="${AMD_REMOTE_LOG:-/var/log/smol-ladder}"
VENV="$ROOT/.venv"
log() { printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*" >&2; }
die() { log "FATAL: $*"; exit 1; }
export DEBIAN_FRONTEND=noninteractive PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_INPUT=1
# shellcheck source=common.sh
source "$ROOT/ops/amd/common.sh"
amd_load_env "$ROOT/.env"

# ── 1b. apt: only what is missing ────────────────────────────────────────────────
MISSING=()
for pkg in curl jq ca-certificates procps; do dpkg -s "$pkg" >/dev/null 2>&1 || MISSING+=("$pkg"); done
if (( ${#MISSING[@]} )); then
  log "apt-get install ${MISSING[*]}"
  amd_apt update -qq && amd_apt install -y -qq "${MISSING[@]}"
fi

# ── 3. the image's python: the one that has torch AND vllm ───────────────────────
SYSPY=""
for cand in python3 python /opt/venv/bin/python /opt/conda/bin/python /usr/local/bin/python3 \
            /root/venv/bin/python /opt/rocm/venv/bin/python; do
  command -v "$cand" >/dev/null 2>&1 || continue
  if "$cand" -c 'import torch, vllm' >/dev/null 2>&1; then SYSPY="$(command -v "$cand")"; break; fi
done
if [[ -z "$SYSPY" ]]; then
  die "the container's python has no torch + vllm: wrong image (AMD_IMAGE=$AMD_IMAGE)"
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
# --system-site-packages: the image's torch/vllm/triton stay visible, and the venv's own newer
# transformers/peft/trl win for training (Qwen3.5 needs transformers 5; vLLM pins <5 and runs on
# the image python, never the venv). The constraints file pins everything the image owns, so pip
# can fail but cannot replace torch.
if [[ -x "$VENV/bin/python" ]] && ! grep -q 'include-system-site-packages = true' "$VENV/pyvenv.cfg"; then rm -rf "$VENV"; fi
[[ -x "$VENV/bin/python" ]] || "$SYSPY" -m venv --system-site-packages "$VENV"
"$SYSPY" - > "$LOGDIR/constraints.txt" <<'PYC'
import importlib.metadata as m
for n in ("torch", "torchvision", "torchaudio", "triton", "vllm"):
    try: print(f"{n}=={m.version(n)}")
    except m.PackageNotFoundError: pass
PYC
if ! "$VENV/bin/python" -c 'import trl, peft, transformers, accelerate, datasets; assert int(transformers.__version__.split(".")[0]) >= 5' >/dev/null 2>&1; then
  log "installing the training stack (image packages pinned by $LOGDIR/constraints.txt)"
  "$VENV/bin/python" -m pip install -q -c "$LOGDIR/constraints.txt" \
    "transformers>=5.17" "trl>=1.13" "peft>=0.21" "accelerate>=1.0" "datasets>=4.0" huggingface_hub \
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
session = os.environ.get("AMD_SESSION", "s2")      # common.sh's amd_hub_name, ops/amd/plan.py's hub_name
for name in (f"smol-ladder-sft-a-{session}", f"smol-ladder-sft-b-{session}", f"smol-ladder-sft-ab-{session}"):
    api.create_repo(f"{ns}/{name}", repo_type="model", private=True, exist_ok=True)
    assert api.model_info(f"{ns}/{name}").private, f"{ns}/{name} is not private"
ds = f"{ns}/smol-ladder-runs-{session}"
api.create_repo(ds, repo_type="dataset", private=True, exist_ok=True)
# exist_ok=True leaves a PRE-EXISTING repo as it was, public or not: assert, as for the models.
assert api.dataset_info(ds).private, f"{ds} is not private"
path = snapshot_download(os.environ.get("AMD_BASE_MODEL", "Qwen/Qwen3.5-2B"))
print("base model at", path)
PYHF

