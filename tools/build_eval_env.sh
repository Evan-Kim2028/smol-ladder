#!/usr/bin/env bash
# Build the environment the `--agent bash` sandbox runs the model's commands under: Python 3.12 and
# the package versions of the container the SFT rows were recorded in (tools/eval_env_requirements.txt,
# from the rows' own `pip list`). Evidence and what it does not cover: docs/LOCAL_MODELS.md.
#
#   tools/build_eval_env.sh [DIR]      default /var/tmp/smol-ladder/eval-env-py312
#
# No pyarrow, openpyxl, plotly or xgboost: the recording container had none (the rows show
# `No module named 'xgboost'`), and a missing import must fail the same way here.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
dir="${1:-/var/tmp/smol-ladder/eval-env-py312}"
uv venv --python 3.12 --seed "$dir"
uv pip install --python "$dir/bin/python" -r "$here/eval_env_requirements.txt"
"$dir/bin/python" -m pip install -q pip==25.0.1
"$dir/bin/python" -m pip list 2>/dev/null | sed -n '1,4p;/^pandas /p;/^numpy /p;/^scikit-learn /p'
