#!/usr/bin/env bash
# Preflight (data files per vast/data_manifest.txt, RAM >= 64 GB, GPU), then the eval bundle.
# Arguments go to both, e.g. bash vast/run.sh --datasets family. Needs no API key.
#
#   bash vast/run.sh              # from the repo root, inside tmux: the full run takes hours
set -euo pipefail
cd "$(dirname "$0")/.."

# shellcheck disable=SC1091
source "${CONDA_DIR:-$HOME/miniforge3}/etc/profile.d/conda.sh"
conda activate hybkgqa

python vast/preflight.py "$@"

mkdir -p results/eval_bundle
log="results/eval_bundle/run_$(date +%Y%m%d_%H%M%S).log"
echo "==> running notebooks/run_eval_bundle.py, log: $log"
PYTHONUNBUFFERED=1 python -m notebooks.run_eval_bundle "$@" 2>&1 | tee "$log"

echo
echo "==> done. Download before destroying the instance:"
ls -1 results/eval_bundle/NOTE_*.md results/calibration_results/*_localpush.pkl "$log"
