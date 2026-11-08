#!/usr/bin/env bash
# Build the hybkgqa conda env on a blank Linux GPU machine (vast.ai), from GoG/hybkgqa.yaml.
# Installs Miniforge under $CONDA_DIR (default ~/miniforge3) when it is missing, so the image's own
# conda (and its 'defaults' channel terms of service) is never used. Safe to rerun.
#
#   bash vast/setup.sh            # from the repo root
set -euo pipefail
cd "$(dirname "$0")/.."

CONDA_DIR="${CONDA_DIR:-$HOME/miniforge3}"
ENV_NAME=hybkgqa

for tool in curl git tar; do
    command -v "$tool" >/dev/null || { apt-get update -qq && apt-get install -y -qq "$tool"; }
done

if [ ! -x "$CONDA_DIR/bin/conda" ]; then
    echo "==> installing Miniforge into $CONDA_DIR"
    curl -fsSL -o /tmp/miniforge.sh \
        https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
    bash /tmp/miniforge.sh -b -p "$CONDA_DIR"
fi
# shellcheck disable=SC1091
source "$CONDA_DIR/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "==> env $ENV_NAME exists; delete it with 'conda env remove -n $ENV_NAME' to rebuild"
else
    echo "==> creating env $ENV_NAME from GoG/hybkgqa.yaml (torch wheels are large: ~10 min)"
    if ! conda env create -n "$ENV_NAME" -f GoG/hybkgqa.yaml; then
        # Only the Python patch version can be missing from conda-forge; no package version changes.
        echo "!! python=3.11.15 failed; retrying with python=3.11 (the run note records the version used)"
        conda env remove -n "$ENV_NAME" -y >/dev/null 2>&1 || true
        sed 's/python=3.11.15/python=3.11/' GoG/hybkgqa.yaml > GoG/.hybkgqa_py311.yaml
        conda env create -n "$ENV_NAME" -f GoG/.hybkgqa_py311.yaml
        rm -f GoG/.hybkgqa_py311.yaml
    fi
fi
conda activate "$ENV_NAME"

echo "==> checking installed versions against vast/requirements-lock.txt"
python - <<'EOF'
import importlib.metadata as md, re, sys
mismatch = []
for line in open("vast/requirements-lock.txt", encoding="utf-8"):
    line = line.strip()
    if not line or line.startswith(("#", "-")):
        continue
    name, want = line.split("==")
    try:
        have = md.version(name)
    except md.PackageNotFoundError:
        have = None
    if have != want:
        mismatch.append(f"{name}: want {want}, have {have}")
print(f"python {sys.version.split()[0]}")
if mismatch:
    sys.exit("version mismatch:\n  " + "\n  ".join(mismatch))
import torch, torch_scatter, torch_geometric  # noqa: F401
print(f"torch {torch.__version__}, CUDA build {torch.version.cuda}, GPU available: {torch.cuda.is_available()}")
EOF

echo "==> caching the sentence-transformers model (needs internet)"
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

echo "==> setup done. Next: tar -xzf <data tar> in the repo root, then bash vast/run.sh"
