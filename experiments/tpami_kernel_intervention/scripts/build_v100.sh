#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
EXPERIMENT_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
GS_ROOT="$EXPERIMENT_ROOT/vendor/gaussian-splatting"

export TORCH_CUDA_ARCH_LIST="7.0"

command -v nvcc >/dev/null || { echo "nvcc is not available" >&2; exit 1; }
python - <<'PY'
import torch
print(f"PyTorch: {torch.__version__}")
print(f"PyTorch CUDA: {torch.version.cuda}")
print(f"CUDA available: {torch.cuda.is_available()}")
PY
nvcc --version

python -m pip install --no-build-isolation -e "$GS_ROOT/submodules/diff-gaussian-rasterization"
python -m pip install --no-build-isolation -e "$GS_ROOT/submodules/simple-knn"
python -m pip install --no-build-isolation -e "$GS_ROOT/submodules/fused-ssim"

echo "Built Experiment 5 CUDA extensions for TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
