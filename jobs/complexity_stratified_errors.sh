#!/bin/bash
#SBATCH --job-name=complexity-stratified-errors
#SBATCH --partition=nodes
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/home/rizk_lab/shared/kiran/kernel-specialization-3d-splatting/jobs/logs/complexity_stratified_errors_%j.out
#SBATCH --error=/home/rizk_lab/shared/kiran/kernel-specialization-3d-splatting/jobs/logs/complexity_stratified_errors_%j.err

set -eo pipefail

PROJECT_ROOT=/home/rizk_lab/shared/kiran/kernel-specialization-3d-splatting
ENV_PATH=/home/rizk_lab/shared/kiran/envs/kernel_splat
OUTPUT_DIR=$PROJECT_ROOT/results/cross_scene/complexity_stratified_errors_p32

mkdir -p "$PROJECT_ROOT/jobs/logs" "$OUTPUT_DIR"
cd "$PROJECT_ROOT"

source /home/usd.local/kiran.prasannannair/miniforge3/etc/profile.d/conda.sh
set +u
conda activate "$ENV_PATH"
set -u

export TMPDIR=${TMPDIR:-/tmp}

python --version
python - <<'PY'
import numpy
from PIL import Image
print(f"NumPy: {numpy.__version__}")
print("Pillow import: ok")
try:
    import scipy
    print(f"SciPy: {scipy.__version__}")
except Exception as exc:
    print(f"SciPy unavailable; using fallback Spearman p-values: {exc}")
PY

python -m compileall scripts/evaluate/complexity_stratified_errors.py
bash -n jobs/complexity_stratified_errors.sh

/usr/bin/time -v python scripts/evaluate/complexity_stratified_errors.py \
  --scene garden=results/garden/3dgs_vs_ges_vs_drk_p32/patches.csv \
  --scene bicycle=results/bicycle/3dgs_vs_ges_vs_drk_p32/patches.csv \
  --scene room=results/room/3dgs_vs_ges_vs_drk_p32/patches.csv \
  --output-dir "$OUTPUT_DIR" \
  --bootstrap-replicates 2000 \
  --seed 1729

find "$OUTPUT_DIR" -maxdepth 2 -type f | sort
