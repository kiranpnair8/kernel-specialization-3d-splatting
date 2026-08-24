#!/bin/bash
#SBATCH --job-name=cross-scene-specialization
#SBATCH --partition=nodes
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=jobs/logs/cross_scene_specialization_%j.out
#SBATCH --error=jobs/logs/cross_scene_specialization_%j.err

set -eo pipefail

PROJECT_ROOT=/home/rizk_lab/shared/kiran/kernel-specialization-3d-splatting
ENV_PATH=/home/rizk_lab/shared/kiran/envs/kernel_splat
OUTPUT_DIR=$PROJECT_ROOT/results/cross_scene/garden_vs_bicycle_p32

mkdir -p "$PROJECT_ROOT/jobs/logs"
cd "$PROJECT_ROOT"

source /home/rizk_lab/shared/kiran/miniconda3/etc/profile.d/conda.sh
conda activate "$ENV_PATH"

python --version
python - <<'PY'
import numpy
from PIL import Image
print(f"NumPy: {numpy.__version__}")
print("Pillow import: ok")
PY

/usr/bin/time -v python scripts/evaluate/cross_scene_specialization.py \
  --scene garden=results/garden/3dgs_vs_ges_vs_drk_p32 \
  --scene bicycle=results/bicycle/3dgs_vs_ges_vs_drk_p32 \
  --output-dir "$OUTPUT_DIR"

find "$OUTPUT_DIR" -maxdepth 2 -type f | sort
