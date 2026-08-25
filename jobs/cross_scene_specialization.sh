#!/bin/bash
#SBATCH --job-name=cross-scene-specialization
#SBATCH --partition=nodes
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/home/rizk_lab/shared/kiran/kernel-specialization-3d-splatting/jobs/logs/cross_scene_specialization_%j.out
#SBATCH --error=/home/rizk_lab/shared/kiran/kernel-specialization-3d-splatting/jobs/logs/cross_scene_specialization_%j.err

set -eo pipefail

PROJECT_ROOT=/home/rizk_lab/shared/kiran/kernel-specialization-3d-splatting
ENV_PATH=/home/rizk_lab/shared/kiran/envs/kernel_splat
OUTPUT_DIR=$PROJECT_ROOT/results/cross_scene/garden_bicycle_room_p32

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
PY

/usr/bin/time -v python scripts/evaluate/cross_scene_specialization.py \
  --scene garden=results/garden/3dgs_vs_ges_vs_drk_p32 \
  --scene bicycle=results/bicycle/3dgs_vs_ges_vs_drk_p32 \
  --scene room=results/room/3dgs_vs_ges_vs_drk_p32 \
  --output-dir "$OUTPUT_DIR"

find "$OUTPUT_DIR" -maxdepth 2 -type f | sort
