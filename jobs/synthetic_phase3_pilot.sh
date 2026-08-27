#!/bin/bash
#SBATCH --job-name=synthetic-phase3-pilot
#SBATCH --partition=nodes
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/home/rizk_lab/shared/kiran/kernel-specialization-3d-splatting/jobs/logs/synthetic_phase3_pilot_%j.out
#SBATCH --error=/home/rizk_lab/shared/kiran/kernel-specialization-3d-splatting/jobs/logs/synthetic_phase3_pilot_%j.err

set -eo pipefail

PROJECT_ROOT=/home/rizk_lab/shared/kiran/kernel-specialization-3d-splatting
ENV_PATH=/home/rizk_lab/shared/kiran/envs/kernel_splat

mkdir -p "$PROJECT_ROOT/jobs/logs"
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

python -m compileall scripts/synthetic/generate_controlled_pilot.py
bash -n jobs/synthetic_phase3_pilot.sh

/usr/bin/time -v python scripts/synthetic/generate_controlled_pilot.py \
  --config configs/synthetic/phase3_controlled_pilot.json

find "$PROJECT_ROOT/datasets/synthetic/phase3_controlled_pilot" -maxdepth 2 -type f | sort | tail -50
