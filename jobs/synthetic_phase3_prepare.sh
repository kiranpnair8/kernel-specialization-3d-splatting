#!/bin/bash
#SBATCH --job-name=synthetic-phase3-prepare
#SBATCH --partition=nodes
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/home/rizk_lab/shared/kiran/kernel-specialization-3d-splatting/jobs/logs/synthetic_phase3_prepare_%j.out
#SBATCH --error=/home/rizk_lab/shared/kiran/kernel-specialization-3d-splatting/jobs/logs/synthetic_phase3_prepare_%j.err

set -eo pipefail

PROJECT_ROOT=/home/rizk_lab/shared/kiran/kernel-specialization-3d-splatting
ENV_PATH=/home/rizk_lab/shared/kiran/envs/kernel_splat
DATASET_ROOT=$PROJECT_ROOT/datasets/synthetic/phase3_controlled_pilot

mkdir -p "$PROJECT_ROOT/jobs/logs"
cd "$PROJECT_ROOT"

source /home/usd.local/kiran.prasannannair/miniforge3/etc/profile.d/conda.sh
set +u
conda activate "$ENV_PATH"
set -u

export TMPDIR=${TMPDIR:-/tmp}

python --version
python -m compileall scripts/synthetic/generate_controlled_pilot.py scripts/synthetic/prepare_nerf_synthetic_inputs.py
bash -n jobs/synthetic_phase3_prepare.sh jobs/synthetic_phase3_loader_check.sh jobs/synthetic_phase3_3dgs_array.sh jobs/synthetic_phase3_ges_array.sh jobs/synthetic_phase3_drk_array.sh

/usr/bin/time -v python scripts/synthetic/generate_controlled_pilot.py \
  --config configs/synthetic/phase3_controlled_pilot.json

/usr/bin/time -v python scripts/synthetic/prepare_nerf_synthetic_inputs.py \
  --dataset-root "$DATASET_ROOT" \
  --point-count 100000 \
  --seed 0

python -m json.tool "$DATASET_ROOT/manifest.json" >/dev/null
python -m json.tool "$DATASET_ROOT/synthetic_input_preparation.json" >/dev/null
find "$DATASET_ROOT" -maxdepth 2 -type f | sort | tail -50
