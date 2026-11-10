#!/bin/bash
#SBATCH --job-name=rebuttal-seed-analysis
#SBATCH --partition=nodes
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/home/rizk_lab/shared/kiran/kernel-specialization-3d-splatting/jobs/logs/rebuttal_multiseed/analysis_%j.out
#SBATCH --error=/home/rizk_lab/shared/kiran/kernel-specialization-3d-splatting/jobs/logs/rebuttal_multiseed/analysis_%j.err

set -eo pipefail
PROJECT_ROOT=/home/rizk_lab/shared/kiran/kernel-specialization-3d-splatting
EXPERIMENT=$PROJECT_ROOT/experiments/rebuttal_multiseed
CONFIG=$EXPERIMENT/configs/room_multiseed.json

source /home/usd.local/kiran.prasannannair/miniforge3/etc/profile.d/conda.sh
set +u
conda activate /home/rizk_lab/shared/kiran/envs/kernel_splat

cd "$PROJECT_ROOT"
python "$EXPERIMENT/scripts/evaluate_multiseed.py" --config "$CONFIG"
python "$EXPERIMENT/scripts/analyze_within_family.py" --config "$CONFIG"
python "$EXPERIMENT/scripts/analyze_cross_family.py" --config "$CONFIG"
python "$EXPERIMENT/scripts/generate_rebuttal_table.py" --config "$CONFIG"
