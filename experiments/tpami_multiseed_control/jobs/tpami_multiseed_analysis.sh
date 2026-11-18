#!/bin/bash
#SBATCH --job-name=tpami-ms-analysis
#SBATCH --partition=nodes
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/home/rizk_lab/shared/kiran/kernel-specialization-3d-splatting/jobs/logs/tpami_multiseed/analysis_%j.out
#SBATCH --error=/home/rizk_lab/shared/kiran/kernel-specialization-3d-splatting/jobs/logs/tpami_multiseed/analysis_%j.err

set -eo pipefail
PROJECT_ROOT=/home/rizk_lab/shared/kiran/kernel-specialization-3d-splatting
EXPERIMENT=$PROJECT_ROOT/experiments/tpami_multiseed_control
CONFIG=$EXPERIMENT/configs/tpami_multiseed_control.json

source /home/usd.local/kiran.prasannannair/miniforge3/etc/profile.d/conda.sh
set +u
conda activate /home/rizk_lab/shared/kiran/envs/kernel_splat
export MPLBACKEND=Agg
export TMPDIR=/tmp/tpami_multiseed_analysis_${SLURM_JOB_ID}
mkdir -p "$TMPDIR"

cd "$PROJECT_ROOT"
python "$EXPERIMENT/scripts/evaluate_six_models.py" --config "$CONFIG"
python "$EXPERIMENT/scripts/analyze_matched_oracles.py" --config "$CONFIG"
