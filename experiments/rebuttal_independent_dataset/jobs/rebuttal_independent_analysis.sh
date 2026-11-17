#!/bin/bash
#SBATCH --job-name=rebuttal-independent-analysis
#SBATCH --partition=nodes
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/home/rizk_lab/shared/kiran/kernel-specialization-3d-splatting/jobs/logs/rebuttal_independent/analysis_%j.out
#SBATCH --error=/home/rizk_lab/shared/kiran/kernel-specialization-3d-splatting/jobs/logs/rebuttal_independent/analysis_%j.err

set -eo pipefail
PROJECT_ROOT=/home/rizk_lab/shared/kiran/kernel-specialization-3d-splatting
EXPERIMENT=$PROJECT_ROOT/experiments/rebuttal_independent_dataset
CONFIG=$EXPERIMENT/configs/independent_scenes.json

source /home/usd.local/kiran.prasannannair/miniforge3/etc/profile.d/conda.sh
set +u
conda activate /home/rizk_lab/shared/kiran/envs/kernel_splat
export MPLBACKEND=Agg
export TMPDIR=/tmp/rebuttal_independent_analysis_${SLURM_JOB_ID}
mkdir -p "$TMPDIR"

cd "$PROJECT_ROOT"
python "$EXPERIMENT/scripts/audit_datasets.py" --config "$CONFIG" --strict
python "$EXPERIMENT/scripts/evaluate_independent.py" --config "$CONFIG"
python "$EXPERIMENT/scripts/analyze_independent.py" --config "$CONFIG"
