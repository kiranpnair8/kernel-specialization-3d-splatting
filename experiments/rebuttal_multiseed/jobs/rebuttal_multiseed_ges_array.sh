#!/bin/bash
#SBATCH --job-name=rebuttal-ges-seeds
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodelist=gpu004
#SBATCH --array=0-2%1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/rizk_lab/shared/kiran/kernel-specialization-3d-splatting/jobs/logs/rebuttal_multiseed/ges_%A_%a.out
#SBATCH --error=/home/rizk_lab/shared/kiran/kernel-specialization-3d-splatting/jobs/logs/rebuttal_multiseed/ges_%A_%a.err

set -eo pipefail
PROJECT_ROOT=/home/rizk_lab/shared/kiran/kernel-specialization-3d-splatting
EXPERIMENT=$PROJECT_ROOT/experiments/rebuttal_multiseed
METHOD_ROOT=$PROJECT_ROOT/external/ges-splatting
CONFIG=$EXPERIMENT/configs/room_multiseed.json
SEED=${SLURM_ARRAY_TASK_ID}

module purge
module load cuda/12.3
source /home/usd.local/kiran.prasannannair/miniforge3/etc/profile.d/conda.sh
set +u
conda activate /home/rizk_lab/shared/kiran/envs/ges_splat

export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc
export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
export CUDAHOSTCXX=$CXX
export TORCH_CUDA_ARCH_LIST="7.0"
export PYTHONHASHSEED=$SEED
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export WANDB_MODE=offline
export WANDB_SILENT=true
LOCAL_SCRATCH=${SLURM_TMPDIR:-/tmp}/ges_rebuttal_${SLURM_JOB_ID:-$$}_${SEED}
export WANDB_DIR=$LOCAL_SCRATCH/wandb
export WANDB_CACHE_DIR=$LOCAL_SCRATCH/wandb_cache
export WANDB_CONFIG_DIR=$LOCAL_SCRATCH/wandb_config
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR"

python "$EXPERIMENT/scripts/apply_seed_control.py" --method ges --root "$METHOD_ROOT"
python "$EXPERIMENT/scripts/run_multiseed.py" --config "$CONFIG" --method ges --seed "$SEED"
