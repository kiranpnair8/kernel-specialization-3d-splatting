#!/bin/bash
#SBATCH --job-name=rebuttal-3dgs-seeds
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodelist=gpu005
#SBATCH --array=0-2%1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/rizk_lab/shared/kiran/kernel-specialization-3d-splatting/jobs/logs/rebuttal_multiseed/3dgs_%A_%a.out
#SBATCH --error=/home/rizk_lab/shared/kiran/kernel-specialization-3d-splatting/jobs/logs/rebuttal_multiseed/3dgs_%A_%a.err

set -eo pipefail
PROJECT_ROOT=/home/rizk_lab/shared/kiran/kernel-specialization-3d-splatting
EXPERIMENT=$PROJECT_ROOT/experiments/rebuttal_multiseed
METHOD_ROOT=$PROJECT_ROOT/external/gaussian-splatting
CONFIG=$EXPERIMENT/configs/room_multiseed.json
SEED=${SLURM_ARRAY_TASK_ID}

module purge
module load cuda/12.3
source /home/usd.local/kiran.prasannannair/miniforge3/etc/profile.d/conda.sh
set +u
conda activate /home/rizk_lab/shared/kiran/envs/kernel_splat

export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc
export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
export CUDAHOSTCXX=$CXX
export TORCH_CUDA_ARCH_LIST="7.0"
export PYTHONHASHSEED=$SEED
export CUBLAS_WORKSPACE_CONFIG=:4096:8

python "$EXPERIMENT/scripts/apply_seed_control.py" --method 3dgs --root "$METHOD_ROOT"
python "$EXPERIMENT/scripts/run_multiseed.py" --config "$CONFIG" --method 3dgs --seed "$SEED"
