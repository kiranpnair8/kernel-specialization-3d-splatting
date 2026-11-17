#!/bin/bash
#SBATCH --job-name=rebuttal-independent
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodelist=gpu005
#SBATCH --array=0-5%1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/home/rizk_lab/shared/kiran/kernel-specialization-3d-splatting/jobs/logs/rebuttal_independent/train_%A_%a.out
#SBATCH --error=/home/rizk_lab/shared/kiran/kernel-specialization-3d-splatting/jobs/logs/rebuttal_independent/train_%A_%a.err

set -eo pipefail
PROJECT_ROOT=/home/rizk_lab/shared/kiran/kernel-specialization-3d-splatting
EXPERIMENT=$PROJECT_ROOT/experiments/rebuttal_independent_dataset
CONFIG=$EXPERIMENT/configs/independent_scenes.json

SCENES=(tandt_train tandt_train tandt_train deepblending_drjohnson deepblending_drjohnson deepblending_drjohnson)
METHODS=(3dgs ges drk 3dgs ges drk)
SCENE=${SCENES[$SLURM_ARRAY_TASK_ID]}
METHOD=${METHODS[$SLURM_ARRAY_TASK_ID]}

module purge
module load cuda/12.3
source /home/usd.local/kiran.prasannannair/miniforge3/etc/profile.d/conda.sh
set +u
case "$METHOD" in
  3dgs) ENV_PATH=/home/rizk_lab/shared/kiran/envs/kernel_splat; METHOD_ROOT=$PROJECT_ROOT/external/gaussian-splatting ;;
  ges)  ENV_PATH=/home/rizk_lab/shared/kiran/envs/ges_splat;    METHOD_ROOT=$PROJECT_ROOT/external/ges-splatting ;;
  drk)  ENV_PATH=/home/rizk_lab/shared/kiran/envs/drk_splat;    METHOD_ROOT=$PROJECT_ROOT/external/drk ;;
esac
conda activate "$ENV_PATH"

export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc
export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
export CUDAHOSTCXX=$CXX
export TORCH_CUDA_ARCH_LIST="7.0"
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export WANDB_MODE=offline
export WANDB_SILENT=true
export WANDB_DIR=/tmp/rebuttal_independent_wandb_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
export WANDB_CACHE_DIR=$WANDB_DIR/cache
export WANDB_CONFIG_DIR=$WANDB_DIR/config
mkdir -p "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR"

echo "scene=$SCENE method=$METHOD host=$(hostname) job=${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
nvidia-smi
python - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable")
properties = torch.cuda.get_device_properties(0)
print("PyTorch:", torch.__version__, "CUDA:", torch.version.cuda)
print("GPU:", properties.name, "capability:", f"{properties.major}.{properties.minor}", "memory_GiB:", properties.total_memory / 2**30)
if (properties.major, properties.minor) != (7, 0):
    raise SystemExit("This job requires a V100/sm_70 GPU")
if properties.total_memory < 30 * 2**30:
    raise SystemExit("This experiment is configured for a 32-GB V100")
PY
python "$EXPERIMENT/scripts/audit_datasets.py" --config "$CONFIG" --strict
python "$PROJECT_ROOT/experiments/rebuttal_multiseed/scripts/apply_seed_control.py" --method "$METHOD" --root "$METHOD_ROOT"
python "$EXPERIMENT/scripts/run_independent.py" --config "$CONFIG" --scene "$SCENE" --method "$METHOD"
