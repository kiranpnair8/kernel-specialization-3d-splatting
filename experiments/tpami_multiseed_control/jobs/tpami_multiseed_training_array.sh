#!/bin/bash
#SBATCH --job-name=tpami-ms-control
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodelist=gpu005
#SBATCH --array=0-3%1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/home/rizk_lab/shared/kiran/kernel-specialization-3d-splatting/jobs/logs/tpami_multiseed/train_%A_%a.out
#SBATCH --error=/home/rizk_lab/shared/kiran/kernel-specialization-3d-splatting/jobs/logs/tpami_multiseed/train_%A_%a.err

set -eo pipefail
PROJECT_ROOT=/home/rizk_lab/shared/kiran/kernel-specialization-3d-splatting
EXPERIMENT=$PROJECT_ROOT/experiments/tpami_multiseed_control
EXP2=$PROJECT_ROOT/experiments/rebuttal_independent_dataset
METHODS=(3dgs 3dgs ges ges)
SEEDS=(1 2 1 2)
METHOD=${METHODS[$SLURM_ARRAY_TASK_ID]}
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
CONFIG=$EXPERIMENT/configs/drjohnson_seed${SEED}.json

module purge
module load cuda/12.3
source /home/usd.local/kiran.prasannannair/miniforge3/etc/profile.d/conda.sh
set +u
case "$METHOD" in
  3dgs) ENV_PATH=/home/rizk_lab/shared/kiran/envs/kernel_splat; METHOD_ROOT=$PROJECT_ROOT/external/gaussian-splatting ;;
  ges) ENV_PATH=/home/rizk_lab/shared/kiran/envs/ges_splat; METHOD_ROOT=$PROJECT_ROOT/external/ges-splatting ;;
esac
conda activate "$ENV_PATH"

export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc
export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
export CUDAHOSTCXX=$CXX
export TORCH_CUDA_ARCH_LIST="7.0"
export PYTHONHASHSEED=$SEED
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export WANDB_MODE=offline
export WANDB_SILENT=true
export WANDB_DIR=/tmp/tpami_multiseed_wandb_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
export WANDB_CACHE_DIR=$WANDB_DIR/cache
export WANDB_CONFIG_DIR=$WANDB_DIR/config
mkdir -p "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR"

echo "method=$METHOD seed=$SEED host=$(hostname) job=${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
nvidia-smi
python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable")
p = torch.cuda.get_device_properties(0)
print("PyTorch", torch.__version__, "CUDA", torch.version.cuda, "GPU", p.name, "capability", (p.major, p.minor), "GiB", p.total_memory / 2**30)
if (p.major, p.minor) != (7, 0) or p.total_memory < 30 * 2**30:
    raise SystemExit("A 32-GB V100/sm_70 GPU is required")
PY

python "$EXP2/scripts/audit_datasets.py" --config "$CONFIG" --strict
python "$PROJECT_ROOT/experiments/rebuttal_multiseed/scripts/apply_seed_control.py" --method "$METHOD" --root "$METHOD_ROOT"
python "$PROJECT_ROOT/experiments/rebuttal_multiseed/scripts/apply_seed_control.py" --method "$METHOD" --root "$METHOD_ROOT" --verify-only
RUN_ARGS=()
if [[ "${TPAMI_RETRY:-0}" == "1" ]]; then
  RUN_ARGS+=(--retry)
fi
python "$EXP2/scripts/run_independent.py" --config "$CONFIG" --scene deepblending_drjohnson --method "$METHOD" "${RUN_ARGS[@]}"
