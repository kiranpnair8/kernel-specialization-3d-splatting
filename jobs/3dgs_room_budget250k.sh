#!/bin/bash
#SBATCH --job-name=3dgs-room-250k
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodelist=gpu005
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/rizk_lab/shared/kiran/kernel-specialization-3d-splatting/jobs/logs/3dgs_room_budget250k_%j.out
#SBATCH --error=/home/rizk_lab/shared/kiran/kernel-specialization-3d-splatting/jobs/logs/3dgs_room_budget250k_%j.err

set -eo pipefail

PROJECT_ROOT=/home/rizk_lab/shared/kiran/kernel-specialization-3d-splatting
GS_ROOT=$PROJECT_ROOT/external/gaussian-splatting
DATASET=$PROJECT_ROOT/datasets/mipnerf360/room
OUTPUT=$PROJECT_ROOT/outputs/3dgs/room_budget250k
MAX_PRIMITIVES=250000

mkdir -p "$OUTPUT"
mkdir -p "$PROJECT_ROOT/jobs/logs"

module purge
module load cuda/12.3

source /home/usd.local/kiran.prasannannair/miniforge3/etc/profile.d/conda.sh
set +u
conda activate /home/rizk_lab/shared/kiran/envs/kernel_splat
set -u

export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc
export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
export CUDAHOSTCXX=$CXX
export TORCH_CUDA_ARCH_LIST="7.0"
export WANDB_MODE=offline
export WANDB_SILENT=true

echo "============================================================"
echo "3DGS ROOM 250K PRIMITIVE-BUDGET VARIANT"
echo "============================================================"
echo "Job ID:          ${SLURM_JOB_ID:-unknown}"
echo "Node:            $(hostname)"
echo "Started:         $(date)"
echo "Dataset:         $DATASET"
echo "Output:          $OUTPUT"
echo "Max primitives:  $MAX_PRIMITIVES"
echo "3DGS root:       $GS_ROOT"
echo

python "$PROJECT_ROOT/scripts/setup/apply_primitive_budget_control.py" --project-root "$PROJECT_ROOT" --install --smoke-test

nvidia-smi

echo
python - <<'PY'
import sys
import torch
print("Python:", sys.version.replace("\n", " "))
print("Executable:", sys.executable)
print("PyTorch:", torch.__version__)
print("Torch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("Capability:", torch.cuda.get_device_capability(0))
import diff_gaussian_rasterization
from simple_knn._C import distCUDA2
print("3DGS rasterizer: OK")
print("3DGS simple_knn: OK")
PY

cd "$GS_ROOT"

echo
echo "============================================================"
echo "TRAINING"
echo "Started: $(date)"
echo "============================================================"

/usr/bin/time -v python train.py \
    -s "$DATASET" \
    -m "$OUTPUT" \
    --eval \
    --max_primitives "$MAX_PRIMITIVES"

echo
echo "Training completed: $(date)"

echo
echo "============================================================"
echo "RENDERING"
echo "Started: $(date)"
echo "============================================================"

python render.py \
    -m "$OUTPUT" \
    --skip_train

echo
echo "Rendering completed: $(date)"

echo
echo "============================================================"
echo "METRICS"
echo "Started: $(date)"
echo "============================================================"

python metrics.py \
    -m "$OUTPUT"

echo
echo "Metrics completed: $(date)"

echo
echo "============================================================"
echo "REPRESENTATION STATISTICS"
echo "============================================================"

du -sh "$OUTPUT"

python - <<PY
from pathlib import Path
from plyfile import PlyData
root = Path("$OUTPUT")
plys = list(root.glob("point_cloud/iteration_*/point_cloud.ply"))
if not plys:
    print("No PLY found.")
else:
    def iteration(p):
        return int(p.parent.name.split("_")[-1])
    path = max(plys, key=iteration)
    ply = PlyData.read(str(path))
    print("Final PLY:", path)
    print("Final iteration:", iteration(path))
    print("Final primitive count:", len(ply["vertex"]))
    print("PLY size (MB):", path.stat().st_size / (1024**2))
    if len(ply["vertex"]) > int("$MAX_PRIMITIVES"):
        raise SystemExit("Final primitive count exceeds cap")
PY

echo
echo "============================================================"
echo "3DGS ROOM 250K PRIMITIVE-BUDGET VARIANT COMPLETE"
echo "Finished: $(date)"
echo "============================================================"
