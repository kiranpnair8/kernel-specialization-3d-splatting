#!/bin/bash
#SBATCH --job-name=drk-garden
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodelist=gpu006
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/home/rizk_lab/shared/kiran/kernel-specialization-3d-splatting/jobs/logs/drk_garden_%j.out
#SBATCH --error=/home/rizk_lab/shared/kiran/kernel-specialization-3d-splatting/jobs/logs/drk_garden_%j.err

set -eo pipefail

PROJECT_ROOT=/home/rizk_lab/shared/kiran/kernel-specialization-3d-splatting
DRK_ROOT=$PROJECT_ROOT/external/drk
DATASET=$PROJECT_ROOT/datasets/mipnerf360/garden
OUTPUT=$PROJECT_ROOT/outputs/drk/garden_baseline

mkdir -p "$OUTPUT"
mkdir -p "$PROJECT_ROOT/jobs/logs"

# -----------------------------
# Environment
# -----------------------------

module purge
module load cuda/12.3

source /home/usd.local/kiran.prasannannair/miniforge3/etc/profile.d/conda.sh
set +u
conda activate /home/rizk_lab/shared/kiran/envs/drk_splat

export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc
export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
export CUDAHOSTCXX=$CXX
export TORCH_CUDA_ARCH_LIST="7.0"

export WANDB_MODE=offline
export WANDB_SILENT=true

# -----------------------------
# Helpers
# -----------------------------

resolve_model_path() {
    python - "$OUTPUT" <<'PY'
from pathlib import Path
import sys

requested = Path(sys.argv[1])

def looks_like_model(path):
    return (
        (path / "cfg_args").exists()
        or (path / "cameras.json").exists()
        or any(path.glob("point_cloud/iteration_*/point_cloud.ply"))
    )

if looks_like_model(requested):
    print(requested)
    raise SystemExit(0)

parent = requested.parent
prefix = requested.name
candidates = [
    path for path in parent.glob(prefix + "*")
    if path.is_dir() and looks_like_model(path)
]

if not candidates:
    print(requested)
    raise SystemExit(0)

def latest_mtime(path):
    mtimes = [p.stat().st_mtime for p in path.rglob("*") if p.exists()]
    mtimes.append(path.stat().st_mtime)
    return max(mtimes)

print(max(candidates, key=latest_mtime))
PY
}

# -----------------------------
# Run information
# -----------------------------

echo "============================================================"
echo "DRK GARDEN BASELINE"
echo "============================================================"
echo "Job ID:   ${SLURM_JOB_ID:-unknown}"
echo "Node:     $(hostname)"
echo "Started:  $(date)"
echo "Dataset:  $DATASET"
echo "Output:   $OUTPUT"
echo "DRK root: $DRK_ROOT"
echo

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
PY

# -----------------------------
# Training
# -----------------------------

cd "$DRK_ROOT"

echo
echo "============================================================"
echo "TRAINING"
echo "Started: $(date)"
echo "============================================================"

/usr/bin/time -v python train.py \
    -s "$DATASET" \
    -m "$OUTPUT" \
    --eval \
    --gs_type DRK \
    --kernel_density dense \
    --cache_sort \
    --is_unbounded

echo
echo "Training completed: $(date)"

ACTUAL_OUTPUT=$(resolve_model_path)

echo
echo "Requested output: $OUTPUT"
echo "Actual output:    $ACTUAL_OUTPUT"

# -----------------------------
# Rendering
# -----------------------------

echo
echo "============================================================"
echo "RENDERING"
echo "Started: $(date)"
echo "============================================================"

python render.py \
    -m "$ACTUAL_OUTPUT" \
    --skip_train

echo
echo "Rendering completed: $(date)"

# -----------------------------
# Metrics
# -----------------------------

echo
echo "============================================================"
echo "METRICS"
echo "Started: $(date)"
echo "============================================================"

python metrics.py \
    -m "$ACTUAL_OUTPUT"

echo
echo "Metrics completed: $(date)"

# -----------------------------
# Representation statistics
# -----------------------------

echo
echo "============================================================"
echo "REPRESENTATION STATISTICS"
echo "============================================================"

du -sh "$ACTUAL_OUTPUT"

python - <<PY
from pathlib import Path
from plyfile import PlyData

root = Path("$ACTUAL_OUTPUT")

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
PY

echo
echo "============================================================"
echo "DRK GARDEN BASELINE COMPLETE"
echo "Finished: $(date)"
echo "============================================================"
