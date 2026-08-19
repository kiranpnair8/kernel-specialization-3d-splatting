# DRK Baseline - Mip-NeRF 360 Garden

## Purpose

Establish the Deformable Radial Kernel (DRK) Garden baseline for the
kernel-specialization study and integrate its held-out outputs into the
family-agnostic local comparison pipeline.

## Implementation

Repository:
`external/drk`

Upstream:
`VAST-AI-Research/Deformable-Radial-Kernel-Splatting`

Training/evaluation script:
`train.py`

## Dataset

Dataset: Mip-NeRF 360
Scene: garden

Local path:
`datasets/mipnerf360/garden`

Evaluation:
LLFF holdout / 24 test views

## Compute Environment

GPU: NVIDIA Tesla V100 / sm_70
CUDA module: cuda/12.3
PyTorch: 2.2.2+cu121

DRK Conda environment:
`/home/rizk_lab/shared/kiran/envs/drk_splat`

## Configuration

The validated Garden baseline uses the DRK repository's documented Mip-NeRF 360
settings:

```bash
python train.py \
    -s "$DATASET" \
    -m "$OUTPUT" \
    --eval \
    --gs_type DRK \
    --kernel_density dense \
    --cache_sort \
    --is_unbounded
```

## Evaluation Path

DRK evaluation uses the repository's internal metric pathway:

```bash
python train.py \
    -s "$DATASET" \
    -m "$OUTPUT" \
    --eval \
    --gs_type DRK \
    --kernel_density dense \
    --cache_sort \
    --is_unbounded \
    --metric \
    --load_iteration -1
```

This is intentionally different from the 3DGS/GES `render.py` plus `metrics.py`
path. DRK appends `_${gs_type}` internally when constructing the actual model
directory, so evaluation should pass the same base `-m "$OUTPUT"` path used for
training.

## Output

Requested output:
`outputs/drk/garden_baseline`

Actual output:
`outputs/drk/garden_baseline_DRK`

Held-out outputs:
`outputs/drk/garden_baseline_DRK/metric/test/`

The held-out directory contains 24 each of:

- `gt_XXXXX.png`
- `render_XXXXX.png`
- `errormap_XXXXX.png`

## Final Metrics

Source:
`outputs/drk/garden_baseline_DRK/metric/test_35000.txt`

- PSNR: 25.43460
- SSIM: 0.91927
- LPIPS: 0.18802
- L1: 0.03511
- Final primitive count: 614,345
- Final iteration: 35,000
- Training wall time: approximately 3h 53m

## Integration Notes

Before using DRK in local patch comparisons, run the read-only GT alignment
audit against the 3DGS Garden GT frames. Do not assume that DRK
`gt_00000.png` through `gt_00023.png` correspond to 3DGS/GES `00000.png`
through `00023.png` without confirming pixel-level equivalence.
