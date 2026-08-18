# GES Baseline — Mip-NeRF 360 Garden

## Purpose

Establish the Generalized Exponential Splatting (GES) baseline on the
Mip-NeRF 360 Garden scene for the kernel-specialization study.

## Implementation

Repository:
ajhamdi/ges-splatting

Commit:
c05fc7dbb22e270a5a6f490e7040adac4af02c96

Training script:
train_ges.py

## Dataset

Dataset: Mip-NeRF 360
Scene: garden

Local path:
datasets/mipnerf360/garden

Images:
images -> images_4

COLMAP:
sparse -> sparse_4

Evaluation:
LLFF holdout / 24 test views

## Compute Environment

GPU: NVIDIA Tesla V100-PCIE-32GB
GPU memory: 32 GB

PyTorch: 2.5.1+cu121
CUDA runtime used by PyTorch: 12.1
CUDA toolkit: 12.3

TORCH_CUDA_ARCH_LIST=7.0

GES Conda environment:
envs/ges_splat

## Training

Iterations: 40,000
Wall-clock training time: 39m 29.5s

Final training loss:
0.0286078

Output directory:
outputs/ges/garden_baseline_00_2026-08-17--17-16-41

## Evaluation Results

PSNR: 27.1016693 dB
SSIM: 0.8524919
LPIPS: 0.1347301

Test views: 24

## Representation Statistics

Final primitive count:
2,709,992

Output directory size:
1.3 GB

## Notes

GES automatically appends the experiment set and timestamp to the
requested model path.

Requested path:

outputs/ges/garden_baseline

Actual path:

outputs/ges/garden_baseline_00_2026-08-17--17-16-41

The original Slurm workflow therefore completed training successfully
but failed during the subsequent rendering step because it searched for
cfg_args in the unsuffixed output directory.

Rendering and metric evaluation were subsequently completed manually
using the actual timestamped model directory.

## Baseline Status

Training: PASS
Rendering: PASS
Evaluation: PASS

This baseline is considered reproducible and may be used as the initial
GES Garden reference for subsequent experiments.