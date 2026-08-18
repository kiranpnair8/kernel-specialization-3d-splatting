# Codex Instructions

## Project

This repository supports Paper 1:
"Beyond a Universal Kernel Family: Understanding Local Representation Bias in 3D Splatting."

Use `kiranpnair8/kernel-specialization-3d-splatting` on `main` as the canonical
working repository.

The scientific objective is to test whether different mathematical
kernel families retain complementary local representation biases even
after strong within-family adaptation.

High-level project context lives in `docs/project_plan_paper1.md`.

## Do not assume the hypothesis is true

The study must allow the possibility that one adaptive family, such as
GES or DRK, captures essentially all useful specialization.

## Scope

This repository should contain:
- experiment orchestration
- standardized method wrappers
- evaluation scripts
- matched-budget protocols
- local error analysis
- kernel-family winner maps
- local structural descriptors
- oracle cross-family analysis
- statistical analysis
- experiment documentation

## Do NOT

- implement MoKES unless explicitly requested
- implement a heterogeneous multi-kernel rasterizer unless explicitly requested
- silently change method hyperparameters
- commit datasets or large checkpoints
- modify external research repositories unless explicitly requested
- copy upstream method code into this repository unless explicitly requested
- remove the gitignore policy for external repositories, datasets, outputs,
  checkpoints, renders, or experiment logs

## External repositories

External implementations are expected to exist as gitignored directories inside
the working HPC checkout. They are not tracked by this repository, so Codex will
not discover them from Git alone.

Expected layout:

```text
kernel-specialization-3d-splatting/
|-- external/
|   |-- gaussian-splatting/
|   `-- ges-splatting/
```

Pinned upstream sources:

- `external/gaussian-splatting`
  - repository: `https://github.com/graphdeco-inria/gaussian-splatting`
  - commit: `54c035f7834b564019656c3e3fcc3646292f727d`
- `external/ges-splatting`
  - repository: `https://github.com/ajhamdi/ges-splatting`
  - commit: `c05fc7dbb22e270a5a6f490e7040adac4af02c96`

Each external method may use its own environment.

The directories `external/`, `datasets/`, `outputs/`, `checkpoints/`,
`renders/`, and experiment logs are intentionally not tracked by Git.

## Validated baselines

- 3DGS Garden, Mip-NeRF 360 `images_4`, 30k iterations:
  PSNR 27.4781, SSIM 0.8680687, LPIPS 0.1060733, 4,146,866 primitives,
  approximately 44 minutes.
- GES Garden, Mip-NeRF 360 `images_4`, 40k iterations:
  PSNR 27.1016693, SSIM 0.8524919, LPIPS 0.1347301, 2,709,992 primitives,
  approximately 39.5 minutes.

## Experimental principles

Comparisons must prioritize:
- identical scene/train-test splits
- matched primitive budgets where appropriate
- matched model size / parameter budget where appropriate
- reproducible configurations
- exact Git commit tracking
- exact environment tracking
- identical evaluation code whenever possible
