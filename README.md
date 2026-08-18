# Kernel Specialization in 3D Splatting

Research code for a hypothesis-driven characterization study of
cross-family kernel specialization in 3D scene representation.

Canonical working repository: `kiranpnair8/kernel-specialization-3d-splatting`
on `main`.

## Central question

Is one locally adaptive kernel family sufficient for heterogeneous
3D scene structure, or do fundamentally different kernel families
retain complementary representation biases?

## Project scope

Paper 1 focuses on:
- controlled kernel-family comparisons
- matched-budget evaluation
- local reconstruction error analysis
- local winner maps
- structural descriptors
- oracle cross-family selection
- statistical characterization

Paper 1 does NOT implement MoKES or a heterogeneous routing architecture.

See [docs/project_plan_paper1.md](docs/project_plan_paper1.md) for the
high-level project framing and go/no-go checkpoints.

## Local HPC checkout

The working HPC checkout is expected to contain gitignored upstream method
repositories under `external/`:

- `external/gaussian-splatting`:
  `https://github.com/graphdeco-inria/gaussian-splatting` at
  `54c035f7834b564019656c3e3fcc3646292f727d`
- `external/ges-splatting`:
  `https://github.com/ajhamdi/ges-splatting` at
  `c05fc7dbb22e270a5a6f490e7040adac4af02c96`

The directories `external/`, `datasets/`, `outputs/`, `checkpoints/`,
`renders/`, and experiment logs are intentionally gitignored. Do not copy
upstream code or large experiment artifacts into this repository unless
explicitly requested.

Current validated Garden baselines are documented in
[docs/baseline_3dgs_garden.md](docs/baseline_3dgs_garden.md) and
[docs/baseline_ges_garden.md](docs/baseline_ges_garden.md).
