# Paper 1 Project Plan

Source context: `Is Gaussian Enough_ .pdf`, provided as high-level project context.

## Canonical Working Repository

Use `kiranpnair8/kernel-specialization-3d-splatting` on `main` as the canonical
working repository for Paper 1.

## Paper Framing

Paper 1 is a hypothesis-driven characterization study of cross-family kernel
specialization in 3D splatting. It is not a "beat Gaussian" methods paper.

The central question is:

> Is one locally adaptive kernel family sufficient for heterogeneous 3D scene
> structure, or do fundamentally different kernel families retain complementary
> representation biases?

The study should test, rather than assume, the hypothesis that the best kernel
family depends on local scene structure and representation budget:

```text
F*(S, B) = preferred kernel family
S = measurable local scene structure
B = representation/resource budget
```

## Competing Possibilities

Paper 1 must preserve both possible outcomes:

- Cross-family specialization exists: different structures consistently prefer
  different kernel families, even after each family is fully optimized.
- One adaptive family is sufficient: GES, DRK, or another expressive family
  captures most or all useful specialization.

Either result is scientifically meaningful. The paper should not be designed to
force a heterogeneous-kernel conclusion.

## Core Experimental Standard

The important comparison is not simply Gaussian versus a heterogeneous method.
Gaussian should remain a baseline, but modern adaptive methods are stronger
scientific competitors.

The critical comparison is:

```text
best adaptive single-family representation
vs.
oracle cross-family local selection
```

Candidate families should ideally include mathematically distinct properties:

- generalized exponential / exponential-decay families
- heavy-tailed families
- compact-support families
- highly deformable radial families such as DRK
- possibly oscillatory or sinc-like families

Analyze specialization against local structure and budget using matched and
tracked experimental conditions.

## Research Questions

- RQ1: Does cross-family specialization exist after strong within-family
  adaptation?
- RQ2: Which mathematical properties explain specialization?
- RQ3: Does representation budget change the preferred family?
- RQ4: Can local structure predict the preferred family?
- RQ5: How much heterogeneous headroom exists relative to the strongest adaptive
  homogeneous representation?
- RQ6: Can one sufficiently flexible family eliminate most or all cross-family
  benefits?

## Go / No-Go Checkpoints

- Around Aug 23: Can at least 3-4 families be trained and evaluated reliably?
- Around Aug 29: Do real scenes exhibit meaningful local cross-family
  specialization?
- Around Sep 5: Does the cross-family oracle meaningfully outperform the
  strongest adaptive family?

If these checkpoints fail, the project scope or ICLR story should change rather
than forcing the original hypothesis.

## GPU Session Context

The project-plan PDF records this as the standard GPU setup context for future
sessions/jobs:

```bash
module load cuda/12.3
conda activate envs/kernel_splat
export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc
export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
export CUDAHOSTCXX=$CXX
export TORCH_CUDA_ARCH_LIST="7.0"
```

Individual external method repositories may still require method-specific
environments when documented by an experiment.
