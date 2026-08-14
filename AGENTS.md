# Codex Instructions

## Project

This repository supports Paper 1:
"Beyond a Universal Kernel Family: Understanding Local Representation Bias in 3D Splatting."

The scientific objective is to test whether different mathematical
kernel families retain complementary local representation biases even
after strong within-family adaptation.

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

## External repositories

External implementations should remain outside this repository as sibling directories.

Expected layout:

~/kernel-specialization-3d-splatting
~/external/gaussian-splatting
~/external/ges-splatting
~/external/drk

Each external method may use its own environment.

## Experimental principles

Comparisons must prioritize:
- identical scene/train-test splits
- matched primitive budgets where appropriate
- matched model size / parameter budget where appropriate
- reproducible configurations
- exact Git commit tracking
- exact environment tracking
- identical evaluation code whenever possible
