# Stage-0 Pipeline Validation

The real Garden 3DGS-vs-GES local comparison pipeline has been validated on all
24 held-out views.

This stage exists to validate the analysis machinery:

- paired held-out view loading
- per-image and patch-level error computation
- winner maps and margin maps
- structural descriptor export
- post-hoc oracle summaries
- classical descriptor-to-winner diagnostics
- patch-size and tie-threshold sensitivity aggregation

Do not interpret Stage-0 3DGS-vs-GES winners as evidence of cross-family kernel
specialization. Gaussian is nested inside GES, and this experiment is only a
pipeline validation step before adding additional independently motivated
families.
