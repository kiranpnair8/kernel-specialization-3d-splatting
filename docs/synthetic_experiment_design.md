# Phase III Controlled Synthetic-Scene Pilot

## Purpose

This pilot creates a reproducible synthetic benchmark for Paper-1 that changes one structural factor at a time while holding cameras, lighting, scene scale, resolution, train/test split, and unrelated appearance factors fixed. It is a controlled benchmark scaffold, not a downstream training sweep.

The pilot is intended to test whether the local-comparison machinery can be applied to scenes whose structure is known by construction. It should not be used to claim causal kernel specialization until the generated datasets, method training, alignment, local comparisons, sensitivity analysis, and replication checks have all been validated.

## Dataset Format And Conversion Path

The generator writes a NeRF-synthetic style dataset:

- RGB PNG images under `train/` and `test/`.
- `transforms_train.json` and `transforms_test.json` with `camera_angle_x`, `file_path`, and camera-to-world `transform_matrix` entries.
- One `metadata.json` file per scene.
- Root-level `manifest.csv` and `manifest.json` across all pilot scenes.

This is the cleanest first target because the 3DGS lineage commonly supports Blender/NeRF-synthetic transform datasets in addition to COLMAP-style scenes, and the exact intrinsics/extrinsics are analytic. The upstream method repositories remain gitignored under `external/`, so the final accepted invocation for 3DGS, GES, and DRK must still be confirmed against each local HPC checkout before training jobs are added.

If any method fork requires a COLMAP directory instead, the conversion path is deterministic: use the known focal length, image size, and camera-to-world poses from the transforms files to emit `cameras`, `images`, and an optional sparse point scaffold. That converter should be added only after pilot generation is validated.

## What Varies

| Sweep family | Controlled parameter | Low | Medium | High |
| --- | --- | ---: | ---: | ---: |
| `edge_sharpness` | `transition_width` | 0.32 | 0.12 | 0.035 |
| `spatial_frequency` | `cycles_per_scene_width` | 2.0 | 6.0 | 14.0 |
| `curvature` | `paraboloid_amplitude` | 0.02 | 0.18 | 0.30 |

`edge_sharpness` renders a planar scene with a smooth-to-sharp color transition. Only the transition width changes within a seed.

`spatial_frequency` renders identical planar geometry with an analytic sinusoidal texture. Only the texture frequency changes within a seed; material amplitude and lighting stay fixed.

`curvature` renders a bounded paraboloid-like surface. Only the geometry curvature amplitude changes within a seed.

## What Is Fixed

- Resolution: `256x256`.
- Train views: `24` fixed orbit cameras.
- Test views: `8` fixed orbit cameras offset from train views.
- Camera radius: `2.8`.
- Camera elevation: `35` degrees.
- Horizontal field of view: `50` degrees.
- Scene extent: `1.15`.
- Background color: white.
- Lighting direction and diffuse/ambient shading rule.
- Scene scale and coordinate system.

## Seed Replication Protocol

Seed `0` is the completed canonical pilot. Phase III-B adds seeds `1,2,3,4`, giving five independent scene realizations per condition while preserving the same cameras, resolution, train/test counts, rendering protocol, scene scale, lighting, and controlled factor levels.

Seed-level nuisance variation is deterministic and fixed across low/medium/high levels within each sweep family:

- edge sharpness: transition orientation and small transition offset vary by seed;
- spatial frequency: texture orientation and sinusoidal phases vary by seed;
- curvature: low-contrast albedo phase/orientation varies by seed, while the curvature amplitudes stay `0.02`, `0.18`, and `0.30`;
- deterministic sparse point initialization uses the same seed as the rendered scene.

The invalid original curvature-high amplitude `0.42` seed-0 scene remains on disk for provenance but is excluded from canonical analysis. The corrected high-curvature condition uses amplitude `0.30` for every seed and is named with `phase3_curvature_high_corrected_seedXXXX`.

## Curvature Renderer Audit

The initial Phase-III stimulus audit exposed a confound in the original curvature-high scene: `paraboloid_amplitude=0.42` produced foreground fraction about `0.0067`, while curvature low/medium were about `0.5409` and `0.5443`. The high-curvature image was therefore almost entirely background and is not a valid controlled comparison.

Diagnosis: the renderer intersects rays with an infinite paraboloid and the first pilot implementation selected the nearest positive quadratic root before applying the finite square scene-extent test. At high curvature, many rays first intersect the unbounded paraboloid outside the intended finite square; those hits are rejected as out-of-bounds, and the alternate positive root that may lie inside the bounded stimulus is never considered. This makes foreground occupancy collapse as amplitude increases.

Smallest geometric fix: keep the same cameras, lighting, scene extent, appearance, resolution, and paraboloid equation, but choose the nearest positive ray-paraboloid root that is inside the finite square support. This preserves the bounded stimulus footprint while still increasing curvature from low to medium to high.

A second candidate validation showed that the corrected `0.42` high-amplitude case still under-filled the view relative to the fixed target: low `0.02` foreground `0.5409`, medium `0.18` foreground `0.5443`, high `0.42` foreground `0.4785`. The occupancy tolerance must not be loosened. Instead, the validation mode supports a curvature-high amplitude sweep over `[0.22, 0.26, 0.30, 0.34, 0.38, 0.42]`, reports foreground occupancy and stimulus diagnostics for every candidate, and identifies the largest amplitude satisfying `0.54 +/- 0.03`.

The accepted replacement high-curvature candidate is amplitude `0.30`. It is promoted as a distinct canonical scene named `phase3_curvature_high_corrected_seed0000`, leaving the original invalid `phase3_curvature_high_seed0000` untouched for provenance.

The surface-normal path avoids invalid divide warnings by using masked `np.divide` operations and validating finite geometry, normals, and rendered images. Candidate validation exits nonzero if any generated geometry, normal, or image contains NaN/Inf.

## Scene Structure

The default output root is `datasets/synthetic/phase3_controlled_pilot/`, which is intentionally under the gitignored dataset area. Seed-0 pilot scenes were named:

```text
phase3_<sweep_family>_<level>_seed0000/
```

Phase III-B uses the same naming for new seed levels except corrected curvature high:

```text
phase3_curvature_high_corrected_seed0001/
```

Example:

```text
datasets/synthetic/phase3_controlled_pilot/
  manifest.csv
  manifest.json
  phase3_edge_sharpness_low_seed0001/
    metadata.json
    transforms_train.json
    transforms_test.json
    points3d.ply
    train/r_000.png
    test/r_000.png
```

The five-seed experiment has `3 sweep families x 3 levels x 5 seeds = 45 canonical scenes`. Across 3DGS, GES, and DRK, this is 135 method-scene trainings once Phase III-B is complete.

## Manifest Fields

The root manifest records:

- `scene_id`
- `sweep_family`
- `level`
- `parameter_value`
- `seed`
- `train_view_count`
- `test_view_count`
- `resolution`
- `dataset_path`

Each scene-level `metadata.json` additionally records the parameter name, dataset format, nuisance parameters for that seed, and fixed factors used for reproducibility.

## Hypotheses For Validation

- Edge sharpness: sharper transitions may expose method-dependent differences in local reconstruction near boundaries.
- Spatial frequency: higher periodic frequencies may stress methods differently in textured regions with identical geometry.
- Curvature: increasing surface curvature may reveal differences in local geometric representation under fixed appearance protocol.

These are controlled-measurement hypotheses. They are not claims about causal kernel specialization by themselves.

## Null Outcomes

Useful null outcomes include:

- Winner fractions or global metric differences do not change systematically with the controlled parameter.
- Trends are seed-specific and do not replicate across independent scene realizations.
- Oracle gains are negligible across all levels.
- Any apparent effect disappears under patch-size or tie-threshold sensitivity checks.

These outcomes would still validate the benchmark machinery if alignment, rendering, and measurement behave as expected.

## Statistical Unit

For Phase III-B, independent scene realizations are the primary statistical units. Per-condition summaries should report mean and standard deviation across seeds, and method comparisons should use paired deltas across matched seeds. View-level bootstrap intervals are secondary diagnostics only and must not be interpreted as independent seed replication.

## Commands

Tiny local/HPC smoke generation:

```bash
python scripts/synthetic/generate_controlled_pilot.py \
  --config configs/synthetic/phase3_controlled_pilot.json \
  --tiny-test \
  --output-root /tmp/phase3_synthetic_tiny_$USER
```

Full seed-0 pilot generation, no training:

```bash
python scripts/synthetic/generate_controlled_pilot.py \
  --config configs/synthetic/phase3_controlled_pilot.json
```

Generate and prepare Phase III-B seeds 1-4, no training:

```bash
sbatch jobs/synthetic_phase3b_generate_prepare.sh
```

Train Phase III-B seeds 1-4:

```bash
sbatch jobs/synthetic_phase3b_3dgs_array.sh
sbatch jobs/synthetic_phase3b_ges_array.sh
sbatch jobs/synthetic_phase3b_drk_array.sh
```

Evaluate after all five seeds are present:

```bash
python scripts/synthetic/inventory_phase3_outputs.py --fail-on-incomplete
python scripts/synthetic/evaluate_phase3_results.py
python scripts/synthetic/analyze_phase3_seed_statistics.py
```
