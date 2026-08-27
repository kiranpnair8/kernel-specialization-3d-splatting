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
| `curvature` | `paraboloid_amplitude` | 0.02 | 0.18 | 0.42 |

`edge_sharpness` renders a planar scene with a smooth-to-sharp color transition. Only the transition width changes.

`spatial_frequency` renders identical planar geometry with an analytic sinusoidal texture. Only the texture frequency changes; material amplitude and lighting stay fixed.

`curvature` renders a fixed-color paraboloid-like surface. Only the geometry curvature amplitude changes.

## What Is Fixed

- Seed: `0` for the pilot.
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

## Scene Structure

The default output root is `datasets/synthetic/phase3_controlled_pilot/`, which is intentionally under the gitignored dataset area. Each generated scene is named:

```text
phase3_<sweep_family>_<level>_seed0000/
```

Example:

```text
datasets/synthetic/phase3_controlled_pilot/
  manifest.csv
  manifest.json
  phase3_edge_sharpness_low_seed0000/
    metadata.json
    transforms_train.json
    transforms_test.json
    train/r_000.png
    test/r_000.png
```

The pilot has `3 sweep families x 3 levels x 1 seed = 9 scenes`.

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

Each scene-level `metadata.json` additionally records the parameter name, dataset format, and fixed factors used for reproducibility.

## Hypotheses For Pilot Validation

- Edge sharpness: sharper transitions may expose method-dependent differences in local reconstruction near boundaries.
- Spatial frequency: higher periodic frequencies may stress methods differently in textured regions with identical geometry.
- Curvature: increasing surface curvature may reveal differences in local geometric representation under fixed appearance.

These are pilot hypotheses for controlled measurement. They are not claims about causal kernel specialization by themselves.

## Null Outcomes

Useful null outcomes include:

- Winner fractions do not change systematically with the controlled parameter.
- Descriptor-to-winner trends are absent or inconsistent across train/test views.
- Oracle gains are negligible across all levels.
- Any apparent effect disappears under patch-size or tie-threshold sensitivity checks.

These outcomes would still validate the benchmark machinery if alignment, rendering, and measurement behave as expected.

## Planned Expansion

Do not expand immediately. After the 9-scene pilot is validated end to end, the planned full expansion is:

```text
3 sweep families x 5 levels x 3 seeds = 45 scenes
```

The expansion should happen only after the pilot confirms that generated datasets are accepted by 3DGS, GES, and DRK; training/evaluation jobs are reproducible; GT alignment passes; and local comparisons plus sensitivity checks produce stable outputs.

## Commands

Tiny local/HPC smoke generation:

```bash
python scripts/synthetic/generate_controlled_pilot.py \
  --config configs/synthetic/phase3_controlled_pilot.json \
  --tiny-test \
  --output-root /tmp/phase3_synthetic_tiny_$USER
```

Full pilot generation, no training:

```bash
python scripts/synthetic/generate_controlled_pilot.py \
  --config configs/synthetic/phase3_controlled_pilot.json
```

Reproducible Slurm wrapper:

```bash
sbatch jobs/synthetic_phase3_pilot.sh
```
