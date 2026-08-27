# Phase III Synthetic Pilot Training Setup

## Loader Inspection Summary

The Phase-III pilot datasets are generated under:

```text
datasets/synthetic/phase3_controlled_pilot/
```

Each scene contains `transforms_train.json`, `transforms_test.json`, `train/*.png`, and `test/*.png`. The split is fixed by the generated transform files: 24 training views and 8 held-out test views.

Direct NeRF-synthetic loading is supported by all three current upstream pipelines:

| Method | Loader result | Relevant behavior |
| --- | --- | --- |
| 3DGS | Direct `transforms_train.json` / `transforms_test.json` support | `scene/__init__.py` chooses the Blender loader when `transforms_train.json` exists; `--eval` preserves test frames from `transforms_test.json`. |
| GES | Direct `transforms_train.json` / `transforms_test.json` support | The GES fork keeps the same Blender loader branch and preserves test frames under `--eval`. |
| DRK | Direct `transforms_train.json` / `transforms_test.json` support | DRK chooses the Blender loader when `transforms_train.json` exists and preserves test frames under `--eval`; it also supports optional NSVF, which is not used here. |

A COLMAP conversion is not required for this pilot. If a future local fork rejects the direct transforms format, the deterministic conversion path is to emit a COLMAP sparse model from the known analytic intrinsics/extrinsics while preserving the existing transform-defined split.

## NeRF Loader Compatibility Patch

The current HPC NumPy/Pillow stack rejects signed int8 RGB arrays. The pinned 3DGS loader and pinned GES loader both construct synthetic RGB images with:

```python
Image.fromarray(np.array(arr * 255.0, dtype=np.byte), "RGB")
```

`np.byte` is signed int8, which can raise:

```text
TypeError: Cannot handle this data type: (1, 1, 3), |i1
```

DRK's inspected NeRF-synthetic loader already uses `np.uint8` in the corresponding Pillow conversion path. The tracked compatibility script below idempotently patches gitignored local upstream checkouts and then verifies that no legacy signed-byte conversion remains:

```bash
python scripts/synthetic/patch_nerf_synthetic_loader_dtype.py --project-root "$PROJECT_ROOT"
python scripts/synthetic/patch_nerf_synthetic_loader_dtype.py --project-root "$PROJECT_ROOT" --verify-only
```

This changes only the local external loader dtype cast from `np.byte` to `np.uint8`. It does not change camera handling, rendering logic, training hyperparameters, datasets, or method protocol. The loader-check job and all synthetic training arrays apply/verify this compatibility patch before using the loaders.

## Image And Path Conventions

The generator writes frame paths without an image extension, for example:

```json
{"file_path": "./train/r_000"}
```

This matches the 3DGS and GES loaders, which append `.png`. DRK also accepts paths without an extension and appends `.png` when needed. The images are stored as:

```text
train/r_000.png ... train/r_023.png
test/r_000.png ... test/r_007.png
```

The train/test split must not be recomputed using LLFF holdout rules for synthetic scenes. It is encoded directly by `transforms_train.json` and `transforms_test.json`.

## Deterministic Input Preparation

Although direct loading works, all three loaders auto-create `points3d.ply` if it is missing. The defaults are method-specific and random; 3DGS/GES use a 100k random cloud, while DRK's public loader uses a smaller default. To avoid hidden initialization differences, run:

```bash
python scripts/synthetic/prepare_nerf_synthetic_inputs.py \
  --dataset-root datasets/synthetic/phase3_controlled_pilot \
  --point-count 100000 \
  --seed 0
```

This validates every scene and writes a deterministic shared `points3d.ply` into each scene directory. It does not modify external method repositories or training code.

## Compatibility Check

Before launching training arrays, validate one pilot scene through the actual method loaders:

```bash
sbatch jobs/synthetic_phase3_loader_check.sh
```

The check targets:

```text
phase3_edge_sharpness_low_seed0000
```

It verifies that each method sees 24 train cameras and 8 test cameras through its Blender/NeRF-synthetic loader.

## Job Structure

Preparation and validation:

```bash
sbatch jobs/synthetic_phase3_prepare.sh
sbatch jobs/synthetic_phase3_loader_check.sh
```

Training/evaluation arrays:

```bash
sbatch jobs/synthetic_phase3_3dgs_array.sh
sbatch jobs/synthetic_phase3_ges_array.sh
sbatch jobs/synthetic_phase3_drk_array.sh
```

Each array has 9 tasks, one per pilot scene, capped at 3 concurrent tasks by `%3`.

## Output Roots

Requested output roots are:

```text
outputs/synthetic/phase3_controlled_pilot/3dgs/<scene_id>/
outputs/synthetic/phase3_controlled_pilot/ges/<scene_id>/
outputs/synthetic/phase3_controlled_pilot/drk/<scene_id>/
```

3DGS writes directly into the requested path. GES may create an internally suffixed or timestamped model path; the job resolves the actual output before rendering and metrics. DRK appends `_${gs_type}`, so the actual DRK model path is expected to be:

```text
outputs/synthetic/phase3_controlled_pilot/drk/<scene_id>_DRK/
```

## Protocol Notes

The 3DGS and GES jobs keep the same `train -> render -> metrics` protocol used for Garden/Bicycle/Room. DRK keeps the same `train.py -> train.py --metric` protocol used for the real-scene DRK baselines.

The documented DRK Mip-NeRF 360 flag `--is_unbounded` is intentionally omitted for the synthetic pilot because these scenes are bounded analytic objects, not unbounded 360 captures. This is a scene-format exception, not a classifier or methodology change.

No large training sweep should be launched until the preparation job and loader compatibility job succeed.
