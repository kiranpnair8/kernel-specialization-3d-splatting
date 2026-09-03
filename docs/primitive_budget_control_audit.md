# Primitive Budget Control Audit

Date: 2026-09-03

Scope: audit and setup record for controlling representation capacity and final primitive count in the 3DGS, GES, and DRK implementations used in Paper-1. The tracked repository contains only patch/setup tooling and Slurm wrappers; upstream implementation edits are applied locally to gitignored `external/` checkouts only when the patch installer is run.

Provenance note: the local `external/` directory was not present in the Codex workspace inspected for the initial audit, consistent with the repository policy that `external/` is gitignored. Source-level conclusions below are based on the pinned/public upstream implementations and our tracked Slurm wrappers. 3DGS and GES were inspected at the commits recorded in `docs/project_plan_paper1.md`; DRK was inspected from the public VAST/CVMI DRK implementation that matches the command-line interface used by our wrappers (`train.py --gs_type DRK --kernel_density dense --cache_sort --is_unbounded`).

## Completed Baseline Densification Settings

The completed real-scene jobs used each method's standard/default densification behavior. No completed Garden/Bicycle/Room baseline wrapper passed a primitive-budget override, a maximum primitive count, or a non-default 3DGS/GES densification schedule.

| Method | Wrapper evidence | Training command settings relevant to budget |
| --- | --- | --- |
| 3DGS | `jobs/3dgs_bicycle_baseline.sh`, `jobs/3dgs_room_baseline.sh`, Garden baseline docs | `python train.py -s ... -m ... --eval`; default 30k iterations; no densification overrides |
| GES | `jobs/ges_garden_baseline.sh`, `jobs/ges_bicycle_baseline.sh`, `jobs/ges_room_baseline.sh` | `python train_ges.py -s ... -m ... --eval`; default 40k iterations; no densification overrides |
| DRK | `jobs/drk_garden_baseline.sh`, `jobs/drk_bicycle_baseline.sh`, `jobs/drk_room_baseline.sh` | `python train.py -s ... -m ... --eval --gs_type DRK --kernel_density dense --cache_sort --is_unbounded`; default 35k iterations with DRK dense/unbounded schedule |

Recorded final primitive counts from tracked documentation:

| Scene | 3DGS | GES | DRK |
| --- | ---: | ---: | ---: |
| Garden | 4,146,866 | 2,709,992 | 614,345 |
| Bicycle | 4,925,950 | 2,773,438 | 1,578,713 |
| Room | 1,314,498 | 777,374 | 259,045 |

## 3DGS

Implementation inspected: `graphdeco-inria/gaussian-splatting` commit `54c035f7834b564019656c3e3fcc3646292f727d`.

### Initial Primitive-count Initialization

3DGS initializes one Gaussian per point in the input scene point cloud. In `Scene.__init__`, when no checkpoint is loaded, the scene calls `gaussians.create_from_pcd(scene_info.point_cloud, ...)`. In `GaussianModel.create_from_pcd`, `_xyz`, SH features, scale, rotation, and opacity tensors are initialized from `pcd.points`; the code prints `Number of points at initialisation`. Therefore the initial primitive count is the number of COLMAP/synthetic initialization points, not a fixed command-line parameter.

For Garden, the tracked baseline doc records 138,766 initial COLMAP points.

### Densification / Splitting / Cloning Mechanism

During training, if `iteration < densify_until_iter`, 3DGS accumulates view-space gradient statistics for visible points. Every `densification_interval` after `densify_from_iter`, it calls:

`gaussians.densify_and_prune(densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii)`

`densify_and_prune` does three things:

1. `densify_and_clone`: duplicates high-gradient primitives whose scale is at or below `percent_dense * scene_extent`.
2. `densify_and_split`: splits high-gradient primitives whose scale is above `percent_dense * scene_extent`; default split count is `N=2`, then the original selected parent is pruned.
3. Prunes low-opacity and, later in training, overly large screen/world-space primitives.

### Relevant Densification Arguments

From `OptimizationParams`:

- `--iterations`, default `30000`
- `--percent_dense`, default `0.01`
- `--densification_interval`, default `100`
- `--opacity_reset_interval`, default `3000`
- `--densify_from_iter`, default `500`
- `--densify_until_iter`, default `15000`
- `--densify_grad_threshold`, default `0.0002`

### Pruning Mechanism and Arguments

3DGS pruning is embedded in `densify_and_prune`:

- low opacity: `get_opacity < 0.005`; this threshold is hard-coded at the training call site, not exposed as an argument in the inspected upstream.
- large screen-space radius: `max_radii2D > size_threshold`; `size_threshold` is `20` after the opacity reset interval, otherwise `None`.
- large world-space scale: `max(get_scaling) > 0.1 * extent` when `size_threshold` is active.
- opacity reset is controlled by `--opacity_reset_interval` and indirectly affects later pruning/growth dynamics.

### Existing Support for Budget Control

| Question | 3DGS answer |
| --- | --- |
| Disable densification? | Yes, indirectly. Set `--densify_until_iter` less than or equal to `--densify_from_iter`, or `--densify_until_iter 0`, so the densification block never reaches a call site. This also disables the in-window pruning/reset behavior tied to that block. |
| Stop at chosen iteration? | Yes. Use `--densify_until_iter N`. The default is 15000. |
| Change frequency/threshold? | Yes. Use `--densification_interval`, `--densify_grad_threshold`, and `--percent_dense`. |
| Explicit maximum primitive count? | No. The inspected upstream has no hard cap or target count argument. |

### Can Final Counts Be Matched Using Existing Arguments Only?

Not reliably. 3DGS can influence growth by changing thresholds, intervals, and stop iteration, but final count is an emergent result of scene content, gradients, opacity pruning, and scale pruning. A matched final primitive count would require trial-and-error retraining and still would not guarantee exact matching.

### Smallest Hard-budget Code Change

Add a `--max_primitives` optimization argument and enforce it immediately after `densify_and_prune`. If the count exceeds the budget, prune the excess with an existing score. The least invasive score would be opacity because it is already available without extra rendering passes. A slightly more principled but more expensive score would be visibility/contribution over train cameras.

Minimal behavior-preserving patch shape:

1. Add `self.max_primitives = -1` to `OptimizationParams`.
2. After each densification/pruning call, if `max_primitives > 0 and get_xyz.shape[0] > max_primitives`, prune `count - max_primitives` lowest-score primitives.
3. Reuse `prune_points` so optimizer tensors and accumulators stay consistent.
4. Log the forced prune event and saved final count.

This is a real methodological intervention: it introduces an additional pruning policy beyond standard 3DGS, so it must be reported as a budget-control variant rather than a baseline.

## GES

Implementation inspected: `ajhamdi/ges-splatting` commit `c05fc7dbb22e270a5a6f490e7040adac4af02c96`.

### Initial Primitive-count Initialization

GES initializes one generalized/Laplian primitive per point in the input point cloud. `train_ges.py` constructs `LaplacianModel(dataset.sh_degree)`, and `Scene` initializes it from `scene_info.point_cloud`. `LaplacianModel.create_from_pcd` creates `_xyz`, features, scale, rotation, opacity, and a learnable `_shape` tensor from the same input point set. The initial primitive count is therefore the input point-cloud size.

### Densification / Splitting / Cloning Mechanism

GES retains the 3DGS clone/split pattern in `LaplacianModel`:

1. gradient accumulation over visible points;
2. `densify_and_clone` for high-gradient small primitives;
3. `densify_and_split` for high-gradient large primitives, carrying the extra shape parameter into child primitives;
4. low-opacity and large-primitive pruning inside `densify_and_prune`.

GES adds shape-specific behavior in the same densification window:

- `size_prune(prune_shape_threshold)` every `shape_pruning_interval` after `densify_from_iter`;
- `reset_shape()` every `shape_reset_interval`.

### Relevant Densification Arguments

From `OptimizationParams`:

- `--iterations`, default `40000`
- `--percent_dense`, default `0.01`
- `--densification_interval`, default `100`
- `--opacity_reset_interval`, default `3000`
- `--densify_from_iter`, default `500`
- `--densify_until_iter`, default `15000`
- `--densify_grad_threshold`, default `0.0003`

GES-specific shape controls relevant to pruning/capacity:

- `--shape_lr`, default `0.001`
- `--shape_reset_interval`, default `1000`
- `--shape_pruning_interval`, default `100`
- `--prune_shape_threshold`, default `0.005`
- `--prune_opacity_threshold`, default `0.005`
- `--shape_strngth`, default `0.1` in the inspected code spelling

### Pruning Mechanism and Arguments

GES pruning includes:

- opacity pruning in `densify_and_prune` using `--prune_opacity_threshold`;
- large screen/world primitive pruning when `size_threshold` is active after opacity reset interval;
- shape pruning through `size_prune`, using `get_shape < --prune_shape_threshold`.

### Existing Support for Budget Control

| Question | GES answer |
| --- | --- |
| Disable densification? | Yes, indirectly via `--densify_until_iter` less than or equal to `--densify_from_iter`, or `--densify_until_iter 0`. As in 3DGS, this also affects pruning/reset operations that live inside the densification block. |
| Stop at chosen iteration? | Yes. Use `--densify_until_iter N`. |
| Change frequency/threshold? | Yes. Use `--densification_interval`, `--densify_grad_threshold`, `--percent_dense`, plus GES-specific shape pruning/reset thresholds. |
| Explicit maximum primitive count? | No. There is no inspected hard maximum primitive argument. |

### Can Final Counts Be Matched Using Existing Arguments Only?

Not reliably. GES gives more pruning knobs than 3DGS because of `prune_shape_threshold` and `shape_pruning_interval`, but final primitive count remains emergent. Matching exact budgets against 3DGS/DRK would still require iterative retraining and would confound capacity with changed growth/pruning dynamics.

### Smallest Hard-budget Code Change

Add a `--max_primitives` argument and call a shared lowest-score pruning helper after both standard `densify_and_prune` and `size_prune`. The hard-budget prune must include `_shape` in optimizer/tensor pruning. The smallest viable score is opacity or a composite of opacity and shape; opacity-only is closest to 3DGS symmetry, while opacity-times-shape is more GES-aware but changes the method more.

Recommended minimal variant for comparability: start with lowest opacity, document it, and keep the existing shape pruning untouched.

## DRK

Implementation inspected: public DRK implementation matching our wrappers: `train.py --gs_type DRK --kernel_density dense --cache_sort --is_unbounded`.

### Initial Primitive-count Initialization

DRK uses the same scene-level point-cloud initialization route. `Trainer` selects `DRKModel` when `--gs_type DRK`, constructs it with `kernel_K`, and `Scene` calls `create_from_pcd` if no trained iteration is loaded. `DRKModel.create_from_pcd` first initializes base Gaussian tensors from input points, then adds DRK-specific parameters:

- `_acutance`
- `_thetas`
- `_l1l2_rates`
- expanded per-kernel scaling with `kernel_K`

The initial primitive count remains the number of input point-cloud points; DRK adds more parameters per primitive rather than changing the initial primitive count by default.

### Densification / Splitting / Cloning Mechanism

DRK extends the Gaussian clone/split mechanism with DRK-specific attributes and scheduling:

1. `DRKModel.update(iteration)` moves through staged acutance/radial-kernel schedules.
2. During the densification window, the trainer accumulates position and optional opacity-gradient statistics.
3. At scheduled intervals, it either calls standard `densify_and_prune` or, if MCMC is enabled and active, calls `mcmc_densify` depending on `mcmc_strategy`.
4. `DRKModel.densify_and_prune` clones/splits using a combined gradient and carries DRK attributes into child primitives.
5. Additional pruning handles sharpened opacity, low visibility/low opacity, large primitive culling, optional opacity-binarization pruning, optional final contribution prune, and optional MCMC cap pruning.

`--kernel_density` changes DRK's effective densification/pruning thresholds. For unbounded scenes, inspected defaults are:

- `dense`: `densify_grad_threshold=1e-3`, `prune_threshold=1e-1`
- `middle`: `densify_grad_threshold=1.5e-3`, `prune_threshold=1e-1`
- `sparse`: `densify_grad_threshold=2e-3`, `prune_threshold=1e-1`

For bounded scenes, the dense threshold is lower and sparse has stronger pruning.

### Relevant Densification Arguments

Core/shared controls:

- `--iterations`, default `35000`
- `--densification_interval`, default `100`, unless overwritten by DRK stage scheduling
- `--opacity_reset_interval`, default `3000`, unless overwritten by DRK stage scheduling
- `--densify_from_iter`, default `500`
- `--densify_until_iter`, default `15000`
- `--densify_grad_threshold`, default `0.0005` before DRK model-specific override
- `--percent_dense`, default `0.01` for GS path
- `--percent_drk_dense`, default `1e-3` for DRK path
- `--kernel_density`, one of `dense`, `middle`, `sparse`
- `--is_unbounded`
- `--keep_manual_densification_interval`
- `--keep_manual_opacity_reset_interval`

DRK/MCMC controls relevant to primitive count:

- `--use_mcmc`
- `--mcmc_strategy`, default `replace`
- `--mcmc_start_iter`, `--mcmc_end_iter`
- `--mcmc_cap_max`, default `-1`
- `--mcmc_growth_rate`, default `1.05`
- `--mcmc_prune_min_opacity`
- `--mcmc_prune_score`

Final/large-prune controls:

- `--final_prune_target`, default `-1`
- `--final_prune_at_iter`, default `-1`
- `--final_prune_score`, default `visible_area_sharp`
- `--final_prune_split`, default `train`
- `--prune_big_screen_px`
- `--prune_big_world_scale_k`
- `--prune_big_from_iter`
- `--prune_big_interval`
- `--prune_big_combine`
- `--opacity_prune_thresh`
- `--opacity_prune_interval`
- `--opacity_prune_from_iter`

### Pruning Mechanism and Arguments

DRK pruning includes several layers:

- sharpened-opacity pruning in `densify_and_prune` using `get_sharpen_opacity < min_opacity`;
- low-visibility/low-opacity floater pruning;
- large screen/world primitive pruning;
- optional opacity-binarization pruning;
- optional MCMC pruning and MCMC cap pruning;
- optional scheduled or final contribution prune through `final_prune_target`.

### Existing Support for Budget Control

| Question | DRK answer |
| --- | --- |
| Disable densification? | Yes, via `--densify_until_iter` less than or equal to `--densify_from_iter`, or `--densify_until_iter 0`. MCMC should also remain disabled unless intentionally used. |
| Stop at chosen iteration? | Yes. Use `--densify_until_iter N`. |
| Change frequency/threshold? | Yes. Use `--kernel_density`, `--densification_interval`, `--keep_manual_densification_interval`, `--densify_grad_threshold`, `--percent_drk_dense`, and related pruning controls. Without `--keep_manual_densification_interval`, DRK staged scheduling may override the manual interval. |
| Explicit maximum primitive count? | Partly. `--final_prune_target` can force a final/scheduled contribution prune to a target count. `--mcmc_cap_max` caps MCMC growth only when MCMC is enabled; it does not cap standard clone/split densification. |

### Can Final Counts Be Matched Using Existing Arguments Only?

DRK can be forced to a final target using `--final_prune_target`, but 3DGS and GES cannot. Therefore matching final primitive counts across all three methods cannot be done cleanly with existing arguments only.

Also, DRK's `final_prune_target` is a post-hoc contribution prune. It matches final count, but it does not make the training trajectory use the same budget throughout optimization. That distinction matters for any capacity-control claim.

### Smallest Hard-budget Code Change

For DRK, no code change is required for a final target count if post-training final-count matching is sufficient: use `--final_prune_target B`. If a hard maximum during training is required for symmetry with 3DGS/GES, add the same `--max_primitives` enforcement after every densification/MCMC addition, using existing pruning utilities. This avoids allowing transient growth above budget.

## Cross-method Capacity Matching Assessment

Existing arguments are enough to create coarse capacity variants, but not enough to guarantee matched final primitive counts across 3DGS, GES, and DRK.

What can be done without code changes:

- stop densification early for all methods using `--densify_until_iter`;
- disable densification for all methods by ending the densification window before it starts;
- make DRK coarser/finer with `--kernel_density sparse|middle|dense`;
- post-hoc target DRK final count using `--final_prune_target`;
- tune 3DGS/GES thresholds manually through repeated pilot runs.

What cannot be done cleanly without code changes:

- impose a shared hard maximum primitive count on 3DGS/GES;
- guarantee the same final count for 3DGS/GES/DRK in one run;
- keep all methods under a shared budget throughout training.

Scientific interpretation: capacity matching by threshold twiddling alone would be weak because threshold changes alter each method's training dynamics in method-specific ways. A hard-budget intervention is cleaner for count control but is still a new budget-constrained variant, not the original baseline protocol.

## Implemented Minimal Hard-budget Patch Design

Tracked installer: `scripts/setup/apply_primitive_budget_control.py`.

The installer patches the gitignored local upstream checkouts in place. It is idempotent, can upgrade the first installed helper version, and refuses to proceed if expected source anchors are absent. It does not commit or copy upstream code into this repository.

Implemented intervention:

1. Add `--max_primitives` to each method's optimization parameters, default `-1`.
2. When `max_primitives > 0`, call `enforce_max_primitives(...)` immediately after operations that can increase primitive count.
3. Prune exactly `count_before - max_primitives` primitives with the lowest current opacity.
4. Reuse each method's existing `prune_points` implementation, preserving optimizer-state and method-specific tensor consistency.
5. For current 3DGS compatibility, provide a temporary same-length `tmp_radii` buffer when cap pruning is called after `densify_and_prune` has cleared `tmp_radii`; the helper restores it to `None` immediately after pruning.
6. Log every enforcement event with iteration, count before, count after, and number pruned.
7. Preserve exact default behavior when `--max_primitives=-1`.

Patch command:

```bash
python scripts/setup/apply_primitive_budget_control.py --project-root "$PWD" --install --smoke-test
```

Smoke-test command:

```bash
python scripts/setup/apply_primitive_budget_control.py --project-root "$PWD" --smoke-test
```

The smoke test verifies that the CLI default is installed, the disabled `-1` path is present, expected cap-enforcement call sites are present, the helper is `tmp_radii`-safe, and a deterministic simulated densification event is pruned to the requested cap by lowest opacity.

## Room 250k Budget-control Variant

Status: configured only. Training has not been submitted from this repository task.

Scene: MiP-NeRF 360 Room.

Budget: `max_primitives=250000`.

Natural Room primitive counts from saved baseline outputs:

| Method | Natural Room final primitives | 250k cap as fraction of natural count |
| --- | ---: | ---: |
| 3DGS | 1,314,498 | 19.0% |
| GES | 777,374 | 32.2% |
| DRK | 259,045 | 96.5% |

Why 250k: it is just below the natural DRK Room count while substantially constraining 3DGS and GES. This creates a small, deadline-conscious robustness probe of whether the Room local-complementarity pattern survives when all methods are forced into roughly the same primitive-count regime. The setting is intentionally a budget-controlled variant, not a replacement baseline.

Training jobs:

- `jobs/3dgs_room_budget250k.sh`
- `jobs/ges_room_budget250k.sh`
- `jobs/drk_room_budget250k.sh`

Output roots:

- `outputs/3dgs/room_budget250k`
- `outputs/ges/room_budget250k` with GES timestamp suffix appended internally
- `outputs/drk/room_budget250k_DRK`

Launch commands, when ready:

```bash
sbatch jobs/3dgs_room_budget250k.sh
sbatch jobs/ges_room_budget250k.sh
sbatch jobs/drk_room_budget250k.sh
```

Interpretation rule: these runs introduce a hard opacity-pruning intervention that is not part of the original baseline implementations. They may be used as a representation-capacity robustness check, but should be reported separately from the standard 3DGS/GES/DRK baselines.

## Recommended Minimal Budget-Control Experiment

Goal: test whether the observed real-scene local complementarity is robust to representation capacity, without launching a broad new sweep before the ICLR deadline.

Recommended design:

1. Use Room only for the first capacity-control probe.
   - Rationale: Room has the largest observed p32 oracle relative MSE gain vs 3DGS in the tracked ledger (31.1564%) and the most balanced winner distribution among the three real scenes: 3DGS 41.7171%, GES 27.2448%, DRK 13.1509%, tie 17.8872%.
   - This makes it the most sensitive single-scene probe for whether capacity explains the local-complementarity signal.
2. Start with one matched hard cap: `250000` primitives.
   - This is below the natural Room DRK count and strongly constrains 3DGS/GES.
   - It minimizes new GPU training while still probing the central capacity confound.
3. Preserve each method's standard optimization behavior as much as possible.
   - Keep iteration counts, losses, camera split, evaluation path, and method-specific rendering protocol fixed.
   - Only add the primitive-budget constraint and report it as a controlled-capacity variant.
4. Evaluate with the already validated p32 local pipeline.
   - Run alignment audit if outputs are regenerated into new directories.
   - Run p32 local comparison with maps/predictors disabled for speed.
   - Run characterization only if the local-comparison signal persists.
   - Run a reduced sensitivity check at `patch=32`, `tie=1e-5` first; expand to the full 9-condition sensitivity grid only if the result becomes central.

Scientifically defensible stopping rule:

- If the same qualitative local-complementarity and descriptor-association patterns persist in Room under the 250k cap, capacity alone is unlikely to explain the observed Room signal.
- If the signal collapses or reverses under the cap, Paper-1 should frame primitive-count/capacity as a major confound and keep the original real-scene conclusions observational.

Do not start with all scenes or a large budget sweep. A single-scene, one-budget pilot is the smallest useful experiment. Expand only if the Room pilot is promising and time remains.
