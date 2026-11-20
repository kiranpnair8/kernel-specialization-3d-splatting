# Experiment 5: controlled fixed-beta radial kernels

This isolated experiment changes only the screen-space radial falloff in a
vendored 3D Gaussian Splatting renderer. It does **not** modify `external/`, and
it does not claim equivalence to the separate GES implementation.

## Intervention

For projected conic distance

```text
q = a dx^2 + 2 b dx dy + c dy^2,
```

the renderer uses

```text
K_beta(q) = exp(-0.5 q^(beta/2)), beta in {1, 2, 4}.
```

`--kernel_beta` is a fixed integer pipeline option (default `2`). It is passed
from `arguments/PipelineParams`, through `GaussianRasterizationSettings` and
the C++ binding, to CUDA preprocessing, forward rasterization, and backward
rasterization. It is not optimized or stored per primitive. Optimizer settings,
initialization, losses, SH evaluation, covariance projection, opacity,
densification, pruning, and training schedules are otherwise untouched.

### Exact Gaussian path

For beta 2, CUDA executes the original expression

```text
power = -0.5 (a dx^2 + c dy^2) - b dx dy
K = exp(power)
```

and retains the original fixed `3 * sqrt(lambda_max)` screen-space radius. The
historical antialiasing path is unchanged and remains disabled by default.
This branch is intentionally separate from generalized arithmetic to avoid
changing baseline floating-point evaluation order.

### Beta-1 center derivative

The mathematical `dK/dq` is singular at `q=0` for beta 1, while the composed
screen-space kernel is nondifferentiable at its center. CUDA uses a symmetric
zero subgradient exactly at `q=0`. For positive `q`, the denominator uses
`sqrt(max(q, 1e-8))`, capping only the near-center derivative magnitude. The
forward value is never epsilon-shifted. This policy is mirrored by
`kernel_reference.py` and covered by finite-value and numerical-gradient tests.

### Support bounds

Beta 2 preserves baseline 3-sigma coverage exactly. For beta 1 and 4, support
is derived from the renderer's alpha rejection threshold:

```text
effective_opacity * K_beta(q_max) = 1/255.
```

With `L = 2 log(effective_opacity / (1/255))`, the radius multiplier on
`sqrt(lambda_max)` is `L` for beta 1 and `L^(1/4)` for beta 4. Effective opacity
includes the existing antialiasing convolution scaling when antialiasing is
enabled. Generalized-kernel primitives whose peak alpha is below the threshold
touch no tiles. This avoids truncating beta-1's heavier tail or over-covering
beta-4's faster tail.

The beta-2 support exception is deliberate: replacing its historical 3-sigma
bound with an opacity-derived bound would change baseline rendered pixels and
visibility/radius-driven densification behavior.

## Files changed inside the vendored copy

- `arguments/__init__.py`: fixed `kernel_beta` CLI option.
- `gaussian_renderer/__init__.py`: validation and rasterizer setting.
- `submodules/diff-gaussian-rasterization/diff_gaussian_rasterization/__init__.py`:
  Python autograd argument propagation.
- `submodules/diff-gaussian-rasterization/rasterize_points.{h,cu}` and
  `cuda_rasterizer/{rasterizer,forward,backward}*`: C++/CUDA propagation.
- `cuda_rasterizer/kernel_utils.cuh`: shared value, derivative, and support math.

All modifications remain under `experiments/tpami_kernel_intervention/`.

## CPU validation

From the project root:

```bash
python -m unittest discover \
  -s experiments/tpami_kernel_intervention/tests \
  -p 'test_*.py' -v
```

The tests cover the original beta-2 expression, analytical `dK/dq`, numerical
screen-coordinate and conic gradients for all three betas, the beta-1 center
policy, threshold-contour support, the baseline 3-sigma bound, and source-level
parameter propagation.

## V100 build and GPU smoke test

Activate the same 3DGS environment used by the project and load its CUDA module.
For the USD V100 nodes:

```bash
cd /home/rizk_lab/shared/kiran/kernel-specialization-3d-splatting
module load cuda/12.3
export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
export CUDAHOSTCXX="$CXX"
bash experiments/tpami_kernel_intervention/scripts/build_v100.sh
bash experiments/tpami_kernel_intervention/scripts/run_gpu_smoke.sh
```

`build_v100.sh` sets `TORCH_CUDA_ARCH_LIST=7.0`. The GPU smoke test renders two
synthetic primitives for beta 1, 2, and 4, checks finite forward/backward
values, checks repeated beta-2 determinism, and compares opacity and projected-
covariance gradients against central finite differences for every beta. It does
not train a scene.

The smoke wrapper also runs `gpu_depth_gradient_regression.py`. That regression
repeats depth-only backward passes 10 times for every beta with both contiguous
and expanded zero-stride upstream gradients, compares both layouts, and checks
depth and RGB opacity/covariance gradients against finite differences.

Training writes `kernel_beta` into the normal `cfg_args`. `render.py` restores
that saved value by default, accepts an explicit `--kernel_beta` override, and
falls back to beta 2 for pre-intervention checkpoints whose config lacks it.

## Optional tiny training smoke test

This command is provided for an explicitly approved, short compatibility run;
it is not run automatically and refuses to overwrite an existing output path:

```bash
bash experiments/tpami_kernel_intervention/scripts/run_training_smoke.sh \
  /absolute/path/to/a/small/scene \
  /tmp/experiment5_beta1_smoke \
  1 20
```

Use beta `1`, `2`, or `4`. Full training sweeps are intentionally out of scope.

## Validation status and limitations

- CPU reference/source tests: **12/12 passed** on 2026-11-20.
- Python byte-compilation and `git diff --check`: **passed**.
- CUDA extension build and GPU smoke: **not run in the local Codex environment**;
  PyTorch and `nvcc` are unavailable there. Do not report GPU validation until
  the V100 commands above pass.
- No dataset training or rendering experiment was launched.
- The beta-1 derivative floor is a documented numerical convention and should
  be held fixed across all beta-1 runs.
- Beta-dependent support changes tile workload substantially, especially for
  beta 1. This is required for correct alpha-threshold coverage and may change
  runtime/memory, but not the stated optimization protocol.

### Inverse-depth backward layout fix

PyTorch may represent the upstream gradient of a reduction such as
`0.01 * invdepth.sum()` as an expanded tensor with zero strides. The CUDA
backward kernel indexes this gradient as a dense `(1,H,W)` array. The binding
therefore materializes both incoming output gradients (`dL/dcolor` and
`dL/dinvdepth`) as named contiguous tensors, validates their dtype/device/shape,
and retains their storage through the CUDA launch. This changes only gradient
layout handling; radial-kernel values, derivatives, support bounds, and the
beta-2 forward path are untouched.
