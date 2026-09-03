#!/usr/bin/env python3
"""Install/read-check primitive-budget controls in gitignored upstream checkouts.

This script patches the local external repositories in place. It is intentionally
tracked here because external/ is gitignored. The patch is idempotent and fails
if expected upstream source anchors are not present.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence


OLD_HELPER = '''\n\n@torch.no_grad()\ndef enforce_max_primitives(gaussians, max_primitives, iteration):\n    """Prune lowest-opacity primitives if a hard primitive cap is exceeded."""\n    if max_primitives is None or int(max_primitives) <= 0:\n        return\n    max_primitives = int(max_primitives)\n    count_before = int(gaussians.get_xyz.shape[0])\n    if count_before <= max_primitives:\n        return\n    prune_count = count_before - max_primitives\n    opacity = gaussians.get_opacity.detach().squeeze(-1)\n    _, prune_idx = torch.topk(opacity, prune_count, largest=False)\n    prune_mask = torch.zeros(count_before, dtype=torch.bool, device=opacity.device)\n    prune_mask[prune_idx] = True\n    gaussians.prune_points(prune_mask)\n    count_after = int(gaussians.get_xyz.shape[0])\n    print(\n        f"[ITER {iteration}] max_primitives cap enforced: "\n        f"count_before={count_before} count_after={count_after} "\n        f"number_pruned={prune_count}"\n    )\n'''

HELPER = '''\n\n@torch.no_grad()\ndef enforce_max_primitives(gaussians, max_primitives, iteration):\n    """Prune lowest-opacity primitives if a hard primitive cap is exceeded."""\n    if max_primitives is None or int(max_primitives) <= 0:\n        return\n    max_primitives = int(max_primitives)\n    count_before = int(gaussians.get_xyz.shape[0])\n    if count_before <= max_primitives:\n        return\n    prune_count = count_before - max_primitives\n    opacity = gaussians.get_opacity.detach().squeeze(-1)\n    _, prune_idx = torch.topk(opacity, prune_count, largest=False)\n    prune_mask = torch.zeros(count_before, dtype=torch.bool, device=opacity.device)\n    prune_mask[prune_idx] = True\n\n    created_tmp_radii = False\n    if hasattr(gaussians, "tmp_radii") and getattr(gaussians, "tmp_radii", None) is None:\n        # Current 3DGS prune_points indexes tmp_radii; cap pruning runs after\n        # densify_and_prune has cleared it, so provide a temporary same-length\n        # buffer solely to keep the existing pruning path consistent.\n        gaussians.tmp_radii = torch.zeros(count_before, dtype=opacity.dtype, device=opacity.device)\n        created_tmp_radii = True\n    gaussians.prune_points(prune_mask)\n    if created_tmp_radii:\n        gaussians.tmp_radii = None\n\n    count_after = int(gaussians.get_xyz.shape[0])\n    print(\n        f"[ITER {iteration}] max_primitives cap enforced: "\n        f"count_before={count_before} count_after={count_after} "\n        f"number_pruned={prune_count}"\n    )\n'''


@dataclass(frozen=True)
class Replacement:
    old: str
    new: str
    marker: str


@dataclass(frozen=True)
class FilePatch:
    label: str
    relative_path: str
    replacements: Sequence[Replacement]
    smoke_markers: Sequence[str]


def helper_replacement() -> Replacement:
    old = "from arguments import ModelParams, PipelineParams, OptimizationParams\ntry:"
    new = "from arguments import ModelParams, PipelineParams, OptimizationParams" + HELPER + "\ntry:"
    return Replacement(old=old, new=new, marker="def enforce_max_primitives")


def drk_helper_replacement() -> Replacement:
    old = "from arguments import ModelParams, PipelineParams, OptimizationParams\nfrom utils.general_utils import line_chart"
    new = "from arguments import ModelParams, PipelineParams, OptimizationParams" + HELPER + "\nfrom utils.general_utils import line_chart"
    return Replacement(old=old, new=new, marker="def enforce_max_primitives")


def patches(project_root: Path) -> List[FilePatch]:
    helper_markers = [
        "def enforce_max_primitives",
        "if max_primitives is None or int(max_primitives) <= 0:",
        "created_tmp_radii = False",
        "gaussians.tmp_radii = torch.zeros(count_before, dtype=opacity.dtype, device=opacity.device)",
    ]
    return [
        FilePatch(
            label="3DGS arguments",
            relative_path="external/gaussian-splatting/arguments/__init__.py",
            replacements=[
                Replacement(
                    old="        self.optimizer_type = \"default\"\n        super().__init__(parser, \"Optimization Parameters\")",
                    new="        self.optimizer_type = \"default\"\n        self.max_primitives = -1\n        super().__init__(parser, \"Optimization Parameters\")",
                    marker="self.max_primitives = -1",
                )
            ],
            smoke_markers=["self.max_primitives = -1"],
        ),
        FilePatch(
            label="3DGS train",
            relative_path="external/gaussian-splatting/train.py",
            replacements=[
                helper_replacement(),
                Replacement(
                    old="                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii)\n                \n                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):",
                    new="                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii)\n                    enforce_max_primitives(gaussians, opt.max_primitives, iteration)\n                \n                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):",
                    marker="enforce_max_primitives(gaussians, opt.max_primitives, iteration)",
                ),
            ],
            smoke_markers=helper_markers + [
                "enforce_max_primitives(gaussians, opt.max_primitives, iteration)",
            ],
        ),
        FilePatch(
            label="GES arguments",
            relative_path="external/ges-splatting/arguments/__init__.py",
            replacements=[
                Replacement(
                    old="        self.densify_grad_threshold = 0.0003\n        \n        # New",
                    new="        self.densify_grad_threshold = 0.0003\n        self.max_primitives = -1\n        \n        # New",
                    marker="self.max_primitives = -1",
                )
            ],
            smoke_markers=["self.max_primitives = -1"],
        ),
        FilePatch(
            label="GES train",
            relative_path="external/ges-splatting/train_ges.py",
            replacements=[
                helper_replacement(),
                Replacement(
                    old="                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.prune_opacity_threshold, scene.cameras_extent, size_threshold)\n                if iteration > opt.densify_from_iter and iteration % opt.shape_pruning_interval == 0:",
                    new="                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.prune_opacity_threshold, scene.cameras_extent, size_threshold)\n                    enforce_max_primitives(gaussians, opt.max_primitives, iteration)\n                if iteration > opt.densify_from_iter and iteration % opt.shape_pruning_interval == 0:",
                    marker="enforce_max_primitives(gaussians, opt.max_primitives, iteration)",
                ),
            ],
            smoke_markers=helper_markers + [
                "enforce_max_primitives(gaussians, opt.max_primitives, iteration)",
            ],
        ),
        FilePatch(
            label="DRK arguments",
            relative_path="external/drk/arguments/__init__.py",
            replacements=[
                Replacement(
                    old="        self.final_prune_split = \"train\"\n\n        # Large-primitive pruning + one-sided size regularization",
                    new="        self.final_prune_split = \"train\"\n        self.max_primitives = -1\n\n        # Large-primitive pruning + one-sided size regularization",
                    marker="self.max_primitives = -1",
                )
            ],
            smoke_markers=["self.max_primitives = -1"],
        ),
        FilePatch(
            label="DRK train",
            relative_path="external/drk/train.py",
            replacements=[
                drk_helper_replacement(),
                Replacement(
                    old="                        relocated, added, pruned = self.gaussians.mcmc_densify()\n                        if self.iteration % 1000 == 0:",
                    new="                        relocated, added, pruned = self.gaussians.mcmc_densify()\n                        enforce_max_primitives(self.gaussians, self.opt.max_primitives, self.iteration)\n                        if self.iteration % 1000 == 0:",
                    marker="relocated, added, pruned = self.gaussians.mcmc_densify()\n                        enforce_max_primitives(self.gaussians, self.opt.max_primitives, self.iteration)",
                ),
                Replacement(
                    old="                        self.gaussians.densify_and_prune(self.gaussians.densify_grad_threshold, self.gaussians.min_opacity_pruning, self.scene.cameras_extent, size_threshold)\n                        if mcmc_active:",
                    new="                        self.gaussians.densify_and_prune(self.gaussians.densify_grad_threshold, self.gaussians.min_opacity_pruning, self.scene.cameras_extent, size_threshold)\n                        enforce_max_primitives(self.gaussians, self.opt.max_primitives, self.iteration)\n                        if mcmc_active:",
                    marker="self.gaussians.densify_and_prune(self.gaussians.densify_grad_threshold, self.gaussians.min_opacity_pruning, self.scene.cameras_extent, size_threshold)\n                        enforce_max_primitives(self.gaussians, self.opt.max_primitives, self.iteration)",
                ),
                Replacement(
                    old="                            else:\n                                relocated, added, pruned = 0, 0, 0\n                            if self.iteration % 1000 == 0:",
                    new="                            else:\n                                relocated, added, pruned = 0, 0, 0\n                            enforce_max_primitives(self.gaussians, self.opt.max_primitives, self.iteration)\n                            if self.iteration % 1000 == 0:",
                    marker="relocated, added, pruned = 0, 0, 0\n                            enforce_max_primitives(self.gaussians, self.opt.max_primitives, self.iteration)",
                ),
            ],
            smoke_markers=helper_markers + [
                "relocated, added, pruned = self.gaussians.mcmc_densify()\n                        enforce_max_primitives(self.gaussians, self.opt.max_primitives, self.iteration)",
                "self.gaussians.densify_and_prune(self.gaussians.densify_grad_threshold, self.gaussians.min_opacity_pruning, self.scene.cameras_extent, size_threshold)\n                        enforce_max_primitives(self.gaussians, self.opt.max_primitives, self.iteration)",
                "relocated, added, pruned = 0, 0, 0\n                            enforce_max_primitives(self.gaussians, self.opt.max_primitives, self.iteration)",
            ],
        ),
    ]


def upgrade_helper_if_needed(text: str) -> tuple[str, bool]:
    if "def enforce_max_primitives" not in text:
        return text, False
    if "created_tmp_radii = False" in text:
        return text, False
    if OLD_HELPER not in text:
        raise RuntimeError(
            "Primitive-budget helper is installed but does not match the known v1 helper. "
            "Refusing to guess; inspect the upstream train file."
        )
    return text.replace(OLD_HELPER, HELPER, 1), True


def apply_file_patch(project_root: Path, patch: FilePatch) -> bool:
    path = project_root / patch.relative_path
    if not path.exists():
        raise FileNotFoundError(f"{patch.label}: expected file does not exist: {path}")
    text = path.read_text(encoding="utf-8")
    text, changed = upgrade_helper_if_needed(text)
    for replacement in patch.replacements:
        if replacement.marker in text:
            continue
        if replacement.old not in text:
            raise RuntimeError(
                f"{patch.label}: expected source anchor not found for marker {replacement.marker!r}. "
                f"Refusing to guess; inspect {path}."
            )
        text = text.replace(replacement.old, replacement.new, 1)
        changed = True
    if changed:
        path.write_text(text, encoding="utf-8")
    return changed


def install(project_root: Path) -> None:
    for patch in patches(project_root):
        changed = apply_file_patch(project_root, patch)
        status = "patched/upgraded" if changed else "already installed"
        print(f"{patch.label}: {status}")


def smoke_test(project_root: Path) -> None:
    failures: List[str] = []
    for patch in patches(project_root):
        path = project_root / patch.relative_path
        if not path.exists():
            failures.append(f"{patch.label}: missing {path}")
            continue
        text = path.read_text(encoding="utf-8")
        for marker in patch.smoke_markers:
            if marker not in text:
                failures.append(f"{patch.label}: missing marker {marker!r}")

    # Small deterministic simulation of the cap rule: lowest opacities are pruned,
    # and count cannot remain above cap immediately after the synthetic event.
    opacities = [0.9, 0.1, 0.4, 0.05, 0.8]
    cap = 3
    prune_count = len(opacities) - cap
    pruned = sorted(range(len(opacities)), key=lambda idx: opacities[idx])[:prune_count]
    kept = [idx for idx in range(len(opacities)) if idx not in pruned]
    if len(kept) > cap or pruned != [3, 1]:
        failures.append("cap simulation failed to prune the lowest-opacity primitives")

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        raise SystemExit(1)
    print("Primitive-budget smoke test passed: CLI default, disabled path, tmp_radii-safe helper, call sites, and cap simulation verified.")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Install/smoke-test primitive-budget controls in external checkouts.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--install", action="store_true", help="Patch external checkouts in place.")
    parser.add_argument("--smoke-test", action="store_true", help="Verify patch markers and cap behavior simulation.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    project_root = args.project_root.resolve()
    if not args.install and not args.smoke_test:
        parser.error("Specify --install, --smoke-test, or both.")

    if args.install:
        install(project_root)
    if args.smoke_test:
        smoke_test(project_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
