#!/usr/bin/env python3
"""Audit Phase-III controlled synthetic stimuli without modifying datasets."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import inventory_phase3_outputs as inventory

FAMILY_ORDER = ["edge_sharpness", "spatial_frequency", "curvature"]
LEVEL_ORDER = ["low", "medium", "high"]
DEFAULT_BACKGROUND = np.asarray([1.0, 1.0, 1.0], dtype=np.float64)


@dataclass
class ViewDiagnostics:
    scene_id: str
    sweep_family: str
    level: str
    parameter_value: str
    seed: int
    view_index: int
    image_path: str
    width: int
    height: int
    mean_intensity: float
    intensity_std: float
    foreground_fraction: float | None
    foreground_fraction_method: str
    mean_gradient_magnitude: float
    laplacian_high_frequency_energy: float


@dataclass
class SceneDiagnostics:
    scene_id: str
    sweep_family: str
    level: str
    parameter_value: str
    seed: int
    test_view_count: int
    width: int
    height: int
    mean_intensity: float
    intensity_std: float
    foreground_fraction: float | None
    mean_gradient_magnitude: float
    laplacian_high_frequency_energy: float
    representative_test_view_index: int
    representative_image_path: str
    foreground_fraction_method: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/synthetic/phase3_controlled_pilot"))
    parser.add_argument("--results-dir", type=Path, default=Path("results/synthetic/phase3_controlled_pilot/stimulus_audit"))
    parser.add_argument("--representative-test-index", type=int, default=0)
    parser.add_argument(
        "--foreground-threshold",
        type=float,
        default=0.03,
        help="RGB Euclidean distance from background used when no alpha mask is available.",
    )
    return parser.parse_args()


def resolve_path(project_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else project_root / path


def rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def as_int(row: Mapping[str, Any], key: str, default: int) -> int:
    return inventory.as_int(dict(row), key, default)


def scene_dir_from_manifest(row: Mapping[str, Any], dataset_root: Path) -> Path:
    candidate = Path(str(row.get("dataset_path", ""))) if row.get("dataset_path") else Path()
    if str(candidate) and candidate.exists():
        return candidate
    return dataset_root / str(row["scene_id"])


def load_metadata(scene_dir: Path) -> dict[str, Any]:
    path = scene_dir / "metadata.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def background_from_metadata(metadata: Mapping[str, Any]) -> np.ndarray | None:
    fixed = metadata.get("fixed_factors") if isinstance(metadata, Mapping) else None
    value = None
    if isinstance(fixed, Mapping):
        value = fixed.get("background_color")
    if value is None and isinstance(metadata, Mapping):
        value = metadata.get("background_color")
    if value is None:
        return DEFAULT_BACKGROUND
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (3,):
        return DEFAULT_BACKGROUND
    return arr


def frame_image_path(scene_dir: Path, frame: Mapping[str, Any]) -> Path:
    raw = str(frame["file_path"])
    relative = raw[2:] if raw.startswith("./") else raw
    path = scene_dir / relative
    if path.suffix:
        return path
    png = path.with_suffix(".png")
    if png.exists():
        return png
    return path


def test_image_paths(scene_dir: Path) -> list[Path]:
    transforms_path = scene_dir / "transforms_test.json"
    if not transforms_path.exists():
        paths = sorted((scene_dir / "test").glob("*.png"))
        if not paths:
            raise FileNotFoundError(f"No transforms_test.json or test PNGs found under {scene_dir}")
        return paths
    transforms = json.loads(transforms_path.read_text(encoding="utf-8"))
    frames = transforms.get("frames", [])
    paths = [frame_image_path(scene_dir, frame) for frame in frames]
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing test images for {scene_dir}: {missing[:3]}")
    return paths


def load_rgba(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    with Image.open(path) as image:
        rgba = np.asarray(image.convert("RGBA"), dtype=np.float64) / 255.0
    rgb = rgba[..., :3]
    alpha = rgba[..., 3]
    if np.any(alpha < 0.999):
        return rgb, alpha
    return rgb, None


def grayscale(rgb: np.ndarray) -> np.ndarray:
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


def gradient_magnitude(gray: np.ndarray) -> np.ndarray:
    gy, gx = np.gradient(gray.astype(np.float64))
    return np.sqrt(gx * gx + gy * gy)


def laplacian(gray: np.ndarray) -> np.ndarray:
    center = -4.0 * gray
    up = np.roll(gray, 1, axis=0)
    down = np.roll(gray, -1, axis=0)
    left = np.roll(gray, 1, axis=1)
    right = np.roll(gray, -1, axis=1)
    result = center + up + down + left + right
    result[0, :] = 0.0
    result[-1, :] = 0.0
    result[:, 0] = 0.0
    result[:, -1] = 0.0
    return result


def foreground_fraction(rgb: np.ndarray, alpha: np.ndarray | None, background: np.ndarray | None, threshold: float) -> tuple[float | None, str]:
    if alpha is not None:
        return float(np.mean(alpha > 0.001)), "alpha>0.001"
    if background is not None:
        distance = np.linalg.norm(rgb - background.reshape(1, 1, 3), axis=-1)
        return float(np.mean(distance > threshold)), f"rgb_distance_from_background>{threshold:g}"
    return None, "unavailable"


def diagnose_image(
    path: Path,
    scene_row: Mapping[str, Any],
    view_index: int,
    background: np.ndarray | None,
    threshold: float,
    project_root: Path,
) -> ViewDiagnostics:
    rgb, alpha = load_rgba(path)
    gray = grayscale(rgb)
    grad = gradient_magnitude(gray)
    lap = laplacian(gray)
    fg_fraction, fg_method = foreground_fraction(rgb, alpha, background, threshold)
    height, width = gray.shape
    return ViewDiagnostics(
        scene_id=str(scene_row["scene_id"]),
        sweep_family=str(scene_row.get("sweep_family", "")),
        level=str(scene_row.get("level", "")),
        parameter_value=str(scene_row.get("parameter_value", "")),
        seed=as_int(scene_row, "seed", 0),
        view_index=view_index,
        image_path=rel(path, project_root),
        width=int(width),
        height=int(height),
        mean_intensity=float(np.mean(gray)),
        intensity_std=float(np.std(gray)),
        foreground_fraction=fg_fraction,
        foreground_fraction_method=fg_method,
        mean_gradient_magnitude=float(np.mean(grad)),
        laplacian_high_frequency_energy=float(np.mean(lap * lap)),
    )


def mean_optional(values: list[float | None]) -> float | None:
    numeric = [value for value in values if value is not None]
    if not numeric:
        return None
    return float(np.mean(np.asarray(numeric, dtype=np.float64)))


def aggregate_scene(row: Mapping[str, Any], diagnostics: list[ViewDiagnostics], representative: ViewDiagnostics) -> SceneDiagnostics:
    first = diagnostics[0]
    return SceneDiagnostics(
        scene_id=str(row["scene_id"]),
        sweep_family=str(row.get("sweep_family", "")),
        level=str(row.get("level", "")),
        parameter_value=str(row.get("parameter_value", "")),
        seed=as_int(row, "seed", 0),
        test_view_count=len(diagnostics),
        width=first.width,
        height=first.height,
        mean_intensity=float(np.mean([d.mean_intensity for d in diagnostics])),
        intensity_std=float(np.mean([d.intensity_std for d in diagnostics])),
        foreground_fraction=mean_optional([d.foreground_fraction for d in diagnostics]),
        mean_gradient_magnitude=float(np.mean([d.mean_gradient_magnitude for d in diagnostics])),
        laplacian_high_frequency_energy=float(np.mean([d.laplacian_high_frequency_energy for d in diagnostics])),
        representative_test_view_index=representative.view_index,
        representative_image_path=representative.image_path,
        foreground_fraction_method=representative.foreground_fraction_method,
    )


def make_grid_figure(scene_rows: list[Mapping[str, Any]], representatives: dict[tuple[str, str], Path], results_dir: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(FAMILY_ORDER), len(LEVEL_ORDER), figsize=(9.4, 8.0), constrained_layout=True)
    for row_index, family in enumerate(FAMILY_ORDER):
        for col_index, level in enumerate(LEVEL_ORDER):
            ax = axes[row_index][col_index]
            path = representatives.get((family, level))
            if path is None:
                raise ValueError(f"Missing representative image for {family}/{level}")
            with Image.open(path) as image:
                ax.imshow(image.convert("RGB"))
            ax.set_xticks([])
            ax.set_yticks([])
            if row_index == 0:
                ax.set_title(level.title(), fontsize=13, pad=8)
            if col_index == 0:
                ax.set_ylabel(family.replace("_", " ").title(), fontsize=13, labelpad=12)
            scene = next(r for r in scene_rows if str(r.get("sweep_family")) == family and str(r.get("level")) == level)
            param = scene.get("parameter_value", "")
            ax.text(
                0.02,
                0.98,
                f"{param}",
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=9,
                color="black",
                bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "none", "alpha": 0.78},
            )
    fig.suptitle("Phase-III Controlled Synthetic Stimuli: Fixed Representative Test View", fontsize=15)
    png = results_dir / "phase3_stimulus_audit_grid.png"
    pdf = results_dir / "phase3_stimulus_audit_grid.pdf"
    results_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return [str(png), str(pdf)]


def sort_manifest(rows: list[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            FAMILY_ORDER.index(str(row.get("sweep_family"))) if str(row.get("sweep_family")) in FAMILY_ORDER else 99,
            LEVEL_ORDER.index(str(row.get("level"))) if str(row.get("level")) in LEVEL_ORDER else 99,
            str(row.get("scene_id")),
        ),
    )


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    dataset_root = resolve_path(project_root, args.dataset_root)
    results_dir = resolve_path(project_root, args.results_dir)

    manifest = sort_manifest(inventory.load_manifest(dataset_root))
    expected_pairs = {(family, level) for family in FAMILY_ORDER for level in LEVEL_ORDER}
    observed_pairs = {(str(row.get("sweep_family")), str(row.get("level"))) for row in manifest}
    missing_pairs = expected_pairs - observed_pairs
    if missing_pairs:
        raise ValueError(f"Manifest is missing expected Phase-III stimuli: {sorted(missing_pairs)}")

    per_view: list[ViewDiagnostics] = []
    per_scene: list[SceneDiagnostics] = []
    representatives: dict[tuple[str, str], Path] = {}

    for row in manifest:
        scene_dir = scene_dir_from_manifest(row, dataset_root)
        metadata = load_metadata(scene_dir)
        background = background_from_metadata(metadata)
        test_paths = test_image_paths(scene_dir)
        if args.representative_test_index < 0 or args.representative_test_index >= len(test_paths):
            raise IndexError(
                f"Representative test index {args.representative_test_index} is outside {scene_dir}'s {len(test_paths)} test views"
            )
        diagnostics = [
            diagnose_image(path, row, index, background, args.foreground_threshold, project_root)
            for index, path in enumerate(test_paths)
        ]
        representative_diag = diagnostics[args.representative_test_index]
        representative_path = test_paths[args.representative_test_index]
        representatives[(str(row.get("sweep_family")), str(row.get("level")))] = representative_path
        per_view.extend(diagnostics)
        per_scene.append(aggregate_scene(row, diagnostics, representative_diag))

    figure_paths = make_grid_figure(manifest, representatives, results_dir)
    scene_rows = [asdict(row) for row in per_scene]
    view_rows = [asdict(row) for row in per_view]

    write_csv(results_dir / "phase3_stimulus_diagnostics.csv", scene_rows)
    write_json(results_dir / "phase3_stimulus_diagnostics.json", scene_rows)
    write_csv(results_dir / "phase3_stimulus_diagnostics_per_view.csv", view_rows)
    write_json(results_dir / "phase3_stimulus_diagnostics_per_view.json", view_rows)
    write_json(
        results_dir / "phase3_stimulus_audit_manifest.json",
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "dataset_root": rel(dataset_root, project_root),
            "results_dir": rel(results_dir, project_root),
            "representative_test_index": args.representative_test_index,
            "foreground_threshold": args.foreground_threshold,
            "figure_paths": [rel(Path(path), project_root) for path in figure_paths],
            "scene_count": len(per_scene),
            "per_view_count": len(per_view),
            "diagnostics": [
                "mean_intensity",
                "intensity_std",
                "foreground_fraction",
                "mean_gradient_magnitude",
                "laplacian_high_frequency_energy",
            ],
        },
    )

    print(f"Phase-III stimulus audit complete: {len(per_scene)} scenes, {len(per_view)} test views")
    print(f"Wrote {rel(results_dir / 'phase3_stimulus_audit_grid.png', project_root)}")
    print(f"Wrote {rel(results_dir / 'phase3_stimulus_diagnostics.csv', project_root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
