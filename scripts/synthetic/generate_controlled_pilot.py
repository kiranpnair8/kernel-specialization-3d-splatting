#!/usr/bin/env python3
"""Generate controlled synthetic pilot scenes for Paper-1 Phase III."""
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CURVATURE_HIGH_CANDIDATES = [0.22, 0.26, 0.30, 0.34, 0.38, 0.42]

DEFAULT_CONFIG: dict[str, Any] = {
    "output_root": "datasets/synthetic/phase3_controlled_pilot",
    "dataset_format": "nerf_synthetic_transforms",
    "seed": 0,
    "resolution": [256, 256],
    "train_view_count": 24,
    "test_view_count": 8,
    "camera_radius": 2.8,
    "camera_elevation_degrees": 35.0,
    "camera_angle_x_degrees": 50.0,
    "scene_extent": 1.15,
    "background_color": [1.0, 1.0, 1.0],
    "sweeps": {
        "edge_sharpness": {
            "parameter_name": "transition_width",
            "levels": {"low": 0.32, "medium": 0.12, "high": 0.035},
        },
        "spatial_frequency": {
            "parameter_name": "cycles_per_scene_width",
            "levels": {"low": 2.0, "medium": 6.0, "high": 14.0},
        },
        "curvature": {
            "parameter_name": "paraboloid_amplitude",
            "levels": {"low": 0.02, "medium": 0.18, "high": 0.42},
        },
    },
}


@dataclass(frozen=True)
class Camera:
    split: str
    index: int
    file_path: str
    transform_matrix: list[list[float]]


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def load_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return json.loads(json.dumps(DEFAULT_CONFIG))
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    merged = json.loads(json.dumps(DEFAULT_CONFIG))
    for key, value in config.items():
        if key == "sweeps":
            merged["sweeps"] = value
        else:
            merged[key] = value
    return merged


def normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        raise ValueError("Cannot normalize a zero vector")
    return vector / norm


def ensure_finite(name: str, values: np.ndarray, mask: np.ndarray | None = None) -> None:
    subset = values if mask is None else values[mask]
    if subset.size and not np.all(np.isfinite(subset)):
        raise FloatingPointError(f"Non-finite values detected in {name}")


def look_at(camera_origin: np.ndarray, target: np.ndarray) -> np.ndarray:
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    forward = normalize(target - camera_origin)
    right = normalize(np.cross(forward, world_up))
    up = normalize(np.cross(right, forward))
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, 0] = right
    matrix[:3, 1] = up
    matrix[:3, 2] = -forward
    matrix[:3, 3] = camera_origin
    return matrix


def make_cameras(config: dict[str, Any], split: str) -> list[Camera]:
    count_key = "train_view_count" if split == "train" else "test_view_count"
    count = int(config[count_key])
    radius = float(config["camera_radius"])
    elevation = math.radians(float(config["camera_elevation_degrees"]))
    offset = 0.0 if split == "train" else math.pi / max(1, count)
    cameras = []
    for index in range(count):
        azimuth = 2.0 * math.pi * index / count + offset
        origin = np.array(
            [
                radius * math.cos(elevation) * math.cos(azimuth),
                radius * math.cos(elevation) * math.sin(azimuth),
                radius * math.sin(elevation),
            ],
            dtype=np.float64,
        )
        transform = look_at(origin, np.zeros(3, dtype=np.float64))
        cameras.append(
            Camera(
                split=split,
                index=index,
                file_path=f"./{split}/r_{index:03d}",
                transform_matrix=transform.tolist(),
            )
        )
    return cameras


def camera_rays(width: int, height: int, camera_angle_x: float, c2w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    focal = 0.5 * width / math.tan(0.5 * camera_angle_x)
    xs = (np.arange(width, dtype=np.float64) + 0.5 - width * 0.5) / focal
    ys = -(np.arange(height, dtype=np.float64) + 0.5 - height * 0.5) / focal
    xx, yy = np.meshgrid(xs, ys)
    dirs_camera = np.stack([xx, yy, -np.ones_like(xx)], axis=-1)
    rotation = c2w[:3, :3]
    dirs_world = dirs_camera @ rotation.T
    dirs_world /= np.linalg.norm(dirs_world, axis=-1, keepdims=True)
    origin = c2w[:3, 3]
    origins = np.broadcast_to(origin, dirs_world.shape)
    return origins, dirs_world


def in_scene_bounds(points: np.ndarray, extent: float) -> np.ndarray:
    return (np.abs(points[..., 0]) <= extent) & (np.abs(points[..., 1]) <= extent)


def intersect_plane(origins: np.ndarray, dirs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    denom = dirs[..., 2]
    denom_ok = np.abs(denom) > 1e-8
    t = np.full(denom.shape, -1.0, dtype=np.float64)
    np.divide(-origins[..., 2], denom, out=t, where=denom_ok)
    valid = denom_ok & (t > 0.0)
    points = origins + dirs * t[..., None]
    points = np.where(valid[..., None], points, 0.0)
    normals = np.zeros_like(points)
    normals[..., 2] = 1.0
    return points, normals, valid


def intersect_paraboloid(
    origins: np.ndarray,
    dirs: np.ndarray,
    amplitude: float,
    extent: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Intersect rays with z = amplitude * (x^2 + y^2).

    With a finite scene extent, root selection is performed after checking whether
    each candidate intersection lies inside the bounded square support. The first
    pilot selected the nearest positive root before the extent check; for high
    curvature, many rays first hit the unbounded paraboloid outside the intended
    square and were then rejected as background even when the other root was useful.
    """
    if abs(amplitude) < 1e-10:
        return intersect_plane(origins, dirs)
    ox, oy, oz = origins[..., 0], origins[..., 1], origins[..., 2]
    dx, dy, dz = dirs[..., 0], dirs[..., 1], dirs[..., 2]
    qa = -amplitude * (dx * dx + dy * dy)
    qb = dz - 2.0 * amplitude * (ox * dx + oy * dy)
    qc = oz - amplitude * (ox * ox + oy * oy)
    disc = qb * qb - 4.0 * qa * qc
    has_real_roots = disc >= 0.0
    sqrt_disc = np.sqrt(np.maximum(disc, 0.0))
    denom = 2.0 * qa
    denom_ok = np.abs(denom) > 1e-10
    t1 = np.full(denom.shape, np.inf, dtype=np.float64)
    t2 = np.full(denom.shape, np.inf, dtype=np.float64)
    np.divide(-qb - sqrt_disc, denom, out=t1, where=denom_ok)
    np.divide(-qb + sqrt_disc, denom, out=t2, where=denom_ok)
    p1 = origins + dirs * t1[..., None]
    p2 = origins + dirs * t2[..., None]
    valid1 = has_real_roots & np.isfinite(t1) & (t1 > 0.0)
    valid2 = has_real_roots & np.isfinite(t2) & (t2 > 0.0)
    if extent is not None:
        valid1 = valid1 & in_scene_bounds(p1, extent)
        valid2 = valid2 & in_scene_bounds(p2, extent)
    t1 = np.where(valid1, t1, np.inf)
    t2 = np.where(valid2, t2, np.inf)
    t = np.minimum(t1, t2)
    valid = np.isfinite(t)

    candidate_points = origins + dirs * t[..., None]
    points = np.zeros_like(origins)
    points[valid] = candidate_points[valid]
    raw_normals = np.stack(
        [-2.0 * amplitude * points[..., 0], -2.0 * amplitude * points[..., 1], np.ones_like(points[..., 2])],
        axis=-1,
    )
    normals_norm = np.linalg.norm(raw_normals, axis=-1, keepdims=True)
    normal_ok = valid & np.isfinite(normals_norm[..., 0]) & (normals_norm[..., 0] > 1e-12)
    normals = np.zeros_like(points)
    normals[..., 2] = 1.0
    np.divide(raw_normals, normals_norm, out=normals, where=normal_ok[..., None])
    valid = valid & normal_ok
    ensure_finite("paraboloid points", points, valid)
    ensure_finite("paraboloid normals", normals, valid)
    return points, normals, valid


def shade(base_color: np.ndarray, normals: np.ndarray, valid: np.ndarray, background: np.ndarray) -> np.ndarray:
    ensure_finite("base color", base_color, valid)
    ensure_finite("normals", normals, valid)
    light_dir = normalize(np.array([-0.35, -0.45, 1.0], dtype=np.float64))
    diffuse = np.clip(np.sum(normals * light_dir, axis=-1), 0.0, 1.0)
    illumination = 0.58 + 0.42 * diffuse
    shaded = base_color * illumination[..., None]
    image = np.broadcast_to(background, shaded.shape).copy()
    image[valid] = shaded[valid]
    image = np.clip(image, 0.0, 1.0)
    ensure_finite("rendered image", image)
    return image


def edge_color(points: np.ndarray, transition_width: float) -> np.ndarray:
    left = np.array([0.18, 0.33, 0.82], dtype=np.float64)
    right = np.array([0.95, 0.63, 0.12], dtype=np.float64)
    mix = 0.5 + 0.5 * np.tanh(points[..., 0] / transition_width)
    return left * (1.0 - mix[..., None]) + right * mix[..., None]


def frequency_color(points: np.ndarray, cycles: float, extent: float) -> np.ndarray:
    base = np.array([0.56, 0.57, 0.60], dtype=np.float64)
    tint = np.array([0.34, 0.30, 0.24], dtype=np.float64)
    phase_x = 2.0 * math.pi * cycles * (points[..., 0] / (2.0 * extent) + 0.5)
    phase_y = 2.0 * math.pi * cycles * (points[..., 1] / (2.0 * extent) + 0.5)
    texture = np.sin(phase_x) * np.sin(phase_y)
    return base + 0.32 * texture[..., None] * tint


def curvature_color(points: np.ndarray) -> np.ndarray:
    color = np.array([0.58, 0.66, 0.78], dtype=np.float64)
    return np.broadcast_to(color, points.shape).copy()


def render_image(config: dict[str, Any], sweep_family: str, parameter_value: float, camera: Camera) -> np.ndarray:
    width, height = [int(value) for value in config["resolution"]]
    extent = float(config["scene_extent"])
    background = np.array(config["background_color"], dtype=np.float64)
    camera_angle_x = math.radians(float(config["camera_angle_x_degrees"]))
    c2w = np.array(camera.transform_matrix, dtype=np.float64)
    origins, dirs = camera_rays(width, height, camera_angle_x, c2w)
    if sweep_family == "curvature":
        points, normals, valid = intersect_paraboloid(origins, dirs, parameter_value, extent=extent)
        colors = curvature_color(points)
    else:
        points, normals, valid = intersect_plane(origins, dirs)
        if sweep_family == "edge_sharpness":
            colors = edge_color(points, parameter_value)
        elif sweep_family == "spatial_frequency":
            colors = frequency_color(points, parameter_value, extent)
        else:
            raise ValueError(f"Unknown sweep family: {sweep_family}")
        valid = valid & in_scene_bounds(points, extent)
    ensure_finite("geometry points", points, valid)
    ensure_finite("geometry normals", normals, valid)
    image = shade(colors, normals, valid, background)
    return (np.clip(image, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def write_transforms(scene_dir: Path, split: str, cameras: list[Camera], camera_angle_x: float) -> None:
    payload = {
        "camera_angle_x": camera_angle_x,
        "frames": [
            {"file_path": camera.file_path, "transform_matrix": camera.transform_matrix}
            for camera in cameras
        ],
    }
    with (scene_dir / f"transforms_{split}.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def scene_id(sweep_family: str, level: str, seed: int) -> str:
    return f"phase3_{sweep_family}_{level}_seed{seed:04d}"


def amplitude_label(value: float) -> str:
    return f"{value:.3g}".replace("-", "m").replace(".", "p")


def generate_scene(
    config: dict[str, Any],
    sweep_family: str,
    level: str,
    parameter_name: str,
    parameter_value: float,
    output_root: Path,
) -> dict[str, Any]:
    seed = int(config["seed"])
    sid = scene_id(sweep_family, level, seed)
    scene_dir = output_root / sid
    train_dir = scene_dir / "train"
    test_dir = scene_dir / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    camera_angle_x = math.radians(float(config["camera_angle_x_degrees"]))
    train_cameras = make_cameras(config, "train")
    test_cameras = make_cameras(config, "test")
    for camera in [*train_cameras, *test_cameras]:
        image = render_image(config, sweep_family, parameter_value, camera)
        Image.fromarray(image, mode="RGB").save(scene_dir / f"{camera.file_path[2:]}.png")
    write_transforms(scene_dir, "train", train_cameras, camera_angle_x)
    write_transforms(scene_dir, "test", test_cameras, camera_angle_x)
    metadata = {
        "scene_id": sid,
        "sweep_family": sweep_family,
        "level": level,
        "parameter_name": parameter_name,
        "parameter_value": parameter_value,
        "seed": seed,
        "train_view_count": len(train_cameras),
        "test_view_count": len(test_cameras),
        "resolution": f"{config['resolution'][0]}x{config['resolution'][1]}",
        "dataset_format": config["dataset_format"],
        "dataset_path": str(scene_dir),
        "fixed_factors": {
            "camera_radius": config["camera_radius"],
            "camera_elevation_degrees": config["camera_elevation_degrees"],
            "camera_angle_x_degrees": config["camera_angle_x_degrees"],
            "scene_extent": config["scene_extent"],
            "background_color": config["background_color"],
            "train_camera_protocol": "fixed orbit shared by every generated scene",
            "test_camera_protocol": "fixed orbit offset shared by every generated scene",
        },
    }
    with (scene_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return metadata


def write_manifest(output_root: Path, rows: list[dict[str, Any]]) -> None:
    csv_path = output_root / "manifest.csv"
    json_path = output_root / "manifest.json"
    fieldnames = [
        "scene_id",
        "sweep_family",
        "level",
        "parameter_value",
        "seed",
        "train_view_count",
        "test_view_count",
        "resolution",
        "dataset_path",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fieldnames})
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2, sort_keys=True)
        handle.write("\n")


def grayscale(rgb: np.ndarray) -> np.ndarray:
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


def gradient_magnitude(gray: np.ndarray) -> np.ndarray:
    gy, gx = np.gradient(gray.astype(np.float64))
    return np.sqrt(gx * gx + gy * gy)


def laplacian(gray: np.ndarray) -> np.ndarray:
    result = -4.0 * gray + np.roll(gray, 1, axis=0) + np.roll(gray, -1, axis=0) + np.roll(gray, 1, axis=1) + np.roll(gray, -1, axis=1)
    result[0, :] = 0.0
    result[-1, :] = 0.0
    result[:, 0] = 0.0
    result[:, -1] = 0.0
    return result


def image_diagnostics(image_path: Path, background: np.ndarray, threshold: float) -> dict[str, float]:
    with Image.open(image_path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0
    ensure_finite("saved RGB image", rgb)
    gray = grayscale(rgb)
    grad = gradient_magnitude(gray)
    lap = laplacian(gray)
    distance = np.linalg.norm(rgb - background.reshape(1, 1, 3), axis=-1)
    return {
        "foreground_fraction": float(np.mean(distance > threshold)),
        "mean_intensity": float(np.mean(gray)),
        "intensity_std": float(np.std(gray)),
        "mean_gradient_magnitude": float(np.mean(grad)),
        "laplacian_high_frequency_energy": float(np.mean(lap * lap)),
    }


def audit_curvature_foreground(
    rows: list[dict[str, Any]],
    config: dict[str, Any],
    output_root: Path,
    target: float,
    tolerance: float,
    threshold: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    background = np.array(config["background_color"], dtype=np.float64)
    audit_rows: list[dict[str, Any]] = []
    high_candidates: list[dict[str, Any]] = []
    for row in rows:
        sid = str(row["scene_id"])
        scene_dir = output_root / sid
        test_images = sorted((scene_dir / "test").glob("*.png"))
        if not test_images:
            raise FileNotFoundError(f"No test images found for curvature validation scene: {scene_dir}")
        per_view = [image_diagnostics(path, background, threshold) for path in test_images]
        mean_foreground = float(np.mean([item["foreground_fraction"] for item in per_view]))
        accepted = abs(mean_foreground - target) <= tolerance
        audit_row = {
            "scene_id": sid,
            "level": row["level"],
            "parameter_value": float(row["parameter_value"]),
            "test_view_count": len(test_images),
            "mean_foreground_fraction": mean_foreground,
            "min_foreground_fraction": float(np.min([item["foreground_fraction"] for item in per_view])),
            "max_foreground_fraction": float(np.max([item["foreground_fraction"] for item in per_view])),
            "mean_intensity": float(np.mean([item["mean_intensity"] for item in per_view])),
            "intensity_std": float(np.mean([item["intensity_std"] for item in per_view])),
            "mean_gradient_magnitude": float(np.mean([item["mean_gradient_magnitude"] for item in per_view])),
            "laplacian_high_frequency_energy": float(np.mean([item["laplacian_high_frequency_energy"] for item in per_view])),
            "target_foreground_fraction": target,
            "tolerance": tolerance,
            "accepted": accepted,
        }
        if str(row["level"]).startswith("high_candidate"):
            high_candidates.append(audit_row)
        audit_rows.append(audit_row)
    accepted_high = [row for row in high_candidates if row["accepted"]]
    selected_high = max(accepted_high, key=lambda row: float(row["parameter_value"])) if accepted_high else None
    summary = {
        "target_foreground_fraction": target,
        "tolerance": tolerance,
        "foreground_threshold": threshold,
        "accepted_control_levels": all(row["accepted"] for row in audit_rows if row["level"] in {"low", "medium"}),
        "accepted_high_candidate_count": len(accepted_high),
        "selected_largest_valid_high_amplitude": selected_high["parameter_value"] if selected_high else None,
        "selected_largest_valid_high_scene_id": selected_high["scene_id"] if selected_high else None,
        "accepted": bool(selected_high) and all(row["accepted"] for row in audit_rows if row["level"] in {"low", "medium"}),
    }
    for row in audit_rows:
        row["selected_largest_valid_high_amplitude"] = summary["selected_largest_valid_high_amplitude"]
        row["selected_largest_valid_high_scene_id"] = summary["selected_largest_valid_high_scene_id"]
    return audit_rows, summary


def write_curvature_validation(output_root: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    csv_path = output_root / "curvature_validation.csv"
    json_path = output_root / "curvature_validation.json"
    summary_path = output_root / "curvature_validation_summary.json"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump({"summary": summary, "rows": rows}, handle, indent=2, sort_keys=True)
        handle.write("\n")
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")


def parse_float_list(value: str) -> list[float]:
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("At least one amplitude candidate is required")
    return values


def curvature_candidate_config(config: dict[str, Any], high_candidates: list[float]) -> dict[str, Any]:
    result = json.loads(json.dumps(config))
    curvature = result["sweeps"]["curvature"]
    levels = {
        "low": curvature["levels"]["low"],
        "medium": curvature["levels"]["medium"],
    }
    for value in high_candidates:
        levels[f"high_candidate_{amplitude_label(value)}"] = float(value)
    result["sweeps"] = {
        "curvature": {
            "parameter_name": curvature["parameter_name"],
            "levels": levels,
        }
    }
    return result


def validate_noncanonical_output_root(config: dict[str, Any], output_root: Path) -> None:
    canonical_root = resolve_path(config["output_root"])
    validation_root = resolve_path(output_root)
    if validation_root == canonical_root or canonical_root in validation_root.parents:
        raise ValueError(
            f"Refusing to write curvature validation candidates under canonical dataset root: {validation_root}"
        )


def generate(config: dict[str, Any], output_root: Path | None = None, tiny_test: bool = False) -> list[dict[str, Any]]:
    config = json.loads(json.dumps(config))
    if output_root is not None:
        config["output_root"] = str(output_root)
    if tiny_test:
        config["resolution"] = [48, 48]
        config["train_view_count"] = 2
        config["test_view_count"] = 1
        config["sweeps"] = {
            "edge_sharpness": {
                "parameter_name": "transition_width",
                "levels": {"low": 0.32},
            }
        }
    root = resolve_path(config["output_root"])
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for sweep_family, sweep in config["sweeps"].items():
        parameter_name = str(sweep["parameter_name"])
        for level, parameter_value in sweep["levels"].items():
            rows.append(generate_scene(config, sweep_family, str(level), parameter_name, float(parameter_value), root))
    write_manifest(root, rows)
    print(f"Generated {len(rows)} synthetic scenes under {root}")
    for row in rows:
        print(f"- {row['scene_id']}: {row['sweep_family']} {row['level']}={row['parameter_value']}")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--tiny-test", action="store_true", help="Generate one tiny scene for validation only.")
    parser.add_argument(
        "--validate-curvature-candidates",
        action="store_true",
        help="Generate curvature controls plus candidate high-amplitude scenes in a temporary output root and audit foreground occupancy.",
    )
    parser.add_argument(
        "--curvature-high-candidates",
        type=parse_float_list,
        default=DEFAULT_CURVATURE_HIGH_CANDIDATES,
        help="Comma-separated high-curvature amplitude candidates for validation mode.",
    )
    parser.add_argument("--foreground-target", type=float, default=0.54)
    parser.add_argument("--foreground-tolerance", type=float, default=0.03)
    parser.add_argument("--foreground-threshold", type=float, default=0.03)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.validate_curvature_candidates:
        output_root = args.output_root or Path(f"/tmp/phase3_curvature_validation_{Path.home().name}")
        validate_noncanonical_output_root(config, output_root)
        config = curvature_candidate_config(config, args.curvature_high_candidates)
        rows = generate(config, output_root, tiny_test=False)
        root = resolve_path(output_root)
        audit_rows, summary = audit_curvature_foreground(
            rows,
            config,
            root,
            target=args.foreground_target,
            tolerance=args.foreground_tolerance,
            threshold=args.foreground_threshold,
        )
        write_curvature_validation(root, audit_rows, summary)
        for row in audit_rows:
            print(
                f"curvature {row['level']}: amplitude={row['parameter_value']:.4f} "
                f"foreground={row['mean_foreground_fraction']:.4f} accepted={row['accepted']} "
                f"mean={row['mean_intensity']:.4f} std={row['intensity_std']:.4f} "
                f"grad={row['mean_gradient_magnitude']:.6f} lap_energy={row['laplacian_high_frequency_energy']:.6f}"
            )
        print(f"Largest accepted high-curvature amplitude: {summary['selected_largest_valid_high_amplitude']}")
        print(f"Selected high-curvature scene: {summary['selected_largest_valid_high_scene_id']}")
        print(f"Wrote {root / 'curvature_validation.csv'}")
        print(f"Wrote {root / 'curvature_validation.json'}")
        print(f"Wrote {root / 'curvature_validation_summary.json'}")
        if not summary["accepted"]:
            print("Curvature validation failed: controls or all high candidates are outside tolerance.")
            return 1
        print("Curvature validation accepted: controls passed and at least one high candidate is within tolerance.")
        return 0
    generate(config, args.output_root, tiny_test=args.tiny_test)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
