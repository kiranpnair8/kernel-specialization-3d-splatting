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
    valid = np.abs(denom) > 1e-8
    t = np.where(valid, -origins[..., 2] / denom, -1.0)
    valid = valid & (t > 0.0)
    points = origins + dirs * t[..., None]
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

    When a finite scene extent is supplied, root selection is performed after checking
    whether each candidate intersection lies inside the bounded square support. The
    original pilot selected the nearest positive root before the extent check; for high
    curvature, many rays first hit the unbounded paraboloid outside the intended square
    and were then rejected as background even though the other root could be in-bounds.
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
    t1 = np.where(np.abs(denom) > 1e-10, (-qb - sqrt_disc) / denom, np.inf)
    t2 = np.where(np.abs(denom) > 1e-10, (-qb + sqrt_disc) / denom, np.inf)
    p1 = origins + dirs * t1[..., None]
    p2 = origins + dirs * t2[..., None]
    valid1 = has_real_roots & np.isfinite(t1) & (t1 > 0.0)
    valid2 = has_real_roots & np.isfinite(t2) & (t2 > 0.0)
    if extent is not None:
        valid1 = valid1 & in_scene_bounds(p1, extent)
        valid2 = valid2 & in_scene_bounds(p2, extent)
    t1 = np.where(valid1, t1, np.inf)
    t2 = np.where(valid2, t2, np.inf)
    use_first = t1 <= t2
    t = np.where(use_first, t1, t2)
    valid = np.isfinite(t)
    points = origins + dirs * t[..., None]
    normals = np.stack(
        [-2.0 * amplitude * points[..., 0], -2.0 * amplitude * points[..., 1], np.ones_like(points[..., 2])],
        axis=-1,
    )
    normals_norm = np.linalg.norm(normals, axis=-1, keepdims=True)
    normals = np.divide(normals, np.maximum(normals_norm, 1e-12))
    return points, normals, valid


def shade(base_color: np.ndarray, normals: np.ndarray, valid: np.ndarray, background: np.ndarray) -> np.ndarray:
    light_dir = normalize(np.array([-0.35, -0.45, 1.0], dtype=np.float64))
    diffuse = np.clip(np.sum(normals * light_dir, axis=-1), 0.0, 1.0)
    illumination = 0.58 + 0.42 * diffuse
    shaded = base_color * illumination[..., None]
    image = np.broadcast_to(background, shaded.shape).copy()
    image[valid] = shaded[valid]
    return np.clip(image, 0.0, 1.0)


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


def foreground_fraction(image_path: Path, background: np.ndarray, threshold: float) -> float:
    with Image.open(image_path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0
    distance = np.linalg.norm(rgb - background.reshape(1, 1, 3), axis=-1)
    return float(np.mean(distance > threshold))


def audit_curvature_foreground(
    config: dict[str, Any],
    output_root: Path,
    target: float,
    tolerance: float,
    threshold: float,
) -> list[dict[str, Any]]:
    background = np.array(config["background_color"], dtype=np.float64)
    seed = int(config["seed"])
    rows: list[dict[str, Any]] = []
    for level in ("low", "medium", "high"):
        sid = scene_id("curvature", level, seed)
        scene_dir = output_root / sid
        test_images = sorted((scene_dir / "test").glob("*.png"))
        if not test_images:
            raise FileNotFoundError(f"No test images found for curvature validation scene: {scene_dir}")
        fractions = [foreground_fraction(path, background, threshold) for path in test_images]
        mean_fraction = float(np.mean(fractions))
        within_tolerance = abs(mean_fraction - target) <= tolerance
        rows.append(
            {
                "scene_id": sid,
                "level": level,
                "parameter_value": config["sweeps"]["curvature"]["levels"][level],
                "test_view_count": len(test_images),
                "mean_foreground_fraction": mean_fraction,
                "min_foreground_fraction": float(np.min(fractions)),
                "max_foreground_fraction": float(np.max(fractions)),
                "target_foreground_fraction": target,
                "tolerance": tolerance,
                "accepted": within_tolerance,
            }
        )
    return rows


def write_curvature_validation(output_root: Path, rows: list[dict[str, Any]]) -> None:
    csv_path = output_root / "curvature_validation.csv"
    json_path = output_root / "curvature_validation.json"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2, sort_keys=True)
        handle.write("\n")


def curvature_only_config(config: dict[str, Any]) -> dict[str, Any]:
    result = json.loads(json.dumps(config))
    result["sweeps"] = {"curvature": result["sweeps"]["curvature"]}
    return result


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
        help="Generate only candidate curvature scenes in a temporary output root and require matched foreground occupancy.",
    )
    parser.add_argument("--foreground-target", type=float, default=0.54)
    parser.add_argument("--foreground-tolerance", type=float, default=0.03)
    parser.add_argument("--foreground-threshold", type=float, default=0.03)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.validate_curvature_candidates:
        config = curvature_only_config(config)
        output_root = args.output_root or Path("/tmp/phase3_curvature_validation")
        generate(config, output_root, tiny_test=False)
        root = resolve_path(output_root)
        rows = audit_curvature_foreground(
            config,
            root,
            target=args.foreground_target,
            tolerance=args.foreground_tolerance,
            threshold=args.foreground_threshold,
        )
        write_curvature_validation(root, rows)
        accepted = all(bool(row["accepted"]) for row in rows)
        for row in rows:
            print(
                f"curvature {row['level']}: foreground={row['mean_foreground_fraction']:.4f} "
                f"target={row['target_foreground_fraction']:.4f} accepted={row['accepted']}"
            )
        print(f"Wrote {root / 'curvature_validation.csv'}")
        print(f"Wrote {root / 'curvature_validation.json'}")
        if not accepted:
            print("Curvature validation failed: foreground occupancy is outside tolerance.")
            return 1
        print("Curvature validation accepted: all levels are within tolerance.")
        return 0
    generate(config, args.output_root, tiny_test=args.tiny_test)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
