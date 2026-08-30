#!/usr/bin/env python3
"""Generate Phase III-B multi-seed controlled synthetic scenes.

This script extends the Phase-III pilot to seeds 1-4 without regenerating or
modifying the canonical seed-0 datasets. It reuses the camera, ray-intersection,
shading, manifest, and diagnostic utilities from generate_controlled_pilot.py,
while adding deterministic seed-level nuisance variation that is held fixed
across low/medium/high levels within each sweep family.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import generate_controlled_pilot as pilot

FAMILIES = ("edge_sharpness", "spatial_frequency", "curvature")
LEVELS = ("low", "medium", "high")
DEFAULT_NEW_SEEDS = (1, 2, 3, 4)
LEVEL_ORDER = {"low": 0, "medium": 1, "high": 2}


def parse_int_list(value: str) -> list[int]:
    seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not seeds:
        raise argparse.ArgumentTypeError("At least one seed is required")
    return seeds


def load_config(path: Path) -> dict[str, Any]:
    config = pilot.load_config(path)
    # Keep the corrected curvature sweep canonical for every new seed.
    config["sweeps"]["curvature"]["levels"]["high"] = 0.30
    return config


def seed_suffix(seed: int) -> str:
    return f"seed{seed:04d}"


def scene_id(sweep_family: str, level: str, seed: int) -> str:
    if sweep_family == "curvature" and level == "high":
        return f"phase3_curvature_high_corrected_{seed_suffix(seed)}"
    return f"phase3_{sweep_family}_{level}_{seed_suffix(seed)}"


def family_seed_rng(sweep_family: str, seed: int) -> np.random.Generator:
    offsets = {"edge_sharpness": 1103, "spatial_frequency": 2207, "curvature": 3301}
    return np.random.default_rng(seed * 1000003 + offsets[sweep_family])


def nuisance_parameters(sweep_family: str, seed: int) -> dict[str, float]:
    if seed == 0:
        return {
            "edge_angle_radians": 0.0,
            "edge_offset": 0.0,
            "spatial_angle_radians": 0.0,
            "spatial_phase_x": 0.0,
            "spatial_phase_y": 0.0,
            "curvature_albedo_angle_radians": 0.0,
            "curvature_albedo_phase": 0.0,
        }
    rng = family_seed_rng(sweep_family, seed)
    if sweep_family == "edge_sharpness":
        return {
            "edge_angle_radians": float(rng.uniform(-math.pi / 7.0, math.pi / 7.0)),
            "edge_offset": float(rng.uniform(-0.10, 0.10)),
        }
    if sweep_family == "spatial_frequency":
        return {
            "spatial_angle_radians": float(rng.uniform(0.0, math.pi)),
            "spatial_phase_x": float(rng.uniform(0.0, 2.0 * math.pi)),
            "spatial_phase_y": float(rng.uniform(0.0, 2.0 * math.pi)),
        }
    if sweep_family == "curvature":
        return {
            "curvature_albedo_angle_radians": float(rng.uniform(0.0, 2.0 * math.pi)),
            "curvature_albedo_phase": float(rng.uniform(0.0, 2.0 * math.pi)),
        }
    raise ValueError(f"Unknown sweep family: {sweep_family}")


def rotate_xy(points: np.ndarray, angle: float) -> tuple[np.ndarray, np.ndarray]:
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    x = points[..., 0] * cos_a + points[..., 1] * sin_a
    y = -points[..., 0] * sin_a + points[..., 1] * cos_a
    return x, y


def edge_color(points: np.ndarray, transition_width: float, nuisance: dict[str, float]) -> np.ndarray:
    left = np.array([0.18, 0.33, 0.82], dtype=np.float64)
    right = np.array([0.95, 0.63, 0.12], dtype=np.float64)
    x_rot, _ = rotate_xy(points, nuisance.get("edge_angle_radians", 0.0))
    coordinate = x_rot - nuisance.get("edge_offset", 0.0)
    mix = 0.5 + 0.5 * np.tanh(coordinate / transition_width)
    return left * (1.0 - mix[..., None]) + right * mix[..., None]


def frequency_color(points: np.ndarray, cycles: float, extent: float, nuisance: dict[str, float]) -> np.ndarray:
    base = np.array([0.56, 0.57, 0.60], dtype=np.float64)
    tint = np.array([0.34, 0.30, 0.24], dtype=np.float64)
    x_rot, y_rot = rotate_xy(points, nuisance.get("spatial_angle_radians", 0.0))
    phase_x = 2.0 * math.pi * cycles * (x_rot / (2.0 * extent) + 0.5) + nuisance.get("spatial_phase_x", 0.0)
    phase_y = 2.0 * math.pi * cycles * (y_rot / (2.0 * extent) + 0.5) + nuisance.get("spatial_phase_y", 0.0)
    texture = np.sin(phase_x) * np.sin(phase_y)
    return base + 0.32 * texture[..., None] * tint


def curvature_color(points: np.ndarray, extent: float, nuisance: dict[str, float]) -> np.ndarray:
    color = np.array([0.58, 0.66, 0.78], dtype=np.float64)
    angle = nuisance.get("curvature_albedo_angle_radians", 0.0)
    phase = nuisance.get("curvature_albedo_phase", 0.0)
    x_rot, _ = rotate_xy(points, angle)
    albedo = 1.0 + 0.035 * np.sin(2.0 * math.pi * (x_rot / (2.0 * extent) + 0.5) + phase)
    return color * albedo[..., None]


def render_image(config: dict[str, Any], sweep_family: str, parameter_value: float, camera: pilot.Camera, seed: int) -> np.ndarray:
    width, height = [int(value) for value in config["resolution"]]
    extent = float(config["scene_extent"])
    background = np.array(config["background_color"], dtype=np.float64)
    camera_angle_x = math.radians(float(config["camera_angle_x_degrees"]))
    c2w = np.array(camera.transform_matrix, dtype=np.float64)
    origins, dirs = pilot.camera_rays(width, height, camera_angle_x, c2w)
    nuisance = nuisance_parameters(sweep_family, seed)

    if sweep_family == "curvature":
        points, normals, valid = pilot.intersect_paraboloid(origins, dirs, parameter_value, extent=extent)
        colors = curvature_color(points, extent, nuisance)
    else:
        points, normals, valid = pilot.intersect_plane(origins, dirs)
        if sweep_family == "edge_sharpness":
            colors = edge_color(points, parameter_value, nuisance)
        elif sweep_family == "spatial_frequency":
            colors = frequency_color(points, parameter_value, extent, nuisance)
        else:
            raise ValueError(f"Unknown sweep family: {sweep_family}")
        valid = valid & pilot.in_scene_bounds(points, extent)

    pilot.ensure_finite("geometry points", points, valid)
    pilot.ensure_finite("geometry normals", normals, valid)
    image = pilot.shade(colors, normals, valid, background)
    return (np.clip(image, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def generate_scene(
    config: dict[str, Any],
    sweep_family: str,
    level: str,
    parameter_name: str,
    parameter_value: float,
    output_root: Path,
    seed: int,
    overwrite: bool,
) -> dict[str, Any]:
    sid = scene_id(sweep_family, level, seed)
    scene_dir = output_root / sid
    if scene_dir.exists() and any(scene_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing scene without --overwrite: {scene_dir}")
    train_dir = scene_dir / "train"
    test_dir = scene_dir / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    camera_angle_x = math.radians(float(config["camera_angle_x_degrees"]))
    train_cameras = pilot.make_cameras(config, "train")
    test_cameras = pilot.make_cameras(config, "test")
    for camera in [*train_cameras, *test_cameras]:
        image = render_image(config, sweep_family, parameter_value, camera, seed)
        Image.fromarray(image, mode="RGB").save(scene_dir / f"{camera.file_path[2:]}.png")
    pilot.write_transforms(scene_dir, "train", train_cameras, camera_angle_x)
    pilot.write_transforms(scene_dir, "test", test_cameras, camera_angle_x)

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
        "phase": "III-B multi-seed replication",
        "nuisance_parameters": nuisance_parameters(sweep_family, seed),
        "fixed_factors": {
            "camera_radius": config["camera_radius"],
            "camera_elevation_degrees": config["camera_elevation_degrees"],
            "camera_angle_x_degrees": config["camera_angle_x_degrees"],
            "scene_extent": config["scene_extent"],
            "background_color": config["background_color"],
            "train_camera_protocol": "fixed orbit shared by every generated scene and seed",
            "test_camera_protocol": "fixed orbit offset shared by every generated scene and seed",
            "lighting": "same diffuse plus ambient rule as Phase III seed 0",
        },
    }
    (scene_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


def read_existing_manifest(root: Path) -> list[dict[str, Any]]:
    csv_path = root / "manifest.csv"
    json_path = root / "manifest.json"
    if csv_path.exists():
        with csv_path.open("r", newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    if json_path.exists():
        data = json.loads(json_path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for key in ("scenes", "records", "manifest"):
                if isinstance(data.get(key), list):
                    return data[key]
    return []


def write_merged_manifest(root: Path, new_rows: list[dict[str, Any]]) -> None:
    existing = read_existing_manifest(root)
    by_scene = {str(row["scene_id"]): dict(row) for row in existing if row.get("scene_id")}
    for row in new_rows:
        by_scene[str(row["scene_id"])] = dict(row)
    rows = sorted(
        by_scene.values(),
        key=lambda row: (
            str(row.get("sweep_family", "")),
            int(row.get("seed", 0)),
            LEVEL_ORDER.get(str(row.get("level", "")), 99),
            str(row.get("scene_id", "")),
        ),
    )
    pilot.write_manifest(root, rows)


def diagnostic_rows(rows: list[dict[str, Any]], config: dict[str, Any], root: Path) -> list[dict[str, Any]]:
    background = np.array(config["background_color"], dtype=np.float64)
    threshold = float(config.get("foreground_threshold", 0.03))
    out: list[dict[str, Any]] = []
    for row in rows:
        scene_dir = root / str(row["scene_id"])
        train_images = sorted((scene_dir / "train").glob("*.png"))
        test_images = sorted((scene_dir / "test").glob("*.png"))
        if len(train_images) != int(config["train_view_count"]):
            raise RuntimeError(f"{row['scene_id']} has {len(train_images)} train images")
        if len(test_images) != int(config["test_view_count"]):
            raise RuntimeError(f"{row['scene_id']} has {len(test_images)} test images")
        per_view = [pilot.image_diagnostics(path, background, threshold) for path in test_images]
        item = {
            "scene_id": row["scene_id"],
            "sweep_family": row["sweep_family"],
            "level": row["level"],
            "parameter_value": row["parameter_value"],
            "seed": row["seed"],
            "train_view_count": len(train_images),
            "test_view_count": len(test_images),
            "mean_foreground_fraction": float(np.mean([v["foreground_fraction"] for v in per_view])),
            "mean_intensity": float(np.mean([v["mean_intensity"] for v in per_view])),
            "intensity_std": float(np.mean([v["intensity_std"] for v in per_view])),
            "mean_gradient_magnitude": float(np.mean([v["mean_gradient_magnitude"] for v in per_view])),
            "laplacian_high_frequency_energy": float(np.mean([v["laplacian_high_frequency_energy"] for v in per_view])),
        }
        if row["sweep_family"] == "curvature":
            target = float(config.get("foreground_target", 0.54))
            tolerance = float(config.get("foreground_tolerance", 0.03))
            item["foreground_target"] = target
            item["foreground_tolerance"] = tolerance
            item["foreground_pass"] = abs(item["mean_foreground_fraction"] - target) <= tolerance
            if not item["foreground_pass"]:
                raise RuntimeError(
                    f"{row['scene_id']} foreground {item['mean_foreground_fraction']:.4f} outside {target} +/- {tolerance}"
                )
        out.append(item)
    return out


def write_validation(root: Path, diagnostics: list[dict[str, Any]]) -> None:
    csv_path = root / "phase3b_stimulus_validation.csv"
    json_path = root / "phase3b_stimulus_validation.json"
    fieldnames: list[str] = []
    for row in diagnostics:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(diagnostics)
    json_path.write_text(json.dumps(diagnostics, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def configured_scene_ids(seeds: list[int]) -> list[str]:
    return [scene_id(family, level, seed) for seed in seeds for family in FAMILIES for level in LEVELS]


def generate(config: dict[str, Any], seeds: list[int], overwrite: bool) -> list[dict[str, Any]]:
    root = pilot.resolve_path(config["output_root"])
    root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        config_for_seed = json.loads(json.dumps(config))
        config_for_seed["seed"] = seed
        for sweep_family, sweep in config_for_seed["sweeps"].items():
            parameter_name = str(sweep["parameter_name"])
            for level, parameter_value in sweep["levels"].items():
                rows.append(
                    generate_scene(
                        config_for_seed,
                        sweep_family,
                        str(level),
                        parameter_name,
                        float(parameter_value),
                        root,
                        seed,
                        overwrite=overwrite,
                    )
                )
    diagnostics = diagnostic_rows(rows, config, root)
    write_validation(root, diagnostics)
    write_merged_manifest(root, rows)
    print(f"Generated {len(rows)} Phase III-B scenes under {root}")
    print(f"Wrote {root / 'phase3b_stimulus_validation.csv'}")
    print(f"Wrote {root / 'manifest.csv'}")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/synthetic/phase3_controlled_multiseed.json"))
    parser.add_argument("--seeds", type=parse_int_list, default=None, help="Comma-separated seeds to generate. Defaults to new_replication_seeds from config, then 1,2,3,4.")
    parser.add_argument("--overwrite", action="store_true", help="Allow regenerating existing scene directories. Never use this for canonical seed 0.")
    parser.add_argument("--print-scene-ids", action="store_true", help="Print configured scene IDs for the requested seeds and exit.")
    args = parser.parse_args()

    config = load_config(args.config)
    seeds = args.seeds
    if seeds is None:
        seeds = [int(seed) for seed in config.get("new_replication_seeds", DEFAULT_NEW_SEEDS)]
    if 0 in seeds and not args.overwrite:
        raise ValueError("Seed 0 is canonical and must not be regenerated without explicit --overwrite.")
    if args.print_scene_ids:
        for sid in configured_scene_ids(seeds):
            print(sid)
        return 0
    generate(config, seeds, overwrite=args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
