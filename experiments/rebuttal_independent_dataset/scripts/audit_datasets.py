from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys

from PIL import Image

from common import atomic_json, load_config, resolve, write_csv


CAMERA_MODELS = {
    0: ("SIMPLE_PINHOLE", 3), 1: ("PINHOLE", 4), 2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5), 4: ("OPENCV", 8), 5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12), 7: ("FOV", 5), 8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5), 10: ("THIN_PRISM_FISHEYE", 12),
}


def digest_text(values: list[str]) -> str:
    return hashlib.sha256(("\n".join(values) + "\n").encode()).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_cameras_binary(path: Path) -> dict[int, dict]:
    cameras: dict[int, dict] = {}
    with path.open("rb") as handle:
        count = struct.unpack("<Q", handle.read(8))[0]
        for _ in range(count):
            camera_id, model_id, width, height = struct.unpack("<iiQQ", handle.read(24))
            if model_id not in CAMERA_MODELS:
                raise ValueError(f"Unknown COLMAP camera model id {model_id}")
            model, parameters = CAMERA_MODELS[model_id]
            handle.read(8 * parameters)
            cameras[camera_id] = {"model": model, "width": width, "height": height}
    return cameras


def read_images_binary(path: Path) -> list[dict]:
    images: list[dict] = []
    with path.open("rb") as handle:
        count = struct.unpack("<Q", handle.read(8))[0]
        for _ in range(count):
            values = struct.unpack("<idddddddi", handle.read(64))
            camera_id = values[-1]
            name = bytearray()
            while True:
                value = handle.read(1)
                if not value:
                    raise EOFError(f"Unterminated image name in {path}")
                if value == b"\x00":
                    break
                name.extend(value)
            points = struct.unpack("<Q", handle.read(8))[0]
            handle.seek(24 * points, 1)
            images.append({"name": name.decode("utf-8"), "camera_id": camera_id})
    return images


def read_cameras_text(path: Path) -> dict[int, dict]:
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        result[int(fields[0])] = {"model": fields[1], "width": int(fields[2]), "height": int(fields[3])}
    return result


def read_images_text(path: Path) -> list[dict]:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line and not line.startswith("#")]
    result = []
    for index in range(0, len(lines), 2):
        fields = lines[index].split()
        result.append({"name": fields[9], "camera_id": int(fields[8])})
    return result


def colmap_records(dataset: Path) -> tuple[dict[int, dict], list[dict], list[Path]]:
    sparse = dataset / "sparse" / "0"
    camera_binary, image_binary = sparse / "cameras.bin", sparse / "images.bin"
    if camera_binary.is_file() and image_binary.is_file():
        return read_cameras_binary(camera_binary), read_images_binary(image_binary), [camera_binary, image_binary]
    camera_text, image_text = sparse / "cameras.txt", sparse / "images.txt"
    if camera_text.is_file() and image_text.is_file():
        return read_cameras_text(camera_text), read_images_text(image_text), [camera_text, image_text]
    raise FileNotFoundError(f"No COLMAP cameras/images model under {sparse}")


def audit_scene(config: dict, key: str, scene: dict) -> dict:
    dataset = resolve(config, scene["dataset"])
    row: dict[str, object] = {
        "scene": key, "label": scene["label"], "dataset": str(dataset),
        "official_archive": scene["official_archive"], "present": dataset.is_dir(),
        "split_policy": "lexicographically sorted COLMAP image names; index modulo 8 == 0 is test",
        "outputs": {method: str(resolve(config, config["output_root"]) / key / method / "seed_0") for method in config["methods"]},
    }
    if not dataset.is_dir():
        row["status"] = "missing"
        return row
    cameras, images, model_files = colmap_records(dataset)
    names = sorted(record["name"] for record in images)
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate COLMAP image names in {dataset}")
    if any(Path(name).name != name for name in names):
        raise ValueError("Nested COLMAP image names are incompatible with GES/DRK basename loading")
    image_root = dataset / "images"
    missing = [name for name in names if not (image_root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} COLMAP images, first={missing[0]}")
    holdout = int(config["split"]["holdout"])
    test_names = names[::holdout]
    test_set = set(test_names)
    train_names = [name for name in names if name not in test_set]
    actual_sizes = set()
    size_mismatches = []
    records_by_name = {record["name"]: record for record in images}
    for name in names:
        with Image.open(image_root / name) as image:
            actual_sizes.add(image.size)
            camera = cameras[records_by_name[name]["camera_id"]]
            if image.size != (int(camera["width"]), int(camera["height"])):
                size_mismatches.append({"image": name, "image_size": image.size, "camera_size": [camera["width"], camera["height"]]})
    if size_mismatches:
        raise ValueError(f"COLMAP/image resolution mismatch in {dataset}; first={size_mismatches[0]}")
    colmap_sizes = {(int(camera["width"]), int(camera["height"])) for camera in cameras.values()}
    models = sorted({camera["model"] for camera in cameras.values()})
    unsupported = sorted(set(models) - {"PINHOLE", "SIMPLE_PINHOLE"})
    if unsupported:
        raise ValueError(f"3DGS/GES common loader does not support camera models {unsupported}")
    sparse = dataset / "sparse" / "0"
    point_files = [path for path in (sparse / "points3D.bin", sparse / "points3D.txt", sparse / "points3D.ply") if path.is_file()]
    if not point_files:
        raise FileNotFoundError(f"Missing COLMAP points3D model under {sparse}")
    row.update({
        "status": "ready", "image_directory": str(image_root), "image_count": len(names),
        "train_view_count": len(train_names), "test_view_count": len(test_names),
        "train_split_sha256": digest_text(train_names), "test_split_sha256": digest_text(test_names),
        "all_views_sha256": digest_text(names), "camera_models": models,
        "colmap_resolutions": sorted([list(value) for value in colmap_sizes]),
        "image_resolutions": sorted([list(value) for value in actual_sizes]),
        "colmap_image_resolution_match": True,
        "colmap_model_files": {str(path): file_sha256(path) for path in [*model_files, *point_files]},
        "scene_normalization": "getNerfppNorm over training camera centers: translate=-mean(center), radius=1.1*max_distance",
        "method_split_equivalence": "3DGS, GES, and DRK all use the same sorted every-8th COLMAP holdout under --eval",
    })
    return row


def build_manifest(config: dict) -> list[dict]:
    return [audit_scene(config, key, scene) for key, scene in config["scenes"].items()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit independent benchmark COLMAP inputs and freeze split hashes.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--strict", action="store_true", help="Exit nonzero unless every configured scene is ready.")
    args = parser.parse_args()
    config = load_config(args.config)
    output = args.output or resolve(config, config["result_root"]) / "dataset_manifest.json"
    rows = build_manifest(config)
    payload = {
        "experiment": config["experiment"], "config": str(args.config.resolve()),
        "config_sha256": file_sha256(args.config), "split": config["split"],
        "training": {name: {key: value for key, value in method.items() if key in {"upstream", "commit", "iterations", "environment"}} for name, method in config["methods"].items()},
        "scenes": rows,
    }
    atomic_json(output, payload)
    write_csv(output.with_suffix(".csv"), rows)
    for row in rows:
        print(f"{row['scene']}: {row['status']} dataset={row['dataset']} test_views={row.get('test_view_count', 'n/a')}")
    if args.strict and any(row["status"] != "ready" for row in rows):
        print("Dataset audit failed; see manifest and official_archive fields.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
