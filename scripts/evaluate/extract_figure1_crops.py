from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.local_error.image_io import MethodSpec, pair_views
from scripts.evaluate.local_compare import load_config, resolve_path

DEFAULT_CONFIG = Path("configs/room_3dgs_ges_drk_budget250k_p32.json")
DEFAULT_OUTPUT_DIR = Path("results/room/3dgs_vs_ges_vs_drk_budget250k_p32/figure1_crops")
DEFAULT_VIEW = "00038.png"
CROP_SIZE = 256
SELECTED_REGIONS = {
    "3dgs": {"selected_x0": 1168, "selected_y0": 528, "selected_x1": 1296, "selected_y1": 656},
    "ges": {"selected_x0": 160, "selected_y0": 480, "selected_x1": 288, "selected_y1": 608},
    "drk": {"selected_x0": 1344, "selected_y0": 160, "selected_x1": 1472, "selected_y1": 288},
}
IMAGE_KEYS = ("gt", "3dgs", "ges", "drk")


def load_paired_view(config_path: Path, config: Mapping[str, object], view: str) -> object:
    gt_dir = resolve_path(config_path, str(config["gt_dir"]))
    methods = [
        MethodSpec(
            name=str(item["name"]),
            render_dir=resolve_path(config_path, str(item["render_dir"])),
            render_name_template=item.get("render_name_template"),
        )
        for item in config["methods"]
    ]
    paired = {item.view: item for item in pair_views(gt_dir, methods)}
    if view not in paired:
        raise ValueError(f"View {view} is not present in paired GT/render views")
    return paired[view]


def centered_crop_box(region: Mapping[str, int], image_width: int, image_height: int, crop_size: int) -> tuple[int, int, int, int]:
    selected_width = int(region["selected_x1"]) - int(region["selected_x0"])
    selected_height = int(region["selected_y1"]) - int(region["selected_y0"])
    if selected_width <= 0 or selected_height <= 0:
        raise ValueError(f"Invalid selected region: {region}")
    if crop_size > image_width or crop_size > image_height:
        raise ValueError(f"Requested crop_size={crop_size} does not fit image size {image_width}x{image_height}")

    center_x = 0.5 * (int(region["selected_x0"]) + int(region["selected_x1"]))
    center_y = 0.5 * (int(region["selected_y0"]) + int(region["selected_y1"]))
    x0 = int(round(center_x - crop_size / 2.0))
    y0 = int(round(center_y - crop_size / 2.0))
    x0 = max(0, min(x0, image_width - crop_size))
    y0 = max(0, min(y0, image_height - crop_size))
    return x0, y0, x0 + crop_size, y0 + crop_size


def open_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def save_crop(image: Image.Image, box: tuple[int, int, int, int], output_path: Path, overwrite: bool) -> None:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file without --overwrite: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.crop(box).save(output_path)


def write_csv(path: Path, rows: Sequence[Mapping[str, object]], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file without --overwrite: {path}")
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, rows: Sequence[Mapping[str, object]], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file without --overwrite: {path}")
    with path.open("w", encoding="utf-8") as handle:
        json.dump(list(rows), handle, indent=2, sort_keys=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract unlabeled publication-ready crops for Figure 1(b) from matched-budget Room renders."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--view", default=DEFAULT_VIEW)
    parser.add_argument("--crop-size", type=int, default=CROP_SIZE)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = load_config(config_path)
    view_paths = load_paired_view(config_path, config, args.view)
    source_paths: Dict[str, Path] = {
        "gt": view_paths.gt_path,
        "3dgs": view_paths.render_paths["3dgs"],
        "ges": view_paths.render_paths["ges"],
        "drk": view_paths.render_paths["drk"],
    }
    images = {key: open_rgb(path) for key, path in source_paths.items()}
    sizes = {key: image.size for key, image in images.items()}
    if len(set(sizes.values())) != 1:
        raise ValueError(f"Source image dimensions do not match for {args.view}: {sizes}")
    image_width, image_height = images["gt"].size

    metadata: List[Dict[str, object]] = []
    for region_name, selected in SELECTED_REGIONS.items():
        crop_box = centered_crop_box(selected, image_width, image_height, args.crop_size)
        crop_width = crop_box[2] - crop_box[0]
        crop_height = crop_box[3] - crop_box[1]
        region_outputs: Dict[str, str] = {}
        crop_sizes: Dict[str, tuple[int, int]] = {}
        for key in IMAGE_KEYS:
            output_path = args.output_dir / f"region_{region_name}_{key}.png"
            save_crop(images[key], crop_box, output_path, args.overwrite)
            region_outputs[f"{key}_crop_path"] = str(output_path)
            crop_sizes[key] = Image.open(output_path).size
        if len(set(crop_sizes.values())) != 1:
            raise ValueError(f"Crop dimensions do not match for region {region_name}: {crop_sizes}")
        metadata.append(
            {
                "region": region_name,
                "view": args.view,
                **selected,
                "crop_x0": crop_box[0],
                "crop_y0": crop_box[1],
                "crop_x1": crop_box[2],
                "crop_y1": crop_box[3],
                "crop_width": crop_width,
                "crop_height": crop_height,
                "image_width": image_width,
                "image_height": image_height,
                "gt_source_path": str(source_paths["gt"]),
                "3dgs_source_path": str(source_paths["3dgs"]),
                "ges_source_path": str(source_paths["ges"]),
                "drk_source_path": str(source_paths["drk"]),
                **region_outputs,
            }
        )

    csv_path = args.output_dir / "figure1_crop_metadata.csv"
    json_path = args.output_dir / "figure1_crop_metadata.json"
    write_csv(csv_path, metadata, args.overwrite)
    write_json(json_path, metadata, args.overwrite)

    print("source_paths:")
    print(json.dumps({key: str(path) for key, path in source_paths.items()}, indent=2, sort_keys=True))
    print("final_crop_coordinates:")
    for row in metadata:
        print(
            f"{row['region']}: "
            f"x0={row['crop_x0']} y0={row['crop_y0']} "
            f"x1={row['crop_x1']} y1={row['crop_y1']} "
            f"size={row['crop_width']}x{row['crop_height']}"
        )
    print(f"wrote: {csv_path}")
    print(f"wrote: {json_path}")


if __name__ == "__main__":
    main()
