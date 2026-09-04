from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Mapping, Sequence

from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.local_error.image_io import MethodSpec, ensure_same_shape, load_rgb, pair_views
from analysis.local_error.patches import Patch
from analysis.winner_maps.visualize import save_winner_map
from scripts.evaluate.local_compare import load_config, method_colors, resolve_path

DEFAULT_CONFIG = Path("configs/room_3dgs_ges_drk_budget250k_p32.json")
DEFAULT_PATCHES_CSV = Path("results/room/3dgs_vs_ges_vs_drk_budget250k_p32/patches.csv")
DEFAULT_OUTPUT_DIR = Path("results/room/3dgs_vs_ges_vs_drk_budget250k_p32/figure1_candidates")
DEFAULT_VIEWS = ("00030.png", "00038.png", "00023.png", "00019.png", "00027.png")
TIE_COLOR = (160, 160, 160)
PANEL_LABELS = ("GT", "3DGS", "GES", "DRK", "Local Winner Map")


def require_columns(fieldnames: Iterable[str] | None, required: Iterable[str]) -> List[str]:
    if fieldnames is None:
        raise ValueError("patches.csv has no header row")
    fields = list(fieldnames)
    missing = [column for column in required if column not in fields]
    if missing:
        raise ValueError(
            "patches.csv is missing required columns "
            f"{missing}; observed columns: {fields}"
        )
    return fields


def load_patch_rows(patches_csv: Path, candidate_views: Sequence[str]) -> tuple[Dict[str, List[Mapping[str, str]]], List[str]]:
    wanted = set(candidate_views)
    rows_by_view: Dict[str, List[Mapping[str, str]]] = {view: [] for view in candidate_views}
    with patches_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = require_columns(reader.fieldnames, ["view", "x", "y", "width", "height", "winner"])
        for row_index, row in enumerate(reader, start=2):
            view = (row.get("view") or "").strip()
            if view not in wanted:
                continue
            winner = (row.get("winner") or "").strip().lower()
            if winner not in {"3dgs", "ges", "drk", "tie"}:
                raise ValueError(
                    f"row {row_index} has unexpected winner={winner!r}; "
                    "expected one of 3dgs, ges, drk, tie"
                )
            rows_by_view[view].append(row)
    missing = [view for view, rows in rows_by_view.items() if not rows]
    if missing:
        raise ValueError(f"No patch rows found for candidate views: {missing}")
    return rows_by_view, fields


def patches_from_rows(rows: Sequence[Mapping[str, str]], patch_size: int) -> tuple[List[Patch], List[str]]:
    patches: List[Patch] = []
    winners: List[str] = []
    for row in rows:
        patch = Patch(
            x=int(float(row["x"])),
            y=int(float(row["y"])),
            width=int(float(row["width"])),
            height=int(float(row["height"])),
        )
        if patch.width != patch_size or patch.height != patch_size:
            raise ValueError(
                f"Patch geometry mismatch for view={row.get('view')}: "
                f"observed {patch.width}x{patch.height}, expected {patch_size}x{patch_size}"
            )
        patches.append(patch)
        winners.append(str(row["winner"]).strip().lower())
    return patches, winners


def load_paired_views(config_path: Path, config: Mapping[str, object]) -> Dict[str, object]:
    gt_dir = resolve_path(config_path, str(config["gt_dir"]))
    methods = [
        MethodSpec(
            name=str(item["name"]),
            render_dir=resolve_path(config_path, str(item["render_dir"])),
            render_name_template=item.get("render_name_template"),
        )
        for item in config["methods"]
    ]
    return {view.view: view for view in pair_views(gt_dir, methods)}


def open_rgb_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def draw_label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font: ImageFont.ImageFont) -> None:
    draw.text((xy[0] + 8, xy[1] + 8), text, fill=(20, 20, 20), font=font)


def draw_legend(draw: ImageDraw.ImageDraw, xy: tuple[int, int], colors: Mapping[str, tuple[int, int, int]], font: ImageFont.ImageFont) -> None:
    x, y = xy
    for label in ("3dgs", "ges", "drk", "tie"):
        color = colors[label]
        draw.rectangle((x, y + 3, x + 14, y + 17), fill=color, outline=(40, 40, 40))
        draw.text((x + 20, y), label.upper() if label != "tie" else "tie", fill=(20, 20, 20), font=font)
        x += 78 if label != "3dgs" else 86


def compose_sheet(
    output_path: Path,
    panels: Sequence[Image.Image],
    method_color_map: Mapping[str, tuple[int, int, int]],
    gutter: int = 8,
    header: int = 44,
) -> None:
    if len(panels) != len(PANEL_LABELS):
        raise ValueError(f"Expected {len(PANEL_LABELS)} panels, got {len(panels)}")
    widths = [panel.width for panel in panels]
    heights = [panel.height for panel in panels]
    canvas = Image.new(
        "RGB",
        (sum(widths) + gutter * (len(panels) - 1), max(heights) + header),
        (255, 255, 255),
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    x = 0
    for label, panel in zip(PANEL_LABELS, panels):
        draw_label(draw, (x, 0), label, font)
        if label == "Local Winner Map":
            draw_legend(draw, (x + 150, 8), method_color_map, font)
        canvas.paste(panel, (x, header))
        x += panel.width + gutter
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def write_sources_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fieldnames = [
        "view",
        "gt_path",
        "3dgs_render_path",
        "ges_render_path",
        "drk_render_path",
        "winner_map_path",
        "sheet_path",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Figure 1(b) candidate qualitative sheets from an existing local-comparison patches.csv."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--patches-csv", type=Path, default=DEFAULT_PATCHES_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--views", nargs="+", default=list(DEFAULT_VIEWS))
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = load_config(config_path)
    patch_size = int(config.get("patch_size", 32))
    paired_views = load_paired_views(config_path, config)
    rows_by_view, fields = load_patch_rows(args.patches_csv, args.views)
    print(f"inspected patches schema: {fields}")

    methods = [str(item["name"]) for item in config["methods"]]
    colors = method_colors(methods)
    colors["tie"] = TIE_COLOR
    missing_methods = [method for method in ("3dgs", "ges", "drk") if method not in colors]
    if missing_methods:
        raise ValueError(f"Config is missing expected methods for Figure 1(b): {missing_methods}")

    source_rows: List[Dict[str, object]] = []
    for view_name in args.views:
        if view_name not in paired_views:
            raise ValueError(f"Candidate view {view_name} is not present in paired GT/render views")
        view_paths = paired_views[view_name]
        gt = load_rgb(view_paths.gt_path)
        renders = {name: load_rgb(path) for name, path in view_paths.render_paths.items()}
        ensure_same_shape([gt, *renders.values()], view_name)
        height, width = gt.shape[:2]

        patches, winners = patches_from_rows(rows_by_view[view_name], patch_size)
        winner_map_path = args.output_dir / f"{Path(view_name).stem}_winner_map.png"
        sheet_path = args.output_dir / f"{Path(view_name).stem}_qualitative_sheet.png"
        save_winner_map(winner_map_path, (height, width), patches, winners, colors, tie_color=TIE_COLOR)

        panels = [
            open_rgb_image(view_paths.gt_path),
            open_rgb_image(view_paths.render_paths["3dgs"]),
            open_rgb_image(view_paths.render_paths["ges"]),
            open_rgb_image(view_paths.render_paths["drk"]),
            open_rgb_image(winner_map_path),
        ]
        compose_sheet(sheet_path, panels, colors)

        record = {
            "view": view_name,
            "gt_path": str(view_paths.gt_path),
            "3dgs_render_path": str(view_paths.render_paths["3dgs"]),
            "ges_render_path": str(view_paths.render_paths["ges"]),
            "drk_render_path": str(view_paths.render_paths["drk"]),
            "winner_map_path": str(winner_map_path),
            "sheet_path": str(sheet_path),
        }
        source_rows.append(record)
        print(json.dumps(record, sort_keys=True))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_sources_csv(args.output_dir / "figure1_candidate_sources.csv", source_rows)
    with (args.output_dir / "figure1_candidate_sources.json").open("w", encoding="utf-8") as handle:
        json.dump(source_rows, handle, indent=2, sort_keys=True)
    print(f"wrote sheets and winner maps under: {args.output_dir}")


if __name__ == "__main__":
    main()
