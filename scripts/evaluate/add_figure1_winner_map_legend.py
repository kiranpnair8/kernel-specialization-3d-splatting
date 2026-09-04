from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
import sys
from typing import Dict, Tuple

from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.winner_maps.visualize import save_winner_map
from scripts.evaluate.local_compare import method_colors

DEFAULT_INPUT = Path(
    "results/room/3dgs_vs_ges_vs_drk_budget250k_p32/figure1_candidates/00038_winner_map.png"
)
DEFAULT_OUTPUT = Path(
    "results/room/3dgs_vs_ges_vs_drk_budget250k_p32/figure1_candidates/00038_winner_map_legend.png"
)
METHODS = ["3dgs", "ges", "drk"]
LABELS = [("3dgs", "3DGS"), ("ges", "GES"), ("drk", "DRK"), ("tie", "Tie")]


def winner_colors() -> Dict[str, Tuple[int, int, int]]:
    colors = method_colors(METHODS)
    signature = inspect.signature(save_winner_map)
    tie_default = signature.parameters["tie_color"].default
    if tie_default is inspect.Parameter.empty:
        raise ValueError("Could not recover save_winner_map tie_color default")
    colors["tie"] = tuple(int(value) for value in tie_default)
    return colors


def load_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def legend_size(draw: ImageDraw.ImageDraw, font: ImageFont.ImageFont, swatch: int, item_gap: int, label_gap: int) -> tuple[int, int]:
    width = 0
    height = swatch
    for index, (_, label) in enumerate(LABELS):
        label_width, label_height = text_size(draw, label, font)
        width += swatch + label_gap + label_width
        if index < len(LABELS) - 1:
            width += item_gap
        height = max(height, label_height)
    return width, height


def add_margin_legend(input_path: Path, output_path: Path, overwrite: bool) -> Dict[str, object]:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file without --overwrite: {output_path}")
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Output path must not be the same as the input winner map")

    image = Image.open(input_path).convert("RGB")
    colors = winner_colors()
    font = load_font(24)
    measure = Image.new("RGB", (1, 1), (255, 255, 255))
    measure_draw = ImageDraw.Draw(measure)

    swatch = 18
    item_gap = 34
    label_gap = 8
    horizontal_pad = 20
    vertical_pad = 14
    legend_width, legend_height = legend_size(measure_draw, font, swatch, item_gap, label_gap)
    margin_height = legend_height + 2 * vertical_pad

    output = Image.new("RGB", (image.width, image.height + margin_height), (255, 255, 255))
    output.paste(image, (0, 0))
    draw = ImageDraw.Draw(output)

    x = max(horizontal_pad, (image.width - legend_width) // 2)
    y = image.height + vertical_pad
    for index, (key, label) in enumerate(LABELS):
        color = colors[key]
        label_width, label_height = text_size(draw, label, font)
        swatch_y = y + max(0, (legend_height - swatch) // 2)
        text_y = y + max(0, (legend_height - label_height) // 2) - 1
        draw.rectangle((x, swatch_y, x + swatch, swatch_y + swatch), fill=color)
        draw.text((x + swatch + label_gap, text_y), label, fill=(20, 20, 20), font=font)
        x += swatch + label_gap + label_width
        if index < len(LABELS) - 1:
            x += item_gap

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.save(output_path)
    return {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "input_dimensions": [image.width, image.height],
        "output_dimensions": [output.width, output.height],
        "legend_placement": "centered in a narrow white bottom margin; original winner-map pixels are unchanged above the margin",
        "colors": {key: list(value) for key, value in colors.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Add a compact four-class legend below an existing Figure 1 winner map.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    report = add_margin_legend(args.input, args.output, args.overwrite)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
