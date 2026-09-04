from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, Tuple

from PIL import Image, ImageDraw, ImageFont

DEFAULT_INPUT_DIR = Path("results/room/3dgs_vs_ges_vs_drk_budget250k_p32/figure1_crops")
DEFAULT_OUTPUT_NAME = "figure1_crop_contact_sheet.png"
ROWS = (
    ("3DGS-favored region", "3dgs"),
    ("GES-favored region", "ges"),
    ("DRK-favored region", "drk"),
)
COLUMNS = (
    ("GT", "gt"),
    ("3DGS", "3dgs"),
    ("GES", "ges"),
    ("DRK", "drk"),
)


def load_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def load_crops(input_dir: Path) -> Dict[tuple[str, str], Image.Image]:
    crops: Dict[tuple[str, str], Image.Image] = {}
    for _, row_key in ROWS:
        for _, column_key in COLUMNS:
            path = input_dir / f"region_{row_key}_{column_key}.png"
            if not path.exists():
                raise FileNotFoundError(f"Missing expected crop: {path}")
            crops[(row_key, column_key)] = Image.open(path).convert("RGB")
    sizes = {crop.size for crop in crops.values()}
    if len(sizes) != 1:
        raise ValueError(f"Expected all crops to share one size; observed: {sorted(sizes)}")
    return crops


def make_contact_sheet(input_dir: Path, output_path: Path, spacing: int = 12) -> None:
    crops = load_crops(input_dir)
    crop_width, crop_height = next(iter(crops.values())).size
    column_font = load_font(34)
    row_font = load_font(30)

    measure_canvas = Image.new("RGB", (1, 1), (255, 255, 255))
    measure_draw = ImageDraw.Draw(measure_canvas)
    left_margin = max(text_size(measure_draw, label, row_font)[0] for label, _ in ROWS) + 28
    top_margin = max(text_size(measure_draw, label, column_font)[1] for label, _ in COLUMNS) + 28

    canvas_width = left_margin + len(COLUMNS) * crop_width + (len(COLUMNS) - 1) * spacing
    canvas_height = top_margin + len(ROWS) * crop_height + (len(ROWS) - 1) * spacing
    canvas = Image.new("RGB", (canvas_width, canvas_height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    for column_index, (column_label, _) in enumerate(COLUMNS):
        x = left_margin + column_index * (crop_width + spacing)
        label_width, _ = text_size(draw, column_label, column_font)
        draw.text((x + (crop_width - label_width) / 2, 12), column_label, fill=(20, 20, 20), font=column_font)

    for row_index, (row_label, row_key) in enumerate(ROWS):
        y = top_margin + row_index * (crop_height + spacing)
        _, label_height = text_size(draw, row_label, row_font)
        draw.text((12, y + (crop_height - label_height) / 2), row_label, fill=(20, 20, 20), font=row_font)
        for column_index, (_, column_key) in enumerate(COLUMNS):
            x = left_margin + column_index * (crop_width + spacing)
            canvas.paste(crops[(row_key, column_key)], (x, y))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    print(f"wrote: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a labeled inspection contact sheet for Figure 1(b) crops.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--spacing", type=int, default=12)
    args = parser.parse_args()

    output = args.output if args.output is not None else args.input_dir / DEFAULT_OUTPUT_NAME
    make_contact_sheet(args.input_dir, output, spacing=args.spacing)


if __name__ == "__main__":
    main()
