from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np
from PIL import Image


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


@dataclass(frozen=True)
class MethodSpec:
    name: str
    render_dir: Path


@dataclass(frozen=True)
class ViewPaths:
    view: str
    gt_path: Path
    render_paths: Mapping[str, Path]


def list_images(directory: Path) -> Dict[str, Path]:
    if not directory.exists():
        raise FileNotFoundError(f"Image directory does not exist: {directory}")
    if not directory.is_dir():
        raise NotADirectoryError(f"Expected image directory: {directory}")

    images: Dict[str, Path] = {}
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            images[path.name] = path
    if not images:
        raise FileNotFoundError(f"No images found in {directory}")
    return images


def pair_views(gt_dir: Path, methods: Sequence[MethodSpec]) -> List[ViewPaths]:
    gt_images = list_images(gt_dir)
    method_images = {method.name: list_images(method.render_dir) for method in methods}

    common = set(gt_images)
    for images in method_images.values():
        common &= set(images)

    if not common:
        details = [f"gt={len(gt_images)}"]
        details.extend(f"{name}={len(images)}" for name, images in method_images.items())
        raise ValueError("No common held-out view filenames across GT and renders: " + ", ".join(details))

    missing_messages = []
    for name, images in method_images.items():
        missing_gt = sorted(set(gt_images) - set(images))
        extra_render = sorted(set(images) - set(gt_images))
        if missing_gt:
            missing_messages.append(f"{name} missing {len(missing_gt)} GT filenames; first={missing_gt[:3]}")
        if extra_render:
            missing_messages.append(f"{name} has {len(extra_render)} extra render filenames; first={extra_render[:3]}")

    if missing_messages:
        raise ValueError("View filename mismatch:\n" + "\n".join(missing_messages))

    paired = []
    for view in sorted(common):
        paired.append(
            ViewPaths(
                view=view,
                gt_path=gt_images[view],
                render_paths={name: images[view] for name, images in method_images.items()},
            )
        )
    return paired


def load_rgb(path: Path) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    return np.asarray(image, dtype=np.float32) / 255.0


def save_rgb(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(image, 0.0, 1.0)
    Image.fromarray((clipped * 255.0 + 0.5).astype(np.uint8)).save(path)


def ensure_same_shape(images: Iterable[np.ndarray], label: str) -> None:
    shapes = {image.shape for image in images}
    if len(shapes) != 1:
        raise ValueError(f"Shape mismatch for {label}: {sorted(shapes)}")
