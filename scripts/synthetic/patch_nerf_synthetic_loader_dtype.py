#!/usr/bin/env python3
"""Patch NeRF-synthetic Pillow dtype compatibility in gitignored loaders."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

TARGETS = [
    ("3DGS", "external/gaussian-splatting/scene/dataset_readers.py"),
    ("GES", "external/ges-splatting/scene/dataset_readers.py"),
    ("DRK", "external/drk/scene/dataset_readers.py"),
]

LEGACY_PATTERN = re.compile(
    r"(Image\.fromarray\(\s*np\.array\(\s*arr\s*\*\s*255\.0\s*,\s*dtype=)np\.byte(\s*\)\s*,\s*['\"]RGBA?['\"]\s*\))"
)
PATCHED_PATTERN = re.compile(
    r"Image\.fromarray\(\s*np\.array\(\s*arr\s*\*\s*255\.0\s*,\s*dtype=np\.uint8\s*\)\s*,\s*['\"]RGBA?['\"]\s*\)"
)


def patch_file(path: Path, label: str, verify_only: bool) -> str:
    if not path.exists():
        raise FileNotFoundError(f"{label} loader not found: {path}")
    text = path.read_text(encoding="utf-8")
    if "readCamerasFromTransforms" not in text or "transforms_train.json" not in text:
        raise RuntimeError(f"{label} does not look like the expected NeRF-synthetic loader: {path}")

    legacy_matches = list(LEGACY_PATTERN.finditer(text))
    patched_matches = list(PATCHED_PATTERN.finditer(text))

    if legacy_matches:
        if verify_only:
            raise RuntimeError(f"{label} still uses signed np.byte in Pillow RGB/RGBA conversion: {path}")
        updated = LEGACY_PATTERN.sub(r"\1np.uint8\2", text)
        path.write_text(updated, encoding="utf-8")
        return f"patched {len(legacy_matches)} np.byte conversion(s)"

    if patched_matches:
        return f"already compatible ({len(patched_matches)} np.uint8 conversion(s))"

    raise RuntimeError(
        f"{label} expected NeRF-synthetic Image.fromarray(arr * 255.0, dtype=...) line was not found in {path}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--verify-only", action="store_true", help="Fail if a legacy signed-byte conversion is present.")
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    print("NeRF-synthetic loader dtype compatibility check")
    for label, relative_path in TARGETS:
        status = patch_file(project_root / relative_path, label, args.verify_only)
        print(f"{label}: {status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
