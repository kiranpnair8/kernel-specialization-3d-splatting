from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.local_error.image_io import IMAGE_EXTENSIONS


def list_images(directory: Path) -> Dict[str, Path]:
    if not directory.exists():
        raise FileNotFoundError(f"Image directory does not exist: {directory}")
    images = {
        path.name: path
        for path in sorted(directory.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }
    if not images:
        raise FileNotFoundError(f"No images found in {directory}")
    return images


def image_signature(path: Path) -> Tuple[Tuple[int, int], str]:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        digest = hashlib.sha256(rgb.tobytes()).hexdigest()
        return rgb.size, digest


def candidate_name(reference_name: str, template: Optional[str]) -> str:
    if not template:
        return reference_name
    path = Path(reference_name)
    return template.format(name=reference_name, stem=path.stem, suffix=path.suffix)


def resolve_repo_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def audit_alignment(
    reference_dir: Path,
    candidate_dir: Path,
    candidate_name_template: Optional[str],
) -> Dict[str, object]:
    reference_images = list_images(reference_dir)
    candidate_images = list_images(candidate_dir)

    reference_signatures = {name: image_signature(path) for name, path in reference_images.items()}
    candidate_signatures = {name: image_signature(path) for name, path in candidate_images.items()}
    signature_to_candidates: Dict[Tuple[Tuple[int, int], str], List[str]] = {}
    for name, signature in candidate_signatures.items():
        signature_to_candidates.setdefault(signature, []).append(name)

    rows = []
    exact_template_matches = 0
    hash_mapped_matches = 0
    mismatches = []

    for reference_name in sorted(reference_images):
        reference_signature = reference_signatures[reference_name]
        expected_candidate = candidate_name(reference_name, candidate_name_template)
        matched_candidate = None
        status = "missing"

        if expected_candidate in candidate_signatures:
            matched_candidate = expected_candidate
            if candidate_signatures[expected_candidate] == reference_signature:
                status = "exact"
                exact_template_matches += 1
            else:
                status = "template_name_hash_mismatch"
        else:
            candidates = signature_to_candidates.get(reference_signature, [])
            if len(candidates) == 1:
                matched_candidate = candidates[0]
                status = "hash_mapped"
                hash_mapped_matches += 1
            elif len(candidates) > 1:
                status = "ambiguous_hash"

        if status not in {"exact", "hash_mapped"}:
            mismatches.append(reference_name)

        rows.append(
            {
                "reference": reference_name,
                "candidate": matched_candidate,
                "status": status,
                "reference_size": reference_signature[0],
                "candidate_size": candidate_signatures[matched_candidate][0] if matched_candidate else None,
                "same_hash": (
                    candidate_signatures.get(matched_candidate) == reference_signature
                    if matched_candidate
                    else False
                ),
            }
        )

    candidate_only = sorted(set(candidate_images) - {row["candidate"] for row in rows if row["candidate"]})
    exact = not mismatches and len(reference_images) == len(rows)
    return {
        "reference_dir": str(reference_dir),
        "candidate_dir": str(candidate_dir),
        "reference_count": len(reference_images),
        "candidate_count": len(candidate_images),
        "exact_or_unambiguous": exact,
        "exact_template_matches": exact_template_matches,
        "hash_mapped_matches": hash_mapped_matches,
        "mismatch_count": len(mismatches),
        "candidate_only_count": len(candidate_only),
        "candidate_only_first": candidate_only[:5],
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only GT alignment audit between rendered evaluation outputs.")
    parser.add_argument("--reference-gt-dir", required=True, type=Path)
    parser.add_argument("--candidate-gt-dir", required=True, type=Path)
    parser.add_argument("--candidate-name-template", default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    try:
        reference_dir = resolve_repo_path(args.reference_gt_dir)
        candidate_dir = resolve_repo_path(args.candidate_gt_dir)
        result = audit_alignment(reference_dir, candidate_dir, args.candidate_name_template)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2, sort_keys=True))
    if result["rows"]:
        print("first_rows:")
        for row in result["rows"][:5]:
            print(json.dumps(row, sort_keys=True))
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, sort_keys=True)

    if not result["exact_or_unambiguous"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
