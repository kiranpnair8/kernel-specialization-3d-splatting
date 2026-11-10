from __future__ import annotations

import argparse
import json
from pathlib import Path


MARKER = "REBUTTAL_MULTI_SEED_CONTROL_V1"


def replace_once(path: Path, old: str, new: str) -> bool:
    text = path.read_text(encoding="utf-8")
    if new in text:
        return False
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"Expected exactly one patch anchor in {path}, found {count}: {old!r}")
    path.write_text(text.replace(old, new), encoding="utf-8")
    return True


def patch_safe_state(path: Path, existing_signature: str) -> bool:
    text = path.read_text(encoding="utf-8")
    if MARKER in text:
        return False
    if "import os\n" not in text:
        if "import sys\n" not in text:
            raise RuntimeError(f"Cannot install required os import in {path}")
        text = text.replace("import sys\n", "import sys\nimport os\n", 1)
    old_body = f"def safe_state({existing_signature}):\n    old_f = sys.stdout\n"
    new_body = f"def safe_state(silent, seed=0):\n    # {MARKER}\n    os.environ[\"PYTHONHASHSEED\"] = str(seed)\n    old_f = sys.stdout\n"
    if old_body not in text:
        raise RuntimeError(f"safe_state signature/body anchor not found in {path}")
    text = text.replace(old_body, new_body, 1)
    if existing_signature == "silent, seed=0":
        replacements = {
            "torch.manual_seed(seed)": "torch.manual_seed(seed)\n    torch.cuda.manual_seed_all(seed)\n    torch.backends.cudnn.deterministic = True\n    torch.backends.cudnn.benchmark = False",
        }
    else:
        replacements = {
            "random.seed(0)": "random.seed(seed)",
            "np.random.seed(0)": "np.random.seed(seed)",
            "torch.manual_seed(0)": "torch.manual_seed(seed)\n    torch.cuda.manual_seed_all(seed)\n    torch.backends.cudnn.deterministic = True\n    torch.backends.cudnn.benchmark = False",
        }
    for old, new in replacements.items():
        if old not in text:
            raise RuntimeError(f"Expected RNG anchor {old!r} in {path}")
        text = text.replace(old, new, 1)
    path.write_text(text, encoding="utf-8")
    return True


def patch_method(root: Path, method: str) -> list[str]:
    changed: list[str] = []
    general = root / "utils" / "general_utils.py"
    train = root / ("train_ges.py" if method == "ges" else "train.py")
    if patch_safe_state(general, "silent, seed=0" if method == "ges" else "silent"):
        changed.append(str(general))
    if method in {"3dgs", "drk"}:
        if replace_once(
            train,
            '    parser.add_argument("--quiet", action="store_true")\n',
            '    parser.add_argument("--quiet", action="store_true")\n    parser.add_argument("--seed", type=int, default=0)\n',
        ):
            changed.append(str(train))
        if replace_once(train, "    safe_state(args.quiet)\n", "    safe_state(args.quiet, args.seed)\n"):
            if str(train) not in changed:
                changed.append(str(train))
    return changed


def verify(root: Path, method: str) -> None:
    general = (root / "utils" / "general_utils.py").read_text(encoding="utf-8")
    train = (root / ("train_ges.py" if method == "ges" else "train.py")).read_text(encoding="utf-8")
    required = ["import os", MARKER, "random.seed(seed)", "np.random.seed(seed)", "torch.manual_seed(seed)", "torch.cuda.manual_seed_all(seed)"]
    missing = [token for token in required if token not in general]
    if method in {"3dgs", "drk"}:
        missing.extend(token for token in ['--seed', "safe_state(args.quiet, args.seed)"] if token not in train)
    elif "safe_state(args.quiet, args.seed)" not in train:
        missing.append("GES seed propagation")
    if missing:
        raise RuntimeError(f"Seed-control verification failed for {method}: {missing}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Idempotently install/verify rebuttal seed controls in ignored upstream checkouts.")
    parser.add_argument("--method", choices=["3dgs", "ges", "drk"], required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    changed = [] if args.verify_only else patch_method(root, args.method)
    verify(root, args.method)
    print(json.dumps({"method": args.method, "root": str(root), "changed": changed, "verified": True}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
