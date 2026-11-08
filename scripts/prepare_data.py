"""Make deterministic stratified splits, removing exact duplicates first.

Usage:
python scripts/prepare_data.py --source /content/PlantVillage/color --output data/split
"""
import argparse
import hashlib
import json
import random
import shutil
from pathlib import Path

from PIL import Image

SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_dataset(source: Path, output: Path, seed: int = 42, expected_classes: int = 10):
    source = source.resolve()
    output = output.resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Not a directory: {source}")
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("The source and output directories must not be nested in one another.")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"{output} already contains files. Use a new output directory.")
    folders = sorted(p for p in source.iterdir() if p.is_dir() and p.name.startswith("Tomato___"))
    if len(folders) != expected_classes:
        raise ValueError(f"Expected {expected_classes} Tomato___ class folders; found {len(folders)} in {source}.")
    rng = random.Random(seed)
    seen_hashes = {}
    summary = {"seed": seed, "source": str(source), "classes": {}, "exact_duplicates_removed": 0, "invalid_images_skipped": 0}
    assignments = []
    for folder in folders:
        unique = []
        raw_count = 0
        for path in sorted(folder.iterdir()):
            if not path.is_file() or path.suffix.lower() not in SUFFIXES:
                continue
            raw_count += 1
            try:
                with Image.open(path) as im:
                    im.verify()
                fingerprint = file_sha256(path)
            except (OSError, ValueError):
                summary["invalid_images_skipped"] += 1
                continue
            existing_class = seen_hashes.get(fingerprint)
            if existing_class is not None:
                if existing_class != folder.name:
                    raise ValueError(f"Same image appears under conflicting labels: {path} vs {existing_class}")
                summary["exact_duplicates_removed"] += 1
                continue
            seen_hashes[fingerprint] = folder.name
            unique.append((path, fingerprint))
        if len(unique) < 3:
            raise ValueError(f"At least 3 unique images needed in {folder.name}; found {len(unique)}")
        rng.shuffle(unique)
        n = len(unique)
        n_train = max(1, min(n - 2, round(n * 0.70)))
        n_val = max(1, min(n - n_train - 1, round(n * 0.15)))
        n_test = n - n_train - n_val
        groups = {"train": unique[:n_train], "val": unique[n_train:n_train + n_val], "test": unique[n_train + n_val:]}
        for split, rows in groups.items():
            for path, fingerprint in rows:
                assignments.append((split, folder.name, path, fingerprint))
        summary["classes"][folder.name] = {
            "raw_count": raw_count, "unique_count": n, "train": n_train, "val": n_val, "test": n_test
        }
    for split, cls, path, fingerprint in assignments:
        dest = output / split / cls
        dest.mkdir(parents=True, exist_ok=True)
        name = f"{path.stem}_{fingerprint[:12]}{path.suffix.lower()}"
        shutil.copy2(path, dest / name)
    (output / "split_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Split PlantVillage Tomato color images without exact duplicate leakage.")
    parser.add_argument("--source", required=True, help="Directory containing the 10 Tomato___ class directories.")
    parser.add_argument("--output", default="data/split")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    summary = prepare_dataset(Path(args.source), Path(args.output), args.seed)
    for label, counts in summary["classes"].items():
        print(f"{label}: {counts['unique_count']} images -> {counts['train']}/{counts['val']}/{counts['test']}")
    print(f"Exact duplicates removed: {summary['exact_duplicates_removed']}")
    print(f"Invalid images skipped: {summary['invalid_images_skipped']}")


if __name__ == "__main__":
    main()
