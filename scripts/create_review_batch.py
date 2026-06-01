"""Create a controlled review batch for hard examples.

Expected usage:
    python scripts/create_review_batch.py --batch-name review_batch_001 --images data/processed/images/046.jpg --output review_batches
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path


MANIFEST_COLUMNS = ["image_filename", "source_path", "status", "notes"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a review workspace for selected hard examples."
    )
    parser.add_argument(
        "--batch-name",
        required=True,
        help="Review batch name, for example review_batch_001.",
    )
    parser.add_argument(
        "--images",
        nargs="+",
        type=Path,
        required=True,
        help="One or more image paths to copy into the review batch.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("review_batches"),
        help="Root output folder for review batches. Default: review_batches.",
    )
    return parser.parse_args()


def create_batch_dirs(batch_dir: Path) -> dict[str, Path]:
    dirs = {
        "images": batch_dir / "images",
        "labels_initial": batch_dir / "labels_initial",
        "labels_fixed": batch_dir / "labels_fixed",
        "notes": batch_dir / "notes",
    }
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    return dirs


def copy_images(image_paths: list[Path], batch_images_dir: Path) -> list[dict[str, str]]:
    manifest_rows: list[dict[str, str]] = []
    for image_path in image_paths:
        if not image_path.exists() or not image_path.is_file():
            print(f"Warning: image not found, skipped: {image_path}")
            continue

        destination = batch_images_dir / image_path.name
        shutil.copy2(image_path, destination)
        manifest_rows.append(
            {
                "image_filename": image_path.name,
                "source_path": str(image_path),
                "status": "needs_review",
                "notes": "",
            }
        )
        print(f"Copied image: {image_path} -> {destination}")

    return manifest_rows


def write_manifest(manifest_path: Path, rows: list[dict[str, str]]) -> None:
    with manifest_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def print_next_steps(batch_dir: Path) -> None:
    print("\nReview batch created")
    print("-" * 32)
    print(f"Batch folder: {batch_dir}")
    print(f"Images:       {batch_dir / 'images'}")
    print(f"Initial YOLO: {batch_dir / 'labels_initial'}")
    print(f"Fixed YOLO:   {batch_dir / 'labels_fixed'}")
    print(f"Manifest:     {batch_dir / 'manifest.csv'}")
    print("\nNext CVAT steps:")
    print("1. Convert model predictions into labels_initial if needed.")
    print("2. Package labels_initial with package_cvat_yolo_import.py.")
    print("3. Import the zip into CVAT using YOLO format.")
    print("4. Fix labels manually in CVAT.")
    print("5. Export fixed YOLO labels into labels_fixed.")


def main() -> int:
    args = parse_args()
    batch_dir = args.output / args.batch_name
    dirs = create_batch_dirs(batch_dir)
    rows = copy_images(args.images, dirs["images"])
    write_manifest(batch_dir / "manifest.csv", rows)
    print_next_steps(batch_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

