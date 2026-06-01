"""Accept human-fixed labels from a review batch into processed data.

Expected usage:
    python scripts/accept_fixed_labels.py --batch review_batches/review_batch_001 --processed-images data/processed/images --processed-labels data/processed/labels
"""

from __future__ import annotations

import argparse
import csv
import shutil
from datetime import datetime
from pathlib import Path


MANIFEST_COLUMNS = ["image_filename", "source_path", "status", "notes"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy fixed review labels into processed data with backups."
    )
    parser.add_argument("--batch", type=Path, required=True, help="Review batch folder.")
    parser.add_argument(
        "--processed-images",
        type=Path,
        required=True,
        help="Processed images folder.",
    )
    parser.add_argument(
        "--processed-labels",
        type=Path,
        required=True,
        help="Processed labels folder.",
    )
    return parser.parse_args()


def validate_batch(batch_dir: Path) -> tuple[Path, Path, Path]:
    images_dir = batch_dir / "images"
    labels_fixed_dir = batch_dir / "labels_fixed"
    manifest_path = batch_dir / "manifest.csv"

    if not images_dir.exists() or not images_dir.is_dir():
        raise FileNotFoundError(f"Batch images folder not found: {images_dir}")
    if not labels_fixed_dir.exists() or not labels_fixed_dir.is_dir():
        raise FileNotFoundError(f"Batch labels_fixed folder not found: {labels_fixed_dir}")
    return images_dir, labels_fixed_dir, manifest_path


def load_manifest(manifest_path: Path) -> list[dict[str, str]]:
    if not manifest_path.exists():
        return []
    with manifest_path.open("r", newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def save_manifest(manifest_path: Path, rows: list[dict[str, str]]) -> None:
    with manifest_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def update_manifest_rows(rows: list[dict[str, str]], accepted_images: set[str]) -> list[dict[str, str]]:
    row_by_image = {row.get("image_filename", ""): row for row in rows}
    for image_name in accepted_images:
        row = row_by_image.get(image_name)
        if row is None:
            row = {
                "image_filename": image_name,
                "source_path": "",
                "status": "accepted",
                "notes": "",
            }
            rows.append(row)
        else:
            row["status"] = "accepted"
    return rows


def find_matching_image(images_dir: Path, label_path: Path) -> Path | None:
    for image_path in images_dir.iterdir():
        if image_path.is_file() and image_path.stem == label_path.stem:
            return image_path
    return None


def backup_existing_label(label_path: Path, backup_dir: Path) -> None:
    if label_path.exists():
        backup_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(label_path, backup_dir / label_path.name)


def accept_labels(
    batch_dir: Path,
    processed_images_dir: Path,
    processed_labels_dir: Path,
) -> list[str]:
    images_dir, labels_fixed_dir, manifest_path = validate_batch(batch_dir)
    processed_images_dir.mkdir(parents=True, exist_ok=True)
    processed_labels_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = processed_labels_dir.parent / "labels_backups" / timestamp
    accepted_images: set[str] = set()

    for fixed_label_path in sorted(labels_fixed_dir.glob("*.txt")):
        image_path = find_matching_image(images_dir, fixed_label_path)
        if image_path is None:
            print(f"Warning: fixed label has no matching batch image: {fixed_label_path}")
            continue

        output_image_path = processed_images_dir / image_path.name
        output_label_path = processed_labels_dir / fixed_label_path.name

        backup_existing_label(output_label_path, backup_dir)
        shutil.copy2(image_path, output_image_path)
        shutil.copy2(fixed_label_path, output_label_path)
        accepted_images.add(image_path.name)
        print(f"Accepted: {image_path.name} -> {output_label_path}")

    rows = update_manifest_rows(load_manifest(manifest_path), accepted_images)
    save_manifest(manifest_path, rows)
    return sorted(accepted_images)


def main() -> int:
    args = parse_args()

    try:
        accepted = accept_labels(args.batch, args.processed_images, args.processed_labels)
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        return 1

    print("\nAccepted fixed labels")
    print("-" * 32)
    if accepted:
        for image_name in accepted:
            print(f"- {image_name}")
    else:
        print("- None")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

