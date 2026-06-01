"""Force selected images from val/test into the YOLO train split.

Expected usage:
    python scripts/force_train_images.py --dataset data/yolo --images 046.jpg
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Move selected images and labels from val/test to train."
    )
    parser.add_argument("--dataset", type=Path, required=True, help="YOLO dataset root.")
    parser.add_argument("--images", nargs="+", required=True, help="Image filenames to force into train.")
    return parser.parse_args()


def move_pair_to_train(dataset_dir: Path, image_name: str) -> str:
    train_image_path = dataset_dir / "images" / "train" / image_name
    train_label_path = dataset_dir / "labels" / "train" / f"{Path(image_name).stem}.txt"

    if train_image_path.exists():
        if train_label_path.exists():
            return f"OK already in train: {image_name}"
        return f"Warning: image already in train but label is missing: {image_name}"

    for split in ["val", "test"]:
        image_path = dataset_dir / "images" / split / image_name
        label_path = dataset_dir / "labels" / split / f"{Path(image_name).stem}.txt"
        if not image_path.exists():
            continue

        train_image_path.parent.mkdir(parents=True, exist_ok=True)
        train_label_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(image_path), str(train_image_path))

        if label_path.exists():
            shutil.move(str(label_path), str(train_label_path))
        else:
            return f"Moved image to train, warning label missing: {image_name}"

        return f"Moved to train: {image_name}"

    return f"Warning: image not found in train/val/test: {image_name}"


def main() -> int:
    args = parse_args()
    if not args.dataset.exists():
        print(f"Error: dataset folder not found: {args.dataset}")
        return 1

    print("\nForce Train Images")
    print("-" * 32)
    for image_name in args.images:
        print(move_pair_to_train(args.dataset, image_name))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

