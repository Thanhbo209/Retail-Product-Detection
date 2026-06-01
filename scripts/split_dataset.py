"""Split image-label pairs into a YOLO train/val/test folder structure.

Expected usage:
    python scripts/split_dataset.py --images data/processed/images --labels data/processed/labels --output data/yolo --copy
"""

from __future__ import annotations

import argparse
import random
import shutil
from dataclasses import dataclass
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class ImageLabelPair:
    image_path: Path
    label_path: Path


@dataclass
class SplitResult:
    train: list[ImageLabelPair]
    val: list[ImageLabelPair]
    test: list[ImageLabelPair]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split a flat image/label folder into YOLO train/val/test folders."
    )
    parser.add_argument(
        "--images",
        type=Path,
        required=True,
        help="Input folder containing image files.",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        required=True,
        help="Input folder containing YOLO .txt label files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output YOLO dataset folder.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.70,
        help="Train split ratio. Default: 0.70.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.20,
        help="Validation split ratio. Default: 0.20.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.10,
        help="Test split ratio. Default: 0.10.",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        default=True,
        help="Copy files instead of moving them. Copy mode is the safe default.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible splits. Default: 42.",
    )
    return parser.parse_args()


def validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    ratios = {
        "train-ratio": train_ratio,
        "val-ratio": val_ratio,
        "test-ratio": test_ratio,
    }
    for name, value in ratios.items():
        if value < 0:
            raise ValueError(f"{name} must be greater than or equal to 0.")

    total = train_ratio + val_ratio + test_ratio
    if total <= 0:
        raise ValueError("Split ratios must add up to a positive value.")
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"Split ratios must add up to 1.0. Current total: {total:.4f}"
        )


def find_images(images_dir: Path) -> dict[str, Path]:
    if not images_dir.exists():
        raise FileNotFoundError(f"Images folder not found: {images_dir}")
    if not images_dir.is_dir():
        raise NotADirectoryError(f"Images path is not a folder: {images_dir}")

    images: dict[str, Path] = {}
    for image_path in images_dir.iterdir():
        if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
            images[image_path.stem] = image_path
    return images


def find_labels(labels_dir: Path) -> dict[str, Path]:
    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels folder not found: {labels_dir}")
    if not labels_dir.is_dir():
        raise NotADirectoryError(f"Labels path is not a folder: {labels_dir}")

    labels: dict[str, Path] = {}
    for label_path in labels_dir.iterdir():
        if label_path.is_file() and label_path.suffix.lower() == ".txt":
            labels[label_path.stem] = label_path
    return labels


def match_pairs(
    images: dict[str, Path],
    labels: dict[str, Path],
) -> tuple[list[ImageLabelPair], list[Path], list[Path]]:
    pairs: list[ImageLabelPair] = []
    skipped_images: list[Path] = []
    orphan_labels: list[Path] = []

    for stem, image_path in sorted(images.items()):
        label_path = labels.get(stem)
        if label_path is None:
            skipped_images.append(image_path)
            continue
        pairs.append(ImageLabelPair(image_path=image_path, label_path=label_path))

    for stem, label_path in sorted(labels.items()):
        if stem not in images:
            orphan_labels.append(label_path)

    return pairs, skipped_images, orphan_labels


def split_pairs(
    pairs: list[ImageLabelPair],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> SplitResult:
    shuffled_pairs = pairs.copy()
    random.Random(seed).shuffle(shuffled_pairs)

    total = len(shuffled_pairs)
    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)

    train_pairs = shuffled_pairs[:train_count]
    val_pairs = shuffled_pairs[train_count : train_count + val_count]
    test_pairs = shuffled_pairs[train_count + val_count :]

    return SplitResult(train=train_pairs, val=val_pairs, test=test_pairs)


def create_yolo_folders(output_dir: Path) -> None:
    for split in SPLITS:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)


def copy_pair(pair: ImageLabelPair, output_dir: Path, split: str) -> None:
    image_destination = output_dir / "images" / split / pair.image_path.name
    label_destination = output_dir / "labels" / split / pair.label_path.name

    shutil.copy2(pair.image_path, image_destination)
    shutil.copy2(pair.label_path, label_destination)


def write_pairs(split_result: SplitResult, output_dir: Path) -> None:
    create_yolo_folders(output_dir)
    split_mapping = {
        "train": split_result.train,
        "val": split_result.val,
        "test": split_result.test,
    }

    for split, pairs in split_mapping.items():
        for pair in pairs:
            copy_pair(pair, output_dir, split)


def print_warning_list(title: str, paths: list[Path]) -> None:
    if not paths:
        return

    print(f"\n{title}")
    print("-" * len(title))
    for path in paths[:20]:
        print(f"- {path}")
    if len(paths) > 20:
        print(f"... {len(paths) - 20} more")


def print_summary(
    split_result: SplitResult,
    skipped_images: list[Path],
    orphan_labels: list[Path],
    output_dir: Path,
) -> None:
    total_pairs = (
        len(split_result.train)
        + len(split_result.val)
        + len(split_result.test)
    )

    print("\nDataset Split Summary")
    print("=" * 32)
    print(f"Output folder:  {output_dir}")
    print(f"Total pairs:    {total_pairs}")
    print(f"Train count:    {len(split_result.train)}")
    print(f"Val count:      {len(split_result.val)}")
    print(f"Test count:     {len(split_result.test)}")
    print(f"Skipped images: {len(skipped_images)}")
    print(f"Orphan labels:  {len(orphan_labels)}")

    print_warning_list("Images without matching label files", skipped_images)
    print_warning_list("Label files without matching images", orphan_labels)


def main() -> int:
    args = parse_args()

    try:
        validate_ratios(args.train_ratio, args.val_ratio, args.test_ratio)
        images = find_images(args.images)
        labels = find_labels(args.labels)
    except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
        print(f"Error: {exc}")
        return 1

    pairs, skipped_images, orphan_labels = match_pairs(images, labels)
    split_result = split_pairs(
        pairs=pairs,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    write_pairs(split_result, args.output)
    print_summary(split_result, skipped_images, orphan_labels, args.output)

    if not args.copy:
        print("\nNote: move mode is not implemented. Files were copied safely.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

