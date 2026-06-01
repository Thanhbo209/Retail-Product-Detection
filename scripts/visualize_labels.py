"""Visualize YOLO labels by drawing boxes and class names on images.

Expected usage:
    python scripts/visualize_labels.py --dataset data/yolo --classes configs/classes.txt --split train --count 20 --output outputs/visualizations
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass(frozen=True)
class YoloBox:
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float


@dataclass
class VisualizationStats:
    available_labeled_images: int = 0
    selected_images: int = 0
    saved_images: int = 0
    skipped_images: int = 0
    invalid_label_lines: int = 0
    missing_label_files: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw YOLO bounding boxes on random labeled images from one split."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to YOLO dataset root, for example data/yolo.",
    )
    parser.add_argument(
        "--classes",
        type=Path,
        required=True,
        help="Path to classes.txt with one class name per line.",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to visualize. Default: train.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=20,
        help="Number of labeled images to sample. Default: 20.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/visualizations"),
        help="Folder where visualized images will be saved.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling. Default: 42.",
    )
    return parser.parse_args()


def load_classes(classes_path: Path) -> list[str]:
    if not classes_path.exists():
        raise FileNotFoundError(f"Classes file not found: {classes_path}")

    classes = [
        line.strip()
        for line in classes_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    if not classes:
        raise ValueError(f"No classes found in classes file: {classes_path}")
    return classes


def find_images(image_dir: Path) -> dict[str, Path]:
    if not image_dir.exists():
        raise FileNotFoundError(f"Image split folder not found: {image_dir}")
    if not image_dir.is_dir():
        raise NotADirectoryError(f"Image split path is not a folder: {image_dir}")

    images: dict[str, Path] = {}
    for image_path in image_dir.iterdir():
        if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
            images[image_path.stem] = image_path
    return images


def select_labeled_images(
    images: dict[str, Path],
    label_dir: Path,
    count: int,
    seed: int,
) -> tuple[list[Path], int, int]:
    labeled_images: list[Path] = []
    missing_label_files = 0

    for stem, image_path in sorted(images.items()):
        label_path = label_dir / f"{stem}.txt"
        if label_path.exists():
            labeled_images.append(image_path)
        else:
            missing_label_files += 1

    random.Random(seed).shuffle(labeled_images)
    if count < 0:
        count = len(labeled_images)
    selected_images = labeled_images[:count]
    return selected_images, len(labeled_images), missing_label_files


def parse_yolo_label(
    label_path: Path,
    class_count: int,
) -> tuple[list[YoloBox], list[str]]:
    boxes: list[YoloBox] = []
    warnings: list[str] = []

    try:
        lines = label_path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        return boxes, [f"{label_path}: could not read label file as UTF-8 text"]

    for line_number, line_text in enumerate(lines, start=1):
        stripped_line = line_text.strip()
        if not stripped_line:
            continue

        parts = stripped_line.split()
        if len(parts) != 5:
            warnings.append(
                f"{label_path}:{line_number}: expected 5 values, got {len(parts)}"
            )
            continue

        try:
            class_id = int(parts[0])
        except ValueError:
            warnings.append(
                f"{label_path}:{line_number}: class_id is not an integer"
            )
            continue

        if class_id < 0 or class_id >= class_count:
            warnings.append(
                f"{label_path}:{line_number}: class_id {class_id} is out of range"
            )
            continue

        try:
            x_center, y_center, width, height = [float(value) for value in parts[1:]]
        except ValueError:
            warnings.append(
                f"{label_path}:{line_number}: coordinates must be numeric"
            )
            continue

        coordinates = (x_center, y_center, width, height)
        if any(value < 0 or value > 1 for value in coordinates):
            warnings.append(
                f"{label_path}:{line_number}: coordinates must be between 0 and 1"
            )
            continue

        if width <= 0 or height <= 0:
            warnings.append(
                f"{label_path}:{line_number}: width and height must be greater than 0"
            )
            continue

        boxes.append(
            YoloBox(
                class_id=class_id,
                x_center=x_center,
                y_center=y_center,
                width=width,
                height=height,
            )
        )

    return boxes, warnings


def yolo_to_pixel_box(box: YoloBox, image_width: int, image_height: int) -> tuple[int, int, int, int]:
    x_center = box.x_center * image_width
    y_center = box.y_center * image_height
    width = box.width * image_width
    height = box.height * image_height

    x1 = int(round(x_center - width / 2))
    y1 = int(round(y_center - height / 2))
    x2 = int(round(x_center + width / 2))
    y2 = int(round(y_center + height / 2))

    x1 = max(0, min(x1, image_width - 1))
    y1 = max(0, min(y1, image_height - 1))
    x2 = max(0, min(x2, image_width - 1))
    y2 = max(0, min(y2, image_height - 1))

    return x1, y1, x2, y2


def class_color(class_id: int) -> tuple[int, int, int]:
    palette = [
        (43, 57, 255),
        (43, 170, 255),
        (43, 255, 170),
        (43, 255, 57),
        (170, 255, 43),
        (255, 170, 43),
        (255, 57, 43),
        (255, 43, 170),
        (170, 43, 255),
        (57, 43, 255),
    ]
    return palette[class_id % len(palette)]


def draw_boxes(
    image_path: Path,
    boxes: list[YoloBox],
    classes: list[str],
    output_path: Path,
) -> bool:
    try:
        import cv2
    except ImportError:
        print("Error: opencv-python is not installed. Run: pip install -r requirements.txt")
        return False

    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Warning: could not read image: {image_path}")
        return False

    image_height, image_width = image.shape[:2]
    for box in boxes:
        x1, y1, x2, y2 = yolo_to_pixel_box(box, image_width, image_height)
        color = class_color(box.class_id)
        class_name = classes[box.class_id]

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        label = f"{class_name}"
        text_size, baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            2,
        )
        text_width, text_height = text_size
        label_y1 = max(0, y1 - text_height - baseline - 4)
        label_y2 = label_y1 + text_height + baseline + 4
        label_x2 = min(image_width - 1, x1 + text_width + 8)

        cv2.rectangle(image, (x1, label_y1), (label_x2, label_y2), color, -1)
        cv2.putText(
            image,
            label,
            (x1 + 4, label_y2 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(output_path), image))


def visualize_dataset(
    dataset_root: Path,
    classes: list[str],
    split: str,
    count: int,
    output_dir: Path,
    seed: int,
) -> VisualizationStats:
    image_dir = dataset_root / "images" / split
    label_dir = dataset_root / "labels" / split
    if not label_dir.exists():
        raise FileNotFoundError(f"Label split folder not found: {label_dir}")
    if not label_dir.is_dir():
        raise NotADirectoryError(f"Label split path is not a folder: {label_dir}")

    images = find_images(image_dir)
    selected_images, available_labeled_images, missing_label_files = select_labeled_images(
        images=images,
        label_dir=label_dir,
        count=count,
        seed=seed,
    )

    stats = VisualizationStats(
        available_labeled_images=available_labeled_images,
        selected_images=len(selected_images),
        missing_label_files=missing_label_files,
    )

    for image_path in selected_images:
        label_path = label_dir / f"{image_path.stem}.txt"
        boxes, warnings = parse_yolo_label(label_path, len(classes))
        stats.invalid_label_lines += len(warnings)
        for warning in warnings:
            print(f"Warning: {warning}")

        if not boxes:
            print(f"Warning: no valid boxes to draw for {image_path}")
            stats.skipped_images += 1
            continue

        output_path = output_dir / split / f"{image_path.stem}_labels{image_path.suffix}"
        if draw_boxes(image_path, boxes, classes, output_path):
            stats.saved_images += 1
        else:
            stats.skipped_images += 1

    return stats


def print_summary(split: str, output_dir: Path, stats: VisualizationStats) -> None:
    print("\nLabel Visualization Summary")
    print("=" * 32)
    print(f"Split:                  {split}")
    print(f"Available labeled imgs: {stats.available_labeled_images}")
    print(f"Selected images:        {stats.selected_images}")
    print(f"Saved visualizations:   {stats.saved_images}")
    print(f"Skipped images:         {stats.skipped_images}")
    print(f"Missing label files:    {stats.missing_label_files}")
    print(f"Invalid label lines:    {stats.invalid_label_lines}")
    print(f"Output folder:          {output_dir}")


def main() -> int:
    args = parse_args()

    if args.count == 0:
        print("No images requested because --count is 0.")
        return 0

    try:
        classes = load_classes(args.classes)
        stats = visualize_dataset(
            dataset_root=args.dataset,
            classes=classes,
            split=args.split,
            count=args.count,
            output_dir=args.output,
            seed=args.seed,
        )
    except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
        print(f"Error: {exc}")
        return 1

    print_summary(args.split, args.output, stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
