"""Validate a YOLO object detection dataset.

Expected usage:
    python scripts/validate_yolo_dataset.py --dataset data/yolo --classes configs/classes.txt
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
DEFAULT_SPLITS = ("train", "val", "test")


@dataclass
class ValidationIssue:
    severity: str
    split: str
    file: str
    line: int | None
    issue_type: str
    message: str


@dataclass
class SplitStats:
    images: int = 0
    label_files: int = 0
    objects: int = 0
    empty_label_files: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate image-label pairs and YOLO label files."
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
        "--output-dir",
        type=Path,
        default=Path("outputs/reports"),
        help="Directory where validation reports will be saved.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(DEFAULT_SPLITS),
        help="Dataset splits to validate. Default: train val test.",
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
        return {}

    images: dict[str, Path] = {}
    for image_path in image_dir.iterdir():
        if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
            images[image_path.stem] = image_path
    return images


def find_labels(label_dir: Path) -> dict[str, Path]:
    if not label_dir.exists():
        return {}

    labels: dict[str, Path] = {}
    for label_path in label_dir.iterdir():
        if label_path.is_file() and label_path.suffix.lower() == ".txt":
            labels[label_path.stem] = label_path
    return labels


def add_missing_folder_issue(
    issues: list[ValidationIssue],
    split: str,
    folder: Path,
    expected_kind: str,
) -> None:
    issues.append(
        ValidationIssue(
            severity="error",
            split=split,
            file=str(folder),
            line=None,
            issue_type="missing_folder",
            message=f"Missing {expected_kind} folder: {folder}",
        )
    )


def validate_pairing(
    split: str,
    images: dict[str, Path],
    labels: dict[str, Path],
    issues: list[ValidationIssue],
) -> None:
    for stem, image_path in sorted(images.items()):
        if stem not in labels:
            issues.append(
                ValidationIssue(
                    severity="error",
                    split=split,
                    file=str(image_path),
                    line=None,
                    issue_type="missing_label_file",
                    message="Image file has no matching label file.",
                )
            )

    for stem, label_path in sorted(labels.items()):
        if stem not in images:
            issues.append(
                ValidationIssue(
                    severity="error",
                    split=split,
                    file=str(label_path),
                    line=None,
                    issue_type="missing_image_file",
                    message="Label file has no matching image file.",
                )
            )


def parse_class_id(raw_value: str) -> int | None:
    try:
        class_id = int(raw_value)
    except ValueError:
        return None

    if str(class_id) != raw_value:
        return None
    return class_id


def parse_float(raw_value: str) -> float | None:
    try:
        return float(raw_value)
    except ValueError:
        return None


def validate_label_line(
    line_text: str,
    line_number: int,
    label_path: Path,
    split: str,
    class_count: int,
    issues: list[ValidationIssue],
) -> int | None:
    parts = line_text.split()
    if len(parts) != 5:
        issues.append(
            ValidationIssue(
                severity="error",
                split=split,
                file=str(label_path),
                line=line_number,
                issue_type="invalid_yolo_format",
                message="YOLO label line must have exactly 5 values: class_id x_center y_center width height.",
            )
        )
        return None

    class_id = parse_class_id(parts[0])
    if class_id is None:
        issues.append(
            ValidationIssue(
                severity="error",
                split=split,
                file=str(label_path),
                line=line_number,
                issue_type="invalid_class_id",
                message=f"Class ID must be an integer, got: {parts[0]}",
            )
        )
    elif class_id < 0 or class_id >= class_count:
        issues.append(
            ValidationIssue(
                severity="error",
                split=split,
                file=str(label_path),
                line=line_number,
                issue_type="class_id_out_of_range",
                message=f"Class ID {class_id} is outside the valid range 0 to {class_count - 1}.",
            )
        )

    coordinate_names = ("x_center", "y_center", "width", "height")
    coordinates: list[float | None] = []
    for name, raw_value in zip(coordinate_names, parts[1:]):
        value = parse_float(raw_value)
        coordinates.append(value)
        if value is None:
            issues.append(
                ValidationIssue(
                    severity="error",
                    split=split,
                    file=str(label_path),
                    line=line_number,
                    issue_type="non_numeric_coordinate",
                    message=f"{name} must be numeric, got: {raw_value}",
                )
            )
            continue

        if value < 0 or value > 1:
            issues.append(
                ValidationIssue(
                    severity="error",
                    split=split,
                    file=str(label_path),
                    line=line_number,
                    issue_type="coordinate_out_of_range",
                    message=f"{name} must be between 0 and 1, got: {value}",
                )
            )

    width = coordinates[2]
    height = coordinates[3]
    if width is not None and width <= 0:
        issues.append(
            ValidationIssue(
                severity="error",
                split=split,
                file=str(label_path),
                line=line_number,
                issue_type="invalid_box_width",
                message=f"width must be greater than 0, got: {width}",
            )
        )
    if height is not None and height <= 0:
        issues.append(
            ValidationIssue(
                severity="error",
                split=split,
                file=str(label_path),
                line=line_number,
                issue_type="invalid_box_height",
                message=f"height must be greater than 0, got: {height}",
            )
        )

    if class_id is None or class_id < 0 or class_id >= class_count:
        return None
    return class_id


def validate_label_file(
    label_path: Path,
    split: str,
    class_count: int,
    issues: list[ValidationIssue],
) -> list[int]:
    class_ids: list[int] = []
    try:
        lines = label_path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        issues.append(
            ValidationIssue(
                severity="error",
                split=split,
                file=str(label_path),
                line=None,
                issue_type="label_file_read_error",
                message="Could not read label file as UTF-8 text.",
            )
        )
        return class_ids

    non_empty_lines = [line for line in lines if line.strip()]
    if not non_empty_lines:
        issues.append(
            ValidationIssue(
                severity="warning",
                split=split,
                file=str(label_path),
                line=None,
                issue_type="empty_label_file",
                message="Label file is empty. This is valid only for images with no objects.",
            )
        )
        return class_ids

    for line_number, line_text in enumerate(lines, start=1):
        stripped_line = line_text.strip()
        if not stripped_line:
            continue
        class_id = validate_label_line(
            stripped_line,
            line_number,
            label_path,
            split,
            class_count,
            issues,
        )
        if class_id is not None:
            class_ids.append(class_id)
    return class_ids


def validate_dataset(
    dataset_root: Path,
    classes: list[str],
    splits: Iterable[str],
) -> tuple[dict[str, SplitStats], dict[str, int], list[ValidationIssue]]:
    issues: list[ValidationIssue] = []
    split_stats: dict[str, SplitStats] = {}
    object_counts = {class_name: 0 for class_name in classes}

    for split in splits:
        stats = SplitStats()
        split_stats[split] = stats

        image_dir = dataset_root / "images" / split
        label_dir = dataset_root / "labels" / split

        if not image_dir.exists():
            add_missing_folder_issue(issues, split, image_dir, "image")
        if not label_dir.exists():
            add_missing_folder_issue(issues, split, label_dir, "label")

        images = find_images(image_dir)
        labels = find_labels(label_dir)

        stats.images = len(images)
        stats.label_files = len(labels)

        validate_pairing(split, images, labels, issues)

        for label_path in sorted(labels.values()):
            class_ids = validate_label_file(label_path, split, len(classes), issues)
            if not class_ids:
                try:
                    if not label_path.read_text(encoding="utf-8").strip():
                        stats.empty_label_files += 1
                except UnicodeDecodeError:
                    pass
            stats.objects += len(class_ids)
            for class_id in class_ids:
                object_counts[classes[class_id]] += 1

    return split_stats, object_counts, issues


def write_json_report(
    output_path: Path,
    dataset_root: Path,
    classes_path: Path,
    classes: list[str],
    split_stats: dict[str, SplitStats],
    object_counts: dict[str, int],
    issues: list[ValidationIssue],
) -> None:
    report = {
        "dataset": str(dataset_root),
        "classes_file": str(classes_path),
        "class_count": len(classes),
        "classes": classes,
        "summary": {
            "total_images": sum(stats.images for stats in split_stats.values()),
            "total_label_files": sum(stats.label_files for stats in split_stats.values()),
            "total_objects": sum(stats.objects for stats in split_stats.values()),
            "total_issues": len(issues),
            "errors": sum(issue.severity == "error" for issue in issues),
            "warnings": sum(issue.severity == "warning" for issue in issues),
        },
        "splits": {
            split: asdict(stats) for split, stats in split_stats.items()
        },
        "objects_per_class": object_counts,
        "issues": [asdict(issue) for issue in issues],
    }

    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def write_csv_issues(output_path: Path, issues: list[ValidationIssue]) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        fieldnames = ["severity", "split", "file", "line", "issue_type", "message"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for issue in issues:
            writer.writerow(asdict(issue))


def print_summary(
    split_stats: dict[str, SplitStats],
    object_counts: dict[str, int],
    issues: list[ValidationIssue],
    json_report_path: Path,
    csv_report_path: Path,
) -> None:
    total_images = sum(stats.images for stats in split_stats.values())
    total_label_files = sum(stats.label_files for stats in split_stats.values())
    total_objects = sum(stats.objects for stats in split_stats.values())
    error_count = sum(issue.severity == "error" for issue in issues)
    warning_count = sum(issue.severity == "warning" for issue in issues)

    print("\nYOLO Dataset Validation Summary")
    print("=" * 32)
    print(f"Total images:      {total_images}")
    print(f"Total label files: {total_label_files}")
    print(f"Total objects:     {total_objects}")
    print(f"Errors:            {error_count}")
    print(f"Warnings:          {warning_count}")

    print("\nSplit Summary")
    print("-" * 32)
    for split, stats in split_stats.items():
        print(
            f"{split:>5}: images={stats.images}, "
            f"label_files={stats.label_files}, "
            f"objects={stats.objects}, "
            f"empty_labels={stats.empty_label_files}"
        )

    print("\nObjects Per Class")
    print("-" * 32)
    for class_name, count in object_counts.items():
        print(f"{class_name:>16}: {count}")

    if issues:
        print("\nIssues")
        print("-" * 32)
        for issue in issues[:20]:
            line_text = f":{issue.line}" if issue.line is not None else ""
            print(
                f"[{issue.severity.upper()}] {issue.split} "
                f"{issue.issue_type} {issue.file}{line_text} - {issue.message}"
            )
        if len(issues) > 20:
            print(f"... {len(issues) - 20} more issues saved in the reports.")
    else:
        print("\nNo validation issues found.")

    print("\nReports saved:")
    print(f"- {json_report_path}")
    print(f"- {csv_report_path}")


def main() -> int:
    args = parse_args()

    try:
        classes = load_classes(args.classes)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}")
        return 1

    dataset_root = args.dataset
    if not dataset_root.exists():
        print(f"Error: Dataset folder not found: {dataset_root}")
        return 1

    split_stats, object_counts, issues = validate_dataset(
        dataset_root=dataset_root,
        classes=classes,
        splits=args.splits,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_report_path = args.output_dir / "validation_report.json"
    csv_report_path = args.output_dir / "validation_issues.csv"

    write_json_report(
        output_path=json_report_path,
        dataset_root=dataset_root,
        classes_path=args.classes,
        classes=classes,
        split_stats=split_stats,
        object_counts=object_counts,
        issues=issues,
    )
    write_csv_issues(csv_report_path, issues)

    print_summary(
        split_stats=split_stats,
        object_counts=object_counts,
        issues=issues,
        json_report_path=json_report_path,
        csv_report_path=csv_report_path,
    )

    return 1 if any(issue.severity == "error" for issue in issues) else 0


if __name__ == "__main__":
    raise SystemExit(main())

