"""Convert Supervisely-style rectangle annotations to YOLO labels.

Expected usage:
    python scripts/convert_supervisely_to_yolo.py --images data/raw/supermarket_shelves/images --annotations data/raw/supermarket_shelves/annotations --output-images data/processed/images --output-labels data/processed/labels
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
TARGET_GEOMETRY_TYPE = "rectangle"
TARGET_CLASS_TITLE = "product"
PRODUCT_CLASS_ID = 0
REPORT_DIR = Path("outputs/reports")


@dataclass(frozen=True)
class YoloBox:
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float


@dataclass
class FileReport:
    image_name: str
    annotation_name: str
    objects_in_json: int
    converted_boxes: int
    skipped_boxes: int
    output_image_path: str
    output_label_path: str


@dataclass
class ConversionIssue:
    image_name: str
    annotation_name: str
    issue_type: str
    message: str


@dataclass
class ConversionStats:
    total_images: int = 0
    converted_images: int = 0
    total_boxes: int = 0
    skipped_boxes: int = 0
    missing_annotations: int = 0
    orphan_annotations: int = 0


@dataclass(frozen=True)
class AnnotationResult:
    object_count: int
    boxes: list[YoloBox]
    skipped_boxes: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Supervisely rectangle annotations to YOLO format."
    )
    parser.add_argument("--images", type=Path, required=True, help="Input image folder.")
    parser.add_argument(
        "--annotations",
        type=Path,
        required=True,
        help="Input annotation folder containing .json files.",
    )
    parser.add_argument(
        "--output-images",
        type=Path,
        required=True,
        help="Output folder for copied images.",
    )
    parser.add_argument(
        "--output-labels",
        type=Path,
        required=True,
        help="Output folder for YOLO .txt label files.",
    )
    return parser.parse_args()


def validate_input_folder(path: Path, name: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{name} folder not found: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"{name} path is not a folder: {path}")


def find_images(images_dir: Path) -> dict[str, Path]:
    images: dict[str, Path] = {}
    for image_path in sorted(images_dir.iterdir()):
        if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
            images.setdefault(image_path.stem, image_path)
    return images


def annotation_image_stem(annotation_path: Path) -> str:
    """Map both 001.json and 001.jpg.json to image stem 001."""
    return Path(annotation_path.stem).stem


def find_annotations(annotations_dir: Path) -> dict[str, Path]:
    annotations: dict[str, Path] = {}
    for annotation_path in sorted(annotations_dir.iterdir()):
        if annotation_path.is_file() and annotation_path.suffix.lower() == ".json":
            stem = annotation_image_stem(annotation_path)
            annotations.setdefault(stem, annotation_path)
    return annotations


def load_annotation(annotation_path: Path, issues: list[ConversionIssue]) -> dict[str, Any] | None:
    try:
        data = json.loads(annotation_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        issues.append(
            ConversionIssue(
                image_name="",
                annotation_name=annotation_path.name,
                issue_type="invalid_json",
                message="Annotation file is not valid JSON.",
            )
        )
        return None
    except UnicodeDecodeError:
        issues.append(
            ConversionIssue(
                image_name="",
                annotation_name=annotation_path.name,
                issue_type="invalid_encoding",
                message="Annotation file is not valid UTF-8 text.",
            )
        )
        return None

    if not isinstance(data, dict):
        issues.append(
            ConversionIssue(
                image_name="",
                annotation_name=annotation_path.name,
                issue_type="invalid_json_root",
                message="Annotation root must be a JSON object.",
            )
        )
        return None
    return data


def get_image_size(
    annotation: dict[str, Any],
    annotation_path: Path,
    issues: list[ConversionIssue],
) -> tuple[float, float] | None:
    size = annotation.get("size")
    if not isinstance(size, dict):
        issues.append(
            ConversionIssue("", annotation_path.name, "missing_size", "Missing size object.")
        )
        return None

    width = size.get("width")
    height = size.get("height")
    if not isinstance(width, (int, float)) or not isinstance(height, (int, float)):
        issues.append(
            ConversionIssue(
                "",
                annotation_path.name,
                "invalid_size",
                "size.width and size.height must be numeric.",
            )
        )
        return None
    if width <= 0 or height <= 0:
        issues.append(
            ConversionIssue(
                "",
                annotation_path.name,
                "invalid_size",
                "size.width and size.height must be greater than 0.",
            )
        )
        return None

    return float(width), float(height)


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(value, maximum))


def is_product_rectangle(obj: dict[str, Any]) -> bool:
    geometry_type = str(obj.get("geometryType", "")).strip().lower()
    class_title = str(obj.get("classTitle", "")).strip().lower()
    return geometry_type == TARGET_GEOMETRY_TYPE and class_title == TARGET_CLASS_TITLE


def is_xy_point(value: Any) -> bool:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return False
    return isinstance(value[0], (int, float)) and isinstance(value[1], (int, float))


def object_to_yolo_box(
    obj: dict[str, Any],
    image_width: float,
    image_height: float,
) -> YoloBox | None:
    points = obj.get("points")
    if not isinstance(points, dict):
        return None

    exterior = points.get("exterior")
    if not isinstance(exterior, list) or len(exterior) < 2:
        return None

    first_point = exterior[0]
    second_point = exterior[1]
    if not is_xy_point(first_point) or not is_xy_point(second_point):
        return None

    x1 = clamp(float(first_point[0]), 0.0, image_width)
    y1 = clamp(float(first_point[1]), 0.0, image_height)
    x2 = clamp(float(second_point[0]), 0.0, image_width)
    y2 = clamp(float(second_point[1]), 0.0, image_height)

    left = min(x1, x2)
    right = max(x1, x2)
    top = min(y1, y2)
    bottom = max(y1, y2)

    box_width = right - left
    box_height = bottom - top
    if box_width <= 0 or box_height <= 0:
        return None

    return YoloBox(
        class_id=PRODUCT_CLASS_ID,
        x_center=(left + box_width / 2) / image_width,
        y_center=(top + box_height / 2) / image_height,
        width=box_width / image_width,
        height=box_height / image_height,
    )


def convert_annotation(
    annotation_path: Path,
    issues: list[ConversionIssue],
) -> AnnotationResult:
    annotation = load_annotation(annotation_path, issues)
    if annotation is None:
        return AnnotationResult(0, [], 0)

    image_size = get_image_size(annotation, annotation_path, issues)
    if image_size is None:
        return AnnotationResult(0, [], 0)

    objects = annotation.get("objects", [])
    if not isinstance(objects, list):
        issues.append(
            ConversionIssue("", annotation_path.name, "invalid_objects", "objects must be a list.")
        )
        return AnnotationResult(0, [], 0)

    image_width, image_height = image_size
    boxes: list[YoloBox] = []
    skipped_boxes = 0

    for obj in objects:
        if not isinstance(obj, dict) or not is_product_rectangle(obj):
            continue

        box = object_to_yolo_box(obj, image_width, image_height)
        if box is None:
            skipped_boxes += 1
            issues.append(
                ConversionIssue(
                    "",
                    annotation_path.name,
                    "invalid_box",
                    "Product rectangle had invalid or zero-area coordinates.",
                )
            )
            continue
        boxes.append(box)

    return AnnotationResult(len(objects), boxes, skipped_boxes)


def format_yolo_box(box: YoloBox) -> str:
    return (
        f"{box.class_id} {box.x_center:.6f} {box.y_center:.6f} "
        f"{box.width:.6f} {box.height:.6f}"
    )


def write_label_file(label_path: Path, boxes: list[YoloBox]) -> None:
    lines = [format_yolo_box(box) for box in boxes]
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def save_reports(
    stats: ConversionStats,
    file_reports: list[FileReport],
    issues: list[ConversionIssue],
) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    report = {
        "summary": asdict(stats),
        "files": [asdict(item) for item in file_reports],
        "issues": [asdict(issue) for issue in issues],
    }
    (REPORT_DIR / "conversion_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    with (REPORT_DIR / "conversion_issues.csv").open("w", newline="", encoding="utf-8") as csv_file:
        fieldnames = ["image_name", "annotation_name", "issue_type", "message"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for issue in issues:
            writer.writerow(asdict(issue))


def convert_dataset(
    images_dir: Path,
    annotations_dir: Path,
    output_images_dir: Path,
    output_labels_dir: Path,
) -> tuple[ConversionStats, list[FileReport], list[ConversionIssue]]:
    validate_input_folder(images_dir, "Images")
    validate_input_folder(annotations_dir, "Annotations")

    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_labels_dir.mkdir(parents=True, exist_ok=True)

    images = find_images(images_dir)
    annotations = find_annotations(annotations_dir)
    stats = ConversionStats(total_images=len(images))
    file_reports: list[FileReport] = []
    issues: list[ConversionIssue] = []

    for stem, image_path in sorted(images.items()):
        annotation_path = annotations.get(stem)
        output_image_path = output_images_dir / image_path.name
        output_label_path = output_labels_dir / f"{stem}.txt"

        shutil.copy2(image_path, output_image_path)

        if annotation_path is None:
            stats.missing_annotations += 1
            write_label_file(output_label_path, [])
            issues.append(
                ConversionIssue(
                    image_name=image_path.name,
                    annotation_name="",
                    issue_type="missing_annotation",
                    message="Image has no matching JSON annotation.",
                )
            )
            result = AnnotationResult(0, [], 0)
        else:
            result = convert_annotation(annotation_path, issues)
            for issue in issues:
                if not issue.image_name and issue.annotation_name == annotation_path.name:
                    issue.image_name = image_path.name
            write_label_file(output_label_path, result.boxes)
            stats.converted_images += 1
            stats.total_boxes += len(result.boxes)
            stats.skipped_boxes += result.skipped_boxes

        file_reports.append(
            FileReport(
                image_name=image_path.name,
                annotation_name=annotation_path.name if annotation_path else "",
                objects_in_json=result.object_count,
                converted_boxes=len(result.boxes),
                skipped_boxes=result.skipped_boxes,
                output_image_path=str(output_image_path),
                output_label_path=str(output_label_path),
            )
        )
        print(
            f"{image_path.name}: objects_in_json={result.object_count}, "
            f"converted_boxes={len(result.boxes)}, skipped_boxes={result.skipped_boxes}"
        )

    for stem, annotation_path in sorted(annotations.items()):
        if stem not in images:
            stats.orphan_annotations += 1
            issues.append(
                ConversionIssue(
                    image_name="",
                    annotation_name=annotation_path.name,
                    issue_type="orphan_annotation",
                    message="JSON annotation has no matching image.",
                )
            )
            print(f"Warning: JSON annotation has no matching image: {annotation_path}")

    save_reports(stats, file_reports, issues)
    return stats, file_reports, issues


def print_summary(stats: ConversionStats, output_images_dir: Path, output_labels_dir: Path) -> None:
    print("\nSupervisely to YOLO Conversion Summary")
    print("=" * 40)
    print(f"Total images:         {stats.total_images}")
    print(f"Converted images:     {stats.converted_images}")
    print(f"Total boxes:          {stats.total_boxes}")
    print(f"Skipped boxes:        {stats.skipped_boxes}")
    print(f"Missing annotations:  {stats.missing_annotations}")
    print(f"Orphan annotations:   {stats.orphan_annotations}")
    print(f"Output images folder: {output_images_dir}")
    print(f"Output labels folder: {output_labels_dir}")
    print(f"Conversion report:    {REPORT_DIR / 'conversion_report.json'}")
    print(f"Issues CSV:           {REPORT_DIR / 'conversion_issues.csv'}")


def main() -> int:
    args = parse_args()

    try:
        stats, _, _ = convert_dataset(
            images_dir=args.images,
            annotations_dir=args.annotations,
            output_images_dir=args.output_images,
            output_labels_dir=args.output_labels,
        )
    except (FileNotFoundError, NotADirectoryError) as exc:
        print(f"Error: {exc}")
        return 1

    print_summary(stats, args.output_images, args.output_labels)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

