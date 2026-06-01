"""Convert model prediction JSON to YOLO label files for CVAT pre-label import.

Expected usage:
    python scripts/convert_predictions_json_to_yolo.py --predictions outputs/inference_046/predictions.json --images data/processed/images --output-labels cvat_import_046/obj_train_data --min-conf 0.25
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from PIL import Image


REPORT_PATH = Path("outputs/reports/prediction_to_yolo_report.json")


@dataclass(frozen=True)
class YoloBox:
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float


@dataclass
class ImageExportStats:
    image_name: str
    exported_boxes: int
    skipped_boxes: int
    output_label_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert prediction JSON detections to YOLO .txt label files."
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="Prediction JSON file from inference.",
    )
    parser.add_argument(
        "--images",
        type=Path,
        required=True,
        help="Folder containing the source images referenced in the prediction JSON.",
    )
    parser.add_argument(
        "--output-labels",
        type=Path,
        required=True,
        help="Output folder for YOLO .txt label files.",
    )
    parser.add_argument(
        "--min-conf",
        type=float,
        default=0.25,
        help="Minimum confidence threshold. Default: 0.25.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.predictions.exists():
        raise FileNotFoundError(f"Prediction JSON not found: {args.predictions}")
    if not args.images.exists():
        raise FileNotFoundError(f"Images folder not found: {args.images}")
    if not args.images.is_dir():
        raise NotADirectoryError(f"Images path is not a folder: {args.images}")
    if args.min_conf < 0 or args.min_conf > 1:
        raise ValueError("--min-conf must be between 0 and 1.")


def load_prediction_records(predictions_path: Path) -> list[dict[str, Any]]:
    try:
        data = json.loads(predictions_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid prediction JSON: {predictions_path}") from exc

    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        records: list[dict[str, Any]] = []
        for index, item in enumerate(data):
            if not isinstance(item, dict):
                print(f"Warning: skipping non-object prediction item at index {index}")
                continue
            records.append(item)
        return records

    raise ValueError("Prediction JSON must be either an object or a list of objects.")


def read_image_size(image_path: Path) -> tuple[int, int]:
    with Image.open(image_path) as image:
        return image.size


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(value, maximum))


def bbox_xyxy_to_yolo(
    bbox_xyxy: Any,
    class_id: int,
    image_width: int,
    image_height: int,
) -> YoloBox | None:
    if not isinstance(bbox_xyxy, list) or len(bbox_xyxy) != 4:
        return None

    try:
        x1, y1, x2, y2 = [float(value) for value in bbox_xyxy]
    except (TypeError, ValueError):
        return None

    x1 = clamp(x1, 0.0, float(image_width))
    y1 = clamp(y1, 0.0, float(image_height))
    x2 = clamp(x2, 0.0, float(image_width))
    y2 = clamp(y2, 0.0, float(image_height))

    left = min(x1, x2)
    right = max(x1, x2)
    top = min(y1, y2)
    bottom = max(y1, y2)

    box_width = right - left
    box_height = bottom - top
    if box_width <= 0 or box_height <= 0:
        return None

    return YoloBox(
        class_id=class_id,
        x_center=(left + box_width / 2) / image_width,
        y_center=(top + box_height / 2) / image_height,
        width=box_width / image_width,
        height=box_height / image_height,
    )


def detection_to_yolo_box(
    detection: dict[str, Any],
    image_width: int,
    image_height: int,
    min_conf: float,
) -> YoloBox | None:
    try:
        confidence = float(detection.get("confidence", 0.0))
    except (TypeError, ValueError):
        return None

    if confidence < min_conf:
        return None

    try:
        class_id = int(detection["class_id"])
    except (KeyError, TypeError, ValueError):
        return None

    return bbox_xyxy_to_yolo(
        bbox_xyxy=detection.get("bbox_xyxy"),
        class_id=class_id,
        image_width=image_width,
        image_height=image_height,
    )


def format_yolo_box(box: YoloBox) -> str:
    return (
        f"{box.class_id} "
        f"{box.x_center:.6f} "
        f"{box.y_center:.6f} "
        f"{box.width:.6f} "
        f"{box.height:.6f}"
    )


def write_label_file(output_label_path: Path, boxes: list[YoloBox]) -> None:
    lines = [format_yolo_box(box) for box in boxes]
    output_label_path.parent.mkdir(parents=True, exist_ok=True)
    output_label_path.write_text(
        "\n".join(lines) + ("\n" if lines else ""),
        encoding="utf-8",
    )


def convert_record(
    record: dict[str, Any],
    images_dir: Path,
    output_labels_dir: Path,
    min_conf: float,
) -> ImageExportStats | None:
    image_name = record.get("image")
    if not isinstance(image_name, str) or not image_name:
        print("Warning: skipping prediction record without a valid image name")
        return None

    image_path = images_dir / image_name
    output_label_path = output_labels_dir / f"{Path(image_name).stem}.txt"

    if not image_path.exists():
        print(f"Warning: source image not found for prediction record: {image_path}")
        return ImageExportStats(
            image_name=image_name,
            exported_boxes=0,
            skipped_boxes=len(record.get("detections", []) or []),
            output_label_path=output_label_path,
        )

    image_width, image_height = read_image_size(image_path)
    detections = record.get("detections", [])
    if not isinstance(detections, list):
        print(f"Warning: detections must be a list for image: {image_name}")
        detections = []

    boxes: list[YoloBox] = []
    skipped_boxes = 0
    for detection in detections:
        if not isinstance(detection, dict):
            skipped_boxes += 1
            continue

        box = detection_to_yolo_box(
            detection=detection,
            image_width=image_width,
            image_height=image_height,
            min_conf=min_conf,
        )
        if box is None:
            skipped_boxes += 1
            continue
        boxes.append(box)

    write_label_file(output_label_path, boxes)
    return ImageExportStats(
        image_name=image_name,
        exported_boxes=len(boxes),
        skipped_boxes=skipped_boxes,
        output_label_path=output_label_path,
    )


def convert_predictions(
    predictions_path: Path,
    images_dir: Path,
    output_labels_dir: Path,
    min_conf: float,
) -> list[ImageExportStats]:
    output_labels_dir.mkdir(parents=True, exist_ok=True)
    records = load_prediction_records(predictions_path)
    stats: list[ImageExportStats] = []

    for record in records:
        image_stats = convert_record(
            record=record,
            images_dir=images_dir,
            output_labels_dir=output_labels_dir,
            min_conf=min_conf,
        )
        if image_stats is not None:
            stats.append(image_stats)

    return stats


def print_summary(stats: list[ImageExportStats]) -> None:
    print("\nPrediction JSON to YOLO Export Summary")
    print("=" * 40)
    for item in stats:
        print(
            f"{item.image_name}: exported_boxes={item.exported_boxes}, "
            f"skipped_boxes={item.skipped_boxes}, "
            f"output_label_path={item.output_label_path}"
        )

    total_exported = sum(item.exported_boxes for item in stats)
    total_skipped = sum(item.skipped_boxes for item in stats)
    print("-" * 40)
    print(f"Images processed: {len(stats)}")
    print(f"Total exported boxes: {total_exported}")
    print(f"Total skipped boxes:  {total_skipped}")
    print(f"Report saved to:      {REPORT_PATH}")


def save_report(stats: list[ImageExportStats], report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "summary": {
            "images_processed": len(stats),
            "total_exported_boxes": sum(item.exported_boxes for item in stats),
            "total_skipped_boxes": sum(item.skipped_boxes for item in stats),
        },
        "files": [
            {
                **asdict(item),
                "output_label_path": str(item.output_label_path),
            }
            for item in stats
        ],
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()

    try:
        validate_args(args)
        stats = convert_predictions(
            predictions_path=args.predictions,
            images_dir=args.images,
            output_labels_dir=args.output_labels,
            min_conf=args.min_conf,
        )
        save_report(stats, REPORT_PATH)
    except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
        print(f"Error: {exc}")
        return 1

    print_summary(stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
