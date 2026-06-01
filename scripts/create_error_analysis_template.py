"""Create an error analysis CSV template for object detection review.

Expected usage:
    python scripts/create_error_analysis_template.py

Optional prediction prefill:
    python scripts/create_error_analysis_template.py --predictions outputs/inference/predictions.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_PATH = Path("outputs/reports/error_analysis_template.csv")
CSV_COLUMNS = [
    "image_filename",
    "error_type",
    "predicted_class",
    "true_class",
    "confidence",
    "description",
    "possible_cause",
    "suggested_fix",
    "reviewed_by",
    "review_date",
]
ERROR_TYPES = [
    "missed_detection",
    "wrong_class",
    "bad_box",
    "duplicate_detection",
    "false_positive",
    "low_confidence",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an error analysis CSV template."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"CSV output path. Default: {DEFAULT_OUTPUT_PATH}.",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        help="Optional prediction JSON from scripts/run_inference.py.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output CSV if it already exists.",
    )
    return parser.parse_args()


def validate_output_path(output_path: Path, overwrite: bool) -> None:
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}. "
            "Use --overwrite to replace it."
        )


def load_prediction_json(predictions_path: Path) -> list[dict[str, Any]]:
    if not predictions_path.exists():
        raise FileNotFoundError(f"Prediction JSON not found: {predictions_path}")

    try:
        data = json.loads(predictions_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid prediction JSON: {predictions_path}") from exc

    if not isinstance(data, list):
        raise ValueError("Prediction JSON must contain a list of image records.")

    return data


def blank_row(error_type: str = "") -> dict[str, str]:
    row = {column: "" for column in CSV_COLUMNS}
    row["error_type"] = error_type
    return row


def rows_from_predictions(prediction_records: list[dict[str, Any]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    for record in prediction_records:
        image_filename = str(record.get("image", ""))
        detections = record.get("detections", [])

        if not isinstance(detections, list) or not detections:
            row = blank_row()
            row["image_filename"] = image_filename
            row["description"] = "No detections found. Review for possible missed detections."
            rows.append(row)
            continue

        for detection in detections:
            if not isinstance(detection, dict):
                continue

            row = blank_row()
            row["image_filename"] = image_filename
            row["predicted_class"] = str(detection.get("class_name", ""))
            confidence = detection.get("confidence", "")
            row["confidence"] = "" if confidence == "" else str(confidence)
            rows.append(row)

    return rows


def default_template_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for error_type in ERROR_TYPES:
        row = blank_row(error_type)
        row["description"] = f"Example row for {error_type}. Replace this with a reviewed mistake."
        rows.append(row)
    return rows


def write_csv(output_path: Path, rows: list[dict[str, str]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def create_template(
    output_path: Path,
    predictions_path: Path | None,
    overwrite: bool,
) -> int:
    validate_output_path(output_path, overwrite)

    if predictions_path is not None:
        prediction_records = load_prediction_json(predictions_path)
        rows = rows_from_predictions(prediction_records)
        if not rows:
            rows = default_template_rows()
    else:
        rows = default_template_rows()

    write_csv(output_path, rows)
    return len(rows)


def main() -> int:
    args = parse_args()

    try:
        row_count = create_template(
            output_path=args.output,
            predictions_path=args.predictions,
            overwrite=args.overwrite,
        )
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}")
        return 1

    print(f"Error analysis template saved to: {args.output}")
    print(f"Rows written: {row_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

