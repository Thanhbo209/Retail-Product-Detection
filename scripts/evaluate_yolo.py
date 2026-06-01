"""Evaluate a trained YOLOv8 object detection model.

Expected usage:
    python scripts/evaluate_yolo.py --model runs/experiments/product_yolov8n/weights/best.pt --data configs/product_detection.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REPORT_PATH = Path("outputs/reports/evaluation_report.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained YOLOv8 model.")
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to trained model weights, for example runs/experiments/product_yolov8n/weights/best.pt.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("configs/product_detection.yaml"),
        help="YOLO dataset YAML path. Default: configs/product_detection.yaml.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Validation image size. Default: 640.",
    )
    parser.add_argument(
        "--split",
        default="val",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate. Default: val.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.model.exists():
        raise FileNotFoundError(f"Model file not found: {args.model}")
    if not args.data.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {args.data}")
    if args.imgsz <= 0:
        raise ValueError("--imgsz must be greater than 0.")


def metric_value(metrics: Any, attribute: str) -> float | None:
    box_metrics = getattr(metrics, "box", None)
    value = getattr(box_metrics, attribute, None)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "ultralytics is not installed. Run: pip install -r requirements.txt"
        ) from exc

    model = YOLO(str(args.model))
    metrics = model.val(
        data=str(args.data),
        imgsz=args.imgsz,
        split=args.split,
    )

    return {
        "model": str(args.model),
        "data": str(args.data),
        "imgsz": args.imgsz,
        "split": args.split,
        "metrics": {
            "precision": metric_value(metrics, "mp"),
            "recall": metric_value(metrics, "mr"),
            "mAP50": metric_value(metrics, "map50"),
            "mAP50-95": metric_value(metrics, "map"),
        },
    }


def save_report(report: dict[str, Any], report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def print_report(report: dict[str, Any], report_path: Path) -> None:
    metrics = report["metrics"]

    print("\nYOLOv8 Evaluation Summary")
    print("=" * 32)
    print(f"Model:      {report['model']}")
    print(f"Data YAML:  {report['data']}")
    print(f"Split:      {report['split']}")
    print(f"Image size: {report['imgsz']}")

    print("\nMetrics")
    print("-" * 32)
    print(f"Precision:  {format_metric(metrics['precision'])}")
    print(f"Recall:     {format_metric(metrics['recall'])}")
    print(f"mAP50:      {format_metric(metrics['mAP50'])}")
    print(f"mAP50-95:   {format_metric(metrics['mAP50-95'])}")
    print(f"\nReport saved to: {report_path}")


def format_metric(value: float | None) -> str:
    if value is None:
        return "not available"
    return f"{value:.4f}"


def main() -> int:
    args = parse_args()

    try:
        validate_args(args)
        report = evaluate(args)
        save_report(report, REPORT_PATH)
    except (FileNotFoundError, ValueError, ImportError) as exc:
        print(f"Error: {exc}")
        return 1

    print_report(report, REPORT_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

