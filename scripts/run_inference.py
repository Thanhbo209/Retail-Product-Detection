"""Run YOLO inference on one image or a folder of images.

Expected usage:
    python scripts/run_inference.py --model runs/experiments/product_yolov8n/weights/best.pt --source app/sample_images --conf 0.25 --output outputs/inference
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
PREDICTIONS_FILENAME = "predictions.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a trained YOLO model on an image or folder of images."
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to trained model weights, for example runs/experiments/product_yolov8n/weights/best.pt.",
    )
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Image file or folder of images to predict.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold. Default: 0.25.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/inference"),
        help="Output folder for annotated images and JSON results.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.model.exists():
        raise FileNotFoundError(f"Model file not found: {args.model}")
    if not args.source.exists():
        raise FileNotFoundError(f"Source path not found: {args.source}")
    if args.conf < 0 or args.conf > 1:
        raise ValueError("--conf must be between 0 and 1.")


def collect_images(source: Path) -> list[Path]:
    if source.is_file():
        if source.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(
                f"Unsupported image extension: {source.suffix}. "
                f"Supported extensions: {', '.join(sorted(IMAGE_EXTENSIONS))}"
            )
        return [source]

    if not source.is_dir():
        raise ValueError(f"Source is not a file or folder: {source}")

    images = [
        path
        for path in sorted(source.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if not images:
        raise ValueError(f"No supported image files found in: {source}")
    return images


def class_name_from_model(model_names: Any, class_id: int) -> str:
    if isinstance(model_names, dict):
        return str(model_names.get(class_id, f"class_{class_id}"))
    if isinstance(model_names, list) and 0 <= class_id < len(model_names):
        return str(model_names[class_id])
    return f"class_{class_id}"


def load_model(model_path: Path) -> Any:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "ultralytics is not installed. Run: pip install -r requirements.txt"
        ) from exc

    return YOLO(str(model_path))


def result_to_detections(result: Any, model_names: Any) -> list[dict[str, Any]]:
    detections: list[dict[str, Any]] = []
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return detections

    for box in boxes:
        class_id = int(box.cls[0].item())
        confidence = float(box.conf[0].item())
        xyxy = box.xyxy[0].tolist()
        bbox_xyxy = [int(round(value)) for value in xyxy]

        detections.append(
            {
                "class_id": class_id,
                "class_name": class_name_from_model(model_names, class_id),
                "confidence": round(confidence, 4),
                "bbox_xyxy": bbox_xyxy,
            }
        )

    return detections


def count_detections(detections: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(detection["class_name"] for detection in detections)
    return dict(sorted(counts.items()))


def save_annotated_image(result: Any, output_path: Path) -> bool:
    try:
        import cv2
    except ImportError:
        print("Error: opencv-python is not installed. Run: pip install -r requirements.txt")
        return False

    annotated_image = result.plot()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(output_path), annotated_image))


def run_inference(
    model: Any,
    image_paths: list[Path],
    conf: float,
    output_dir: Path,
) -> list[dict[str, Any]]:
    annotated_dir = output_dir / "annotated"
    prediction_records: list[dict[str, Any]] = []
    model_names = getattr(model, "names", {})

    for image_path in image_paths:
        results = model.predict(
            source=str(image_path),
            conf=conf,
            verbose=False,
        )
        if not results:
            print(f"Warning: no result returned for {image_path}")
            prediction_records.append(
                {"image": image_path.name, "detections": [], "counts": {}}
            )
            continue

        result = results[0]
        detections = result_to_detections(result, model_names)
        counts = count_detections(detections)

        annotated_path = annotated_dir / image_path.name
        if not save_annotated_image(result, annotated_path):
            print(f"Warning: failed to save annotated image: {annotated_path}")

        prediction_records.append(
            {
                "image": image_path.name,
                "detections": detections,
                "counts": counts,
            }
        )

    return prediction_records


def save_predictions_json(records: list[dict[str, Any]], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / PREDICTIONS_FILENAME
    json_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    return json_path


def print_summary(
    records: list[dict[str, Any]],
    output_dir: Path,
    json_path: Path,
) -> None:
    total_detections = sum(len(record["detections"]) for record in records)
    total_counts: Counter[str] = Counter()
    for record in records:
        total_counts.update(record["counts"])

    print("\nYOLO Inference Summary")
    print("=" * 32)
    print(f"Images processed:       {len(records)}")
    print(f"Total detections:       {total_detections}")
    print(f"Annotated images:       {output_dir / 'annotated'}")
    print(f"Prediction JSON:        {json_path}")

    print("\nDetected Products")
    print("-" * 32)
    if total_counts:
        for class_name, count in sorted(total_counts.items()):
            print(f"{class_name:>16}: {count}")
    else:
        print("No detections.")


def main() -> int:
    args = parse_args()

    try:
        validate_args(args)
        image_paths = collect_images(args.source)
        model = load_model(args.model)
        records = run_inference(
            model=model,
            image_paths=image_paths,
            conf=args.conf,
            output_dir=args.output,
        )
        json_path = save_predictions_json(records, args.output)
    except (FileNotFoundError, ValueError, ImportError) as exc:
        print(f"Error: {exc}")
        return 1

    print_summary(records, args.output, json_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

