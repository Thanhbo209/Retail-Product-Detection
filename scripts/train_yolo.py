"""Train a YOLOv8 object detection model with Ultralytics.

Expected usage:
    python scripts/train_yolo.py --model yolov8n.pt --data configs/product_detection.yaml --epochs 30 --imgsz 640 --batch 8
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a YOLOv8 model.")
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="YOLO model checkpoint or model name. Default: yolov8n.pt.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("configs/product_detection.yaml"),
        help="YOLO dataset YAML path. Default: configs/product_detection.yaml.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of training epochs. Default: 30.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Training image size. Default: 640.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=8,
        help="Training batch size. Default: 8.",
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=Path("runs/experiments"),
        help="Directory for experiment outputs. Default: runs/experiments.",
    )
    parser.add_argument(
        "--name",
        default="product_yolov8n",
        help="Experiment name. Default: product_yolov8n.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.data.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {args.data}")
    if args.epochs <= 0:
        raise ValueError("--epochs must be greater than 0.")
    if args.imgsz <= 0:
        raise ValueError("--imgsz must be greater than 0.")
    if args.batch <= 0:
        raise ValueError("--batch must be greater than 0.")


def print_training_config(args: argparse.Namespace) -> None:
    print("\nYOLOv8 Training Configuration")
    print("=" * 32)
    print(f"Model:      {args.model}")
    print(f"Data YAML:  {args.data}")
    print(f"Epochs:     {args.epochs}")
    print(f"Image size: {args.imgsz}")
    print(f"Batch size: {args.batch}")
    print(f"Project:    {args.project}")
    print(f"Run name:   {args.name}")


def get_training_save_dir(model: Any, results: Any, project: Path, name: str) -> Path:
    """Find the actual save directory created by Ultralytics.

    Ultralytics may increment the run name when a folder already exists, so the
    trainer save directory is the most reliable source after training.
    """
    trainer = getattr(model, "trainer", None)
    trainer_save_dir = getattr(trainer, "save_dir", None)
    if trainer_save_dir is not None:
        return Path(trainer_save_dir)

    results_save_dir = getattr(results, "save_dir", None)
    if results_save_dir is not None:
        return Path(results_save_dir)

    return project / name


def train(args: argparse.Namespace) -> Path:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "ultralytics is not installed. Run: pip install -r requirements.txt"
        ) from exc

    model = YOLO(args.model)
    results = model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=str(args.project),
        name=args.name,
    )

    save_dir = get_training_save_dir(model, results, args.project, args.name)
    return save_dir / "weights" / "best.pt"


def main() -> int:
    args = parse_args()

    try:
        validate_args(args)
        print_training_config(args)
        best_model_path = train(args)
    except (FileNotFoundError, ValueError, ImportError) as exc:
        print(f"Error: {exc}")
        return 1

    print("\nTraining complete.")
    if best_model_path.exists():
        print(f"Best model saved to: {best_model_path}")
    else:
        print(f"Expected best model path: {best_model_path}")
        print("Warning: best.pt was not found. Check the Ultralytics training logs.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

