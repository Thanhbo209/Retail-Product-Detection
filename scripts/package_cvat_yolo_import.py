"""Package YOLO txt labels into a CVAT-importable zip.

Expected usage:
    python scripts/package_cvat_yolo_import.py --images review_batches/review_batch_001/images --labels review_batches/review_batch_001/labels_initial --output-zip review_batches/review_batch_001/cvat_import_flat.zip --layout flat --class-name product
"""

from __future__ import annotations

import argparse
import tempfile
import zipfile
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Package YOLO labels for CVAT import.")
    parser.add_argument("--images", type=Path, required=True, help="Folder containing images.")
    parser.add_argument("--labels", type=Path, required=True, help="Folder containing YOLO txt labels.")
    parser.add_argument("--output-zip", type=Path, required=True, help="Output zip path.")
    parser.add_argument(
        "--layout",
        choices=["flat", "darknet"],
        default="flat",
        help="Zip layout. Default: flat.",
    )
    parser.add_argument(
        "--class-name",
        default="product",
        help="Class name to write into obj.names. Default: product.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.images.exists() or not args.images.is_dir():
        raise FileNotFoundError(f"Images folder not found: {args.images}")
    if not args.labels.exists() or not args.labels.is_dir():
        raise FileNotFoundError(f"Labels folder not found: {args.labels}")


def find_images(images_dir: Path) -> list[Path]:
    return [
        path
        for path in sorted(images_dir.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def write_metadata_files(temp_dir: Path, image_paths: list[Path], layout: str, class_name: str) -> None:
    (temp_dir / "obj.names").write_text(f"{class_name}\n", encoding="utf-8")
    (temp_dir / "obj.data").write_text(
        "classes = 1\n"
        "train = train.txt\n"
        "names = obj.names\n"
        "backup = backup/\n",
        encoding="utf-8",
    )

    if layout == "flat":
        train_lines = [image_path.name for image_path in image_paths]
    else:
        train_lines = [f"obj_train_data/{image_path.name}" for image_path in image_paths]
    (temp_dir / "train.txt").write_text("\n".join(train_lines) + "\n", encoding="utf-8")


def add_file_to_zip(zip_file: zipfile.ZipFile, source_path: Path, archive_name: str) -> None:
    zip_file.write(source_path, archive_name)


def package_zip(
    image_paths: list[Path],
    labels_dir: Path,
    output_zip: Path,
    layout: str,
    class_name: str,
) -> list[str]:
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    archive_names: list[str] = []

    with tempfile.TemporaryDirectory() as temp_name:
        temp_dir = Path(temp_name)
        write_metadata_files(temp_dir, image_paths, layout, class_name)

        with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as zip_file:
            for metadata_name in ["obj.names", "obj.data", "train.txt"]:
                add_file_to_zip(zip_file, temp_dir / metadata_name, metadata_name)
                archive_names.append(metadata_name)

            for image_path in image_paths:
                label_path = labels_dir / f"{image_path.stem}.txt"
                if not label_path.exists():
                    print(f"Warning: missing label for image, skipped: {image_path.name}")
                    continue

                if layout == "flat":
                    archive_name = label_path.name
                else:
                    archive_name = f"obj_train_data/{label_path.name}"

                add_file_to_zip(zip_file, label_path, archive_name)
                archive_names.append(archive_name)

    return archive_names


def print_summary(output_zip: Path, archive_names: list[str]) -> None:
    print("\nCVAT YOLO Import Package")
    print("-" * 32)
    print(f"Output zip: {output_zip}")
    print("Zip contents:")
    for archive_name in archive_names:
        print(f"- {archive_name}")


def main() -> int:
    args = parse_args()

    try:
        validate_args(args)
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        return 1

    image_paths = find_images(args.images)
    archive_names = package_zip(
        image_paths=image_paths,
        labels_dir=args.labels,
        output_zip=args.output_zip,
        layout=args.layout,
        class_name=args.class_name,
    )
    print_summary(args.output_zip, archive_names)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

