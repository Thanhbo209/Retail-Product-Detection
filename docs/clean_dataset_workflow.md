# Clean Dataset Workflow

This workflow keeps raw source data immutable and prevents model predictions from becoming training labels without human review.

## Core Policy

- Raw data is immutable: `data/raw`.
- Processed images live in `data/processed/images`.
- Processed labels live in `data/processed/labels`.
- Processed labels are accepted labels only.
- Prediction labels are pre-labels, not ground truth.
- Pre-labels go into `review_batches/<batch>/labels_initial`.
- Human-corrected labels go into `review_batches/<batch>/labels_fixed`.
- Fixed labels are accepted with `scripts/accept_fixed_labels.py`.
- Validate and visualize before training.

## Workflow

```text
raw Supervisely data
  -> convert to YOLO
  -> accepted processed labels
  -> review batch for hard examples
  -> model predictions as labels_initial
  -> CVAT human correction
  -> labels_fixed
  -> accept fixed labels
  -> split train/val/test
  -> force hard examples into train
  -> validate
  -> visualize
  -> train
```

## A. Reset Generated Data

This deletes generated folders only. It does not delete `data/raw`, scripts, configs, docs, app files, or requirements.

```bash
python scripts/reset_generated_data.py --yes
```

## B. Convert Raw Supervisely Annotations

The raw dataset is expected at:

```text
data/raw/supermarket_shelves/images
data/raw/supermarket_shelves/annotations
```

Annotation files may be named `001.json` or `001.jpg.json`. Both are supported.

```bash
python scripts/convert_supervisely_to_yolo.py --images data/raw/supermarket_shelves/images --annotations data/raw/supermarket_shelves/annotations --output-images data/processed/images --output-labels data/processed/labels
```

Generated reports:

```text
outputs/reports/conversion_report.json
outputs/reports/conversion_issues.csv
```

## C. Create Review Batch

Use review batches for hard examples such as images with false positives, missed products, or suspicious labels.

```bash
python scripts/create_review_batch.py --batch-name review_batch_001 --images data/processed/images/046.jpg --output review_batches
```

This creates:

```text
review_batches/review_batch_001/images
review_batches/review_batch_001/labels_initial
review_batches/review_batch_001/labels_fixed
review_batches/review_batch_001/notes
review_batches/review_batch_001/manifest.csv
```

## D. Convert Prediction JSON To Pre-Labels

Use low confidence for review so more possible mistakes are visible.

```bash
python scripts/convert_predictions_json_to_yolo.py --predictions outputs/inference_046/predictions.json --images data/processed/images --output-labels review_batches/review_batch_001/labels_initial --min-conf 0.25
```

Generated report:

```text
outputs/reports/prediction_to_yolo_report.json
```

## E. Package CVAT Import

The default `flat` layout places YOLO `.txt` files at the zip root because this is often the simplest CVAT import path.

```bash
python scripts/package_cvat_yolo_import.py --images review_batches/review_batch_001/images --labels review_batches/review_batch_001/labels_initial --output-zip review_batches/review_batch_001/cvat_import_flat.zip --layout flat --class-name product
```

## F. Human Fixes In CVAT

In CVAT:

- Create a task.
- Add label `product`.
- Upload the image from the review batch.
- Import `cvat_import_flat.zip` using YOLO format.
- Delete false positives.
- Fix bad boxes.
- Add missing product boxes.
- Export YOLO labels.
- Save fixed `.txt` files into `review_batches/review_batch_001/labels_fixed`.

## G. Accept Fixed Labels

This copies reviewed labels into processed data and backs up any overwritten label.

```bash
python scripts/accept_fixed_labels.py --batch review_batches/review_batch_001 --processed-images data/processed/images --processed-labels data/processed/labels
```

Backups are written to:

```text
data/processed/labels_backups/<timestamp>
```

## H. Split Dataset

```bash
python scripts/split_dataset.py --images data/processed/images --labels data/processed/labels --output data/yolo --copy
```

## I. Force Hard Examples Into Train

```bash
python scripts/force_train_images.py --dataset data/yolo --images 046.jpg
```

## J. Validate

```bash
python scripts/validate_yolo_dataset.py --dataset data/yolo --classes configs/classes.txt
```

## K. Visualize

```bash
python scripts/visualize_labels.py --dataset data/yolo --classes configs/classes.txt --split train --count 30 --output outputs/visualizations/train_review
```

## L. Smoke Train

Run only after validation and visualization look correct.

```bash
python scripts/train_yolo.py --model yolov8n.pt --data configs/product_detection.yaml --epochs 3 --imgsz 640 --batch 4 --name clean_smoke_v1
```

## M. Full Train

```bash
python scripts/train_yolo.py --model yolov8n.pt --data configs/product_detection.yaml --epochs 30 --imgsz 640 --batch 4 --name clean_product_v1
```

Use a more specific run name, such as `clean_product_after_label_fix_v1`, when documenting a model trained after a label-fix cycle.
