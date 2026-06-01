# Retail Product Detection Dataset & YOLO Evaluation Pipeline

Experimental portfolio project for building a clean object detection workflow around retail shelf images. The project detects and counts visible retail product packages with a one-class YOLO model.

This is a dataset engineering and model evaluation project, not a production retail analytics system.

## Project Status

- Status: experimental portfolio project
- Task: object detection
- Current scope: one class, `product`
- Annotation format: YOLO object detection
- Review tool: CVAT
- Current trained model: `clean_product_after_label_fix_v1`

Current reported metrics:

| Metric | Value |
| --- | ---: |
| Precision | 0.635 |
| Recall | 0.441 |
| mAP50 | 0.408 |
| mAP50-95 | 0.189 |

Example inference check: at confidence `> 0.50`, one shelf image counted `117` products. A manual visual estimate was approximately `120`.

## What The Model Does

- Detects visible retail product packages on shelf images.
- Counts detections by the single class `product`.
- Produces bounding boxes, confidence scores, annotated images, and JSON predictions.
- Supports a CVAT review loop for correcting model-assisted pre-labels.

## What The Model Does Not Do

- It does not classify product categories such as milk, snack, coffee, or tea.
- It does not identify SKUs, brands, prices, barcodes, or package text.
- It does not decide whether a product is in stock.
- It is not production-ready and should not be used for business decisions without more data, review, and testing.

## Why This Project Matters

This project demonstrates practical skills used in IT Labeling, AI Data Annotation, and beginner Computer Vision roles:

- converting source annotations into YOLO format
- keeping raw data immutable
- separating model predictions from accepted labels
- reviewing hard examples in CVAT
- validating label quality before training
- visualizing annotations before training
- recording model errors and label fixes
- training, evaluating, and testing a YOLO baseline

## Repository Structure

```text
app/
  streamlit_app.py
configs/
  classes.txt
  product_detection.yaml
data/
  raw/
  processed/
  yolo/
docs/
outputs/
  reports/
  visualizations/
  inference/
review_batches/
scripts/
```

Large data, generated outputs, model weights, and review exports are intentionally ignored by Git unless you explicitly choose to track them with Git LFS.

## Installation

```bash
pip install -r requirements.txt
```

Core dependencies include Python, OpenCV, Pillow, pandas, NumPy, PyYAML, Matplotlib, Ultralytics YOLO, and Streamlit.

## Configuration

Class list:

```text
configs/classes.txt
```

Expected content:

```text
product
```

YOLO dataset config:

```text
configs/product_detection.yaml
```

Expected one-class config:

```yaml
path: data/yolo
train: images/train
val: images/val
test: images/test

nc: 1

names:
  0: product
```

## Clean Dataset Workflow

The clean workflow is:

```text
raw data
  -> convert Supervisely annotations
  -> processed accepted labels
  -> review batches for suspicious labels
  -> CVAT correction
  -> accept fixed labels
  -> split train/val/test
  -> validate
  -> visualize
  -> train
  -> evaluate
  -> inference/demo
```

Raw data is immutable. Model predictions are not ground truth. Prediction labels must go into `labels_initial`, human-corrected labels go into `labels_fixed`, and only reviewed fixed labels should be copied into `data/processed/labels`.

## Commands

Reset generated data:

```bash
python scripts/reset_generated_data.py --yes
```

Convert raw Supervisely annotations:

```bash
python scripts/convert_supervisely_to_yolo.py --images data/raw/supermarket_shelves/images --annotations data/raw/supermarket_shelves/annotations --output-images data/processed/images --output-labels data/processed/labels
```

Create a review batch for a hard example:

```bash
python scripts/create_review_batch.py --batch-name review_batch_001 --images data/processed/images/046.jpg --output review_batches
```

Convert prediction JSON into YOLO pre-labels for review:

```bash
python scripts/convert_predictions_json_to_yolo.py --predictions outputs/inference_046/predictions.json --images data/processed/images --output-labels review_batches/review_batch_001/labels_initial --min-conf 0.25
```

Package YOLO pre-labels for CVAT import:

```bash
python scripts/package_cvat_yolo_import.py --images review_batches/review_batch_001/images --labels review_batches/review_batch_001/labels_initial --output-zip review_batches/review_batch_001/cvat_import_flat.zip --layout flat --class-name product
```

Accept human-fixed CVAT labels:

```bash
python scripts/accept_fixed_labels.py --batch review_batches/review_batch_001 --processed-images data/processed/images --processed-labels data/processed/labels
```

Split accepted processed data:

```bash
python scripts/split_dataset.py --images data/processed/images --labels data/processed/labels --output data/yolo --copy
```

Force hard examples into train:

```bash
python scripts/force_train_images.py --dataset data/yolo --images 046.jpg
```

Validate YOLO dataset:

```bash
python scripts/validate_yolo_dataset.py --dataset data/yolo --classes configs/classes.txt
```

Visualize labels before training:

```bash
python scripts/visualize_labels.py --dataset data/yolo --classes configs/classes.txt --split train --count 30 --output outputs/visualizations/train_review
```

Smoke train:

```bash
python scripts/train_yolo.py --model yolov8n.pt --data configs/product_detection.yaml --epochs 3 --imgsz 640 --batch 4 --name clean_smoke_v1
```

Full train:

```bash
python scripts/train_yolo.py --model yolov8n.pt --data configs/product_detection.yaml --epochs 30 --imgsz 640 --batch 4 --name clean_product_v1
```

The latest reported metrics in this README came from `clean_product_after_label_fix_v1`, trained after selected label fixes.

Evaluate:

```bash
python scripts/evaluate_yolo.py --model runs/experiments/clean_product_after_label_fix_v1/weights/best.pt --data configs/product_detection.yaml
```

Run inference:

```bash
python scripts/run_inference.py --model runs/experiments/clean_product_after_label_fix_v1/weights/best.pt --source data/processed/images/046.jpg --conf 0.50 --output outputs/inference_046
```

Run Streamlit demo:

```bash
streamlit run app/streamlit_app.py
```

## CVAT Review Workflow

1. Create a review batch with selected hard images.
2. Convert prediction JSON into YOLO pre-labels under `labels_initial`.
3. Package `labels_initial` into a CVAT-importable zip.
4. Create a CVAT task with label `product`.
5. Import the zip using YOLO format.
6. Delete false positives, fix bad boxes, and add missed products.
7. Export YOLO labels from CVAT.
8. Save fixed `.txt` files into `labels_fixed`.
9. Run `accept_fixed_labels.py` to copy reviewed labels into `data/processed/labels`.

## Confidence Threshold Guidance

Confidence threshold is a trade-off:

| Threshold | Use Case | Trade-off |
| ---: | --- | --- |
| 0.25 | Error review | More detections, more false positives |
| 0.50 | Balanced demo default | Reasonable count check for current model |
| 0.65-0.75 | Cleaner high-confidence display | Fewer false positives, more missed products |

Recommended default demo threshold: `0.50`.

## Error Analysis

Use [docs/error_analysis.md](docs/error_analysis.md) and generated CSV templates to track:

- false positives
- missed detections
- bad boxes
- duplicate detections
- annotation noise
- low-confidence correct detections

The goal is to connect each model mistake to a concrete dataset action: fix labels, add missing boxes, remove false positives, or collect harder examples.

## Limitations

- One-class detection only.
- No category, SKU, or brand recognition.
- Current metrics show the model is still weak on recall and strict localization.
- Shelf images can contain tiny, occluded, blurry, or reflective products.
- Product counts depend heavily on confidence threshold and label quality.
- The dataset and model are not production-ready.

## Next Improvements

- Review more hard examples in CVAT.
- Add label quality checks for duplicate and tiny boxes.
- Improve split strategy for shelf images from similar scenes.
- Compare YOLOv8n and YOLOv8s after labels are cleaner.
- Add a saved error-analysis report for reviewed predictions.
- Improve the Streamlit demo with batch uploads and saved reports.

## GitHub Data Policy

Do not commit large raw datasets, generated YOLO folders, model weights, training runs, review batches, CVAT zips, or inference outputs unless intentionally using Git LFS.

Keep source code, configs, and documentation in Git. Keep generated artifacts reproducible through scripts.
