# Retail Product Detection Dataset & YOLO Evaluation Pipeline

This project is a practical computer vision portfolio project for retail product object detection. It focuses on dataset preparation, annotation quality, YOLO-format validation, and repeatable documentation before model training.

## Project Goal

Build a complete product image labeling workflow similar to an IT Labeling or AI Data Annotation task:

- collect supermarket shelf and consumer product images
- label products with bounding boxes
- export annotations in YOLO format
- validate dataset quality before training
- document annotation rules and dataset limitations
- prepare the project for YOLOv8 training, evaluation, inference, and error analysis in later phases

## Classes

The current class list is stored in [configs/classes.txt](configs/classes.txt):

| ID | Class |
| --- | --- |
| 0 | milk |
| 1 | water_bottle |
| 2 | snack |
| 3 | coffee |
| 4 | tea |
| 5 | juice |
| 6 | instant_noodle |
| 7 | canned_food |
| 8 | box_product |
| 9 | other_product |

## Dataset Layout

YOLO dataset files should be placed under:

```text
data/yolo/
  images/
    train/
    val/
    test/
  labels/
    train/
    val/
    test/
```

Each image should have a matching `.txt` label file with the same stem:

```text
data/yolo/images/train/example_001.jpg
data/yolo/labels/train/example_001.txt
```

Each YOLO label line must use this format:

```text
class_id x_center y_center width height
```

All coordinates must be normalized values between `0` and `1`.

## Setup

Create and activate a Python environment, then install dependencies:

```bash
pip install -r requirements.txt
```

## Validate Dataset

Run the Phase 1 validator:

```bash
python scripts/validate_yolo_dataset.py --dataset data/yolo --classes configs/classes.txt
```

The script checks image-label matching, YOLO label format, coordinate ranges, class IDs, object counts, split counts, and empty label files.

Validation output is saved to:

```text
outputs/reports/validation_report.json
outputs/reports/validation_issues.csv
```

## Current Phase

Phase 1 includes:

- project folder structure
- class configuration
- project configuration
- documentation templates
- basic YOLO dataset validation script

Training, inference, Streamlit demo, and model evaluation are intentionally not implemented yet.

