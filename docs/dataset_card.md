# Dataset Card

## Dataset Name

Retail Product Detection Dataset

## Project Status

Experimental portfolio dataset for one-class retail shelf object detection.

## Task

Detect visible retail product packages in shelf images.

This dataset supports object detection only. It does not support product category classification, brand recognition, SKU recognition, barcode recognition, or price recognition.

## Class

| ID | Class | Meaning |
| ---: | --- | --- |
| 0 | product | Visible retail product package |

Class config:

```text
configs/classes.txt
```

## Source Format

Raw annotations use Supervisely-style JSON rectangles:

```text
data/raw/supermarket_shelves/images
data/raw/supermarket_shelves/annotations
```

Annotation files may be named like:

```text
001.json
001.jpg.json
```

The converter reads:

- `data["size"]["width"]`
- `data["size"]["height"]`
- `data["objects"]`
- rectangle objects where `classTitle` is `Product`

## YOLO Format

Processed labels use YOLO object detection format:

```text
class_id x_center y_center width height
```

Coordinates are normalized to image width and height.

## Dataset Folders

Raw data:

```text
data/raw
```

Accepted processed data:

```text
data/processed/images
data/processed/labels
```

YOLO train/val/test data:

```text
data/yolo/images/train
data/yolo/images/val
data/yolo/images/test
data/yolo/labels/train
data/yolo/labels/val
data/yolo/labels/test
```

Review batches:

```text
review_batches/<batch>/images
review_batches/<batch>/labels_initial
review_batches/<batch>/labels_fixed
```

## Labeling Policy

- `product` means visible retail product package.
- One product instance gets one box.
- Empty shelf gaps, price tags, shelf rails, and background areas get no box.
- Partially occluded products are labeled only when recognizable.
- Tiny uncertain fragments are skipped.
- Model predictions are not ground truth until reviewed and fixed.

## Current Model Context

Latest reported model:

```text
clean_product_after_label_fix_v1
```

Latest reported metrics:

| Metric | Value |
| --- | ---: |
| Precision | 0.635 |
| Recall | 0.441 |
| mAP50 | 0.408 |
| mAP50-95 | 0.189 |

## Intended Use

- portfolio demonstration
- dataset conversion workflow
- YOLO label validation
- annotation quality review
- baseline object detection training and evaluation

## Not Intended For

- production inventory counting
- SKU-level analytics
- brand or category classification
- price or barcode detection
- high-stakes business automation

## Known Limitations

- one-class labels only
- no semantic product category labels
- small and occluded shelf products are difficult
- counts are sensitive to confidence threshold
- label quality depends on review completeness
- current recall and mAP50-95 indicate room for improvement

## Data Policy

Do not commit large raw datasets, generated YOLO folders, model weights, review batches, CVAT exports, or inference outputs unless intentionally using Git LFS.

