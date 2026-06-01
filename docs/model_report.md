# Model Report

## Current Reported Model

| Field | Value |
| --- | --- |
| Model name | `clean_product_after_label_fix_v1` |
| Task | One-class object detection |
| Class | `product` |
| Base model | YOLOv8n |
| Dataset config | `configs/product_detection.yaml` |
| Status | Experimental portfolio model |

## Metrics

Latest reported metrics:

| Metric | Value |
| --- | ---: |
| Precision | 0.635 |
| Recall | 0.441 |
| mAP50 | 0.408 |
| mAP50-95 | 0.189 |

Interpretation:

- Precision is moderate: many predictions are usable, but false positives still occur.
- Recall is low: the model still misses visible products.
- mAP50 is modest: boxes are partially useful at a loose IoU threshold.
- mAP50-95 is low: strict localization quality needs improvement.

## Example Count Check

At confidence `> 0.50`, one shelf image counted `117` products. A manual visual estimate was approximately `120`.

This is a useful sanity check, not proof of production readiness.

## Confidence Threshold Guidance

| Threshold | Recommended Use | Trade-off |
| ---: | --- | --- |
| 0.25 | Error review | More detections and more false positives |
| 0.50 | Balanced demo default | Better product count demonstration for current model |
| 0.65-0.75 | Conservative display | Fewer false positives and more missed products |

## Evaluation Command

```bash
python scripts/evaluate_yolo.py --model runs/experiments/clean_product_after_label_fix_v1/weights/best.pt --data configs/product_detection.yaml
```

The evaluation script writes:

```text
outputs/reports/evaluation_report.json
```

## Inference Command

```bash
python scripts/run_inference.py --model runs/experiments/clean_product_after_label_fix_v1/weights/best.pt --source data/processed/images/046.jpg --conf 0.50 --output outputs/inference_046
```

## Common Failure Modes

- false positives on empty shelf gaps
- missed products in crowded or lower shelf regions
- bad boxes that include neighboring products
- duplicate detections on one product
- annotation noise inherited from converted labels
- low confidence on small or occluded products

## Improvement Plan

- Continue reviewing hard examples in CVAT.
- Remove false positives from accepted labels.
- Add missed visible product boxes.
- Keep model predictions in `labels_initial` until reviewed.
- Accept only human-fixed labels into `data/processed/labels`.
- Re-split, validate, and visualize before each training run.
- Compare model metrics after each label-fix cycle.
