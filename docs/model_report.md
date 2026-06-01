# Model Report

Use this file to document each YOLO training run. Keep one copy per important experiment or paste completed sections into a final portfolio report.

## Model

- model name:
- base checkpoint:
- training date:
- dataset version:
- trained weights path:
- experiment folder:

## Training Config

- dataset YAML:
- image size:
- epochs:
- batch size:
- optimizer:
- learning rate:
- random seed:
- hardware:

## Dataset Summary

| Split | Images | Objects |
| --- | ---: | ---: |
| train | TBD | TBD |
| val | TBD | TBD |
| test | TBD | TBD |

## Metrics

Report validation or test metrics from `scripts/evaluate_yolo.py`.

| Metric | Value |
| --- | ---: |
| precision | TBD |
| recall | TBD |
| mAP50 | TBD |
| mAP50-95 | TBD |
| inference speed | TBD |

## Per-Class Notes

| Class | Observation | Action |
| --- | --- | --- |
| milk | TBD | TBD |
| water_bottle | TBD | TBD |
| snack | TBD | TBD |
| coffee | TBD | TBD |
| tea | TBD | TBD |
| juice | TBD | TBD |
| instant_noodle | TBD | TBD |
| canned_food | TBD | TBD |
| box_product | TBD | TBD |
| other_product | TBD | TBD |

## Common Errors

- missed detections:
- false positives:
- wrong class predictions:
- bad bounding boxes:
- duplicate detections:
- low-confidence correct detections:

## Error Analysis Summary

- most common error type:
- most affected class:
- likely dataset issue:
- likely model issue:

## Improvement Plan

- collect more examples for:
- relabel or fix:
- adjust class definitions:
- try training change:
- next experiment name:
