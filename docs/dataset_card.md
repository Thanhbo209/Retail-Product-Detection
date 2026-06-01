# Dataset Card

## Dataset Name

Retail Product Detection Dataset

## Purpose

This dataset is intended for object detection experiments on supermarket shelf images and consumer product images. It is designed for a beginner computer vision portfolio project focused on annotation quality and YOLO evaluation.

## Task

Object detection with bounding boxes.

## Annotation Format

YOLO format:

```text
class_id x_center y_center width height
```

Coordinates are normalized to image width and height.

## Classes

See `configs/classes.txt`.

## Data Sources

Describe where the images came from:

- source name:
- collection date:
- collection method:
- license or usage permission:
- number of images:

## Dataset Split

| Split | Images | Labels | Notes |
| --- | ---: | ---: | --- |
| train | TBD | TBD |  |
| val | TBD | TBD |  |
| test | TBD | TBD |  |

## Labeling Policy

Summarize the key labeling rules from `docs/annotation_guideline.md`.

## Known Limitations

Document limitations honestly:

- limited store types
- limited camera angles
- possible class imbalance
- ambiguous packaging
- small or occluded shelf products
- lighting variation

## Intended Use

- portfolio demonstration
- YOLO object detection baseline
- dataset validation workflow
- annotation quality review

## Not Intended For

- production retail analytics without more data
- price recognition
- brand-level recognition
- barcode recognition
- high-risk decision making

## Version History

| Version | Date | Notes |
| --- | --- | --- |
| 0.1 | TBD | Initial project setup |

