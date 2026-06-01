# Error Analysis Workflow

Use this document after model evaluation or inference. The goal is to turn model mistakes into concrete dataset and labeling improvements.

## Review Metadata

- dataset version:
- model version:
- model weights path:
- evaluation date:
- reviewer:
- prediction source:
- prediction JSON:
- confidence threshold:

## Error Types

Use one of these values in `outputs/reports/error_analysis_template.csv`.

| Error Type | Meaning |
| --- | --- |
| missed_detection | A real product is visible but the model did not detect it |
| wrong_class | The model detected the product but assigned the wrong class |
| bad_box | The class is correct but the bounding box is too loose, too tight, or shifted |
| duplicate_detection | The same product instance was detected more than once |
| false_positive | The model detected something that is not a target product |
| low_confidence | The prediction is correct but confidence is too low |

## Summary Of Common Errors

Fill this section after reviewing predictions.

| Error Type | Count | Most Affected Classes | Notes |
| --- | ---: | --- | --- |
| missed_detection | TBD | TBD | TBD |
| wrong_class | TBD | TBD | TBD |
| bad_box | TBD | TBD | TBD |
| duplicate_detection | TBD | TBD | TBD |
| false_positive | TBD | TBD | TBD |
| low_confidence | TBD | TBD | TBD |

## Missed Detection Examples

| Image | True Class | Description | Possible Cause | Suggested Fix |
| --- | --- | --- | --- | --- |
| TBD | TBD | Product was visible but not detected | Too few examples, small object, occlusion, blur | Add more examples and check labels |

## Wrong Class Examples

| Image | Predicted Class | True Class | Confidence | Possible Cause | Suggested Fix |
| --- | --- | --- | ---: | --- | --- |
| TBD | TBD | TBD | TBD | Similar packaging or unclear class rule | Improve class guideline and add more examples |

## False Positive Examples

| Image | Predicted Class | Confidence | Description | Possible Cause | Suggested Fix |
| --- | --- | ---: | --- | --- | --- |
| TBD | TBD | TBD | Shelf tag, background object, reflection, or poster detected | Background looks similar to product packaging | Add negative examples or fix labels |

## Duplicate Detection Examples

| Image | Class | Description | Possible Cause | Suggested Fix |
| --- | --- | --- | --- | --- |
| TBD | TBD | Same product detected multiple times | Overlapping boxes or low NMS effect | Review labels and tune inference settings |

## Bad Box Examples

| Image | Class | Description | Possible Cause | Suggested Fix |
| --- | --- | --- | --- | --- |
| TBD | TBD | Box cuts off the product or includes nearby products | Inconsistent training labels | Fix annotations and retrain |

## Low Confidence Examples

| Image | Class | Confidence | Description | Possible Cause | Suggested Fix |
| --- | --- | ---: | --- | --- | --- |
| TBD | TBD | TBD | Correct detection but weak confidence | Underrepresented class or hard lighting | Add more similar examples |

## Possible Causes

Common causes to check:

- class imbalance
- inconsistent box tightness
- overuse of `other_product`
- too few examples for one class
- small products on crowded shelves
- blurry images
- strong glare or poor lighting
- occluded packaging
- ambiguous class definitions
- train/val/test leakage or poor split distribution
- incorrect ground-truth labels

## Suggested Fixes

Choose fixes that match the observed mistake:

- correct bad labels
- add more examples for weak classes
- add more crowded shelf images
- collect examples with blur, glare, and occlusion
- simplify confusing class definitions
- reduce overuse of `other_product`
- inspect tiny boxes and duplicate labels
- retrain with a clean dataset version
- compare YOLOv8n and YOLOv8s after labels are clean

## Next Labeling Improvements

Before the next training run:

- review the top 20 missed detections
- review the top 20 false positives
- review all wrong-class predictions for confusing classes
- update `docs/annotation_guideline.md` if class rules are unclear
- fix annotation mistakes in the labeling tool
- export a new dataset version
- run dataset validation again
- retrain and compare metrics

## CSV Columns

The error analysis CSV uses:

```text
image_filename,error_type,predicted_class,true_class,confidence,description,possible_cause,suggested_fix,reviewed_by,review_date
```

Generate the template with:

```bash
python scripts/create_error_analysis_template.py
```

To prefill rows from inference JSON:

```bash
python scripts/create_error_analysis_template.py --predictions outputs/inference/predictions.json --overwrite
```

