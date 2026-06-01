# Error Analysis Workflow

Error analysis connects model mistakes to label fixes and dataset improvements.

## Key Rule

Prediction is not ground truth.

Predictions can be used as pre-labels for review, but they must stay in:

```text
review_batches/<batch>/labels_initial
```

Human-corrected labels belong in:

```text
review_batches/<batch>/labels_fixed
```

Only reviewed fixed labels should be accepted into:

```text
data/processed/labels
```

## Error Taxonomy

| Error Type | Meaning | Typical Action |
| --- | --- | --- |
| false_positive | Model labels empty shelf, background, price tag, or non-product as product | Delete the box |
| missed_detection | Visible product has no predicted box | Add a product box |
| bad_box | Box is too loose, too tight, shifted, or includes neighbors | Adjust the box |
| duplicate_detection | Same product has multiple boxes | Keep one correct box |
| annotation_noise | Accepted ground-truth label is wrong or uncertain | Fix source label and retrain |
| low_confidence | Correct detection exists but confidence is weak | Add more similar examples or improve labels |

## Review Metadata Template

- dataset version:
- model version:
- model weights path:
- evaluation date:
- reviewer:
- prediction source:
- prediction JSON:
- confidence threshold:

## Error Analysis CSV

Generate the template with:

```bash
python scripts/create_error_analysis_template.py
```

Optional prefill from inference JSON:

```bash
python scripts/create_error_analysis_template.py --predictions outputs/inference/predictions.json --overwrite
```

CSV columns:

```text
image_filename,error_type,predicted_class,true_class,confidence,description,possible_cause,suggested_fix,reviewed_by,review_date
```

## Label Fix Log

Use the generated template:

```text
outputs/reports/label_fix_log_template.csv
```

Columns:

```text
image_filename,issue_type,source,action_taken,before_label_count,after_label_count,review_status,reviewed_by,review_date,notes
```

## Common Causes

- empty shelf gaps were labeled or predicted as products
- boxes include shelf background or neighboring products
- small products are difficult to separate
- converted labels contain annotation noise
- hard examples were not forced into train
- confidence threshold is too low for demo use
- confidence threshold is too high for error discovery

## Suggested Fixes

- Review suspicious images in CVAT.
- Delete false positives.
- Add missed product boxes.
- Tighten or resize bad boxes.
- Remove duplicate boxes.
- Accept only fixed labels into processed data.
- Re-run validation and visualization before training.

## Review Commands

Create a review batch:

```bash
python scripts/create_review_batch.py --batch-name review_batch_001 --images data/processed/images/046.jpg --output review_batches
```

Convert predictions to pre-labels:

```bash
python scripts/convert_predictions_json_to_yolo.py --predictions outputs/inference_046/predictions.json --images data/processed/images --output-labels review_batches/review_batch_001/labels_initial --min-conf 0.25
```

Package for CVAT:

```bash
python scripts/package_cvat_yolo_import.py --images review_batches/review_batch_001/images --labels review_batches/review_batch_001/labels_initial --output-zip review_batches/review_batch_001/cvat_import_flat.zip --layout flat --class-name product
```

Accept fixed labels:

```bash
python scripts/accept_fixed_labels.py --batch review_batches/review_batch_001 --processed-images data/processed/images --processed-labels data/processed/labels
```

