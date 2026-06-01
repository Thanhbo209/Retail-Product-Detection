# Error Analysis Template

Use this document with `outputs/reports/error_analysis.csv` in later phases. The goal is to record model mistakes and connect them to dataset improvements.

## Error Types

| Error Type | Meaning |
| --- | --- |
| missed_detection | A real product was not detected |
| false_positive | The model detected something that is not a product |
| wrong_class | The box is on a product but the class is wrong |
| bad_box | The class is correct but the box is poorly placed |
| duplicate_detection | The same product is detected multiple times |
| low_confidence | The prediction is correct but confidence is low |

## CSV Columns

Recommended columns:

```text
image_filename,error_type,true_class,predicted_class,confidence,possible_cause,suggested_fix,notes
```

## Example

```text
image_001.jpg,wrong_class,juice,milk,0.62,similar carton shape,add more juice carton examples,
```

## Review Questions

- Is the problem caused by missing training examples?
- Is the class definition unclear?
- Is the product too small, blurry, or occluded?
- Is the annotation incorrect?
- Does the model need more epochs or better image resolution?
- Should the class taxonomy be simplified?

