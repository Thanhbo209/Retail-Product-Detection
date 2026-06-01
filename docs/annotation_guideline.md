# Annotation Guideline

This project uses one object class:

```text
product
```

The goal is to label visible retail product packages on shelf images for object detection. This is not product category classification, SKU recognition, brand recognition, or price recognition.

## Class Definition

| Class | Definition |
| --- | --- |
| product | A visible retail product package that a shopper could reasonably identify as a product unit or package on the shelf |

Examples include boxes, bottles, cans, cartons, cups, packets, bags, jars, and other packaged retail goods.

## Do Not Label

- empty shelf gaps
- shelf background
- price tags
- shelf rails
- posters or printed shelf graphics
- reflections
- hands, carts, baskets, or people
- tiny uncertain fragments that cannot be confidently localized
- product-like shapes that are not actual visible product packages

## Bounding Box Rules

- Use one bounding box per visible product package.
- Draw the box tightly around the visible product.
- Do not include empty shelf space around the product.
- Do not merge multiple products into one box.
- Do not split one product into multiple boxes.
- If a product is partially hidden, label only the visible region if the product is still recognizable.
- Skip heavily occluded, tiny, or blurry fragments when the product boundary is uncertain.

## Crowded Shelves

Crowded shelves are expected. Label each visible product instance separately when its boundary is clear enough.

If many products overlap:

- label the visible part of each recognizable product
- avoid guessing hidden boundaries
- skip fragments that cannot be separated from neighbors

## Hard Examples

Hard examples are images where labels or predictions are likely to be wrong. Common hard cases include:

- empty shelf gaps detected as products
- duplicate boxes on the same product
- boxes that include neighboring products
- missed products on lower shelves
- small products in dense rows
- blurry or reflective packaging

Hard examples should be reviewed in CVAT before being accepted into `data/processed/labels`.

## Prediction Review Rule

Model predictions are not ground truth.

Prediction labels should be treated as pre-labels only:

```text
review_batches/<batch>/labels_initial
```

After human review in CVAT, corrected labels should be saved as:

```text
review_batches/<batch>/labels_fixed
```

Only fixed labels should be accepted into:

```text
data/processed/labels
```

## Quality Checklist

Before training:

- no boxes on empty shelf gaps
- no boxes on price tags or shelf rails
- one product has one box
- boxes are tight and do not include neighboring products
- obvious visible products are not missed
- uncertain tiny fragments are skipped
- hard examples have been reviewed in CVAT

