# Annotation Guideline

This document defines how to label retail products for object detection. The goal is consistency. A smaller but clean dataset is better than a larger dataset with unclear labels.

## Labeling Tool

Use one of these tools:

- CVAT
- Roboflow
- Label Studio

Export annotations in YOLO format.

## Classes

Use the class list in `configs/classes.txt`.

| Class | Use When |
| --- | --- |
| milk | Milk cartons, bottles, cans, or milk drink packaging |
| water_bottle | Plain bottled water products |
| snack | Chips, biscuits, candy, crackers, and similar snack products |
| coffee | Coffee bags, jars, cans, sachets, and ready-to-drink coffee |
| tea | Tea boxes, bottles, bags, sachets, and ready-to-drink tea |
| juice | Juice cartons, bottles, cans, and boxes |
| instant_noodle | Cup noodles, packet noodles, and instant ramen products |
| canned_food | Food products sold in cans |
| box_product | Boxed products that do not fit the more specific classes |
| other_product | Clearly visible retail products that do not fit another class |

## Bounding Box Rules

- Draw the box tightly around the visible product.
- Include the full visible product packaging, not the shelf space around it.
- Do not include neighboring products in the same box.
- Use one box per product instance.
- For multipacks, label the visible selling unit. If the multipack is sold as one package, draw one box around the pack.

## Occluded Products

- Label a product if enough of it is visible to identify the class.
- Draw the box around only the visible part if the full product boundary is hidden.
- Do not guess hidden boundaries.
- Skip products that are too occluded to classify reliably.

## Blurry Products

- Label blurry products only if the class is still reasonably clear.
- Use `other_product` only when the item is clearly a product but the exact target class is not reliable.
- Skip products that are too blurry to localize or classify.

## Ambiguous Products

- Prefer the most specific class when the product type is clear.
- Use `box_product` for boxed products that do not belong to milk, juice, tea, coffee, instant noodles, snacks, canned food, or water.
- Use `other_product` for visible products outside the defined taxonomy.
- Do not use `other_product` for difficult examples that should belong to a known class.

## Common Mistakes To Avoid

- Drawing boxes too loose around the product.
- Drawing boxes too tight and cutting off packaging.
- Labeling shelf price tags as products.
- Combining multiple products into one box.
- Using inconsistent classes for the same product type.
- Overusing `other_product`.
- Labeling reflections, posters, or background graphics as products.

## Quality Checklist

Before exporting annotations:

- every visible target product has one box
- each box is tight and aligned with the visible product
- class names are consistent
- tiny, unreadable, or heavily occluded products are skipped
- no price tags, shelf labels, hands, or background objects are labeled

