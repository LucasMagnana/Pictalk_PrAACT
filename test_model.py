from transformers import pipeline

mask_filler = pipeline(
    "fill-mask", model="models/pictalk"
)

text = text = "This is a great [MASK]."

preds = mask_filler(text)

for pred in preds:
    print(f">>> {pred['sequence']}")