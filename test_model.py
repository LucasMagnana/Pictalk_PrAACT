from transformers import pipeline

fill_mask = pipeline(
    "fill-mask",
    model="google/mobilebert-uncased",
    tokenizer="google/mobilebert-uncased"
)

print(
    fill_mask(f"HuggingFace is creating a [MASK] that the community uses to solve NLP tasks.")
)