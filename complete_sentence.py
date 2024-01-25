from datasets import load_dataset
from transformers import AutoModelForMaskedLM, AutoTokenizer  
from huggingface_hub import hf_hub_download
import torch
import os
import pickle

model_name = "LucasMagnana/Pictalk"
d_pictos  = load_dataset("LucasMagnana/ARASAAC_CACE")["train"]
model = AutoModelForMaskedLM.from_pretrained(model_name)
layer_file = hf_hub_download(repo_id=model_name, filename="encoded_layer.t")
with open(layer_file, "rb") as infile:
    out_layer = pickle.load(infile)
model.set_output_embeddings(out_layer)
tokenizer = AutoTokenizer.from_pretrained(model_name)

while(True):
    text = input("Please enter a sentence to complete (exit to stop) : ")
    if(text == "exit"):
        exit()
    text += " [MASK]"
    inputs = tokenizer(text, return_tensors="pt")
    token_logits = model(**inputs).logits
    # Find the location of [MASK] and extract its logits
    mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
    mask_token_logits = token_logits[0, mask_token_index, :]
    # Pick the [MASK] candidates with the highest logits
    top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()

    for token in top_5_tokens:
        print(f">>> {text.replace('[MASK]', d_pictos[token]['text'])}(id : {d_pictos[token]['id']})")