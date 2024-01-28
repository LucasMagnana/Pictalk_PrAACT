from datasets import load_dataset
from transformers import AutoModelForMaskedLM, AutoTokenizer  
from huggingface_hub import hf_hub_download
import torch
import os
import pickle
import argparse

repo_name = "LucasMagnana/"
model_name = "Pictalk_large"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-encode", action="store_true")
    args = parser.parse_args()

    d_pictos  = load_dataset(repo_name+"ARASAAC_CACE")["train"]
    model = AutoModelForMaskedLM.from_pretrained(repo_name+model_name)
    if(not args.no_encode):
        #download the custom layer and switch the model decoder layer with it
        layer_file = hf_hub_download(repo_id=repo_name+model_name, filename="encoded_layer.t")
        with open(layer_file, "rb") as infile:
            out_layer = pickle.load(infile)
        model.set_output_embeddings(out_layer)
    tokenizer = AutoTokenizer.from_pretrained(repo_name+model_name)

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
            if(args.no_encode):
                print(f">>> {text.replace('[MASK]', tokenizer.decode([token]))}")
            else:
                print(f">>> {text.replace('[MASK]', d_pictos[token]['text'])}(id : {d_pictos[token]['id']})")