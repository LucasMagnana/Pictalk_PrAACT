from datasets import load_dataset
from transformers import AutoModelForMaskedLM, AutoTokenizer, BertConfig  
import torch  
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--text", type=str, default="I want eat")

    args = parser.parse_args()

    model_name = "LucasMagnana/Pictalk_encoded"
    d_pictos  = load_dataset("LucasMagnana/ARASAAC_CACE")["train"]
    config = BertConfig.from_pretrained(model_name, vocab_size=len(d_pictos))
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    args.text += " [MASK]"
    inputs = tokenizer(args.text, return_tensors="pt")
    token_logits = model(**inputs).logits
    # Find the location of [MASK] and extract its logits
    mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
    mask_token_logits = token_logits[0, mask_token_index, :]
    # Pick the [MASK] candidates with the highest logits
    top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()

    for token in top_5_tokens:
        print(f">>> {args.text.replace('[MASK]', d_pictos[token]['text'])}(picto id : {d_pictos[token]['id']})")