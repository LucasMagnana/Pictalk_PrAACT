from datasets import load_dataset
from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch
from huggingface_hub import HfApi
import pickle
import os

repo_name = "LucasMagnana/"
model_name = "Pictalk_large"

model = AutoModelForMaskedLM.from_pretrained(repo_name+model_name)
tokenizer = AutoTokenizer.from_pretrained(repo_name+model_name)
d_pictos  = load_dataset("LucasMagnana/ARASAAC_CACE")["train"]

text = "I want eat [MASK]"

#test the original model
inputs = tokenizer(text, return_tensors="pt")
token_logits = model(**inputs).logits
# Find the location of [MASK] and extract its logits
mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
mask_token_logits = token_logits[0, mask_token_index, :]
# Pick the [MASK] candidates with the highest logits
top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()

for token in top_5_tokens:
    print(f">>> {text.replace(tokenizer.mask_token, tokenizer.decode([token]))}")


print("========================================")
print("Encoding...")

#create the embedding matrix
input_embeddings = model.get_input_embeddings()
for i in range(len(d_pictos)):
    pictos = d_pictos[i]["text"]
    embbed_matrix = torch.zeros(input_embeddings.weight.shape[1])
    for word in pictos.split(" "):
        t = tokenizer(word, return_tensors="pt")["input_ids"].squeeze()[1]
        embbed_matrix += input_embeddings(t)
    embbed_matrix = (embbed_matrix/len(pictos.split(" "))).unsqueeze(0)
    if(i == 0):
        out_layer_weight = torch.zeros((len(d_pictos), embbed_matrix.shape[-1]))
    out_layer_weight[i] = embbed_matrix

#create the custom decoder layer 
out_layer = torch.nn.Linear(out_layer_weight.shape[1], out_layer_weight.shape[0])
#change its weights to the values stored in the embedding matrix
out_layer.weight = torch.nn.Parameter(out_layer_weight)
#switch the model decoder layer with the custom on for testing
model.set_output_embeddings(out_layer)

#test the model
inputs = tokenizer(text, return_tensors="pt")
token_logits = model(**inputs).logits
# Find the location of [MASK] and extract its logits
mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
mask_token_logits = token_logits[0, mask_token_index, :]
# Pick the [MASK] candidates with the highest logits
top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()

for token in top_5_tokens:
    print(f">>> {text.replace(tokenizer.mask_token, d_pictos[token]['text'])}(id: {d_pictos[token]['id']})")

with open("./encoded_layer.t", "wb") as outfile:
    pickle.dump(out_layer, outfile)

#upload the custom decoder layer on HuggingFace
api = HfApi()
api.upload_file(
    path_or_fileobj="./encoded_layer.t",
    path_in_repo="encoded_layer.t",
    repo_id=repo_name+model_name,
    repo_type="model",
)
os.remove("./encoded_layer.t")

