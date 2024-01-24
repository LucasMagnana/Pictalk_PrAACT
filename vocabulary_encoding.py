from transformers import pipeline
import pickle
from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch

model = AutoModelForMaskedLM.from_pretrained("LucasMagnana/Pictalk_mobile")
tokenizer = AutoTokenizer.from_pretrained("LucasMagnana/Pictalk_mobile")

text = "I want eat [MASK]"

inputs = tokenizer(text, return_tensors="pt")
token_logits = model(**inputs).logits
# Find the location of [MASK] and extract its logits
mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
mask_token_logits = token_logits[0, mask_token_index, :]
# Pick the [MASK] candidates with the highest logits
top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()

for token in top_5_tokens:
    print(f"'>>> {text.replace(tokenizer.mask_token, tokenizer.decode([token]))}'")


with open("pictos.tab", "rb") as infile:
    pictos = pickle.load(infile)




input_embeddings = model.get_input_embeddings()
for i in range(len(pictos)):
    embbed_matrix = torch.zeros(input_embeddings.weight.shape[1])
    for word in pictos[i].split(" "):
        t = tokenizer(word, return_tensors="pt")["input_ids"].squeeze()[1]
        embbed_matrix += input_embeddings(t)
    if(i == 0):
        out_layer_weight = (embbed_matrix/len(pictos[i].split(" "))).unsqueeze(0)
    else:
        out_layer_weight = torch.cat((out_layer_weight, (embbed_matrix/len(pictos[i].split(" "))).unsqueeze(0)))
print(out_layer_weight.shape, model.cls.predictions.decoder.weight.shape)
out_layer = torch.nn.Embedding(out_layer_weight.shape[0], out_layer_weight.shape[1])
out_layer.weight = torch.nn.Parameter(out_layer_weight)
model.set_output_embeddings(out_layer)
print(out_layer_weight.shape, model.cls.predictions.decoder.weight.shape, torch.all(model.cls.predictions.decoder.weight == out_layer_weight))

print(model.cls.predictions.dense.weight.shape, model.cls.predictions.decoder.weight.shape)

inputs = tokenizer(text, return_tensors="pt")
token_logits = model(**inputs).logits
# Find the location of [MASK] and extract its logits
mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
mask_token_logits = token_logits[0, mask_token_index, :]
# Pick the [MASK] candidates with the highest logits
top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()

print(top_5_tokens)

for token in top_5_tokens:
    print(f"'>>> {text.replace(tokenizer.mask_token, tokenizer.decode([token]))}'")
