import requests
import pickle
import datasets

repo_name = "LucasMagnana/"

# Translate each line using Arasaac API
pictos = []
pictos_seen = []
response = requests.get("https://api.arasaac.org/v1/pictograms/all/en")
for p in response.json():
    p_text = p.get("keywords")[0].get("keyword").replace("-", " ")
    if(p_text not in pictos):
        if(len(p_text.split(" ")) <= 3 and p_text not in pictos_seen):
            pictos.append({"text": p_text, "id": p["_id"]})
            pictos_seen.append(p_text)
print("size of dataset :", len(pictos))
d_pictos = datasets.Dataset.from_list(pictos)
d_pictos.push_to_hub(repo_name+"ARASAAC")


