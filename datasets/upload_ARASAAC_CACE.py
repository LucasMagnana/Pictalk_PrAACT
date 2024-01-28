import requests
import datasets

repo_name = "LucasMagnana/"

# Read the input file
input_file = "./datasets/cace/vocab.txt"
with open(input_file, "r") as file:
    lines = file.readlines()

# Translate each line using Arasaac API
pictos = []
pictos_seen = []
for line in lines:
    word = line.strip()
    url = f"https://api.arasaac.org/v1/pictograms/es/search/{word}"
    response = requests.get(url)
    if response.status_code == 200:
        pictogram_id = response.json()[0]["_id"]
        response2 = requests.get(f"https://api.arasaac.org/v1/pictograms/en/{pictogram_id}")
        if response.status_code == 200:
            translated_word = response2.json().get("keywords")[0].get("keyword")
            if(len(translated_word.split(" ")) <= 3 and translated_word not in pictos_seen):
                pictos.append({"text": translated_word, "id": response2.json()["_id"]})
                pictos_seen.append(translated_word)
                    
    else:
        print(f"{word} : translation not available")

# Write the translated lines to a new file
print("size of dataset :", len(pictos))
d_pictos = datasets.Dataset.from_list(pictos)
d_pictos.push_to_hub(repo_name+"ARASAAC_CACE")
