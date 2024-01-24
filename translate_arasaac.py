import requests
import pickle

# Read the input file
input_file = "./datasets/cace-vocab.txt"
with open(input_file, "r") as file:
    lines = file.readlines()

# Translate each line using Arasaac API
pictos = []
for line in lines:
    word = line.strip()
    url = f"https://api.arasaac.org/v1/pictograms/es/search/{word}"
    response = requests.get(url)
    if response.status_code == 200:
        pictogram_id = response.json()[0]["_id"]
        response2 = requests.get(f"https://api.arasaac.org/v1/pictograms/en/{pictogram_id}")
        if response.status_code == 200:
            translated_word = response2.json().get("keywords")[0].get("keyword")
            if(len(translated_word.split(" ")) <= 3):
                pictos.append(translated_word)
    else:
        print(f"{word} : translation not available")

# Write the translated lines to a new file
output_file = "./datasets/pictos.tab"
with open(output_file, "wb") as file:
    pickle.dump(pictos, file)
