import spacy
import datasets
import string

repo_name = "LucasMagnana/"



nlp = spacy.load("en_core_web_sm")

authorized_pos = ["ADJ", "NOUN", "PRON", "VERB", "ADV"]

translator = str.maketrans('', '', string.punctuation) #used to rmeove punctuations

dataset = {}

for set in ["train", "test", "dev"]:

    with open("./datasets/aac_comm/sent_"+set+"_aac.txt", "r") as file:
        list_original_sentences = file.readlines()

    dataset[set] = []
    print(set, "set original size:", len(list_original_sentences))
    for i in range(len(list_original_sentences)):
        original_sentence = list_original_sentences[i].removesuffix('\n') #removes the endline char
        if("," in original_sentence):
            continue
        original_sentence = original_sentence.translate(translator) #removes the punctuation
        final_sentence = ""
        doc = nlp(original_sentence)
        for token in doc:
            if(token.pos_ in authorized_pos):
                if(len(final_sentence) > 0 and len(original_sentence.split()) >= 3):
                    final_sentence += " "
                final_sentence += token.lemma_
        if(len(original_sentence.split()) >= 3 and len(final_sentence)>0):
            dataset[set].append({"text": original_sentence})
            dataset[set].append({"text": final_sentence})
    print(set, "set final size:", len(dataset[set]))
    print("========================")

test_set = dataset["test"]
test_set.extend(dataset["dev"])
dataset = {"train": dataset["train"], "test": test_set}
datasetdict = datasets.DatasetDict()
for k in dataset:
    datasetdict[k] = datasets.Dataset.from_list(dataset[k])
datasetdict.push_to_hub(repo_name+"aactext_text")

    