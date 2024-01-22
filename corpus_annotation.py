import spacy
from os import walk
import xml.etree.ElementTree as ET
import pickle

nlp = spacy.load("en_core_web_sm")

authorized_pos = ["ADJ", "NOUN", "PRON", "VERB", "ADV"]

with open("./datasets/aac_comm/sent_train_aac.txt", "r") as file:
    list_original_sentences = file.readlines()

dataset = []

for i in range(len(list_original_sentences)):
    original_sentence = list_original_sentences[i].removesuffix('\n')
    if("," in original_sentence):
        continue
    final_sentence = ""
    doc = nlp(original_sentence)
    for token in doc:
        if(token.pos_ in authorized_pos):
            if(len(final_sentence) > 0):
                final_sentence += " "
            final_sentence += token.lemma_
    dataset.append(original_sentence)
    if(len(final_sentence) > 0):
        dataset.append(final_sentence)

with open("./datasets/aactext_train.dt", "wb") as outfile:
    pickle.dump(dataset, outfile)

with open("./datasets/aactext_train.dt", "rb") as infile:
    d = pickle.load(infile)

print(d)