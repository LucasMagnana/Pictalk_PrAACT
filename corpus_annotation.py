import spacy
from datasets import load_dataset
import string

nlp = spacy.load("en_core_web_sm")

authorized_pos = ["ADJ", "NOUN", "PRON", "VERB", "ADV"]

translator = str.maketrans('', '', string.punctuation)

for set in ["train", "test", "dev"]:

    with open("./datasets/aac_comm/sent_"+set+"_aac.txt", "r") as file:
        list_original_sentences = file.readlines()

    dataset = ""
    print(set, ":", len(list_original_sentences))
    for i in range(len(list_original_sentences)):
        original_sentence = list_original_sentences[i].removesuffix('\n')
        if("," in original_sentence):
            continue
        original_sentence = original_sentence.translate(translator)
        final_sentence = ""
        doc = nlp(original_sentence)
        for token in doc:
            if(token.pos_ in authorized_pos):
                if(len(final_sentence) > 0):
                    final_sentence += " "
                final_sentence += token.lemma_
        if(len(final_sentence) > 0):
            dataset += original_sentence+"\n"
            dataset += final_sentence+"\n"
    print(set, ":", dataset.count("\n"))
    with open("./datasets/aactext_"+set+".txt", "w") as outfile:
        outfile.write(dataset)

dataset = load_dataset("text", data_files={"train": "./datasets/aactext_train.txt", "test": ["./datasets/aactext_test.txt","./datasets/aactext_dev.txt"]})
dataset.push_to_hub("LucasMagnana/aactext_text")

    