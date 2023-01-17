import pandas as pd
import unicodedata
import re
import random

import torch
import torch.nn as nn
from torch import optim
import matplotlib.pyplot as plt


from Lang import Lang
from NN import RNN

MAX_SENTENCE_LENGTH = 10

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters


def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def sentenceToTensor(s, lang):
    splitted_s = s.split(" ")
    splitted_s.insert(0, "SOS")
    splitted_s[-1] = "EOS"
    indexes = [lang.word2index[word] for word in splitted_s]
    return torch.tensor(indexes, dtype=torch.long)

df = pd.read_csv('fra_sentences.tsv', delimiter="\t",)

sentences = df[df.columns[2]].tolist()

lang = Lang()
list_norm_sentences = []

for s in sentences:
    norm_s = normalizeString(unicodeToAscii(s))
    splitted_norm_s = norm_s.split(" ")
    if(len(splitted_norm_s) <= MAX_SENTENCE_LENGTH):
        list_norm_sentences.append(norm_s)
        lang.addSplittedSentence(splitted_norm_s)


print(len(list_norm_sentences), "sentences, ", lang.n_words, "words.")


model = RNN(lang.n_words, 512)
model_optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.NLLLoss()

tot_iter = 500000
plot_loss = [-1]
loss_tot = 0
plot_every = 5000

for iter in range(1, tot_iter+1):
    s = list_norm_sentences[random.randint(0,len(list_norm_sentences))]

    tens = sentenceToTensor(s, lang)
    model_optimizer.zero_grad()
    hidden = model.initHidden()
    loss = 0
    for i in range(tens.size(0)-1):
        output, hidden = model(tens[i], hidden)
        loss += criterion(output.squeeze(), tens[i+1])

    loss.backward()
    model_optimizer.step()

    loss_tot += loss.item()

    if(iter%plot_every==0):
        plot_loss.append(loss_tot/plot_every)
        loss_tot = 0
        print()
        plt.plot(plot_loss[1:])  
        plt.ylabel('Loss')       
        plt.savefig("./loss.png")
        torch.save(model.state_dict(), './model.n')

    print("\rIter: {}/{}, last loss : {:.2f}".format(iter, tot_iter, plot_loss[-1]), end="")

