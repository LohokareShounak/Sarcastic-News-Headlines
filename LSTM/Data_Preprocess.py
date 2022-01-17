import pandas as pd
import torch

def Prep_Vocab(DATA_LOCATION):
    Data = pd.read_csv(DATA_LOCATION)
    Vocabulary = {}
    i = 0
    for sentences in Data.headline:
        for word in sentences.split():
            if word not in Vocabulary:
                Vocabulary[word] = i
                i += 1
    return Vocabulary

def Convert_Sentence_to_ids(data, vocab):
    id = [vocab[w] for w in data.split()]
    return torch.tensor(id, dtype=torch.long)
