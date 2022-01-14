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


def Padding_Sequences(input, max_len):
    return torch.stack(torch.cat((input, input.new_zeros(max_len - input.size(0)))))
    