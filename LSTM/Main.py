import torch
import torch.nn.functional as F
import torch.optim as optim
from Model import LSTM_MODEL
import torch.nn as nn
from Data_Preprocess import Prep_Vocab, Convert_Sentence_to_ids
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import pickle
DATA_LOC = r"C:\Users\shounak lohokare\Desktop\Python Basics\Project A\Data\Data.csv"
SPLIT_SIZE = 0.7
VOCAB = Prep_Vocab(DATA_LOC)
DATA = pd.read_csv(DATA_LOC)

MSK = np.random.rand(len(DATA)) <= 0.8
DATA_TRAIN = DATA[MSK]
DATA_TEST = DATA[~MSK]

VOCAB_SIZE = len(VOCAB)
EMBED_DIM = 512
HIDDEN_DIM = 80
DROPOUT = 0
TARGET_SIZE = 2
LR_RATE = 0.1
EPOCHS = 1
GPU = True
MAX_LEN = 151 #Calculated Separately from Data


MODEL = LSTM_MODEL(VOCAB_SIZE, HIDDEN_DIM, EMBED_DIM,TARGET_SIZE, Dropout = DROPOUT)
DEVICE = torch.device("cuda") if torch.cuda.is_available else torch.device("cpu")
OPTIM = optim.SGD(MODEL.parameters(), LR_RATE)
LOSS = nn.BCELoss()

def Training(MODEL, OPTIM, LOSS, DATA, EPOCHS, VOCAB, DEVICE, GPU = False):
    if GPU:
        MODEL = MODEL.to(DEVICE)
        LOSS = LOSS.to(DEVICE)
    MODEL.train()
    print(f"The Size of the Data is - {len(DATA)}")
    for epoch in range(EPOCHS): 
        i = 0
        for sentence, answer in zip(DATA.headline, DATA.is_sarcastic):

            MODEL.zero_grad()

            sentence_embed = Convert_Sentence_to_ids(sentence, VOCAB).to(DEVICE)
            #print(sentence_embed)
            #print(sentence_embed.shape)

            scores = MODEL(sentence_embed)
            #print(scores)
            #print(scores[-1])
            if answer == 0:
                targets = [0, 1]
            else:
                targets = [1, 0]
            targets = torch.tensor(targets, dtype=torch.float).to(DEVICE)
            #print(targets)
            loss = LOSS(scores[-1], targets) #[scores[-1], 2]
            #print(loss)
            loss.backward()
            OPTIM.step()
            #print("Sucess")
            print(i , end = '\r')
            if i % 5000 == 0:
                print(f"Iteration Finished - {i}, Loss - {loss}")
            i += 1
        print(f"Epochs Finished - {epoch}")
    pickl = {'model': MODEL}
    pickle.dump( pickl, open( 'model' + ".p", "wb" ) )

def Testing(MODEL, DATA, LOSS, VOCAB,GPU = False):
    print(f"The Size of the Data is - {len(DATA)}")
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        MODEL.eval()
        if GPU:
            MODEL = MODEL.to(DEVICE)
            LOSS = LOSS.to(DEVICE)
        for answer, headlines in zip(DATA.is_sarcastic,DATA.headline):
            sentence_embed = Convert_Sentence_to_ids(headlines, VOCAB).to(DEVICE)
            if answer == 0:
                targets = 0
            else:
                targets = 1
            targets = torch.tensor(targets, dtype=torch.float).to(DEVICE)

            scores = MODEL(sentence_embed)
            Preds = torch.round(scores[-1][0])
            n_samples += 1
            if Preds == answer:
                n_correct += 1
        accuracy = 100*n_correct/n_samples
        print(f"Accuracy of Model - {accuracy}")
if __name__ == "__main__" :
    try:
        with open("model.p", "rb") as pickled:
            data = pickle.load(pickled)
            MODEL = data['model']
        print("Saved Model Found!")
    except:
        print("Started Training")
        Training(MODEL, OPTIM, LOSS, DATA_TRAIN,EPOCHS,VOCAB,DEVICE, GPU)
        print("Finished Training")
    print("Started Testing")
    Testing(MODEL, DATA_TEST, LOSS, VOCAB,GPU)
    print("Test Finished")