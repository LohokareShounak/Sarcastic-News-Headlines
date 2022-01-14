import torch.nn as nn
import torch.nn.functional as F


class LSTM_MODEL(nn.Module):
    def __init__(self, vocab_size, hidden_dim,embedding_dim, target_size, Dropout = 0):
        super(LSTM_MODEL, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.Dropout = Dropout
        #self.max_len = max_len
        #Defining Layers 
        self.Embedding = nn.Embedding(self.vocab_size,self.embedding_dim)
        self.LSTM_1 = nn.LSTM(self.embedding_dim, self.hidden_dim, dropout = self.Dropout, bidirectional = True)
        self.Linear = nn.Linear(self.hidden_dim*2, target_size)

    def forward(self,input):
        embeds = self.Embedding(input)
        #print(embeds.size())
        #print(embeds.view(len(input), 1, -1).size())
        lstm_out, _ = self.LSTM_1(embeds.view(len(input), 1, -1))
        #print(_[1].size())
        #print(lstm_out.size())
        #print(_[0] == lstm_out[-1])
        #print(lstm_out.view(len(input), -1).size())
        lin_out = self.Linear(lstm_out.view(len(input), -1))
        #print(lin_out[-1].size())
        #print(lin_out.size())
        probab_scores = F.softmax(lin_out, dim = 1)
        return probab_scores

