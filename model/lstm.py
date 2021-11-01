import csv
import pandas as pd
import re
import torch
from torchtext import data
from torchtext.vocab import Vectors
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import time

class LSTMTagger(nn.Module):

    def __init__(self,vocab, embedding_dim=300, hidden_dim=256, tagset_size=11,n_layers=1, bidirectional=True):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.word_embeddings = nn.Embedding.from_pretrained(vocab)
        self.word_embeddings.weight.requires_grad = True
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first = True,dropout = 0.3, bidirectional=bidirectional,
                            num_layers = n_layers)
        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

        if bidirectional:
            self.fc = nn.Linear(hidden_dim*2, tagset_size)
        else:
            self.fc = nn.Linear(hidden_dim, tagset_size)

        self.dropout = nn.Dropout(p = 0.3)

    def forward(self, sentence):
        batch_size = sentence.size(0)
        sentence = sentence.long()
        embeds = self.word_embeddings(sentence)
        #    lstm_out, hidden = self.lstm(embeds, hidden)
        lstm_out, _ = self.lstm(embeds.view(batch_size, len(sentence[0]), -1))
        #  print(len(lstm_out[0][0]))
        tag_space = self.fc(lstm_out)
        tag_space = self.dropout(tag_space)
        tag_space = tag_space[:,-1]
        #tag_space = self.dropout(tag_space)

        return tag_space
