import csv
import pandas as pd
import re
import torch
from torchtext import data
from torchtext.vocab import Vectors
from torch.distributions.bernoulli import Bernoulli
from model.DANN import GRL
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import time

class LSTMTagger(nn.Module):

    def __init__(self,vocab, args, embedding_dim=300, hidden_dim=256,n_layers=1, bidirectional=True):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        try:
            self.word_embeddings = nn.Embedding.from_pretrained(vocab.vectors)
        except:
            self.word_embeddings = nn.Embedding(len(vocab),embedding_dim)
        self.word_embeddings.weight.requires_grad = True
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first = True, bidirectional=bidirectional,
                            num_layers = n_layers)
        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim,  args.OUTPUT_DIM)

        if bidirectional:
            self.fc = nn.Linear(hidden_dim*2, args.OUTPUT_DIM)
        else:
            self.fc = nn.Linear(hidden_dim, args.OUTPUT_DIM)

        if args.using_dann == True:
            self.grl = GRL()

        if args.using_rl == True:
            self.fc_rl = nn.Linear(hidden_dim*2, hidden_dim*2)
            #self.bernoulli = Bernoulli()

        self.fc_domain = nn.Linear(hidden_dim*2, 7)
        self.args = args

    def forward(self, sentence,*args,train =True, using_rl = False):
        batch_size = sentence.size(0)
        sentence = sentence.long()
        embeds = self.word_embeddings(sentence)
        #    lstm_out, hidden = self.lstm(embeds, hidden)
        lstm_out, _ = self.lstm(embeds.view(batch_size, len(sentence[0]), -1))
        lstm_out = lstm_out[:,-1]

        domain_pre = 0
        if self.args.using_dann == True and train==True:
            lstm_out = self.grl(lstm_out,args)

        if self.args.multile == True:
            domain_pre = self.fc_domain(lstm_out)
            domain_pre = F.softmax(domain_pre[:,-1],dim=-1)

        if using_rl == True:
            self.action, self.logpro, self.entrop = self.rl(lstm_out)
            lstm_out = lstm_out * self.action

        tag_space = self.fc(lstm_out)
        tag_space = torch.sigmoid(tag_space)
        return tag_space, domain_pre, lstm_out

    def rl(self,x):

        x = torch.sigmoid(self.fc_rl(x))
        m = Bernoulli(x)
        action = m.sample()
        logpro = torch.exp(m.log_prob(action))
        entrop = m.entropy()

        return action, logpro, entrop

    def cal_loss(self,reward,predictions,log_softmax, entropy):

        #baseline
        reward_mean = torch.mean(reward, dim=0)

        # Actor learning rate,'reinforce'
        reward_baseline = reward - reward_mean - predictions.squeeze()
        loss = - torch.mean(torch.mean(reward_baseline.unsqueeze(1) * log_softmax + entropy,dim=1), dim=0)

        return loss




