import torch.nn as nn
from model.DANN import GRL
from model.rl import RL
import torch
from transformers import BertModel
from model.classifier import Classifier
import torch.nn.functional as F


class EncoderBase(nn.Module):
    def __init__(self, hidden_dim,args):
        super().__init__()
        if args.train == 'train_dann':
            self.fc_domain = GRL()

        elif args.train == 'train_dta':
            self.fc_domain = nn.Sequential(nn.Dropout(0.4),nn.Linear(hidden_dim*2, hidden_dim*2))

        self.domain_pre = nn.Linear(hidden_dim, 1)
        self.args = args

    def da(self,x,*args):

        if self.args.train == 'train_dann':
            x = self.fc_domain(x,args)

        elif self.args.train == 'train_dta':
            x1 = self.fc_domain(x)
            x2 = self.fc_domain(x)
            x = x1,x2

        domain = torch.sigmoid(self.domain_pre(x)).squeeze()

        return x, domain


class FastText(EncoderBase):
    def __init__(self, hidden_dim, args, input_dim): #hidden_dim is the dimension goes into classifier
        super(FastText,self).__init__(hidden_dim, args,)

        self.embedding = nn.Embedding(input_dim, args.HIDDEN_DIM)

    def forward(self, x,*args):

        x = self.embedding(x)
        x = F.avg_pool2d(x, (x.shape[1], 1)).squeeze(1)
        domain = self.da(x,args)
        return x, domain

class LSTMTagger(EncoderBase):

    def __init__(self,vocab, args, embedding_dim=300, hidden_dim=256,n_layers=1):
        super(LSTMTagger, self).__init__(hidden_dim*2, args)
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.word_embeddings = nn.Embedding.from_pretrained(vocab.vectors)
        self.word_embeddings.weight.requires_grad = True
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first = True, bidirectional=True,
                            num_layers = n_layers)
        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim,  args.OUTPUT_DIM)

    def forward(self, sentence,*args):

        batch_size = sentence.size(0)
        sentence = sentence.long()
        embeds = self.word_embeddings(sentence)
        #    lstm_out, hidden = self.lstm(embeds, hidden)
        lstm_out, _ = self.lstm(embeds.view(batch_size, len(sentence[0]), -1))
        lstm_out = lstm_out[:,-1]
        domain = self.da(lstm_out,args)
        return lstm_out, domain

class Bert(EncoderBase):
    def __init__(self,config):
        super(EncoderBase, self).__init__()

        self.bert = BertModel(config, add_pooling_layer=True)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)

    def forward(self,input_ids,attention_mask):

        outputs = self.bert(input_ids,attention_mask=attention_mask)

        sequence_output = torch.mean(outputs[0],dim=1)
        sequence_output = self.dropout(sequence_output)
        domain = self.da(sequence_output)

        return sequence_output, domain

