import torch
from torch import nn
from transformers import BertConfig, BertForSequenceClassification

class Gumbel(nn.Module):
    def __init__(self):

        super(Gumbel, self).__init__()

        self.embedding_u = nn.Embedding(11,11)
        torch.nn.init.xavier_uniform_(self.embedding_u.weight)
        self.fc_fea = nn.Linear(11,11)

    def forward(self,source, freq):

        alpha = self.embedding_u(source)
        x = self.fc_fea(freq)
        x = torch.exp(torch.exp(-(x-alpha)))
        return x

class GumbelBert(nn.Module):
    def __init__(self):

        super(GumbelBert, self).__init__()

        config = BertConfig.from_pretrained('bert-base-uncased')
        config.num_labels = 100
        self.bert = BertForSequenceClassification(config)
        self.gumbel = Gumbel()
        self.combine = nn.Linear(111,11)

    def forward(self,review,mask,source, freq):
        review = self.bert(input_ids=review, attention_mask=mask).logits
        x = self.gumbel(source,freq)
        x = self.combine(torch.cat((review,x),dim=-1))
        return x


