import torch
import torch.nn as nn
from model.DANN import GRL
import torch.nn.functional as F
from transformers import BertModel


class BertDann(nn.Module):
    def __init__(self,config,args):
        super(BertDann, self).__init__()

        self.bert = BertModel(config, add_pooling_layer=True)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.args = args

        if args.using_dann == True:
            self.grl = GRL()
            self.fc_domain = nn.Linear(config.hidden_size, 7)

    def forward(self,
                input_ids,
                attention_mask,
                *args,train):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask)

        sequence_output = torch.mean(outputs[0],dim=1)
        domain_pre = 0
        if self.args.using_dann == True and train==True:
            lstm_out = self.grl(sequence_output,args)
            domain_pre = self.fc_domain(lstm_out)
            domain_pre = F.softmax(domain_pre,dim=-1)

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        return logits, domain_pre