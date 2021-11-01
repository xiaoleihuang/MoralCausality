import torch
import sys
sys.path.append('/home/ywu10/Documents/MoralCausality/')
from loguru import logger
from tqdm import tqdm
import torch.nn as nn
from dataloader.analysis import Analysis
from model.gumbel import GumbelBert
import torch.nn.functional as F
from dataloader.preprogress import BertDataLoader
from transformers import BertConfig, BertForSequenceClassification, AdamW

if __name__ == '__main__':

    batch_size = 16
    device = 0
    epochs = 100
    hidden_size = 100
    model = 'bert_gumbel'
    test_file = 'dataset/sentiment/orig/test.tsv'

    analysis = Analysis()
    dist, label, source = analysis.dist, analysis.label, analysis.source
    train_dataset = BertDataLoader(source,label,dist,train=True)
    test_dataset = BertDataLoader(source,label,dist,train=False)
    trainloader = torch.utils.data.DataLoader(train_dataset,
                            batch_size=batch_size, shuffle=True, num_workers=10)
    testloader = torch.utils.data.DataLoader(test_dataset,
                             batch_size=batch_size, shuffle=False, num_workers=10)

    if model == 'bert':
        config = BertConfig.from_pretrained('bert-base-uncased')
        config.num_labels = 11
        model = BertForSequenceClassification(config).cuda(device)
        optimizer = AdamW(model.parameters(),lr=1e-6)

    else:
        model = GumbelBert().cuda(device)
        optimizer = AdamW([{'params':model.gumbel.parameters(),'lr':1e-4},
                            {'params':model.bert.parameters(),'lr':1e-6},])


    for epoch in range(epochs):
        model.train()
        for batch_id, (tid_code, source_code, sentiment, reviews, mask,dist) in enumerate(trainloader):
            reviews, mask, sentiment = reviews.cuda(device), mask.cuda(device), sentiment.cuda(device)
            optimizer.zero_grad()
            if model == 'bert':
                output = model(input_ids=reviews, attention_mask=mask).logits
            else:
                source_code,dist = source_code.cuda(device).long(),dist.cuda(device).float()
                output = model(reviews,mask,source_code,dist)
            loss = F.binary_cross_entropy_with_logits(output,sentiment)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            recall = 0
            precision = 0
            TP = 0
            for batch_id, (tid_code, source_code, sentiment, reviews, mask,dist) in enumerate(trainloader):
                reviews, mask, sentiment = reviews.cuda(device), mask.cuda(device), sentiment.cuda(device)
                optimizer.zero_grad()
                if model == 'bert':
                    output = model(input_ids=reviews, attention_mask=mask).logits
                else:
                    source_code,dist = source_code.cuda(device).long(),dist.cuda(device).float()
                    output = model(reviews,mask,source_code,dist)
                prediction = F.sigmoid(output)
                aa = (prediction > 0.5).float()
                TP = TP + torch.sum(aa * sentiment)
                recall = recall + torch.sum(aa)
                precision = precision + torch.sum(sentiment)
            precison = TP/precision
            recall = TP/recall
            F1 = 2 * (precison*recall)/(precison+recall)
            logger.info('epoch: {}, F1: {}, precision:{}. recall:{}'.format(epoch, F1, precison, recall))
