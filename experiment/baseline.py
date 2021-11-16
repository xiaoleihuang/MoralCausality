import torch
import sys
sys.path.append('/home/ywu10/Documents/MoralCausality/')
from loguru import logger
from tqdm import tqdm
import torch.nn as nn
from model.bertdann import BertDann
from dataloader.analysis import Analysis
import torch.nn.functional as F
import argparse
from dataloader.preprogress import BertDataLoader
from transformers import BertConfig, BertForSequenceClassification, AdamW


parser = argparse.ArgumentParser(
    description='PyTorch TruncatedLoss')

parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--device', default=0, type=int)
parser.add_argument('--epochs', default=150, type=int)
parser.add_argument('--hidden_size',default=100,type=int)
parser.add_argument('--model', default='bertdan', type=str)
parser.add_argument('--using_dann', default=True)
parser.add_argument('--test_file', default='dataset/sentiment/orig/test.tsv', type=str)
args = parser.parse_args()

if __name__ == '__main__':

    analysis = Analysis()
    dist, label, source = analysis.dist, analysis.label, analysis.source
    train_dataset = BertDataLoader(source,label,dist,train=True)
    test_dataset = BertDataLoader(source,label,dist,train=False)
    trainloader = torch.utils.data.DataLoader(train_dataset,
                            batch_size=args.batch_size, shuffle=True, num_workers=10)
    testloader = torch.utils.data.DataLoader(test_dataset,
                             batch_size=args.batch_size, shuffle=False, num_workers=10)

    path = 'bert' + str(args.using_dann) +'.pkl'
    config = BertConfig.from_pretrained('bert-base-uncased')
    config.num_labels = 11
    model = BertDann(config,args).cuda(args.device)
    optimizer = AdamW(model.parameters(),lr=1e-6)
    best_F = 0
    for epoch in range(1,args.epochs+1):
        model.train()
        for batch_id, (tid_code, source_code, sentiment, reviews, mask,dist) in tqdm(enumerate(trainloader)):
            reviews, mask, sentiment, source_code = reviews.cuda(args.device), \
                            mask.cuda(args.device), sentiment.cuda(args.device), source_code.cuda(args.device)
            optimizer.zero_grad()
            p = torch.tensor(epoch/(args.epochs+1)).cuda(args.device)
            #p = torch.tensor(1).cuda()
            train = True if epoch<80 else False
            predictions,domain = model(reviews, mask, p, train=train)
            loss = F.binary_cross_entropy_with_logits(predictions,sentiment)
            if args.using_dann == True and train==True:
                loss1 = 0.3*F.cross_entropy(domain, source_code)
                loss = loss1 + loss

            loss.backward()
            optimizer.step()

        with torch.no_grad():
            recall = 0
            precision = 0
            TP = 0
            for batch_id, (tid_code, source_code, sentiment, reviews, mask,dist) in enumerate(trainloader):
                reviews, mask, sentiment = reviews.cuda(args.device), mask.cuda(args.device), sentiment.cuda(args.device)
                optimizer.zero_grad()
                prediction,_ = model(reviews, attention_mask=mask,train=False)
                prediction = F.sigmoid(prediction)
                aa = (prediction > 0.5).float()
                TP = TP + torch.sum(aa * sentiment)
                recall = recall + torch.sum(aa)
                precision = precision + torch.sum(sentiment)
            precison = TP/precision
            recall = TP/recall
            F1 = 2 * (precison*recall)/(precison+recall)
            logger.info('epoch: {}, F1: {}, precision:{}. recall:{}'.format(epoch, F1, precison, recall))
            if F1>best_F:
                best_F = F1
                torch.save(model.state_dict(),path)

    print(args.using_dann)
