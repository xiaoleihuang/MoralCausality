import sys
sys.path.append('/home/ywu10/Documents/MoralCausality/')
import torch
import numpy as np
import argparse
import pickle
from loguru import logger
import torch.optim as optim
from dataloader.analysis import Analysis
from model import fasttext, lstm
import torch.optim.lr_scheduler as lr_scheduler
from dataloader.preprogress import WordDataset
from torchtext.legacy import data, datasets
import torch.nn.functional as F
import matplotlib.pyplot as plt
from experiment.visualization import paint

parser = argparse.ArgumentParser(
    description='PyTorch TruncatedLoss')

parser.add_argument('--N_GRAMS', default=1, type=int)
parser.add_argument('--MAX_LENGTH', default=None, type=int)
parser.add_argument('--VOCAB_MAX_SIZE', default=None, type=int)
parser.add_argument('--BATCH_SIZE', '-b', default=32,type=int)
parser.add_argument('--N_EPOCHS', default=100, type=int)
parser.add_argument('--HIDDEN_DIM', default=300, type=int)
parser.add_argument('--OUTPUT_DIM', default=11, type=int)
parser.add_argument('--VOCAB_MIN_FREQ', default=1, type=int)
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--model', default='lstm', type=str)
parser.add_argument('--using_glove', default=True)
parser.add_argument('--using_dann', default=False)
parser.add_argument('--using_rl', default=False)
parser.add_argument('--multile', default=False)
parser.add_argument('--stop_domain', default=180)
args = parser.parse_args()


def generate_n_grams(x):

    N_GRAMS = 1
    if args.N_GRAMS <= 1: #no need to create n-grams if we only want uni-grams
        return x
    else:
        n_grams = set(zip(*[x[i:] for i in range(N_GRAMS)]))
        for n_gram in n_grams:
            x.append(' '.join(n_gram))
        return x

if __name__ == '__main__':

    TOKENIZER = lambda s: s.split( )
    TEXT = data.Field(batch_first=True, tokenize=TOKENIZER, preprocessing=generate_n_grams, fix_length=args.MAX_LENGTH)
    LABEL = data.Field(batch_first=True, use_vocab = False,sequential=True)
    LABEL2 = data.Field(batch_first=True,use_vocab = False,sequential=False)

    dist, label, source = Analysis().distribution()
    source_num = source.index('Sandy')
    train = WordDataset(args,TEXT,LABEL,LABEL2,source,label,source_area='Sandy',target_area='Davidson',train=True,tune=False,split=True)
    test = WordDataset(args,TEXT,LABEL,LABEL2,source,label,source_area='Sandy',target_area='Davidson',train=False,tune=False,split=False)
    test2 = WordDataset(args,TEXT,LABEL,LABEL2,source,label,source_area='Sandy',target_area='Sandy',train=False,tune=False,split=True)

    LABEL.build_vocab(train,test, test2)
    LABEL2.build_vocab(train,test, test2)

    if args.using_glove == False:
        TEXT.build_vocab(train,test, test2, max_size=args.VOCAB_MAX_SIZE, min_freq=args.VOCAB_MIN_FREQ)
    else:
        TEXT.build_vocab(train,test, test2, max_size=args.VOCAB_MAX_SIZE, min_freq=args.VOCAB_MIN_FREQ,vectors="glove.6B.300d")

    if args.model == 'lstm':
        model = lstm.LSTMTagger(TEXT.vocab,args)
        optimizer = optim.Adam(model.parameters(), args.lr)

    else:
        model = fasttext.FastText(len(TEXT.vocab), args)
        optimizer = optim.Adam(model.parameters(),lr=args.lr)

    train_iter, test_iter,test_iter2 = data.BucketIterator.splits(
        (train, test,test2),
        batch_size=args.BATCH_SIZE,
        sort_key=lambda x: len(x.review),
        device = None if torch.cuda.is_available() else -1, #device needs to be -1 for CPU, else use default GPU
        repeat=False)

    model = model.cuda()

    #initialize optimizer, scheduler and loss function
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer)
    source_pre = []
    target_pre = []
    tt = []
    best_acc = 0
    for epoch in range(1, args.N_EPOCHS+1):

        #set metric accumulators
        epoch_loss = 0
        epoch_acc = 0
        model.train()
        for idx, batch in enumerate(train_iter):

            x = batch.review.cuda()
            y = batch.label.float().cuda()
            source = batch.source.cuda()

            optimizer.zero_grad()
            p = torch.tensor(1/(args.N_EPOCHS+1)).cuda()
            train = True if epoch<args.stop_domain else False
            predictions,domain,_ = model(x, p, train=train)
            loss = F.binary_cross_entropy(predictions,y)
            if (args.using_dann == True and train==True) or args.multile:
                mask = (source == source_num).float().cuda().unsqueeze(-1)
                loss = F.binary_cross_entropy_with_logits(predictions,y,reduction='none') * mask
                loss = torch.sum(loss)/torch.sum(mask)
                loss1 = 0.3*F.cross_entropy(domain, source)
                loss = loss1 + loss
            loss.backward()
            optimizer.step()
            '''
            if loss < best_loss:
                best_loss = loss
                torch.save(model.state_dict(),'model.pkl')
            '''

        '''
        embeds = []
        labelss = []
        with torch.no_grad():
            recall = 0
            precision = 0
            TP = 0
            for idx, batch in enumerate(train_iter):

                x = batch.review.cuda()
                y = batch.label.cuda()
                prediction,_,embed = model(x,train=False)
                embed = embed.cpu().tolist()
                embeds += embed
                ll = y.cpu().tolist()
                labelss.append(ll)
                prediction = F.sigmoid(prediction)
                aa = (prediction > 0.5).float()
                TP = TP + torch.sum(aa * y)
                recall = recall + torch.sum(y)
                precision = precision + torch.sum(aa)
            precision = TP/precision
            recall = TP/recall
            F1 = 2 * (precision*recall)/(precision+recall)
            tt.append(F1.cpu())
            logger.info('epoch: {}, F1: {}, precision:{}. recall:{}'.format(epoch, F1, precision, recall))
        #model_dict=model.load_state_dict(torch.load('model.pkl'))
        '''

        embeds = []
        labelss = []
        with torch.no_grad():
            recall = 0
            precision = 0
            TP = 0
            for idx, batch in enumerate(test_iter):

                x = batch.review.cuda()
                y = batch.label.cuda()
                prediction,_,embed = model(x,train=False)
                embed = embed.cpu().tolist()
                embeds += embed
                ll = y.cpu().tolist()
                labelss.append(ll)
                #prediction = F.sigmoid(prediction)
                aa = (prediction > 0.5).float()
                TP = TP + torch.sum(aa * y)
                recall = recall + torch.sum(y)
                precision = precision + torch.sum(aa)
            precision = TP/precision
            recall = TP/recall
            F1 = 2 * (precision*recall)/(precision+recall)
            target_pre.append(F1.cpu())
            logger.info('epoch: {}, F1: {}, precision:{}. recall:{}'.format(epoch, F1, precision, recall))
            #paint(embeds,label)

            recall = 0
            precision = 0
            TP = 0
            embeds = []
            labelss = []
            for idx, batch in enumerate(test_iter2):

                x = batch.review.cuda()
                y = batch.label.cuda()
                prediction,_,embed = model(x,train=False)
                #prediction = F.sigmoid(prediction)
                aa = (prediction > 0.5).float()
                TP = TP + torch.sum(aa * y)
                recall = recall + torch.sum(y)
                precision = precision + torch.sum(aa)
                embed = embed.cpu().tolist()
                embeds +=embed
                ll = y.cpu().tolist()
                labelss+=ll
            precision = TP/precision
            recall = TP/recall
            F2 = 2 * (precision*recall)/(precision+recall)
            source_pre.append(F2.cpu())
            logger.info('epoch: {}, F2: {}, precision:{}. recall:{}'.format(epoch, F2, precision, recall))
            print('---------')
            #paint(embeds,label)

            if best_acc < F1+F2:
                best_acc = F1 + F2
                torch.save(model.state_dict(),'wordmodel.pkl')

    #保存
    model_dict=model.load_state_dict(torch.load('wordmodel.pkl'))
    embeds = []
    labelss = []
    with torch.no_grad():
        recall = 0
        precision = 0
        TP = 0
        for idx, batch in enumerate(train_iter):

            x = batch.review.cuda()
            y = batch.label.cuda()
            prediction,_,embed = model(x,train=False)
            embed = embed.cpu().tolist()
            embeds += embed
            ll = y.cpu().tolist()
            labelss.append(ll)

        embeds = np.array(embeds)
        labelss = sum(labelss,[])
        labelss = np.array(labelss)
        a = embeds, labelss
        with open("train_source_word.npy", 'wb') as fo:
            pickle.dump(a, fo)

        recall = 0
        precision = 0
        TP = 0
        embeds = []
        labelss = []
        for idx, batch in enumerate(test_iter):

            x = batch.review.cuda()
            y = batch.label.cuda()
            prediction,_,embed = model(x,train=False)
            embed = embed.cpu().tolist()
            embeds += embed
            ll = y.cpu().tolist()
            labelss.append(ll)

        embeds = np.array(embeds)
        labelss = sum(labelss,[])
        labelss = np.array(labelss)
        a = embeds, labelss
        with open("target_word.npy", 'wb') as fo:
            pickle.dump(a, fo)

        recall = 0
        precision = 0
        TP = 0
        embeds = []
        labelss = []
        for idx, batch in enumerate(test_iter2):

            x = batch.review.cuda()
            y = batch.label.cuda()
            prediction,_,embed = model(x,train=False)
            #prediction = F.sigmoid(prediction)
            aa = (prediction > 0.5).float()
            TP = TP + torch.sum(aa * y)
            recall = recall + torch.sum(y)
            precision = precision + torch.sum(aa)
            embed = embed.cpu().tolist()
            embeds += embed
            ll = y.cpu().tolist()
            labelss+=ll

        embeds = np.array(embeds)
        #labelss = sum(labelss,[])
        labelss = np.array(labelss)
        a = embeds, labelss

        with open("test_source_word.npy", 'wb') as fo:
            pickle.dump(a, fo)


    x = np.arange(0,len(source_pre),1)
    #plt.scatter(x,tt,color=(0.1,0.,0.),label='train')
    plt.scatter(x,source_pre,color=(0.8,0.,0.),label='source')
    plt.scatter(x,target_pre,color=(0.,0.5,0.),label='target')
    plt.legend()
    plt.show()

