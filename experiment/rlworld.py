import sys
sys.path.append('/home/ywu10/Documents/MoralCausality/')
import torch
import numpy as np
import argparse
from model.actorcritic import Critic,CalReward
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
parser.add_argument('--N_EPOCHS', default=180, type=int)
parser.add_argument('--START_TUNE', default=80, type=int)
parser.add_argument('--HIDDEN_DIM', default=300, type=int)
parser.add_argument('--OUTPUT_DIM', default=11, type=int)
parser.add_argument('--VOCAB_MIN_FREQ', default=1, type=int)
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--model', default='lstm', type=str)
parser.add_argument('--using_glove', default=True)
parser.add_argument('--using_dann', default=False)
parser.add_argument('--using_rl', default=True)
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
    train = WordDataset(args,TEXT,LABEL,LABEL2,source,label,source_area='Sandy',target_area='Davidson',train=True,tune=False)
    test = WordDataset(args,TEXT,LABEL,LABEL2,source,label,source_area='Sandy',target_area='Davidson',train=False,tune=False)
    test2 = WordDataset(args,TEXT,LABEL,LABEL2,source,label,source_area='Sandy',target_area='Sandy',train=False,tune=False)
    tune = WordDataset(args,TEXT,LABEL,LABEL2,source,label,source_area='Sandy',target_area='Davidson',train=False,tune=True)

    LABEL.build_vocab(train,test,test2, tune)
    LABEL2.build_vocab(train,test,test2, tune)

    if args.using_glove == False:
        TEXT.build_vocab(train,test,test2, tune, max_size=args.VOCAB_MAX_SIZE, min_freq=args.VOCAB_MIN_FREQ)
    else:
        TEXT.build_vocab(train,test,test2, tune, max_size=args.VOCAB_MAX_SIZE, min_freq=args.VOCAB_MIN_FREQ,vectors="glove.6B.300d")

    if args.model == 'lstm':
        model = lstm.LSTMTagger(TEXT.vocab,args)
        optimizer = optim.Adam(model.parameters(), args.lr)
        if args.using_rl == True:
            critic = Critic(args)
            critic = critic.cuda()
            optimizer_critic = torch.optim.Adam(critic.parameters(), lr=0.001, betas=(0.9, 0.99), eps=0.0000001)
            optimizer_rl = torch.optim.Adam(model.fc_rl.parameters(), lr=0.001, betas=(0.9, 0.99), eps=0.0000001)

    else:
        model = fasttext.FastText(len(TEXT.vocab), args)
        optimizer = optim.Adam(model.parameters(),lr=args.lr)

    train_iter, test_iter, test_iter2, tune = data.BucketIterator.splits(
        (train, test, test2, tune),
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
    best = 10
    best_loss = 10000
    for epoch in range(1, args.N_EPOCHS+1):

        #set metric accumulators
        epoch_loss = 0
        epoch_acc = 0
        model.train()
        #if epoch < args.START_TUNE:
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
                loss = F.binary_cross_entropy(predictions,y,reduction='none') * mask
                loss = torch.sum(loss)/torch.sum(mask)
                loss1 = 0.3*F.cross_entropy(domain, source)
                loss = loss1 + loss
            loss.backward()
            optimizer.step()

        if epoch > args.START_TUNE and args.using_rl:
            #if epoch == args.START_TUNE:
            #model_dict = model.load_state_dict(torch.load('model.pkl'))

            for idx, batch in enumerate(tune):

                x = batch.review.cuda()
                y = batch.label.cuda()
                source = batch.source.cuda()
                optimizer_rl.zero_grad()
                optimizer_critic.zero_grad()

                prediction_old,_,_ = model(x,train=True,using_rl=False)
                prediction_new,_,x = model(x,train=True,using_rl=True)
                action, logpro, entropy = \
                    model.action, model.logpro, model.entrop
                value = CalReward().reward(prediction_old,prediction_new,y,source_num,source)
                prediction = critic(x,prediction_old,prediction_new)
                loss1 = F.mse_loss(prediction.squeeze(),value)
                loss1.backward(retain_graph=True)
                optimizer_critic.step()
                prediction = critic(x,prediction_old,prediction_new)
                loss2 = model.cal_loss(value,prediction,logpro,entropy)
                loss2.backward()
                optimizer_rl.step()

        if epoch > args.START_TUNE and args.using_rl:
            using_rl = True
        else:
            using_rl = False
        labelss = []
        with torch.no_grad():
            recall = 0
            precision = 0
            TP = 0
            for idx, batch in enumerate(test_iter):

                x = batch.review.cuda()
                y = batch.label.cuda()
                prediction,_,_ = model(x,using_rl=using_rl)
                ll = y.cpu().tolist()
                labelss.append(ll)
                aa = (prediction > 0.5).float()
                TP = TP + torch.sum(aa * y)
                recall = recall + torch.sum(y)
                precision = precision + torch.sum(aa)
            precision = TP/precision
            recall = TP/recall
            F1 = 2 * (precision*recall)/(precision+recall)
            #if best < F1:
            #    best_loss = F1
            #    torch.save(model.state_dict(),'model.pkl')
            F1 = F1.cpu()
            target_pre.append(float(F1))
            logger.info('test epoch: {}, F1: {}, precision:{}. recall:{}'.format(epoch, F1, precision, recall))

            recall = 0
            precision = 0
            TP = 0
            for idx, batch in enumerate(test_iter2):

                x = batch.review.cuda()
                y = batch.label.cuda()
                prediction,_,_ = model(x,using_rl=using_rl)
                ll = y.cpu().tolist()
                labelss.append(ll)
                aa = (prediction > 0.5).float()
                TP = TP + torch.sum(aa * y)
                recall = recall + torch.sum(y)
                precision = precision + torch.sum(aa)
            precision = TP/precision
            recall = TP/recall
            F2 = 2 * (precision*recall)/(precision+recall)
            if best > abs(F1-F2): #+ F1 + F2:
                best_loss = abs(F1-F2) #+ F1 + F2
                torch.save(model.state_dict(),'model.pkl')
            F2 = F2.cpu()
            source_pre.append(float(F2))
            logger.info('sandy test epoch: {}, F1: {}, precision:{}. recall:{}'.format(epoch, F2, precision, recall))

            print('------')

    x = np.arange(0,len(source_pre),1)
    #plt.scatter(x,tt,color=(0.1,0.,0.),label='train')
    plt.scatter(x,source_pre,color=(0.8,0.,0.),label='source')
    plt.scatter(x,target_pre,color=(0.,0.5,0.),label='target')
    plt.legend()
    plt.title('Training with RL',color='black')
    plt.savefig('/home/ywu10/Documents/MoralCausality/img/rl.jpg')
    plt.show()