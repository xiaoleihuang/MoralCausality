import torch
import argparse
from loguru import logger
import torch.optim as optim
from model import fasttext, lstm
import torch.optim.lr_scheduler as lr_scheduler
from dataloader.preprogress import WordDataset
from torchtext.legacy import data, datasets
import torch.nn.functional as F

parser = argparse.ArgumentParser(
    description='PyTorch TruncatedLoss')

parser.add_argument('--N_GRAMS', default=1, type=int)
parser.add_argument('--MAX_LENGTH', default=None, type=int)
parser.add_argument('--VOCAB_MAX_SIZE', default=None, type=int)
parser.add_argument('--BATCH_SIZE', '-b', default=32,type=int)
parser.add_argument('--N_EPOCHS', default=240, type=int)
parser.add_argument('--HIDDEN_DIM', default=300, type=int)
parser.add_argument('--VOCAB_MIN_FREQ', default=1, type=int)
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--model', default='fasttext', type=str)
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
    LABEL = data.Field(batch_first=True, pad_token=None, unk_token=None)

    train = WordDataset(args,TEXT,LABEL,train=True)
    test = WordDataset(args,TEXT,LABEL,train=False)
    if args.model == 'lstm':
        TEXT.build_vocab(train,test, max_size=args.VOCAB_MAX_SIZE, min_freq=args.VOCAB_MIN_FREQ,vectors="glove.6B.300d")
        model = lstm.LSTMTagger(TEXT.vocab.vectors)
        params = list(map(id,model.word_embeddings.parameters()))
        rest_params = filter(lambda x:id(x) not in params,model.parameters())
        optimizer = optim.Adam([{'params':model.word_embeddings.parameters(),'lr':1e-6},
                                {'params':rest_params,'lr':args.lr},])

    else:
        TEXT.build_vocab(train,test, max_size=args.VOCAB_MAX_SIZE, min_freq=args.VOCAB_MIN_FREQ)
        model = fasttext.FastText(len(TEXT.vocab), args.HIDDEN_DIM, 11)
        optimizer = optim.Adam(model.parameters())
    LABEL.build_vocab(train,test)

    train_iter, test_iter = data.BucketIterator.splits(
        (train, test),
        batch_size=args.BATCH_SIZE,
        sort_key=lambda x: len(x.review),
        device = None if torch.cuda.is_available() else -1, #device needs to be -1 for CPU, else use default GPU
        repeat=False)

    #initialize model
    model = model.cuda()

    #initialize optimizer, scheduler and loss function
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer)

    for epoch in range(1, args.N_EPOCHS+1):

        #set metric accumulators
        epoch_loss = 0
        epoch_acc = 0
        model.train()
        for idx, batch in enumerate(train_iter):

            x = batch.review
            y = batch.label

            optimizer.zero_grad()
            x = x.cuda()
            y = y.cuda()
            y = (y>0).float()
            predictions = model(x)
            loss = F.binary_cross_entropy_with_logits(predictions,y)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            recall = 0
            precision = 0
            TP = 0
            for idx, batch in enumerate(test_iter):


                x = batch.review.cuda()
                y = batch.label.cuda()
                y = (y>0).float()
                predictions = model(x)

                prediction = F.sigmoid(predictions)
                aa = (prediction > 0.5).float()
                TP = TP + torch.sum(aa * y)
                recall = recall + torch.sum(aa)
                precision = precision + torch.sum(y)
            precison = TP/precision
            recall = TP/recall
            F1 = 2 * (precison*recall)/(precison+recall)
            logger.info('epoch: {}, F1: {}, precision:{}. recall:{}'.format(epoch, F1, precison, recall))