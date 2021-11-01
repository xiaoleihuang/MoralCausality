import json
import numpy as np
import pandas as pd
import torch
import pickle
import os
import torchtext as text
from torchtext import data
from torchtext.legacy import data, datasets
from torchtext.vocab import Vectors
from transformers import BertTokenizer

def preprogress():
    file_address = '/data/datasets/moralcausality/MFTC_V4_text.json'
    file_ = 'data/MFTC_V4_text_preprogress.tsv'
    f = open(file_address)
    data = json.load(f)
    tid = []
    text = []
    source = []
    labels = []
    for corpus in data:
        for id in corpus['Tweets']:
            label = []
            for ll in id['annotations']:
                label += ll['annotation'].split(',')
            rep = list(set(label))
            label_vote = []
            for ll in rep:
                if (np.array(label) == ll).sum() > 1:
                    label_vote.append(ll)
            label_vote = ','.join(label_vote)
            if len(label_vote) > 0:
                source.append(corpus['Corpus'])
                tid.append(id['tweet_id'])
                text.append(id['tweet_text'])
                labels.append(label_vote)
    data = pd.DataFrame({'tid':tid, 'text':text, 'source':source, 'label':labels})
    data.to_csv(file_, index = False)

class WordDataset(data.Dataset):
    def __init__(self,args,TEXT,LABEL,file='data/MFTC_V4_text_preprogress.tsv', train=1,**kwargs):

        self.file = file
        self.train = train
        self.args = args
        self.TEXT = TEXT
        self.LABEL = LABEL
        examples, fields = self.example()

        super(WordDataset, self).__init__(examples, fields, **kwargs)

    def coding(self):

        if os.path.exists(self.file_) == False or os.path.exists(self.file__) == False:
            data = pd.read_csv(self.file)
            tid = list(set(data['tid']))
            tid_code = []
            for code in data['tid']:
                tid_code.append(tid.index(code))

            source = list(set(data['source']))
            source_code = []
            for code in data['source']:
                source_code.append(source.index(code))

            label = list(set(sum([i.split(',') for i in data['label']], [])))
            label_code = []
            for code in data['label']:
                label_ = np.zeros(len(label))
                label_rep = code.split(',')
                for la in label_rep:
                    label_[label.index(la)] = 1
                label_code.append(label_)

            reviews = data['text'].values.tolist() ##提取rating信息并保存为list格式

            train_data = pd.DataFrame({'tid':tid_code, 'review':reviews, 'source':source_code, 'label':label_code})
            train_data.to_csv(self.file_, index = False)

    def example(self):

        examples = []

        fields = [('tid',None),('review',self.TEXT), ('source',None), ('label',self.LABEL)]

        self.coding()

        if self.train:
            tsv_data = pd.read_csv(self.file_)
            split = int(len(tsv_data) * 0.8)
            tsv_data = tsv_data[:split]
        else:
            tsv_data = pd.read_csv(self.file_)
            split = int(len(tsv_data) * 0.8)
            tsv_data = tsv_data[split:]

        for text, label in zip(tsv_data['review'], tsv_data['label']):
            examples.append(data.Example.fromlist([None, text, None,label], fields))

        return examples,fields


class BertDataLoader(data.Dataset):
    def __init__(self,source, label,dist, file = '/home/ywu10/Documents/MoralCausality/data/MFTC_V4_text_preprogress.tsv', \
                 file__ = '/home/ywu10/Documents/MoralCausality/data/MFTC_V4_text_preprogress.pkl',train=True, max_len = None):
        if os.path.exists(file__) == False:
            data = pd.read_csv(file)
            tid = list(set(data['tid']))
            tid_code = []
            for code in data['tid']:
                tid_code.append(tid.index(code))

            source_code = []
            for code in data['source']:
                source_code.append(source.index(code))

            label_code = []
            for code in data['label']:
                label_ = np.zeros(len(label))
                label_rep = code.split(',')
                for la in label_rep:
                    label_[label.index(la)] = 1
                label_code.append(label_)

            if max_len == None:
                len_list = [len(i) for i in data['text']]
                max_len = int(np.percentile(len_list, 85))

            reviews = data['text'].values.tolist() ##提取rating信息并保存为list格式
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            review_ = []
            mask_ = []
            for i in range(len(reviews)):
                review = reviews[i]
                if len(reviews[i])>max_len:
                    review = review[:max_len]
                dd = tokenizer(review,return_tensors="pt")
                pad = torch.tensor(int((max_len - len(dd['input_ids'].squeeze()))) * [0])
                data = torch.cat((dd['input_ids'].squeeze(),pad), dim=0)
                mask = torch.cat((dd['attention_mask'].squeeze(),pad), dim=0)
                review_.append(data)
                mask_.append(mask)

            data = tid_code, source_code, label_code, review_, mask_
            with open(file__, 'wb') as fo:
                pickle.dump(data, fo)

        with open(file__, 'rb') as fo:
            data = pickle.load(fo)

        self.train = train
        self.dist = dist
        self.source = source
        split = int(len(data[0]) * 0.8)
        self.train_data = data[0][:split],data[1][:split],data[2][:split],data[3][:split],data[4][:split]
        self.test_data = data[0][:split],data[1][:split],data[2][:split],data[3][:split],data[4][:split]

    def __getitem__(self, index):

        data = self.train_data if self.train else self.test_data

        source = data[1][index]
        dist = self.dist[self.source[source]]
        return data[0][index], source, data[2][index], \
               data[3][index], data[4][index], dist


    def __len__(self):
        return len(self.train_data[0]) if self.train == True else len(self.test_data[0])
