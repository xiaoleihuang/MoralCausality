import json
import numpy as np
import pandas as pd
import torch
import pickle
import os
import torchtext as text
from torchtext import data
import torch.nn.functional as F
from torchtext.legacy import data, datasets
from torchtext.vocab import Vectors
from transformers import BertTokenizer

def preprogress():
    file_address = '/home/ywu10/Documents/MoralCausality/data/datasets/moralcausality/MFTC_V4_text.json'
    file_ = '/home/ywu10/Documents/MoralCausality/data/MFTC_V4_text_preprogress.tsv'
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
    def __init__(self,args,TEXT,LABEL, source,label, source_area, target_area, \
                 file='/home/ywu10/Documents/MoralCausality/data/MFTC_V4_text_preprogress.tsv', train=True,**kwargs):

        self.file = file
        self.train = train
        self.args = args
        self.TEXT = TEXT
        self.LABEL = LABEL
        self.source = source
        self.label = label
        self.source_area = source_area
        self.target_area = target_area
        self.file_ = '/home/ywu10/Documents/MoralCausality/data/MFTC_V4_text_word.csv'
        examples, fields = self.example()

        super(WordDataset, self).__init__(examples, fields, **kwargs)

    def coding(self):

        if os.path.exists(self.file_) == False:
            data = pd.read_csv(self.file)
            tid = list(set(data['tid']))
            tid_code = []
            for code in data['tid']:
                tid_code.append(tid.index(code))

            source_code = []
            for code in data['source']:
                source_code.append(code)

            reviews = data['text'].values.tolist() ##提取rating信息并保存为list格式

            train_data = pd.DataFrame({'tid':tid_code, 'review':reviews, 'source':source_code, 'label':data['label']})
            train_data.to_csv(self.file_, index = False)

    def example(self):

        examples = []

        fields = [('tid',None),('review_s',self.TEXT), ('label_s',self.LABEL), ('review_t',self.TEXT), ('label_t',self.LABEL)]

        self.coding()
        tsv_data = pd.read_csv(self.file_)

        if self.train:
            data_s = tsv_data[tsv_data['source'] == self.source_area]
            data_t = tsv_data[tsv_data['source'] == self.target_area]
            split = int(min(len(data_s),len(data_t)) * 0.8)
            data_s = data_s[:split]
            data_t = data_t[:split]

        else:
            data_s = tsv_data[tsv_data['source'] == self.target_area]
            data_t = tsv_data[tsv_data['source'] == self.source_area]
            length = min(int(len(data_s) * 0.2), int(len(data_t) * 0.2))
            data_s = data_s[-length:-1]
            data_t = data_t[-length:-1]

        for text_s, label_s, text_t, label_t in zip(data_s['review'], data_s['label'], data_t['review'], data_t['label']):
            labels = 11*[0]
            label_rep = label_s.split(',')
            for la in label_rep:
                labels[self.label.index(la)] = 1

            labelt = 11*[0]
            label_rep = label_s.split(',')
            for la in label_rep:
                labelt[self.label.index(la)] = 1

            examples.append(data.Example.fromlist([None, text_s, labels, text_t, labelt], fields))

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
        tsv_data = pd.DataFrame({'tid':data[0], 'review':data[3],'mask':data[4], 'source':data[1], 'label':data[2]})
        datas = []
        for ss in range(len(self.source)):
            a = tsv_data[tsv_data['source'] == ss]
            split = int(len(a) * 0.8)
            if self.train:
                datas.append(a[:split])
            else:
                datas.append(a[split:])
        datas = pd.concat(datas)
        self.datas = datas['tid'].tolist(),datas['source'].tolist(), \
                     datas['label'].tolist(),datas['review'].tolist(),datas['mask'].tolist()

    def __getitem__(self, index):

        data = self.datas
        source = data[1][index]
        dist = self.dist[self.source[source]]
        return data[0][index], source, data[2][index], \
               data[3][index], data[4][index], dist

    def __len__(self):
        return len(self.datas[0])