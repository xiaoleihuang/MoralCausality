
import os
import pickle
import json

import gensim
from keras_preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from torch.utils.data import Dataset
import numpy as np
from sklearn import metrics

import torch
from transformers import BertTokenizer


def micro_f1_average(y_preds, y_truths):
    precisions = []
    recalls = []
    for idx, (y_pred, y_truth) in enumerate(zip(y_preds, y_truths)):
        true_positives = np.sum(np.logical_and(y_truth, y_pred))

        # compute the sum of tp + fp across training examples and labels
        l_prec_den = np.sum(y_pred)
        if l_prec_den != 0:
            # compute micro-averaged precision
            precisions.append(true_positives / l_prec_den)

        # compute sum of tp + fn across training examples and labels
        l_recall_den = np.sum(y_truth)

        # compute mirco-average recall
        if l_recall_den != 0:
            recalls.append(true_positives / l_recall_den)

    precisions = np.mean(precisions)
    recalls = np.mean(recalls)
    if precisions + recalls == 0:
        return 0
    f1 = 2 * precisions * recalls / (precisions + recalls)
    return precisions, recalls, f1


def label_encoder(raw_label):
    all_labels = [
        'subversion', 'loyalty', 'care', 'cheating',
        'purity', 'fairness', 'degradation', 'betrayal', 'harm', 'authority'
    ]
    encode_label = [0]*(len(all_labels) + 1)
    if type(raw_label) != str:
        encode_label[-1] = 1
        return encode_label
    for label in raw_label.split(','):
        if label not in all_labels:
            encode_label[-1] = 1
        else:
            encode_label[all_labels.index(label)] = 1
    return encode_label


class TorchDataset(Dataset):
    def __init__(self, dataset, domain_name):
        self.dataset = dataset
        self.domain_name = domain_name

    def __len__(self):
        return len(self.dataset['docs'])

    def __getitem__(self, idx):
        if self.domain_name in self.dataset:
            return self.dataset['docs'][idx], self.dataset['labels'][idx], self.dataset[self.domain_name][idx]
        else:
            return self.dataset['docs'][idx], self.dataset['labels'][idx], -1


def data_split(data):
    """
    :param data:
    :return:
    """
    data_indices = list(range(len(data['docs'])))
    np.random.seed(33)  # for reproductive results
    np.random.shuffle(data_indices)

    train_indices = data_indices[:int(.8 * len(data_indices))]
    dev_indices = data_indices[int(.8 * len(data_indices)):int(.9 * len(data_indices))]
    test_indices = data_indices[int(.9 * len(data_indices)):]
    return train_indices, dev_indices, test_indices


def cal_fpr(fp, tn):
    """False positive rate"""
    return fp / (fp + tn)


def cal_fnr(fn, tp):
    """False negative rate"""
    return fn / (fn + tp)


def cal_tpr(tp, fn):
    """True positive rate"""
    return tp / (tp + fn)


def cal_tnr(tn, fp):
    """True negative rate"""
    return tn / (tn + fp)


def fair_eval(true_labels, pred_labels, domain_labels):
    scores = {
        'fned': 0.0,  # gap between fnr
        'fped': 0.0,  # gap between fpr
        'tped': 0.0,  # gap between tpr
        'tned': 0.0,  # gap between tnr
    }

    # get overall confusion matrix
    tn, fp, fn, tp = metrics.confusion_matrix(
        y_true=true_labels, y_pred=pred_labels
    ).ravel()

    # get the unique types of demographic groups
    uniq_types = np.unique(domain_labels)
    for group in uniq_types:
        # calculate group specific confusion matrix
        group_indices = [item for item in range(len(domain_labels)) if domain_labels[item] == group]
        group_labels = [true_labels[item] for item in group_indices]
        group_preds = [pred_labels[item] for item in group_indices]

        g_tn, g_fp, g_fn, g_tp = metrics.confusion_matrix(
            y_true=group_labels, y_pred=group_preds
        ).ravel()

        # calculate and accumulate the gaps
        scores['fned'] = scores['fned'] + abs(
            cal_fnr(fn, tp) - cal_fnr(g_fn, g_tp)
        )
        scores['fped'] = scores['fped'] + abs(
            cal_fpr(fp, tn) - cal_fpr(g_fp, g_tn)
        )
        scores['tped'] = scores['tped'] + abs(
            cal_tpr(tp, fn) - cal_tpr(g_tp, g_fn)
        )
        scores['tned'] = scores['tned'] + abs(
            cal_tnr(tn, fp) - cal_tnr(g_tn, g_fp)
        )
    return json.dumps(scores)


def build_wt(tkn, emb_path, opath):
    """Build weight using word embedding"""
    embed_len = len(tkn.word_index)
    if embed_len > tkn.num_words:
        embed_len = tkn.num_words

    if emb_path.endswith('.bin'):
        embeds = gensim.models.KeyedVectors.load_word2vec_format(
            emb_path, binary=True, unicode_errors='ignore'
        )
        emb_size = embeds.vector_size
        emb_matrix = list(np.zeros((embed_len + 1, emb_size)))
        for pair in zip(embeds.wv.index2word, embeds.wv.syn0):
            if pair[0] in tkn.word_index and \
                    tkn.word_index[pair[0]] < tkn.num_words:
                emb_matrix[tkn.word_index[pair[0]]] = np.asarray([
                    float(item) for item in pair[1]
                ], dtype=np.float32)
    else:
        dfile = open(emb_path)
        line = dfile.readline().strip().split()
        if len(line) < 5:
            line = dfile.readline().strip().split()
        emb_size = len(line[1:])
        emb_matrix = list(np.zeros((embed_len + 1, emb_size)))
        dfile.close()

        with open(emb_path) as dfile:
            for line in dfile:
                line = line.strip().split()
                if line[0] in tkn.word_index and \
                        tkn.word_index[line[0]] < tkn.num_words:
                    emb_matrix[tkn.word_index[line[0]]] = np.asarray([
                        float(item) for item in line[1:]
                    ], dtype=np.float32)
    # emb_matrix = np.array(emb_matrix, dtype=np.float32)
    np.save(opath, emb_matrix)


def build_tok(docs, max_feature, opath):
    if os.path.exists(opath):
        return pickle.load(open(opath, 'rb'))
    else:
        # load corpus
        tkn = Tokenizer(num_words=max_feature)
        tkn.fit_on_texts(docs)

        with open(opath, 'wb') as wfile:
            pickle.dump(tkn, wfile)
        return tkn


class DataEncoder(object):
    def __init__(self, params, mtype='rnn'):
        """
        :param params:
        :param mtype: Model type, rnn or bert
        """
        self.params = params
        self.mtype = mtype
        if self.mtype == 'rnn':
            self.tok = pickle.load(open(
                os.path.join(params['model_dir'], params['dname'] + '.tok'), 'rb'))
        elif self.mtype == 'bert':
            self.tok = BertTokenizer.from_pretrained(params['bert_name'])
        else:
            raise ValueError('Only support BERT and RNN data encoders')

    def __call__(self, batch):
        docs = []
        labels = []
        domains = []
        for text, label, domain in batch:
            if self.mtype == 'bert':
                text = self.tok.encode_plus(
                    text, padding='max_length', max_length=self.params['max_len'],
                    return_tensors='pt', return_token_type_ids=False,
                    truncation=True,
                )
                docs.append(text['input_ids'][0])
            else:
                docs.append(text)
            labels.append(label)
            domains.append(domain)

        labels = torch.tensor(labels, dtype=torch.long)
        domains = torch.tensor(domains, dtype=torch.long)
        if self.mtype == 'rnn':
            # padding and tokenize
            docs = self.tok.texts_to_sequences(docs)
            docs = pad_sequences(docs)
            docs = torch.Tensor(docs).long()
        else:
            docs = torch.stack(docs).long()
        return docs, labels, domains
