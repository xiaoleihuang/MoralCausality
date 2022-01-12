import re
import pickle
import os
import json

import pandas as pd
import warnings
from tqdm import tqdm

from sklearn import metrics
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
import gensim

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from transformers import BertTokenizer
from torch.utils.data import Dataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# nltk.download('stopwords')
spw_set = set(stopwords.words('english'))
spw_set.add('url')
tokenizer = TweetTokenizer()
warnings.filterwarnings("ignore")


def preprocess(tweet):
    """
    Preprocess a single tweet
    :param tweet:
    :return:
    """
    global tokenizer

    # lowercase
    tweet = tweet.lower()
    # noinspection PyUnresolvedReferences
    tweet = re.sub(r"https?:\S+", "URL", tweet)  # replace url
    # replace user
    # tweet = re.sub(r'@\w+', 'USER', tweet)
    # replace hashtag
    # tweet = re.sub(r'#\S+', 'HASHTAG', tweet)
    # tokenize
    return [item.strip() for item in tokenizer.tokenize(tweet) if len(item.strip()) > 0]


def label_encoder(raw_label):
    pre_labels = [
        'subversion', 'loyalty', 'care', 'cheating',
        'purity', 'fairness', 'degradation', 'betrayal', 'harm', 'authority'
    ]
    encode_label = [0]*(len(pre_labels) + 1)
    if type(raw_label) != str:
        encode_label[-1] = 1
        return encode_label
    for label in raw_label.split(','):
        if label not in pre_labels:
            encode_label[-1] = 1
        else:
            encode_label[pre_labels.index(label)] = 1
    return encode_label


def micro_f1_average(y_preds, y_truths):
    precisions = []
    recalls = []
    for idx, (y_pred, y_truth) in enumerate(zip(y_preds, y_truths)):
        # noinspection PyUnresolvedReferences
        true_positives = np.sum(np.logical_and(y_truth, y_pred))

        # compute the sum of tp + fp across training examples and labels
        # noinspection PyUnresolvedReferences
        l_prec_den = np.sum(y_pred)
        if l_prec_den != 0:
            # compute micro-averaged precision
            precisions.append(true_positives / l_prec_den)

        # compute sum of tp + fn across training examples and labels
        # noinspection PyUnresolvedReferences
        l_recall_den = np.sum(y_truth)

        # compute mirco-average recall
        if l_recall_den != 0:
            recalls.append(true_positives / l_recall_den)

    precisions = np.average(precisions)
    recalls = np.average(recalls)
    if precisions + recalls == 0:
        return 0
    f1 = 2 * precisions * recalls / (precisions + recalls)
    return f1


def multi_label_f1(y_preds, y_truths, mode='weighted'):
    preds = dict()
    truths = dict()
    for idx in range(len(y_truths)):
        for jdx in range(len(y_truths[idx])):
            if jdx not in preds:
                preds[jdx] = []
                truths[jdx] = []
            preds[jdx].append(y_preds[idx][jdx])
            truths[jdx].append(y_truths[idx][jdx])
    results = []
    for jdx in preds:
        results.append(metrics.f1_score(preds[jdx], truths[jdx], average=mode))
    return np.average(results)


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
    return emb_matrix


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
                os.path.join(params['tok_dir'], '{}.tok'.format(params['dname'])), 'rb'))
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

        labels = torch.tensor(labels, dtype=torch.float)
        domains = torch.tensor(domains, dtype=torch.long)
        if self.mtype == 'rnn':
            # padding and tokenize
            docs = self.tok.texts_to_sequences(docs)
            docs = pad_sequences(docs)
            docs = torch.Tensor(docs).long()
        else:
            docs = torch.stack(docs).long()
        return docs, labels, domains


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


class RegularRNN(nn.Module):
    def __init__(self, params):
        super(RegularRNN, self).__init__()
        self.params = params

        if 'word_emb_path' in self.params and os.path.exists(self.params['word_emb_path']):
            self.wemb = nn.Embedding.from_pretrained(
                torch.FloatTensor(np.load(
                    self.params['word_emb_path'], allow_pickle=True))
            )
        else:
            self.wemb = nn.Embedding(
                self.params['max_feature'], self.params['emb_dim']
            )
            self.wemb.reset_parameters()
            nn.init.kaiming_uniform_(self.wemb.weight, a=np.sqrt(5))

        if self.params['bidirectional']:
            self.word_hidden_size = self.params['emb_dim'] // 2
        else:
            self.word_hidden_size = self.params['emb_dim']

        # domain adaptation
        self.doc_net_general = nn.GRU(
            self.wemb.embedding_dim, self.word_hidden_size,
            bidirectional=self.params['bidirectional'], dropout=self.params['dp_rate'],
            batch_first=True
        )
        # prediction
        self.predictor = nn.Linear(
            self.params['emb_dim'], self.params['num_label'])

    def forward(self, input_docs):
        # encode the document from different perspectives
        doc_embs = self.wemb(input_docs)
        _, doc_general = self.doc_net_general(doc_embs)  # omit hidden vectors

        # concatenate hidden state
        if self.params['bidirectional']:
            doc_general = torch.cat((doc_general[0, :, :], doc_general[1, :, :]), -1)

        if doc_general.shape[0] == 1:
            doc_general = doc_general.squeeze(dim=0)

        # prediction
        doc_preds = self.predictor(doc_general)
        return doc_preds


class AdaptRNN(nn.Module):
    def __init__(self, params):
        super(AdaptRNN, self).__init__()
        self.params = params

        if 'word_emb_path' in self.params and os.path.exists(self.params['word_emb_path']):
            self.wemb = nn.Embedding.from_pretrained(
                torch.FloatTensor(np.load(
                    self.params['word_emb_path'], allow_pickle=True))
            )
        else:
            self.wemb = nn.Embedding(
                self.params['max_feature'], self.params['emb_dim']
            )
            self.wemb.reset_parameters()
            nn.init.kaiming_uniform_(self.wemb.weight, a=np.sqrt(5))

        if self.params['bidirectional']:
            self.word_hidden_size = self.params['emb_dim'] // 2
        else:
            self.word_hidden_size = self.params['emb_dim']

        # domain adaptation
        self.domain_net = nn.GRU(
            self.wemb.embedding_dim, self.word_hidden_size,
            bidirectional=self.params['bidirectional'], dropout=self.params['dp_rate'],
            batch_first=True
        )
        # two domains, this domain vs others
        self.domain_clf = nn.Linear(
            self.params['emb_dim'], 1
        )

        # regular prediction
        self.document_net = nn.GRU(
            self.wemb.embedding_dim, self.word_hidden_size,
            bidirectional=self.params['bidirectional'], dropout=self.params['dp_rate'],
            batch_first=True
        )
        # prediction
        self.document_predictor = nn.Linear(
            self.params['emb_dim'], self.params['num_label'])

    def forward(self, input_docs):
        # encode the document from different perspectives
        doc_embs = self.wemb(input_docs)
        _, doc_general = self.document_net(doc_embs)  # omit hidden vectors

        # concatenate hidden state
        if self.params['bidirectional']:
            doc_general = torch.cat((doc_general[0, :, :], doc_general[1, :, :]), -1)

        if doc_general.shape[0] == 1:
            doc_general = doc_general.squeeze(dim=0)

        # prediction
        doc_preds = self.document_predictor(doc_general)
        return doc_preds

    def discriminator(self, input_docs):
        # encode the document from different perspectives
        doc_embs = self.wemb(input_docs)
        _, doc_domain = self.domain_net(doc_embs)  # omit hidden vectors
        # concatenate hidden state
        if self.params['bidirectional']:
            doc_domain = torch.cat((doc_domain[0, :, :], doc_domain[1, :, :]), -1)

        if doc_domain.shape[0] == 1:
            doc_domain = doc_domain.squeeze(dim=0)

        # prediction
        domain_preds = self.predictor(doc_domain)
        return domain_preds

    def freeze_layer(self, if_train=True):
        self.wemb.weight.requires_grad = if_train


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


def main(params):
    all_labels = [
        'subversion', 'loyalty', 'care', 'cheating',
        'purity', 'fairness', 'degradation', 'betrayal', 'harm', 'authority'
    ]
    if torch.cuda.is_available() and params['device'] != 'cpu':
        device = torch.device(params['device'])
    else:
        device = torch.device('cpu')
    params['device'] = device
    wfile = open(params['result_path'], 'a')

    print('Loading Data...')
    all_data = pd.read_csv(params['dpath'], sep='\t', dtype=str)
    all_data.tid = all_data.tid.apply(lambda x: str(x))
    all_data = all_data[~all_data.text.isna()]
    all_data = all_data[~all_data.labels.isna()]
    # preprocess tweet and remove short tweet
    all_data.text = all_data.text.apply(lambda x: preprocess(x))
    all_data = all_data[all_data.text.apply(lambda x: len(x) > 3)]
    all_data.text = all_data.text.apply(lambda x: ' '.join(x))
    all_data.labels = all_data.labels.apply(lambda x: label_encoder(x))
    params['unique_domains'] = list(all_data.corpus.unique())
    wfile.write(json.dumps(params) + '\n')

    # load the vaccine data and test the classifier on the vaccine data
    vaccine_df = pd.read_csv('../data/vaccine_morality.csv', dtype=str)
    vaccine_df.text = vaccine_df.text.apply(lambda x: preprocess(x))
    # vaccine_df = vaccine_df[vaccine_df.text.apply(lambda x: len(x) > 3)]
    vaccine_df.text = vaccine_df.text.apply(lambda x: ' '.join(x))
    vaccine_df = vaccine_df.sample(frac=1).reset_index(drop=True)

    # domains
    domain_encoder = list(all_data.corpus.unique()) + ['vaccine']

    # use half of the vaccine as train and half as test
    all_corpus = {
        'docs': all_data.text.to_list(),
        'labels': all_data.labels.to_list(),
        'corpus': all_data.corpus.to_list(),
    }
    all_corpus['corpus'] = [domain_encoder.index(item) for item in all_corpus['corpus']]

    # build tokenizer and weight
    tok_dir = os.path.dirname(params['model_dir'])
    params['tok_dir'] = tok_dir
    params['word_emb_path'] = os.path.join(
        tok_dir, params['dname'] + '.npy'
    )
    tok = build_tok(
        all_data.text.tolist() + vaccine_df.text.tolist(), max_feature=params['max_feature'],
        opath=os.path.join(tok_dir, '{}.tok'.format(params['dname']))
    )
    if not os.path.exists(params['word_emb_path']):
        build_wt(tok, params['emb_path'], params['word_emb_path'])
    data_encoder = DataEncoder(params, mtype='rnn')

    # start to iterate experiments over domains
    print('Run over domains...')
    for didx, domain in enumerate(tqdm(params['unique_domains'])):
        wfile.write('Working on {}, index {} \n'.format(domain, didx))
        in_domain_indices = [item for item in range(len(all_corpus['corpus'])) if all_corpus['corpus'][item] == didx]
        out_domain_indices = [item for item in range(len(all_corpus['corpus'])) if all_corpus['corpus'][item] != didx]

        train_corpus = {
            'docs': [all_corpus['docs'][item] for item in out_domain_indices],
            'labels': [all_corpus['labels'][item] for item in out_domain_indices],
            'corpus': [all_corpus['corpus'][item] for item in out_domain_indices],
        }
        domain_corpus = {
            'docs': train_corpus['docs'],
            'labels': train_corpus['labels'],
            'corpus': [0] * len(train_corpus['docs']),  # first collect documents from out of domain
        }
        in_domain_corpus = {
            'docs': [all_corpus['docs'][item] for item in in_domain_indices],
            'labels': [all_corpus['labels'][item] for item in in_domain_indices],
            'corpus': [all_corpus['corpus'][item] for item in in_domain_indices],
        }
        domain_corpus['docs'].extend(in_domain_corpus['docs'])
        domain_corpus['labels'].extend(in_domain_corpus['labels'])
        domain_corpus['corpus'].extend([1] * len(in_domain_corpus['docs']))

        # 10% for training, 10% for valid, the rest for testing
        test_indices, val_indices, train_indices = data_split(in_domain_corpus)
        in_domain_train = {
            'docs': [in_domain_corpus['docs'][item] for item in train_indices],
            'labels': [in_domain_corpus['labels'][item] for item in train_indices],
            'corpus': [in_domain_corpus['corpus'][item] for item in train_indices]
        }
        train_corpus['docs'].extend(in_domain_train['docs'])
        train_corpus['labels'].extend(in_domain_train['labels'])
        train_corpus['corpus'].extend(in_domain_train['corpus'])

        valid_corpus = {
            'docs': [in_domain_corpus['docs'][item] for item in val_indices],
            'labels': [in_domain_corpus['labels'][item] for item in val_indices],
            'corpus': [in_domain_corpus['corpus'][item] for item in val_indices]
        }
        test_corpus = {
            'docs': [in_domain_corpus['docs'][item] for item in test_indices],
            'labels': [in_domain_corpus['labels'][item] for item in test_indices],
            'corpus': [in_domain_corpus['corpus'][item] for item in test_indices]
        }

        # start to iteratively train and test the proposed approach.
        train_data = TorchDataset(train_corpus, params['domain_name'])
        valid_data = TorchDataset(valid_corpus, params['domain_name'])
        test_data = TorchDataset(test_corpus, params['domain_name'])
        in_domain_train_data = TorchDataset(in_domain_train, params['domain_name'])
        domain_data = TorchDataset(domain_corpus, params['domain_name'])

        train_data_loader = DataLoader(
            train_data, batch_size=params['batch_size'], shuffle=True,
            collate_fn=data_encoder
        )
        valid_data_loader = DataLoader(
            valid_data, batch_size=params['batch_size'], shuffle=True,
            collate_fn=data_encoder
        )
        test_data_loader = DataLoader(
            test_data, batch_size=params['batch_size'], shuffle=True,
            collate_fn=data_encoder
        )
        in_domain_train_data_loader = DataLoader(
            in_domain_train_data, batch_size=params['batch_size'], shuffle=True,
            collate_fn=data_encoder
        )
        domain_data_loader = DataLoader(
            domain_data, batch_size=params['batch_size'], shuffle=True,
            collate_fn=data_encoder
        )

        regular_model = RegularRNN(params)
        regular_model = regular_model.to(device)
        criterion = nn.BCEWithLogitsLoss().to(device)
        domain_criterion = nn.CrossEntropyLoss().to(device)
        regular_optim = torch.optim.RMSprop(regular_model.parameters(), lr=params['lr'])

        adapt_model = AdaptRNN(params)
        criterion_adapt = nn.BCEWithLogitsLoss(reduction='none').to(device)
        pred_params = [param for name, param in adapt_model.named_parameters() if 'domain' not in name]
        adapt_pred_optim = torch.optim.RMSprop(pred_params, lr=params['lr'])
        domain_params = [param for name, param in adapt_model.named_parameters() if 'domain' in name]
        adapt_domain_optim = torch.optim.RMSprop(domain_params, lr=params['lr'])

        # train the networks
        print('Start to train...')
        print(params)
        best_valid_regular = 0.
        best_valid_adapt = 0.

        for epoch in tqdm(range(params['epochs'])):
            train_loss_regular = 0.
            train_loss_adapt = 0.
            adapt_model.train()
            regular_model.train()

            # train discriminator first
            for step, train_batch in enumerate(domain_data_loader):
                train_batch = tuple(t.to(device) for t in train_batch)
                input_docs, input_labels, input_domains = train_batch
                adapt_domain_optim.zero_grad()
                domain_preds = adapt_model.discriminator({
                    'input_docs': input_docs
                })
                domain_loss = domain_criterion(domain_preds, input_domains)
                domain_loss.backward()
                adapt_domain_optim.step()

            # train predictor
            for step, train_batch in enumerate(train_data_loader):
                train_batch = tuple(t.to(device) for t in train_batch)
                input_docs, input_labels, input_domains = train_batch
                regular_optim.zero_grad()
                adapt_pred_optim.zero_grad()
                # adapt_domain_optim.zero_grad()

                # regular models
                regular_preds = regular_model(**{
                    'input_docs': input_docs
                })
                loss = criterion(regular_preds, input_labels)
                train_loss_regular += loss.item()
                loss_avg_regular = train_loss_regular / (step + 1)

                # adapt models
                adapt_preds = adapt_model(**{
                    'input_docs': input_docs
                })
                loss_adapt = criterion_adapt(adapt_preds, input_labels)
                domain_preds = torch.sigmoid(adapt_model.discriminator({'input_docs': input_docs}))
                loss_adapt = domain_preds * loss_adapt
                loss_adapt = loss_adapt.mean()
                train_loss_adapt += loss_adapt.item()
                loss_avg_adapt = train_loss_adapt / (step + 1)

                if (step + 1) % 301 == 0:
                    print('Epoch: {}, Step: {}'.format(epoch, step))
                    print('\tRegular Loss: {}.'.format(loss_avg_regular))
                    print('\tAdapt Loss: {}.'.format(loss_avg_adapt))
                    print('-------------------------------------------------')

                loss_adapt.backward()
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(rnn_model.parameters(), 0.5)
                regular_optim.step()
                adapt_pred_optim.step()

            # fit on in domain corpus.
            for _ in range(5):
                for step, train_batch in enumerate(in_domain_train_data_loader):
                    train_batch = tuple(t.to(device) for t in train_batch)
                    input_docs, input_labels, input_domains = train_batch
                    adapt_pred_optim.zero_grad()
                    adapt_preds = adapt_model({
                        'input_docs': input_docs
                    })
                    loss_adapt = criterion_adapt(adapt_preds, input_labels)
                    loss_adapt.backward()
                    adapt_pred_optim.step()

            # evaluate on valid data
            regular_model.eval()
            adapt_model.eval()
            y_preds_regular = []
            y_preds_adapt = []
            y_trues = []
            for valid_batch in valid_data_loader:
                valid_batch = tuple(t.to(device) for t in valid_batch)
                input_docs, input_labels, input_domains = valid_batch
                with torch.no_grad():
                    preds_regular = regular_model(**{'input_docs': input_docs})
                    preds_adapt = adapt_model(**{'input_docs': input_docs})

                logits_regular = (torch.sigmoid(preds_regular) > .5).long().cpu().numpy()
                logits_adapt = (torch.sigmoid(preds_adapt) > .5).long().cpu().numpy()
                y_preds_regular.extend(logits_regular)
                y_preds_adapt.extend(logits_adapt)
                y_trues.extend(input_labels.to('cpu').numpy())

            eval_score_regular = micro_f1_average(y_preds=y_preds_regular, y_truths=y_trues)
            eval_score_adapt = micro_f1_average(y_preds=y_preds_adapt, y_truths=y_trues)

            # test for regular model
            if eval_score_regular > best_valid_regular:
                best_valid_regular = eval_score_regular
                torch.save(regular_model, params['model_dir'] + '{}.pth'.format(os.path.basename(__file__)))

                # test
                y_preds = []
                y_trues = []
                # evaluate on the test set
                for test_batch in test_data_loader:
                    test_batch = tuple(t.to(device) for t in test_batch)
                    input_docs, input_labels, input_domains = test_batch

                    with torch.no_grad():
                        preds_regular = regular_model(**{
                            'input_docs': input_docs,
                        })
                    logits_regular = (torch.sigmoid(preds_regular) > .5).long().cpu().numpy()
                    y_preds.extend(logits_regular)
                    y_trues.extend(input_labels.to('cpu').numpy())

                test_score_regular = micro_f1_average(y_preds=y_preds, y_truths=y_trues)
                wfile.write(
                    'Test on Regular RNN, Domain {}, Epoch {}, F1-micro-average {}, Valid Score {}\n'.format(
                        domain, epoch, test_score_regular, best_valid_regular)
                )

            if eval_score_adapt > best_valid_adapt:
                best_valid_adapt = eval_score_adapt
                torch.save(adapt_model, params['model_dir'] + '{}.pth'.format(os.path.basename(__file__)))

                # test
                y_preds = []
                y_trues = []
                # evaluate on the test set
                for test_batch in test_data_loader:
                    test_batch = tuple(t.to(device) for t in test_batch)
                    input_docs, input_labels, input_domains = test_batch

                    with torch.no_grad():
                        preds_adapt = adapt_model(**{
                            'input_docs': input_docs,
                        })
                    logits_adapt = (torch.sigmoid(preds_adapt) > .5).long().cpu().numpy()
                    y_preds.extend(logits_adapt)
                    y_trues.extend(input_labels.to('cpu').numpy())

                test_score_adapt = micro_f1_average(y_preds=y_preds, y_truths=y_trues)
                wfile.write(
                    'Test on Adapt RNN, Domain {}, Epoch {}, F1-micro-average {}, Valid Score {}\n'.format(
                        domain, epoch, test_score_adapt, best_valid_adapt)
                )

    # vaccine experiments
    vaccine_data = {
        'docs': [],
        'labels': [],
    }
    wfile.write('\nVaccine Evaluation---\n')

    for idx, row in vaccine_df.iterrows():
        encode_label = [0] * params['num_label']
        for label_index, _ in enumerate(all_labels):
            if np.isnan(np.array(row[all_labels[label_index]], dtype=np.float32)):
                continue
            if int(row[all_labels[label_index]]) == 1:
                encode_label[label_index] = 1
        if sum(encode_label) == 0:
            encode_label[-1] = 1
        vaccine_data['docs'].append(row['text'])
        vaccine_data['labels'].append(encode_label)
    vaccine_train = {
        'docs': vaccine_data['docs'][:250],
        'labels': vaccine_data['labels'][:250],
        'corpus': [1] * 250
    }
    vaccine_test = {
        'docs': vaccine_data['docs'][250:],
        'labels': vaccine_data['labels'][250:],
        'corpus': [1] * 250
    }
    all_train = {
        'docs': all_data.text.to_list() + vaccine_data['docs'][:250],
        'labels': all_data.labels.to_list() + vaccine_data['labels'][:250],
        'corpus': [0] * len(all_data.corpus.to_list()) + [1] * 250
    }
    all_data = {
        'docs': all_data.text.to_list() + vaccine_data['docs'],
        'labels': all_data.labels.to_list() + vaccine_data['labels'],
        'corpus': [0] * len(all_data.corpus.to_list()) + [1] * 250
    }
    vaccine_train_data = TorchDataset(vaccine_train, domain_name=params['domain_name'])
    vaccine_test_data = TorchDataset(vaccine_test, domain_name=params['domain_name'])
    all_train_data = TorchDataset(all_train, domain_name=params['domain_name'])
    all_data_torch = TorchDataset(all_data, domain_name=params['domain_name'])

    train_data_loader = DataLoader(
        all_train_data, batch_size=params['batch_size'], shuffle=True,
        collate_fn=data_encoder
    )
    valid_data_loader = DataLoader(
        vaccine_train_data, batch_size=params['batch_size'], shuffle=True,
        collate_fn=data_encoder
    )
    test_data_loader = DataLoader(
        vaccine_test_data, batch_size=params['batch_size'], shuffle=True,
        collate_fn=data_encoder
    )
    all_data_loader = DataLoader(
        all_data_torch, batch_size=params['batch_size'], shuffle=True,
        collate_fn=data_encoder
    )

    regular_model = RegularRNN(params)
    regular_model = regular_model.to(device)
    criterion = nn.BCEWithLogitsLoss().to(device)
    domain_criterion = nn.CrossEntropyLoss().to(device)
    regular_optim = torch.optim.RMSprop(regular_model.parameters(), lr=params['lr'])

    adapt_model = AdaptRNN(params)
    criterion_adapt = nn.BCEWithLogitsLoss(reduction='none').to(device)
    pred_params = [param for name, param in adapt_model.named_parameters() if 'domain' not in name]
    adapt_pred_optim = torch.optim.RMSprop(pred_params, lr=params['lr'])
    domain_params = [param for name, param in adapt_model.named_parameters() if 'domain' in name]
    adapt_domain_optim = torch.optim.RMSprop(domain_params, lr=params['lr'])

    # train the networks
    print('Start to train...')
    print(params)
    best_valid_regular = 0.
    best_valid_adapt = 0.

    for epoch in tqdm(range(params['epochs'])):
        train_loss_regular = 0.
        train_loss_adapt = 0.
        adapt_model.train()
        regular_model.train()

        # train discriminator first
        for step, train_batch in enumerate(all_data_loader):
            train_batch = tuple(t.to(device) for t in train_batch)
            input_docs, input_labels, input_domains = train_batch
            adapt_domain_optim.zero_grad()
            domain_preds = adapt_model.discriminator({
                'input_docs': input_docs
            })
            domain_loss = domain_criterion(domain_preds, input_domains)
            domain_loss.backward()
            adapt_domain_optim.step()

        # train predictor
        for step, train_batch in enumerate(train_data_loader):
            train_batch = tuple(t.to(device) for t in train_batch)
            input_docs, input_labels, input_domains = train_batch
            regular_optim.zero_grad()
            adapt_pred_optim.zero_grad()
            # adapt_domain_optim.zero_grad()

            # regular models
            regular_preds = regular_model(**{
                'input_docs': input_docs
            })
            loss = criterion(regular_preds, input_labels)
            train_loss_regular += loss.item()
            loss_avg_regular = train_loss_regular / (step + 1)

            # adapt models
            adapt_preds = adapt_model(**{
                'input_docs': input_docs
            })
            loss_adapt = criterion_adapt(adapt_preds, input_labels)
            domain_preds = torch.sigmoid(adapt_model.discriminator({'input_docs': input_docs}))
            loss_adapt = domain_preds * loss_adapt
            loss_adapt = loss_adapt.mean()
            train_loss_adapt += loss_adapt.item()
            loss_avg_adapt = train_loss_adapt / (step + 1)

            if (step + 1) % 301 == 0:
                print('Epoch: {}, Step: {}'.format(epoch, step))
                print('\tRegular Loss: {}.'.format(loss_avg_regular))
                print('\tAdapt Loss: {}.'.format(loss_avg_adapt))
                print('-------------------------------------------------')

            loss_adapt.backward()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(rnn_model.parameters(), 0.5)
            regular_optim.step()
            adapt_pred_optim.step()

        # fit on in domain corpus.
        for _ in range(5):
            for step, train_batch in enumerate(valid_data_loader):
                train_batch = tuple(t.to(device) for t in train_batch)
                input_docs, input_labels, input_domains = train_batch
                adapt_pred_optim.zero_grad()
                adapt_preds = adapt_model({
                    'input_docs': input_docs
                })
                loss_adapt = criterion_adapt(adapt_preds, input_labels)
                loss_adapt.backward()
                adapt_pred_optim.step()

        # evaluate on valid data
        regular_model.eval()
        adapt_model.eval()
        y_preds_regular = []
        y_preds_adapt = []
        y_trues = []
        for valid_batch in valid_data_loader:
            valid_batch = tuple(t.to(device) for t in valid_batch)
            input_docs, input_labels, input_domains = valid_batch
            with torch.no_grad():
                preds_regular = regular_model(**{'input_docs': input_docs})
                preds_adapt = adapt_model(**{'input_docs': input_docs})

            logits_regular = (torch.sigmoid(preds_regular) > .5).long().cpu().numpy()
            logits_adapt = (torch.sigmoid(preds_adapt) > .5).long().cpu().numpy()
            y_preds_regular.extend(logits_regular)
            y_preds_adapt.extend(logits_adapt)
            y_trues.extend(input_labels.to('cpu').numpy())

        eval_score_regular = micro_f1_average(y_preds=y_preds_regular, y_truths=y_trues)
        eval_score_adapt = micro_f1_average(y_preds=y_preds_adapt, y_truths=y_trues)

        # test for regular model
        if eval_score_regular > best_valid_regular:
            best_valid_regular = eval_score_regular
            torch.save(regular_model, params['model_dir'] + '{}.pth'.format(os.path.basename(__file__)))

            # test
            y_preds = []
            y_trues = []
            # evaluate on the test set
            for test_batch in test_data_loader:
                test_batch = tuple(t.to(device) for t in test_batch)
                input_docs, input_labels, input_domains = test_batch

                with torch.no_grad():
                    preds_regular = regular_model(**{
                        'input_docs': input_docs,
                    })
                logits_regular = (torch.sigmoid(preds_regular) > .5).long().cpu().numpy()
                y_preds.extend(logits_regular)
                y_trues.extend(input_labels.to('cpu').numpy())

            test_score_regular = micro_f1_average(y_preds=y_preds, y_truths=y_trues)
            wfile.write(
                'Test on Regular RNN, Domain {}, Epoch {}, F1-micro-average {}, Valid Score {}\n'.format(
                    'vaccine', epoch, test_score_regular, best_valid_regular)
            )

        if eval_score_adapt > best_valid_adapt:
            best_valid_adapt = eval_score_adapt
            torch.save(adapt_model, params['model_dir'] + '{}.pth'.format(os.path.basename(__file__)))

            # test
            y_preds = []
            y_trues = []
            # evaluate on the test set
            for test_batch in test_data_loader:
                test_batch = tuple(t.to(device) for t in test_batch)
                input_docs, input_labels, input_domains = test_batch

                with torch.no_grad():
                    preds_adapt = adapt_model(**{
                        'input_docs': input_docs,
                    })
                logits_adapt = (torch.sigmoid(preds_adapt) > .5).long().cpu().numpy()
                y_preds.extend(logits_adapt)
                y_trues.extend(input_labels.to('cpu').numpy())

            test_score_adapt = micro_f1_average(y_preds=y_preds, y_truths=y_trues)
            wfile.write(
                'Test on Adapt RNN, Domain {}, Epoch {}, F1-micro-average {}, Valid Score {}\n'.format(
                    'vaccine', epoch, test_score_adapt, best_valid_adapt)
            )

    wfile.write('\n\n\n')
    wfile.close()


if __name__ == '__main__':
    all_morality = [
        'subversion', 'loyalty', 'care', 'cheating',
        'purity', 'fairness', 'degradation', 'betrayal', 'harm', 'authority'
    ]

    result_dir = '../resource/results/'
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    model_dir = '../resource/model/'
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    model_dir = model_dir + 'adapt_rnn/'
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    parameters = {
        'result_path': os.path.join(result_dir, 'adapt_rnn.txt'),
        'model_dir': model_dir,
        'dname': 'all',
        'dpath': '../data/dataset.tsv',
        'max_feature': 15000,
        'over_sample': True,
        'domain_name': 'corpus',
        'epochs': 15,
        'batch_size': 64,
        'lr': 9e-5,
        'max_len': 100,
        'dp_rate': .2,
        'optimizer': 'rmsprop',
        'emb_path': '/data/models/glove.twitter.27B.200d.txt',  # adjust for different languages
        'emb_dim': 200,
        'unique_domains': [],
        'bidirectional': False,
        'device': 'cuda',
        'num_label': len(all_morality)+1,  # plus no-moral
    }

    main(parameters)
