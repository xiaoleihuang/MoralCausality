import json
import multiprocessing
import os

import gensim
import pandas as pd


class SentIter_file:
    def __init__(self, datap):
        self.datap = datap

    def __iter__(self):
        with open(self.datap) as dfile:
            for line in dfile:
                line = line.strip().split()

                yield line


class SentIter_list:
    def __init__(self, data):
        self.data = data

    def __iter__(self):
        for line in self.data:
            line = line.strip().split()
            yield line


def train_regular_emb(datap):
    """Train word embedding model.
    """
    docs = SentIter_file(datap)
    model = gensim.models.Word2Vec(
        sentences=docs,
        size=200,
        window=5,
        min_count=5,
        workers=multiprocessing.cpu_count(),
        sg=1,
        iter=15,
    )
    model.save('./embeds/w2v.model')
    model.wv.save_word2vec_format('./embeds/w2v.txt')


def train_dw_emb(datap='./data/json_corpora.txt', skipgram=1):
    """Updated version Train diachronic word embedding model
        skipgram (int): if use skipgram, 1 use, 0 not to use
    """
    data = dict()
    data['general'] = []

    with open(datap) as dfile:
        for line in dfile:
            entry = json.loads(line.strip())

            if entry['year'] not in data:
                data[entry['year']] = []
            data['general'].append(entry['content'])
            data[entry['year']].append(' '.join([word + '#%s#' % entry['year'] for word in entry['content'].split()]))

    model = gensim.models.fasttext.FastText(
        size=200,
        window=5,
        workers=multiprocessing.cpu_count(),
        sg=skipgram,  # use skipgram or not
        min_n=3,
        max_n=6,
        iter=1,
    )
    # build a unified vocab
    for idx, year in enumerate(data.keys()):
        docs = SentIter_list(data[year])
        if idx == 0:
            model.build_vocab(docs, update=False)
        else:
            model.build_vocab(docs, update=True)

    # loop though each year data to train model
    # from each year to the general
    for epoch in range(20):
        # sample the data within the time interval
        # constrain the vocab samples
        for year in sorted(data.keys()):
            docs = SentIter_list(data[year])
            model.train(docs, epochs=1, total_examples=len(data[year]))

    if skipgram == 1:
        model.save('./embeds/dwe/dwe_skip.model')
    else:
        model.save('./embeds/dwe/dwe_cbow.model')


if __name__ == '__main__':
    emb_dir = '../resource/embs'
    if not os.path.exists(emb_dir):
        os.mkdir(emb_dir)
    pass
