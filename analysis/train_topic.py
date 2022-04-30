import numpy as np
import pandas as pd
from gensim.corpora import Dictionary
from gensim.models import LdaModel
import os
import pickle
import re
from nltk.tokenize import TweetTokenizer
tokenizer = TweetTokenizer()


# preprocessing url
def preprocess(tweet):
    """
    Preprocess a single tweet
    :param tweet:
    :return:
    """
    global tokenizer

    # lowercase
    tweet = tweet.lower()
    # replace url
    tweet = re.sub(r"https?:\S+", "URL", tweet)
    # replace user
#     tweet = re.sub(r'@\w+', 'USER', tweet)
    # replace hashtag
#     tweet = re.sub(r'#\S+', 'HASHTAG', tweet)
    # tokenize
    return [item.strip() for item in tokenizer.tokenize(tweet) if len(item.strip())>0]


all_data = pd.read_csv('../data/dataset.tsv', sep='\t', dtype=str)
all_data = all_data[~all_data.text.isna()]
all_data = all_data[~all_data.labels.isna()]
all_data.text = all_data.text.apply(lambda x: preprocess(x))
all_data = all_data[all_data.text.apply(lambda x: len(x) > 3)]
# all_data.text = all_data.text.apply(lambda x: ' '.join(x))
corpus = all_data.text.tolist()

# build dictionary
dictionary = Dictionary(corpus)
dictionary.save('./topic/moral_topic.dict')

# documents to indices
doc_matrix = [dictionary.doc2bow(doc) for doc in corpus]
del corpus # release memory
ldamodel = LdaModel(doc_matrix,
        id2word=dictionary, num_topics=100,
        passes=2, alpha='symmetric', eta=None)
ldamodel.save('./topic/moral_topic.model')
