import numpy as np
import pandas as pd
from loguru import logger
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

if __name__ == '__main__':

    data = pd.read_csv('/home/ywu10/Documents/MoralCausality/data/MFTC_V4_text_preprogress.tsv')
    class_names = list(set(sum([l.split(',') for l in data['label']],[])))
    word_vectorizer = TfidfVectorizer(
        sublinear_tf=True,
        strip_accents='unicode',
        analyzer='word',
        token_pattern=r'\w{1,}',
        stop_words='english',
        ngram_range=(1, 1),
        max_features=10000)
    word_vectorizer.fit(data['text'])
    split = int(len(data['text']) * 0.8)
    train_word_features = word_vectorizer.transform(data['text'][:split])
    test_word_features = word_vectorizer.transform(data['text'][split:])

    TP = 0
    precision = 0
    recall = 0
    for class_name in class_names:
        train_target = [1 if class_name in i else 0 for i in data['label'][:split]]
        test_target = np.array([1 if class_name in i else 0 for i in data['label'][split:]])
        classifier = LogisticRegression(C=0.1, solver='sag')
        classifier.fit(train_word_features, train_target)
        prediction = (classifier.predict_proba(test_word_features)[:,1]) > 0.06
        precision += np.sum(prediction)
        recall += np.sum(test_target)
        TP += np.sum(prediction * test_target)
    precison = TP/precision
    recall = TP/recall
    F1 = 2 * (precison*recall)/(precison+recall)
    logger.info('F1: {}, precision:{}. recall:{}'.format(F1, precison, recall))