from collections import Counter
import re
import pickle
import os
import json

import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.feature_selection import mutual_info_classif

import numpy as np
from nltk.corpus import stopwords
import nltk
from nltk.tokenize import TweetTokenizer, word_tokenize
import gensim
from imblearn.over_sampling import RandomOverSampler

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from torch.utils.data import Dataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# nltk.download('stopwords')
spw_set = set(stopwords.words('english'))
spw_set.add('url')
tokenizer = TweetTokenizer()

