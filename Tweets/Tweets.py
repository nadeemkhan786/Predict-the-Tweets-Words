# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 23:14:55 2021

@author: EI11560
"""

import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import string

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

#%matplotlib inline

# data set
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Method for remove unwanted thinks
def remove_pattern(input_txt, pattern):
  r = re.findall(pattern, input_txt)
  for i in r:
    input_txt = re.sub(i, '', input_txt)

  return input_txt

com = train.append(test, ignore_index=True)

com['tidy_tweet'] = np.vectorize(remove_pattern)(com['tweet'], "@[\w]*")

# Remove special characters, numbers, punctuation

com['tidy_tweet'] = com['tidy_tweet'].str.replace("[^a-zA-Z#]", " " )
     
# Removing short words
     
com['tidy_tweet'] = com['tidy_tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))

tokenized_tweet = com['tidy_tweet'].apply(lambda x: x.split())
print(com.head())
# import nltk
#from nltk.stem.porter import *
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()

tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x])

from sklearn.feature_extraction.text import CountVectorizer
bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')

bow = bow_vectorizer.fit_transform(com['tidy_tweet'])

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')

tfidf = tfidf_vectorizer.fit_transform(com['tidy_tweet'])

# we’ll build with the bag-of-words dataframe
     
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

train_bow = bow[:31962, :]
test_bow = bow[31962:, :]

xtrain_bow, xvalid_bow, ytrain, yvalid = train_test_split(train_bow, train['label'], random_state=42, test_size=0.3)

lreg = LogisticRegression()
lreg.fit(xtrain_bow, ytrain)

prediction = lreg.predict_proba(xvalid_bow)
prediction_int = prediction[:,1] >= 0.3
prediction_int = prediction_int.astype(np.int)

#printing

print (f1_score(yvalid, prediction_int))

train_tfidf = tfidf[:31962, :]
test_tfidf = tfidf[31962:, :]

xtrain_tfidf = train_tfidf[ytrain.index]
xvalid_tfidf = train_tfidf[yvalid.index]

lreg.fit(xtrain_tfidf, ytrain)

prediction = lreg.predict_proba(xvalid_tfidf)
prediction_int = prediction[:, 1] >= 0.3
prediction_int = prediction_int.astype(np.int)

