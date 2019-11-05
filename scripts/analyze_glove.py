#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import argparse
import pprint
import preprocess
import gensim
from glove import Glove
from glove import Corpus

from sklearn.cluster import KMeans
from sklearn.neighbors import KDTree
from wordcloud import WordCloud, ImageColorGenerator
import nltk
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
import sys
import os.path
from os import path
import unicodedata
import re
import nltk.data;
import gensim
import gzip
import pandas as pd;
import numpy as np;
import os;
import re;
import logging;
import csv;
import sqlite3;
import time;
import codecs
import sys;
import matplotlib.pyplot as plt;
from itertools import cycle;

reload(sys)
sys.setdefaultencoding('utf8')

print("Training model...");
start = time.time();


num_features=50
print('Loading pre-trained GloVe model')
glove = Glove.load('glove.model')

num_clusters=10

tables = [];
reviewCount=0
amostra = []
fname=sys.argv[1]
with codecs.open(fname, 'r',encoding='utf-8',errors='replace') as f:
        reader = csv.reader(f, delimiter=';' )
        for row in reader:
                amostra.append(row)
header = [];
header.append('target')
for x in range(int(num_features)):
  header.append(x)

with open('output.csv', 'w') as csvfile:
        csvWriter = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_NONE)
        csvWriter.writerow(header)
        for value in amostra:
                reviewCount+=1
		word_tokens = word_tokenize(preprocess.convert_com(value[2]))
		stop_words = set(stopwords.words('portuguese')) 
		filtered_sentence = [w for w in word_tokens if not w in stop_words]
                reviewVec = []
                validWords=0
                allWordsVec=[]
                for word in filtered_sentence:
                        m = re.match(r"(\w{3,})",word)
                        if bool(m):
                                try:
                                        allWordsVec.append(glove.word_vectors[glove.dictionary[m.group(0)]])
                                        validWords+=1
                                except KeyError as error:
                                        pass
                        else:
                                pass
                if (reviewCount % 500 == 0):
                        print ("Review ",reviewCount," com ",validWords," palavras")
		reviewVec.append(value[3])
                if validWords > 0:
                        reviewVec.extend(np.mean(np.array(allWordsVec), axis=0))
		else:
                        reviewVec.extend([0] * num_features)
		csvWriter.writerow(reviewVec)

print('output to output.csv')
