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
from spellchecker import SpellChecker  

reload(sys)
sys.setdefaultencoding('utf8')

spell = SpellChecker('pt')

if (len(sys.argv) < 3):
	print("usage <command> <file> <column to predict>")
	sys.exit()

print("Training model...");
start = time.time();


num_features=100
print('Loading pre-trained GloVe model')
glove = Glove.load('glove.model')
#from gensim.models.keyedvectors import KeyedVectors
#glove = KeyedVectors.load_word2vec_format("glove_s50.txt", binary=False)

tables = [];
reviewCount=0
amostra = []

fname=sys.argv[1]
column=int(sys.argv[2])
with codecs.open(fname, 'r',encoding='utf-8',errors='strict') as f:
        reader = csv.reader(f, delimiter=';', quoting=csv.QUOTE_NONE)
	try:
	        for row in reader:
        	        amostra.append(row)
	except csv.Error as e:
		sys.exit('file {}, line {}: {}'.format(fname, reader.line_num, e))

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
				#word_spelled = spell.correction(m.group(0))
				word_spelled = m.group(0)
                                try:
                                        allWordsVec.append(glove.word_vectors[glove.dictionary[word_spelled]])
                                        #allWordsVec.append(glove[word_spelled])
                                        validWords+=1
                                except KeyError as error:
                                        pass
                        else:
                                pass
                if (reviewCount % 100 == 0):
                        print ("Review ",reviewCount," com ",validWords," palavras")
		reviewVec.append(value[column])
                if validWords > 0:
                        reviewVec.extend(np.mean(np.array(allWordsVec), axis=0))
		else:
                        reviewVec.extend([0] * num_features)
		csvWriter.writerow(reviewVec)

print('output to output.csv')
