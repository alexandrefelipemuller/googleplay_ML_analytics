#!/usr/bin/env python
# -*- coding: utf-8 -*-
from gensim.models import Word2Vec
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
import nltk.data
import gensim
import gzip
import pandas as pd
import numpy as np
import os
import re
import logging
import csv
import sqlite3
import time
import codecs
import sys
import preprocess
import multiprocessing;
import matplotlib.pyplot as plt;
from itertools import cycle;

reload(sys)
sys.setdefaultencoding('utf8')

if (len(sys.argv) < 3):
	print("usage <command> <file> <column to predict>")
	sys.exit()

print("Training model...");
start = time.time();

num_features=40
#from gensim.models import KeyedVectors
#model = KeyedVectors.load_word2vec_format("cbow_s50.txt");
model_name = "model_bradesco";
if (path.isfile(model_name)):
	model = Word2Vec.load(model_name)
else:
	documents = list (read_input ("./all_comments.txt.gz"))
	num_workers=10
	min_word_count=4
	context=3
	model = Word2Vec.Word2Vec(documents, workers=num_workers, \
            size=num_features, min_count = min_word_count, \
            window = context);
	# Save the model
	# We don't plan on training the model any further, so calling 
	# init_sims will make the model more memory efficient by normalizing the vectors in-place.
	model.init_sims(replace=True);
	model.save(model_name);

print('Total time: ' + str((time.time() - start)) + ' secs')
Z = model.wv.syn0;

num_clusters=10

tables = [];
reviewCount=0
amostra = []
fname=sys.argv[1]
column=sys.argv[2]
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
                                        allWordsVec.append(model[m.group(0)])
                                        validWords+=1
                                except KeyError as error:
                                        pass
                        else:
                                pass
                if (reviewCount % 500 == 0):
                        print ("Review ",reviewCount," com ",validWords," palavras")
		reviewVec.append(value[column])
                if validWords > 0:
                        reviewVec.extend(np.mean(np.array(allWordsVec), axis=0))
		else:
                        reviewVec.extend([0] * num_features)
		csvWriter.writerow(reviewVec)

print('output to output.csv')
