
#%matplotlib inline
import nltk.data;
from gensim.models import word2vec;
import gensim
import gzip
from sklearn.cluster import KMeans;
from sklearn.neighbors import KDTree;
from glove import Glove
import pandas as pd;
import numpy as np;
import os;
import re;
import logging;
import csv;
import time;
import codecs
import sys;
import multiprocessing;
from wordcloud import WordCloud, ImageColorGenerator
import matplotlib.pyplot as plt;
from itertools import cycle;

reload(sys)
sys.setdefaultencoding('utf8')

#print("Training model...");
start = time.time();

num_workers=10
num_features=50
min_word_count=4
context=3

print('Loading pre-trained GloVe model')
glove = Glove.load('glove.model')

tables = [];
allReviews = []
reviewCount=0
amostra = []
fname=sys.argv[1]
with codecs.open(fname, 'r', encoding='utf-8', errors='strict') as f:
        reader = csv.reader(f, delimiter=';', quoting=csv.QUOTE_NONE)
	try:
	        for row in reader:
        	        amostra.append(row)
	except csv.Error as e:
		sys.exit('file {}, line {}: {}'.format(fname, reader.line_num, e))

for value in amostra:
	reviewCount+=1
	value[2] = value[2].split(' ')
	reviewVec = []
	validWords=0
	allWordsVec=[]
	for word in value[2]:
		m = re.match(r"(\w{3,})",word)
		if bool(m):
			try:
                                allWordsVec.append(glove.word_vectors[glove.dictionary[m.group(0)]])
				validWords+=1
			except KeyError as error:
				pass
		else:
			pass
	if validWords > 0:
		reviewVec.extend(np.mean(np.array(allWordsVec), axis=0))
	else:
		reviewVec.extend([0] * num_features)
	allReviews.append(reviewVec);

#print('CSV: %d\nReview count: %d ' % (len(amostra), reviewCount))

from sklearn.metrics import silhouette_score

sil = []
kmin = 5
kmax = 100

for k in range(kmin, kmax):
	kmeans_clustering = KMeans(n_clusters = k, init='k-means++').fit(allReviews); 
	labels = kmeans_clustering.labels_
	sil.append(silhouette_score(allReviews, labels, metric = 'euclidean'))

import operator
index, value = max(enumerate(sil), key=operator.itemgetter(1))
bestk = index+kmin
print ("best k: ",bestk)
kmeans_clustering = KMeans(n_clusters = bestk, init='k-means++');
idx = kmeans_clustering.fit_predict(allReviews); 
centers, clusters = kmeans_clustering.cluster_centers_, idx;

i=0
cls=[]
print('lineno, cluster')
for cluster in clusters:
	i+=1
	print(str(i)+","+str(cluster+1));

