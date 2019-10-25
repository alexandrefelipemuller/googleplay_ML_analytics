
#%matplotlib inline
import nltk.data;
from gensim.models import word2vec;
import gensim
import gzip
from sklearn.cluster import KMeans;
from sklearn.neighbors import KDTree;
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

def read_input(input_file):
        """This method reads the input file which is in gzip format"""
        logging.info("reading file {0}...this may take a while".format(input_file))
        with gzip.open (input_file, 'rb') as f:
                for i, line in enumerate (f):
                        if (i%5000==0):
                                logging.info ("read {0} reviews".format (i))
                        yield gensim.utils.simple_preprocess (line)


documents = list (read_input ("./all_comments.txt.gz"))

#num_clusters=int(sys.argv[2])

num_workers=10
num_features=40
min_word_count=4
context=3

#from gensim.models import KeyedVectors
#model = KeyedVectors.load_word2vec_format("cbow_s50.txt");

model = word2vec.Word2Vec(documents, workers=num_workers, \
            size=num_features, min_count = min_word_count, \
            window = context);

# We don't plan on training the model any further, so calling 
# init_sims will make the model more memory efficient by normalizing the vectors in-place.
#model.init_sims(replace=True);
# Save the model
#model_name = "model_bradesco";
#model.save(model_name);

#print('Total time: ' + str((time.time() - start)) + ' secs')

tables = [];
allReviews = []
reviewCount=0
amostra = []
fname=sys.argv[1]
with codecs.open(fname, 'r',encoding='utf-8',errors='replace') as f:
        reader = csv.reader(f, delimiter=';' )
        for row in reader:
                amostra.append(row)

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
				allWordsVec.append(model[m.group(0)])
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


from sklearn.metrics import silhouette_score

sil = []
kmin = 2
kmax = 25

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


