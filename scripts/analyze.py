#!/usr/bin/env python
# -*- coding: utf-8 -*-
from gensim.models import word2vec;
from sklearn.cluster import KMeans;
from sklearn.neighbors import KDTree;
from wordcloud import WordCloud, ImageColorGenerator
import sys
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
import multiprocessing;
import matplotlib.pyplot as plt;
from itertools import cycle;

reload(sys)
sys.setdefaultencoding('utf8')

print("Training model...");
start = time.time();

def strip_accents(input_str):
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    only_ascii = nfkd_form.encode('ASCII', 'ignore')
    return only_ascii

def convert_com(text):
	text = re.sub('\?', ' pergunta ', text)
	text = re.sub('\!', ' exclama ', text)
	text = re.sub('🤔', ' pensativo ', text)
	text = re.sub('😍', ' apaixonado ', text)
	text = re.sub('🥰', ' papaixonado ', text) 
	text = re.sub('👊', ' soquinho ', text)
	text = re.sub('😢', ' chorando ', text)
	text = re.sub('👏', ' amen ', text)
	text = re.sub('😌', ' satisfeito ', text)
	text = re.sub('😤', ' bufando ', text)
	text = re.sub('😁', ' feliz ', text)
	text = re.sub('🙏', ' amen ', text)
	text = re.sub('🙌', ' exclama ', text)
	text = re.sub('🤙', ' hangloose ', text)
	text = re.sub('👍', ' joinha ', text)
	text = re.sub('☹️,', ' triste ', text)
	text = re.sub('😡', ' bravo ', text)
	text = re.sub('🤢', ' enjoado ', text)
	text = re.sub('❤️', ' coracao ', text)
	text = re.sub('😐', ' serio ', text)
	text = re.sub('\.\.\.', ' reticencias ', text)
	text = text.decode('utf-8').lower()
	text = strip_accents(text)
	text = re.sub('[ ]+', ' ', text)
	text = re.sub('[^0-9a-zA-Z_\ -]', '', text)
	return text

def read_input(input_file):
        """This method reads the input file which is in gzip format"""
        logging.info("reading file {0}...this may take a while".format(input_file))
        with gzip.open (input_file, 'rb') as f:
                for i, line in enumerate (f):
                        if (i%5000==0):
                                print ("read {0} reviews".format (i))
                        yield gensim.utils.simple_preprocess(convert_com(line))

documents = list (read_input ("./all_comments.txt.gz"))

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
model_name = "model_bradesco";
model.save(model_name);

print('Total time: ' + str((time.time() - start)) + ' secs')
Z = model.wv.syn0;
def clustering_on_wordvecs(word_vectors, num_clusters):
    # Initalize a k-means object and use it to extract centroids
    kmeans_clustering = KMeans(n_clusters = num_clusters, init='k-means++');
    idx = kmeans_clustering.fit_predict(word_vectors); 
    return kmeans_clustering.cluster_centers_, idx;

def get_top_words(index2word, k, centers, wordvecs):
    tree = KDTree(wordvecs);
    #Closest points for each Cluster center is used to query the closest 20 points to it.
    closest_points = [tree.query(np.reshape(x, (1, -1)), k=k) for x in centers];
    closest_words_idxs = [x[1] for x in closest_points];
    #Word Index is queried for each position in the above array, and added to a Dictionary.
    closest_words = {};
    for i in range(0, len(closest_words_idxs)):
        closest_words['Cluster #' + str(i)] = [index2word[j] for j in closest_words_idxs[i][0]]
    #A DataFrame is generated from the dictionary.
    df = pd.DataFrame(closest_words);
    df.index = df.index+1
    return df;

def display_cloud(cluster_num, cmap):
    wc = WordCloud(background_color="black", max_words=100, max_font_size=80, colormap=cmap);
    wordcloud = wc.generate(' '.join([word for word in top_words['Cluster #' + str(cluster_num)]]))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.savefig('cluster_' + str(cluster_num), bbox_inches='tight')

num_clusters=10

#centers, clusters = clustering_on_wordvecs(Z, num_clusters);

#top_words = get_top_words(model.wv.index2word, 50, centers, Z);
#print(top_words)

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
		value[2] = convert_com(value[2]).split(' ')
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
                if (reviewCount % 500 == 0):
                        print ("Review ",reviewCount," com ",validWords," palavras")
		reviewVec.append(value[3])
                if validWords > 0:
                        reviewVec.extend(np.mean(np.array(allWordsVec), axis=0))
		else:
                        reviewVec.extend([0] * num_features)
		csvWriter.writerow(reviewVec)

print('output to output.csv')
