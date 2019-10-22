
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
import sqlite3;
import time;
import codecs
import sys;
import multiprocessing;
from wordcloud import WordCloud, ImageColorGenerator
import matplotlib.pyplot as plt;
from itertools import cycle;

reload(sys)
sys.setdefaultencoding('utf8')

print("Training model...");
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

num_workers=10
num_features=36
min_word_count=4
context=3

model = word2vec.Word2Vec(documents, workers=num_workers, \
            size=num_features, min_count = min_word_count, \
            window = context);
# We don't plan on training the model any further, so calling 
# init_sims will make the model more memory efficient by normalizing the vectors in-place.
#model.init_sims(replace=True);
# Save the model
#model_name = "model_bradesco";
#model.save(model_name);

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

#keys = ['bradesco', 'aplicativo', 'problema', 'facil', 'preciso', 'conta', 'cartao', 'nubank', 'compras', 'extrato', 'fatura', 'gasto', ];

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
                if (reviewCount % 500 == 0):
                        print ("Review ",reviewCount," com ",validWords," palavras")
                if validWords > 0:
                        reviewVec.append(value[3])
                        reviewVec.extend(np.mean(np.array(allWordsVec), axis=0))
		else:
			reviewVec.append(0)
                        reviewVec.extend([0] * num_features)
		csvWriter.writerow(reviewVec)

print('output to output.csv')
