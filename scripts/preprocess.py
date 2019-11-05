#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import argparse
import pprint
import gensim
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
import numpy as np;
import os;
import re;
import logging;
import csv;
import time;
import codecs
import sys;
import matplotlib.pyplot as plt;
from itertools import cycle;

def strip_accents(input_str):
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    only_ascii = nfkd_form.encode('ASCII', 'ignore')
    return only_ascii

def convert_com(text):
	text = re.sub('\?', ' pergunto ', text)
	text = re.sub('\!', ' exclamo ', text)
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
	text = re.sub('🙌', ' celebracao ', text)
	text = re.sub('🤙', ' hangloose ', text)
	text = re.sub('👍', ' curti ', text)
	text = re.sub('☹️,', ' triste ', text)
	text = re.sub('😡', ' bravo ', text)
	text = re.sub('🤢', ' enjoado ', text)
	text = re.sub('❤️', ' coracao ', text)
	text = re.sub('😐', ' serio ', text)
	text = re.sub('😀', ' feliz ', text)
	text = re.sub('😂', ' feliz ', text)
	text = re.sub('😃', ' feliz ', text)
	text = re.sub('😄', ' feliz ', text)
	text = re.sub('😅', ' suando ', text)
	text = re.sub('😆', ' feliz ', text)
	text = re.sub('😈', ' endiabrado ', text)
	text = re.sub('😉', ' piscadela ', text)
	text = re.sub('😊', ' corado ', text)
	text = re.sub('😌', ' satisfeito ', text)
	text = re.sub('😎', ' sunglasses ', text)
	text = re.sub('😏', ' desconfiado ', text)
	text = re.sub('🤐', ' quieto ', text)
	text = re.sub('🤑', ' endinheirado ', text)
	text = re.sub('🤒', ' corado ', text)
	text = re.sub('😓', ' suando ', text)
	text = re.sub('🤓', ' feliz ', text)
	text = re.sub('🤔', ' pensativo ', text)
	text = re.sub('😔', ' desapontado ', text)
	text = re.sub('😖', ' bravo ', text)
	text = re.sub('🤗', ' feliz ', text)
	text = re.sub('😘', ' feliz ', text)
	text = re.sub('🤘', ' hellyeah ', text)
	text = re.sub('😙', ' beijinho ', text)
	text = re.sub('😜', ' feliz ', text)
	text = re.sub('🔝', ' top ', text)
	text = re.sub('🤝', ' negocio ', text)
	text = re.sub('😞', ' triste ', text)
	text = re.sub('🤞', ' cruzando ', text)
	text = re.sub('😟', ' assustado ', text)
	text = re.sub('😡', ' bravo ', text)
	text = re.sub('😢', ' chorando ', text)
	text = re.sub('🤢', ' enjoado ', text)
	text = re.sub('😣', ' triste ', text)
	text = re.sub('🤣', ' feliz ', text)
	text = re.sub('😤', ' bufando ', text)
	text = re.sub('😥', ' suando ', text)
	text = re.sub('🤦', ' facepalm ', text)
	text = re.sub('😦', ' assustado ', text)
	text = re.sub('🤨', ' pensativo ', text)
	text = re.sub('😨', ' assustado ', text)
	text = re.sub('😩', ' triste ', text)
	text = re.sub('🤩', ' feliz ', text)
	text = re.sub('😪', ' cancado ', text)
	text = re.sub('😬', ' feliz ', text)
	text = re.sub('🤬', ' bravo ', text)
	text = re.sub('😭', ' chorando ', text)
	text = re.sub('😱', ' assustado ', text)
	text = re.sub('😳', ' assustado ', text)
	text = re.sub('😵', ' assustado ', text)
	text = re.sub('😶', ' serio ', text)
	text = re.sub('👀', ' deolho ', text)
	text = re.sub('🙇', ' amem ', text)
	text = re.sub('👋', ' tchau ', text)
	text = re.sub('👎', ' descurti ', text)
	text = re.sub('❤️', ' coracao ', text)
	text = re.sub('👺', ' diabo ', text)
	text = re.sub('🎉', ' festa ', text)
	text = re.sub('🎊', ' festa ', text)
	text = re.sub('💕', ' coracao ', text)
	text = re.sub('💖', ' coracao ', text)
	text = re.sub('💜', ' coracao ', text)
	text = re.sub('💩', ' coco ', text)
	text = re.sub('💰', ' dinheiro ', text)
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

