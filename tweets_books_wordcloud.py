# coding=utf-8

from os import path, listdir
from os.path import isfile, join
from wordcloud import WordCloud, STOPWORDS
import numpy as np
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

#################################################################
tweets_path = 'TwitterRatings/users_goodreads/'
filenames = [ tweets_path + f for f in listdir(tweets_path) ]

not_words = ['https', 'co', 'goodreads', 'review', 'bit', 'ly', 'id', 'user_status', 'html', 'spref', 'goo', 'gl', 'fb', 'facebook']

text = ''
for filename in filenames:
	tweets = open(filename, 'r', encoding='utf-8', errors='ignore').read()
	text = text + tweets

stop_words=set( list(STOPWORDS) + not_words)

wordcloud = WordCloud(max_font_size=40, stopwords=stop_words, max_words=500).generate(text)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.savefig('tweets_wordcloud.png')

#################################################################