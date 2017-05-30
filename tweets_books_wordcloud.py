# coding=utf-8

from os import path
from os.path import isfile, join
from wordcloud import WordCloud
import numpy as np
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

tweets_path = 'TwitterRatings/users_goodreads/'
filenames = [ tweets_path + f for f in listdir(tweets_path) ]

text = ''
for filename in filenames:
	tweets = open(filename, 'r', encoding='utf-8', errors='ignore').read()
	text = text + tweets


wordcloud = WordCloud(max_font_size=40).generate(text)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.savefig('tweets_wordcloud.png')