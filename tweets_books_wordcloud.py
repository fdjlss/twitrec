# coding=utf-8

from os import path, listdir
from os.path import isfile, join
from wordcloud import WordCloud, STOPWORDS
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

#################################################################
tweets_path = 'TwitterRatings/users_goodreads/'
filenames = [ tweets_path + f for f in listdir(tweets_path) ]

not_words = ['https', 'co', 'goodreads', 'review', 'bit', 'ly', 'id', 'user_status', 'html', 'spref', 'goo', 'gl', 'fb', 'facebook', \
							'show', 'star', 'instagram', 'blog', 'YouTube', 'Marked', 'page', 'bloglovin', 'amazon', 'dp', 'youtu', 'pinterest', 'pin']

text = ''
for filename in filenames:
	tweets = open(filename, 'r', encoding='utf-8', errors='ignore').read()
	text = text + tweets

stop_words=set( list(STOPWORDS) + not_words)
twitter_mask = np.array( Image.open("twitter_mask.png") )


wordcloud = WordCloud(background_color="white", max_font_size=40, stopwords=stop_words, max_words=800, mask=twitter_mask).generate(text)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.savefig('tweets_wordcloud.png')

#################################################################