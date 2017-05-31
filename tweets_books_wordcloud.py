# coding=utf-8

from os import path, listdir
from os.path import isfile, join
from wordcloud import WordCloud, STOPWORDS
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

# Users
#################################################################
# tweets_path = 'TwitterRatings/users_goodreads/'
# filenames = [ tweets_path + f for f in listdir(tweets_path) ]

# not_words = ['https', 'co', 'goodreads', 'review', 'bit', 'ly', 'id', 'user_status', 'html', 'spref', 'goo', 'gl', 'fb', 'facebook', \
# 							'show', 'star', 'instagram', 'blog', 'YouTube', 'Marked', 'page', 'bloglovin', 'amazon', 'dp', 'youtu', 'pinterest', 'pin']

# text = ''
# for filename in filenames:
# 	tweets = open(filename, 'r', encoding='utf-8', errors='ignore').read()
# 	text = text + tweets

# stop_words=set( list(STOPWORDS) + not_words)
# twitter_mask = np.array( Image.open("twitter_mask.png") )


# wordcloud = WordCloud(background_color="white", max_font_size=50, stopwords=stop_words, max_words=500, mask=twitter_mask).generate(text)
# plt.figure()
# plt.imshow(wordcloud, interpolation="bilinear")
# plt.axis("off")
# plt.savefig('tweets_wordcloud.png')
#################################################################


# Books
#################################################################
books_path = 'TwitterRatings/items_goodreads_sampled/'
filenames = [ books_path + f for f in listdir(books_path) ]

text = ''
for i in range(0, len(filenames)):
	info = open(filenames[i], 'r', encoding='utf-8', errors='ignore').read()
	text = text + info

	if i%500==0: print( "Viendo libro {0} de {1}".format(i, len(filenames)) )

stop_words=set( list(STOPWORDS) )
book_mask = np.array( Image.open("book_mask.png") )

print("Generando wordcloud")
wordcloud = WordCloud(background_color="white", max_font_size=50, stopwords=stop_words, max_words=500, mask=book_mask).generate(text)
print("Ploteando")
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
print("Guardando")
plt.savefig('books_wordcloud.png')
#################################################################