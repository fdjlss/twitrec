# coding=utf-8

import gensim.downloader as api
from gensim.models.word2vec import Word2Vec
from gensim.parsing.preprocessing import preprocess_string
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import os
from smart_open import smart_open
from nltk.corpus import stopwords
stop_words = set(stopwords.words('spanish') + stopwords.words('english'))
CUSTOM_FILTERS = [lambda x: x.lower(), strip_tags, strip_punctuation, strip_multiple_whitespaces, strip_numeric]

class SmartCorpus(object):
  def __init__(self, dirname):
    self.dirname = dirname
  def __iter__(self):
    for fname in os.listdir(self.dirname):
      for line in smart_open(os.path.join(self.dirname, fname), 'rb'):
        sentence = preprocess_string(line, CUSTOM_FILTERS)
        yield [w for w in sentence if w not in stop_words]

model = api.load('word2vec-google-news-300')
model.save('w2v-google-news')

spanish_corpus = SmartCorpus('/mnt/f90f82f4-c2c7-4e53-b6af-7acc6eb85058/datasets/word2vec_corpus/SBWCE_corpus/spanish_billion_words/') # SBWCE

model.build_vocab(spanish_corpus, update=True)
model.train(spanish_corpus, total_examples=model.corpus_count, epochs=model.iter)
model.save('w2v-bilingual')
