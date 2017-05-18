"""
Crea modelo word2vec con vocabulario proveniente de tweets e info. de libros
"""

#--------------------------------#
import gensim, logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import smart_open, os
from stop_words import get_stop_words
from gensim.utils import simple_preprocess 
#--------------------------------#


class MySentences(object):
  def __init__(self, dirname):
  	self.dirname = dirname

  def __iter__(self):
  	for path in self.dirname:
	  	for file in os.listdir(path):
	  		for line in open(os.path.join(path, file), 'r', encoding='ISO-8859-1'):
	  			yield simple_preprocess(line)


sentences = MySentences(['TwitterRatings/users_goodreads/', 'TwitterRatings/items_goodreads/']) # a memory-friendly iterator

model = gensim.models.Word2Vec(sentences, min_count=5, size=100, workers=4)
model.save('./tmp/min_count5_size_100.w2v')
print(len(model.vocab))
#####################################################

