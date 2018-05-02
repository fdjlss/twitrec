# coding=utf-8

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import os
import re, json
from urllib.parse import urlencode, quote_plus
from urllib.request import urlopen
from svd_evaluation import mean, stdev, MRR, rel_div, DCG, iDCG, nDCG, P_at_N, AP_at_N, R_precision, consumption, user_ranked_recs, opt_value
import numpy as np
from gensim.models import KeyedVectors
from gensim.models.word2vec import Word2Vec
from gensim.parsing.preprocessing import preprocess_string, strip_tags, strip_punctuation, strip_multiple_whitespaces, strip_numeric
from nltk.corpus import stopwords
from textblob.blob import TextBlob
stop_words = set(stopwords.words('spanish') + stopwords.words('english') + stopwords.words('german') + \
								 stopwords.words('french') + stopwords.words('italian') + stopwords.words('portuguese'))
CUSTOM_FILTERS = [lambda x: x.lower(), strip_tags, strip_punctuation, strip_multiple_whitespaces, strip_numeric]

#-----"PRIVATE" METHODS----------#
# Los mismos que solr_evaluation
# ya que gensim no está para 2.x 
def encoded_itemIds(item_list):
	ids_string = '('
	for itemId in item_list: ids_string += itemId + '%20OR%20'
	ids_string = ids_string[:-8] # para borrar el último "%20OR%20"
	ids_string += ')'
	return ids_string
def flatten_list(list_of_lists, rows):
	"""Eliminamos duplicados manteniendo orden"""
	flattened = []
	for i in range(0, rows): #asumimos que todas las listas tienen largo "rows"
		for j in range(0, len(list_of_lists)):
			try:
				flattened.append( list_of_lists[j][i] )
			except IndexError as e:
				continue
	return sorted(set(flattened), key=lambda x: flattened.index(x))
def remove_consumed(user_consumption, rec_list):
	l = rec_list
	for itemId in rec_list:
		if itemId in user_consumption: l.remove(itemId)
	return l
def doc2vec(document, model):
	flat_doc = ""
	for field in document:
		if not isinstance(document[field], list): continue #No tomamos en cuenta los campos 'id' y '_version_': auto-generados por Solr
		for value in document[field]:

			## Detección y traducción ##
			if field=='author.authors.authorName' or field=='author.authorBio' or field=='description' or field=='quotes.quoteText':
				value_blob = TextBlob(value)
				try:
					if value_blob.detect_language() != 'en':
						try: 
							value = value_blob.translate(to='en')
						except Exception as e: 
							value = value #e = NotTranslated('Translation API returned the input string unchanged.',)
				except Exception as e:
					value = value #e = TranslatorError('Must provide a string with at least 3 characters.')
			############################
			
			flat_doc += str(value)+' ' #Se aplana el documento en un solo string

	flat_doc = preprocess_string(flat_doc, CUSTOM_FILTERS) #Preprocesa el string
	flat_doc = [w for w in flat_doc if w not in stop_words.union(['www', '"'])] #Remueve stop words
	vec_doc = np.zeros((model.vector_size,), dtype=float)
	for token in flat_doc:
		if token not in model.vocab: continue
		vec_doc += model[token]
	return vec_doc
def all_doc2vec(solr, model):
	ids2vec = {}
	url = solr + '/query?q=*:*&rows=100000'
	docs = json.loads( urlopen(url).read().decode('utf8') )
	docs = docs['response']['docs']
	i = 0
	for doc in docs:
		i+=1
		goodreadsId = str( doc['goodreadsId'][0] )
		logging.info("{0} de {1}. Doc: {2}".format(i, len(docs), goodreadsId))
		ids2vec[goodreadsId] = doc2vec(document= doc, model= model)
	del docs
	return ids2vec
#--------------------------------#


def protocol_evaluation(data_path, solr, N, model):
	# userId='113447232'
	test_c  = consumption(ratings_path=data_path+'test/test_N'+str(N)+'.data', rel_thresh=0, with_ratings=True)
	train_c = consumption(ratings_path=data_path+'eval_train_N'+str(N)+'.data', rel_thresh=0, with_ratings=False)
	MRRs   = []
	nDCGs  = []
	APs    = []
	Rprecs = []

	for userId in test_c:
		stream_url = solr + '/query?q=goodreadsId:{ids}'
		ids_string = encoded_itemIds(item_list=train_c[userId])
		url        = stream_url.format(ids=ids_string)
		response   = json.loads( urlopen(url).read().decode('utf8') )
		try:
			docs     = response['response']['docs']
		except TypeError as e:
			continue

		vec_user = np.zeros((model.vector_size,), dtype=float)
		for doc in docs:
			vec_doc = doc2vec(document= doc, model= model)
			vec_user += vec_doc



		book_recs  = [ str(doc['goodreadsId'][0]) for doc in docs] 
		book_recs  = remove_consumed(user_consumption=train_c[userId], rec_list=book_recs)
		try:
			recs     = user_ranked_recs(user_recs=book_recs, user_consumpt=test_c[userId])
		except KeyError as e:
			logging.info("Usuario {0} del fold de train (total) no encontrado en fold de 'test'".format(userId))
			continue

def main():
	data_path = 'TwitterRatings/funkSVD/data/'
	solr = 'http://localhost:8983/solr/grrecsys'
	model_eng = KeyedVectors.load_word2vec_format('/home/jschellman/gensim-data/word2vec-google-news-300/word2vec-google-news-300', binary=True)
	ids2vec =	all_doc2vec(solr= solr, model= model_eng)
	np.save('./w2v-tmp/ids2vec.npy', ids2vec)
	#Por ahora no:
	# model_esp = KeyedVectors.load_word2vec_format('/home/jschellman/fasttext-sbwc.3.6.e20.vec')
	#Sólo por ahora para guardar el diccionario de vectores:

	# for N in [5, 10, 15, 20]:
	# 	protocol_evaluation(data_path= data_path, solr= solr, N=N, model= model_eng)

if __name__ == '__main__':
	main()
