# coding=utf-8

import time
import random
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import os
import json
from urllib.parse import urlencode, quote_plus
from urllib.request import urlopen
from svd_evaluation import mean, stdev, MRR, rel_div, DCG, iDCG, nDCG, P_at_N, AP_at_N, R_precision, consumption, remove_consumed, user_ranked_recs, opt_value
from wmd_evaluation import flat_doc, flat_user
import numpy as np
from scipy import spatial
import operator
from gensim.models import KeyedVectors
from gensim.models.word2vec import Word2Vec
from gensim.parsing.preprocessing import preprocess_string, strip_tags, strip_punctuation, strip_multiple_whitespaces, strip_numeric
from nltk.corpus import stopwords
from textblob.blob import TextBlob
from annoy import AnnoyIndex
stop_words = set(stopwords.words('spanish') + stopwords.words('english') + stopwords.words('german') + stopwords.words('arabic') + \
								 stopwords.words('french') + stopwords.words('italian') + stopwords.words('portuguese') + ['goodreads', 'http', 'https', 'www', '"'])
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
def max_pool(np_matrix):
	rows, cols = np_matrix.shape
	max_pooled = []
	for j in range(cols):
		max_pooled.append( max(np_matrix[:,j]) )
	return np.array(max_pooled)

def doc2vec(list_document, model):
	# MAX POOLING
	matrix_doc = np.zeros((model.vector_size,), dtype=float)
	for token in list_document:
		# Debiera estar manejado en las funciones de wmd_evaluation,
		# pero habría que hacer este manejo si considero el vocab de Twitter
		# y convierto en vector con modelo Wiki, por ej..
		if token not in model.vocab: continue 
		matrix_doc = np.vstack((matrix_doc, model[token]))
	matrix_doc = np.delete(matrix_doc, 0, 0) #Elimina la primera fila de sólo ceros
	vec_doc = max_pool(np_matrix= matrix_doc)
	return vec_doc

def docs2vecs(model):
	ids2vec = {}
	flat_docs = np.load('./w2v-tmp/flattened_docs_fea075b1.npy').item()
	i = 0
	for bookId, flat_doc in flat_docs.items():
		i+=1
		logging.info("{0} de {1}. Doc: {2}".format(i, len(flat_docs), bookId))
		ids2vec[bookId] = doc2vec(list_document= flat_doc, model= model)
	return ids2vec

def users2vecs(model, representation):
	ids2vec = {}
	flat_users = np.load('./w2v-tmp/flattened_users_'+str(representation)+'.npy').item()
	i = 0
	for userId, flat_user in flat_users.items():
		i+=1
		logging.info("USERS 2 VECS. {0} de {1}. User: {2}".format(i, len(flat_users), userId))
		ids2vec[userId] = doc2vec(list_document= flat_user, model= model)
	return ids2vec
#--------------------------------#


def option1_protocol_evaluation(data_path, which_model, metric):
	# userId='113447232' 285597345
	test_c  = consumption(ratings_path=data_path+'test/test_N20.data', rel_thresh=0, with_ratings=True)
	train_c = consumption(ratings_path=data_path+'eval_train_N20.data', rel_thresh=0, with_ratings=False)
	MRRs   = dict((N, []) for N in [5, 10, 15, 20])
	nDCGs  = dict((N, []) for N in [5, 10, 15, 20])
	APs    = dict((N, []) for N in [5, 10, 15, 20])
	Rprecs = dict((N, []) for N in [5, 10, 15, 20])
	
	docs2vec = np.load('./w2v-tmp/'+which_model+'/docs2vec_'+which_model+'.npy').item()
	if which_model == 'twit':
		vector_size = 200
	else:
		vector_size = 300
	t = AnnoyIndex(vector_size, metric=metric)
	t.load('./w2v-tmp/'+which_model+'/doc_vecs_t100_'+metric+'_'+which_model+'.tree')
	num_to_grId = np.load('./w2v-tmp/'+which_model+'/num_to_grId_'+metric+'_'+which_model+'.npy').item()
	grId_to_num = np.load('./w2v-tmp/'+which_model+'/grId_to_num_'+metric+'_'+which_model+'.npy').item()

	i = 1
	for userId in test_c:
		logging.info("MODO 1. {0} de {1}. User ID: {2}".format(i, len(test_c), userId))
		i += 1

		book_recs = []
		for bookId in train_c[userId]:

			try:
				docs = t.get_nns_by_item(grId_to_num[bookId], 500)
				book_recs.append( [ str(num_to_grId[doc_num]) for doc_num in docs ] )
			except KeyError as e:
				logging.info("{} ES UNO DE LOS LIBROS CUYO HTML NO PUDO SER DESCARGADO. PROSIGUIENDO CON EL SIGUIENTE LIBRO..".format(bookId))
				continue

		book_recs = flatten_list(list_of_lists=book_recs, rows=len(book_recs[0])) #rows=len(sorted_sims))
		book_recs = remove_consumed(user_consumption=train_c[userId], rec_list=book_recs)
		try:
			recs    = user_ranked_recs(user_recs=book_recs, user_consumpt=test_c[userId])
		except KeyError as e:
			logging.info("Usuario {0} del fold de train (total) no encontrado en fold de 'test'".format(userId))
			continue

		for N in [5, 10, 15, 20]:
			mini_recs = dict((k, recs[k]) for k in list(recs.keys())[:N])
			MRRs[N].append( MRR(recs=mini_recs, rel_thresh=1) )
			nDCGs[N].append( nDCG(recs=mini_recs, alt_form=False, rel_thresh=False) )		
			APs[N].append( AP_at_N(n=N, recs=recs, rel_thresh=1) )
			Rprecs[N].append( R_precision(n_relevants=N, recs=mini_recs) )

	with open('TwitterRatings/word2vec/option1_protocol_'+which_model+'.txt', 'a') as file:
		file.write( "METRIC: %s\n" % (metric, representation) )	

	for N in [5, 10, 15, 20]:
		with open('TwitterRatings/word2vec/option1_protocol_'+which_model+'.txt', 'a') as file:
			file.write( "N=%s, nDCG=%s, MAP=%s, MRR=%s, R-precision=%s\n" % \
				(N, mean(nDCGs[N]), mean(APs[N]), mean(MRRs[N]), mean(Rprecs[N])) )	



def option2_protocol_evaluation(data_path, which_model, metric, representation):
	test_c  = consumption(ratings_path=data_path+'test/test_N20.data', rel_thresh=0, with_ratings=True)
	train_c = consumption(ratings_path=data_path+'eval_train_N20.data', rel_thresh=0, with_ratings=False)
	MRRs   = dict((N, []) for N in [5, 10, 15, 20])
	nDCGs  = dict((N, []) for N in [5, 10, 15, 20])
	APs    = dict((N, []) for N in [5, 10, 15, 20])
	Rprecs = dict((N, []) for N in [5, 10, 15, 20])
	docs2vec  = np.load('./w2v-tmp/'+which_model+'/docs2vec_'+which_model+'.npy').item()
	users2vec = np.load('./w2v-tmp/'+which_model+'/users2vec_'+representation+'_'+which_model+'.npy').item()
	
	i = 1
	for userId in test_c:
		logging.info("MODO 2. {0} de {1}. User ID: {2}".format(i, len(test_c), userId))
		i += 1

		distances = dict((bookId, 0.0) for bookId in docs2vec)
		for bookId in docs2vec:
			if metric=='angular':
				distances[bookId] = spatial.distance.cosine(users2vec[userId], docs2vec[bookId])
			elif metric=='euclidean':
				distances[bookId] = spatial.distance.euclidean(users2vec[userId], docs2vec[bookId])

		sorted_sims = sorted(distances.items(), key=operator.itemgetter(1), reverse=False) #[(<grId>, MENOR dist), ..., (<grId>, MAYOR dist)]
		book_recs   = [ bookId for bookId, sim in sorted_sims ]
		book_recs   = remove_consumed(user_consumption=train_c[userId], rec_list=book_recs)
		try:
			recs      = user_ranked_recs(user_recs=book_recs, user_consumpt=test_c[userId])
		except KeyError as e:
			logging.info("Usuario {0} del fold de train (total) no encontrado en fold de 'test'".format(userId))
			continue

		for N in [5, 10, 15, 20]:
			mini_recs = dict((k, recs[k]) for k in list(recs.keys())[:N])
			MRRs[N].append( MRR(recs=mini_recs, rel_thresh=1) )
			nDCGs[N].append( nDCG(recs=mini_recs, alt_form=False, rel_thresh=False) )		
			APs[N].append( AP_at_N(n=N, recs=recs, rel_thresh=1) )
			Rprecs[N].append( R_precision(n_relevants=N, recs=mini_recs) )

	with open('TwitterRatings/word2vec/option2_protocol_'+which_model+'.txt', 'a') as file:
		file.write( "METRIC: %s \t REPR: %s\n" % (metric, representation) )	

		with open('TwitterRatings/word2vec/option2_protocol_'+which_model+'.txt', 'a') as file:
			file.write( "N=%s, nDCG=%s, MAP=%s, MRR=%s, R-precision=%s\n" % \
				(N, mean(nDCGs[N]), mean(APs[N]), mean(MRRs[N]), mean(Rprecs[N])) )	

def main():
	data_path = 'TwitterRatings/funkSVD/data/'
	solr = 'http://localhost:8983/solr/grrecsys'
	models = ['google', 'wiki', 'twit']

	## Para modo 1 y 2
	## Mapeo book Id -> vec_book
	# dict_docs =	docs2vecs(model= model)
	# np.save('./w2v-tmp/'+which_model+'/docs2vec_'+which_model+'.npy', dict_docs)
	
	# ## Para modo 2
	# for which_model in models:
	# 	if which_model=='google':
	# 		representation = 'tweets'
	# 		# Modelo w2v Google 300 ##
	# 		model = KeyedVectors.load_word2vec_format('/home/jschellman/gensim-data/word2vec-google-news-300/word2vec-google-news-300', binary=True)
	# 	elif which_model=='wiki':
	# 		representation = 'tweets'
	# 		# Modelo FT Wiki + UMBC webbase + statmt.org news 300 ##
	# 		model = KeyedVectors.load_word2vec_format('/home/jschellman/gensim-data/fasttext-wiki-news-subwords-300/fasttext-wiki-news-subwords-300.gz')
	# 	elif which_model=='twit':
	# 		representation = 'books'
	# 		# Modelo glove (convertido a w2v) Twitter 200 ##
	# 		model = KeyedVectors.load_word2vec_format('/home/jschellman/gensim-data/glove-twitter-200/glove-twitter-200.txt')

	# 	dict_users = users2vecs(model= model, representation=representation)
	# 	np.save('./w2v-tmp/'+which_model+'/users2vec_'+representation+'_'+which_model+'.npy', dict_users)


	# CONVERTIR FLATTENED USERS AS TWEETS A USERS2VEC_TWEET PARA CADA GOOGLE Y WIKI
	# .. PARA TWEET CONVERTIR FLAT USERS AS BOOKS A USERS2VEC_BOOKS

	for metric in ['angular']:
		for which_model in ['twit']:
			for representation in ['books', 'tweets', 'mix']:
				option2_protocol_evaluation(data_path= data_path, which_model=which_model, metric=metric, representation=representation)			

	for metric in ['euclidean']:
		for which_model in ['google', 'wiki', 'twit']:
			# CORRER annoy_indexer ANTES DE..
			# for N in [5, 10, 15, 20]:
			option1_protocol_evaluation(data_path= data_path, which_model=which_model, metric=metric)
				
			for representation in ['books', 'tweets', 'mix']:
				# for N in [5, 10, 15, 20]:
				option2_protocol_evaluation(data_path= data_path, which_model=which_model, metric=metric, representation=representation)
	
	
if __name__ == '__main__':
	main()
