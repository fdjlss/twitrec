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
		matrix_doc = np.vstack((matrix_doc, model[token]))
	matrix_doc = np.delete(matrix_doc, 0, 0) #Elimina la primera fila de sólo ceros
	vec_doc = max_pool(np_matrix= matrix_doc)
	return vec_doc

def docs2vecs(model):
	# ids2vec = {}
	# url = solr + '/query?q=*:*&rows=100000'
	# docs = json.loads( urlopen(url).read().decode('utf8') )
	# docs = docs['response']['docs']
	# i = 0
	# for doc in docs:
	# 	i+=1
	# 	goodreadsId = str( doc['goodreadsId'][0] )
	# 	logging.info("{0} de {1}. Doc: {2}".format(i, len(docs), goodreadsId))
	# 	ids2vec[goodreadsId] = doc2vec(document= doc, model= model)
	# del docs
	# return ids2vec
	ids2vec = {}
	flat_docs = np.load('./w2v-tmp/flattened_docs_fe.npy').item()
	i = 0
	for bookId, flat_doc in flat_docs.items():
		i+=1
		logging.info("{0} de {1}. Doc: {2}".format(i, len(flat_docs), bookId))
		ids2vec[bookId] = doc2vec(list_document= flat_doc, model= model)
	return ids2vec


# Para el modo 1
# MÉTODO DESCARTADO!
def sim_matrix(doc_vecs):
	# sim_matrix = np.zeros(shape=(len(doc_vecs), len(doc_vecs)), dtype=float)
	sim_dict = dict((bookId, {}) for bookId in doc_vecs)
	delta = 0
	key_list = list(doc_vecs.keys())

	i=0
	for bookId1 in key_list:
		i+=1
		initial = time.time()
		logging.info("i={}. delta time={}".format(i, delta))
		sims = {}
		for bookId2 in key_list:
			sims[bookId2] = 1 - spatial.distance.cosine(doc_vecs[bookId1], doc_vecs[bookId2])
		#Dejamos los 100 libros más parecidos
		sims = sorted(sims.items(), key=operator.itemgetter(1), reverse=True) #[(<bookId>, MAYOR sim), ..., (<bookId>, menor sim)]
		for j in range(1000):
			sim_dict[bookId1][sims[j][0]] = sims[j][1]
		delta = time.time() - initial

	return sim_dict

# Para el modo 2
def user2vec(list_user, model):
	# MAX POOLING
	matrix_doc = np.zeros((model.vector_size,), dtype=float)
	for token in list_user:
		matrix_doc = np.vstack((matrix_doc, model[token]))
	matrix_doc = np.delete(matrix_doc, 0, 0) #Elimina la primera fila de puros ceros
	vec_doc = max_pool(np_matrix= matrix_doc)	
	return vec_doc

def users2vecs(model):
	ids2vec = {}
	flat_users = np.load('./w2v-tmp/flattened_users_fe.npy').item()
	i = 0
	for userId, flat_user in flat_users.items():
		i+=1
		logging.info("USERS 2 VECS. {0} de {1}. User: {2}".format(i, len(flat_users), userId))
		ids2vec[userId] = user2vec(list_user= flat_user, model= model)
	return ids2vec
#--------------------------------#


def option1_protocol_evaluation(data_path, N):
	# userId='113447232' 285597345
	test_c  = consumption(ratings_path=data_path+'test/test_N'+str(N)+'.data', rel_thresh=0, with_ratings=True)
	train_c = consumption(ratings_path=data_path+'eval_train_N'+str(N)+'.data', rel_thresh=0, with_ratings=False)
	MRRs   = []
	nDCGs  = []
	APs    = []
	Rprecs = []
	docs2vec = np.load('./w2v-tmp/docs2vec.npy').item()
	num_to_grId = np.load('./w2v-tmp/num_to_grId.npy').item()
	grId_to_num = np.load('./w2v-tmp/grId_to_num.npy').item()
	t = AnnoyIndex(300)
	t.load('./w2v-tmp/doc_vecs_t100.tree')
	# sim_dict = np.load('./w2v-tmp/sim_matrix.npy').item() #OLD 1

	i = 1
	# sampled_user_ids = random.sample(test_c.keys(), 200)
	for userId in test_c: #sampled_user_ids:
		logging.info("MODO 1. {0} de {1}. User ID: {2}".format(i, len(test_c), userId))
		i += 1
		# stream_url = solr + '/query?rows=1000&q=goodreadsId:{ids}'
		# ids_string = encoded_itemIds(item_list=train_c[userId])
		# url        = stream_url.format(ids=ids_string)
		# response   = json.loads( urlopen(url).read().decode('utf8') )
		# try:
		# 	docs     = response['response']['docs']
		# except TypeError as e:
		# 	continue

		book_recs = []
		for bookId in train_c[userId]:#for user_doc in test_c[userId]:#docs:
			# user_bookId = str(user_doc['goodreadsId'][0]) #id de libro consumido por user
			
			# cosines = sim_dict[user_bookId] #OLD 1
			# cosines = dict((bookId, 0.0) for bookId in docs2vec) #OLD 2
			# for bookId in docs2vec: #ids de libros en la DB
			try:
				docs = t.get_nns_by_item(grId_to_num[bookId], 500)
				book_recs.append( [ str(num_to_grId[doc_num]) for doc_num in docs ] )
			except KeyError as e:
				logging.info("{} ES UNO DE LOS LIBROS CUYO HTML NO PUDO SER DESCARGADO. PROSIGUIENDO CON EL SIGUIENTE LIBRO..".format(bookId))
				continue
				#OLD 2
				# if bookId == user_bookId: continue 
				# cosines[bookId] = 1 - spatial.distance.cosine(docs2vec[bookId], docs2vec[user_bookId]) #1 - dist = similarity
			#OLD 2
			# sorted_sims = sorted(cosines.items(), key=operator.itemgetter(1), reverse=True) #[(<grId>, MAYOR sim), ..., (<grId>, menor sim)]
			# book_recs.append( [ bookId for bookId, sim in sorted_sims ] )

		book_recs = flatten_list(list_of_lists=book_recs, rows=len(book_recs[0])) #rows=len(sorted_sims))
		book_recs = remove_consumed(user_consumption=train_c[userId], rec_list=book_recs)
		try:
			recs    = user_ranked_recs(user_recs=book_recs, user_consumpt=test_c[userId])
		except KeyError as e:
			logging.info("Usuario {0} del fold de train (total) no encontrado en fold de 'test'".format(userId))
			continue

		####################################
		mini_recs = dict((k, recs[k]) for k in list(recs.keys())[:N]) #Python 3.x: .keys() devuelve una vista, no una lista
		nDCGs.append( nDCG(recs=mini_recs, alt_form=False, rel_thresh=False) )
		APs.append( AP_at_N(n=N, recs=mini_recs, rel_thresh=1) )
		MRRs.append( MRR(recs=mini_recs, rel_thresh=1) )
		Rprecs.append( R_precision(n_relevants=N, recs=mini_recs) )
		####################################

	with open('TwitterRatings/word2vec/option1_protocol.txt', 'a') as file:
		file.write( "EXTREMES FILTERED N=%s, normal nDCG=%s, MAP=%s, MRR=%s, R-precision=%s\n" % \
				(N, mean(nDCGs), mean(APs), mean(MRRs), mean(Rprecs)) )



def option2_protocol_evaluation(data_path, N):
	test_c  = consumption(ratings_path=data_path+'test/test_N'+str(N)+'.data', rel_thresh=0, with_ratings=True)
	train_c = consumption(ratings_path=data_path+'eval_train_N'+str(N)+'.data', rel_thresh=0, with_ratings=False)
	MRRs   = []
	nDCGs  = []
	APs    = []
	Rprecs = []
	docs2vec  = np.load('./w2v-tmp/docs2vec.npy').item()
	users2vec = np.load('./w2v-tmp/users2vec.npy').item()

	i = 1
	# sampled_user_ids = random.sample(test_c.keys(), 200)
	for userId in test_c:#sampled_user_ids:
		logging.info("MODO 2. {0} de {1}. User ID: {2}".format(i, len(test_c), userId))
		i += 1

		cosines = dict((bookId, 0.0) for bookId in docs2vec)
		for bookId in docs2vec:
			cosines[bookId] = 1 - spatial.distance.cosine(users2vec[userId], docs2vec[bookId]) #1 - dist = similarity

		sorted_sims = sorted(cosines.items(), key=operator.itemgetter(1), reverse=True) #[(<grId>, MAYOR sim), ..., (<grId>, menor sim)]
		book_recs   = [ bookId for bookId, sim in sorted_sims ]
		book_recs   = remove_consumed(user_consumption=train_c[userId], rec_list=book_recs)
		try:
			recs      = user_ranked_recs(user_recs=book_recs, user_consumpt=test_c[userId])
		except KeyError as e:
			logging.info("Usuario {0} del fold de train (total) no encontrado en fold de 'test'".format(userId))
			continue

		####################################
		mini_recs = dict((k, recs[k]) for k in list(recs.keys())[:N]) #Python 3.x: .keys() devuelve una vista, no una lista
		nDCGs.append( nDCG(recs=mini_recs, alt_form=False, rel_thresh=False) )
		APs.append( AP_at_N(n=N, recs=mini_recs, rel_thresh=1) )
		MRRs.append( MRR(recs=mini_recs, rel_thresh=1) )
		Rprecs.append( R_precision(n_relevants=N, recs=mini_recs) )
		####################################

	with open('TwitterRatings/word2vec/option2_protocol.txt', 'a') as file:
		file.write( "EXTREMES FILTERED N=%s, normal nDCG=%s, MAP=%s, MRR=%s, R-precision=%s\n" % \
				(N, mean(nDCGs), mean(APs), mean(MRRs), mean(Rprecs)) )


def main():
	data_path = 'TwitterRatings/funkSVD/data/'
	solr = 'http://localhost:8983/solr/grrecsys'
	## Modelo w2v Google 300 ##
	model_eng = KeyedVectors.load_word2vec_format('/home/jschellman/gensim-data/word2vec-google-news-300/word2vec-google-news-300', binary=True)
	## Modelo FT Wiki + UMBC webbase + statmt.org news 300 ##
	# model = KeyedVectors.load_word2vec_format('/home/jschellman/gensim-data/fasttext-wiki-news-subwords-300/fasttext-wiki-news-subwords-300.gz')
	## Modelo glove Twitter 200 ##
	# model = KeyedVectors.load_word2vec_format('/home/jschellman/gensim-data/glove-twitter-200/glove-twitter-200.txt')

	## Mapeo book Id -> vec_book Para modo 1 y 2 ##:
	dict_docs =	docs2vecs(model= model_eng)
	np.save('./w2v-tmp/docs2vec.npy', dict_docs)
	## DONE ##

	## Para modo 1 ##
	# docs2vec  = np.load('./w2v-tmp/docs2vec.npy').item()
	# dict_sim = sim_matrix(doc_vecs= docs2vec)
	# np.save('./w2v-tmp/sim_matrix.npy', dict_sim)
	## DONE ##

	## Para modo 2
	dict_users = users2vecs(model= model_eng)
	np.save('./w2v-tmp/users2vec.npy', dict_users)
	#Por ahora no:
	# model_esp = KeyedVectors.load_word2vec_format('/home/jschellman/fasttext-sbwc.3.6.e20.vec')

	# CORRER annoy_indexer ANTES DE..
	# for N in [5, 10, 15, 20]:
	# 	option1_protocol_evaluation(data_path= data_path, N=N)
	# 	option2_protocol_evaluation(data_path= data_path, N=N)
	
	
if __name__ == '__main__':
	main()
