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
import numpy as np
from scipy import spatial
import operator
from gensim.models import KeyedVectors
from gensim.models.word2vec import Word2Vec
from gensim.parsing.preprocessing import preprocess_string, strip_tags, strip_punctuation, strip_multiple_whitespaces, strip_numeric
from nltk.corpus import stopwords
from textblob.blob import TextBlob
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
	flat_doc = [w for w in flat_doc if w not in stop_words] #Remueve stop words
	flat_doc = [w for w in flat_doc if w in model.vocab] #Deja sólo palabras del vocabulario
	# MAX POOLING
	matrix_doc = np.zeros((model.vector_size,), dtype=float)
	for token in flat_doc:
		matrix_doc = np.vstack((matrix_doc, model[token]))
	matrix_doc = np.delete(matrix_doc, 0, 0) #Elimina la primera fila de puros ceros
	vec_doc = max_pool(np_matrix= matrix_doc)
	return vec_doc

def docs2vecs(solr, model):
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

# Para el modo 1
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
def user2vec(solr, consumption, model):
	stream_url = solr + '/query?rows=1000&q=goodreadsId:{ids}'
	ids_string = encoded_itemIds(item_list=consumption)
	url        = stream_url.format(ids=ids_string)
	response   = json.loads( urlopen(url).read().decode('utf8') )
	docs       = response['response']['docs']
	flat_doc = ""
	for document in docs:
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
	flat_doc = [w for w in flat_doc if w not in stop_words] #Remueve stop words
	flat_doc = [w for w in flat_doc if w in model.vocab] #Deja sólo palabras del vocabulario
	# MAX POOLING
	matrix_doc = np.zeros((model.vector_size,), dtype=float)
	for token in flat_doc:
		matrix_doc = np.vstack((matrix_doc, model[token]))
	matrix_doc = np.delete(matrix_doc, 0, 0) #Elimina la primera fila de puros ceros
	vec_doc = max_pool(np_matrix= matrix_doc)	
	return vec_doc

def users2vecs(solr, data_path, model):
	train_c = consumption(ratings_path=data_path+'eval_train_N5.data', rel_thresh=0, with_ratings=False)
	i = 0
	ids2vec = {}
	for userId in train_c:
		i+=1
		logging.info("USERS 2 VECS. {0} de {1}. User: {2}".format(i, len(train_c), userId))
		ids2vec[userId] = user2vec(solr= solr, consumption= train_c[userId], model= model)
	return ids2vec
#--------------------------------#


def option1_protocol_evaluation(data_path, solr, N):
	# userId='113447232' 285597345
	test_c  = consumption(ratings_path=data_path+'test/test_N'+str(N)+'.data', rel_thresh=0, with_ratings=True)
	train_c = consumption(ratings_path=data_path+'eval_train_N'+str(N)+'.data', rel_thresh=0, with_ratings=False)
	MRRs   = []
	nDCGs  = []
	APs    = []
	Rprecs = []
	docs2vec = np.load('./w2v-tmp/docs2vec.npy').item()
	# sim_dict = np.load('./w2v-tmp/sim_matrix.npy').item() #THE DREAM

	i = 1
	sampled_user_ids = random.sample(test_c.keys(), 100)
	for userId in sampled_user_ids: #test_c:
		logging.info("MODO 1. {0} de {1}. User ID: {2}".format(i, len(test_c), userId))
		i += 1
		stream_url = solr + '/query?rows=1000&q=goodreadsId:{ids}'
		ids_string = encoded_itemIds(item_list=train_c[userId])
		url        = stream_url.format(ids=ids_string)
		response   = json.loads( urlopen(url).read().decode('utf8') )
		try:
			docs     = response['response']['docs']
		except TypeError as e:
			continue

		book_recs = []
		for user_doc in docs:
			user_bookId = str(user_doc['goodreadsId'][0]) #id de libro consumido por user
			
			# cosines = sim_dict[user_bookId] #THE DREAM
			cosines = dict((bookId, 0.0) for bookId in docs2vec)
			for bookId in docs2vec: #ids de libros en la DB
				if bookId == user_bookId: continue
				cosines[bookId] = 1 - spatial.distance.cosine(docs2vec[bookId], docs2vec[user_bookId]) #1 - dist = similarity
			
			sorted_sims = sorted(cosines.items(), key=operator.itemgetter(1), reverse=True) #[(<grId>, MAYOR sim), ..., (<grId>, menor sim)]
			book_recs.append( [ bookId for bookId, sim in sorted_sims ] )

		book_recs = flatten_list(list_of_lists=book_recs, rows=len(sorted_sims))
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
		file.write( "SAMPLED N=%s, normal nDCG=%s, MAP=%s, MRR=%s, R-precision=%s\n" % \
				(N, mean(nDCGs), mean(APs), mean(MRRs), mean(Rprecs)) )



def option2_protocol_evaluation(data_path, solr, N):
	test_c  = consumption(ratings_path=data_path+'test/test_N'+str(N)+'.data', rel_thresh=0, with_ratings=True)
	train_c = consumption(ratings_path=data_path+'eval_train_N'+str(N)+'.data', rel_thresh=0, with_ratings=False)
	MRRs   = []
	nDCGs  = []
	APs    = []
	Rprecs = []
	docs2vec  = np.load('./w2v-tmp/docs2vec.npy').item()
	users2vec = np.load('./w2v-tmp/users2vec.npy').item()

	i = 1
	sampled_user_ids = random.sample(test_c.keys(), 100)
	for userId in sampled_user_ids: #test_c:
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
		file.write( "SAMPLED N=%s, normal nDCG=%s, MAP=%s, MRR=%s, R-precision=%s\n" % \
				(N, mean(nDCGs), mean(APs), mean(MRRs), mean(Rprecs)) )


def main():
	data_path = 'TwitterRatings/funkSVD/data/'
	solr = 'http://localhost:8983/solr/grrecsys'
	# model_eng = KeyedVectors.load_word2vec_format('/home/jschellman/gensim-data/word2vec-google-news-300/word2vec-google-news-300', binary=True)
	
	## Mapeo book Id -> vec_book ##:
	# dict_docs =	docs2vecs(solr= solr, model= model_eng)
	# np.save('./w2v-tmp/docs2vec.npy', dict_docs)
	## DONE ##

	## Para modo 1 ##
	# docs2vec  = np.load('./w2v-tmp/docs2vec.npy').item()
	# dict_sim = sim_matrix(doc_vecs= docs2vec)
	# np.save('./w2v-tmp/sim_matrix.npy', dict_sim)
	## DONE ##

	## Para modo 2
	# dict_users = users2vecs(solr= solr, data_path= data_path, model= model_eng)
	# np.save('./w2v-tmp/users2vec.npy', dict_users)
	#Por ahora no:
	# model_esp = KeyedVectors.load_word2vec_format('/home/jschellman/fasttext-sbwc.3.6.e20.vec')

	for N in [5, 10, 15, 20]:
		option2_protocol_evaluation(data_path= data_path, solr= solr, N=N)
	
	for N in [5, 10, 15, 20]:
		option1_protocol_evaluation(data_path= data_path, solr= solr, N=N)
	

if __name__ == '__main__':
	main()
