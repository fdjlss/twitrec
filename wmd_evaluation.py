# coding=utf-8

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import os
import json
from urllib.parse import urlencode, quote_plus
from urllib.request import urlopen
from utils_py2 import mean, stdev, MRR, rel_div, DCG, iDCG, nDCG, P_at_N, AP_at_N, R_precision, consumption, remove_consumed, user_ranked_recs, opt_value
from utils_py3 import *
import numpy as np
from scipy import spatial
import operator
from gensim.models import KeyedVectors
from gensim.models.word2vec import Word2Vec
from gensim.parsing.preprocessing import preprocess_string, strip_tags, strip_punctuation, strip_multiple_whitespaces, strip_numeric
from gensim.similarities import WmdSimilarity
from nltk.corpus import stopwords
from textblob.blob import TextBlob
from annoy import AnnoyIndex
from gensim.corpora import Dictionary
# Debería estar dentro del main:
stop_words = set(stopwords.words('spanish') + stopwords.words('english') + stopwords.words('german') + stopwords.words('arabic') + \
								 stopwords.words('french') + stopwords.words('italian') + stopwords.words('portuguese') + ['goodreads', 'http', 'https', 'www', '"'])
CUSTOM_FILTERS = [lambda x: x.lower(), strip_tags, strip_punctuation, strip_multiple_whitespaces, strip_numeric]

def option1_protocol_evaluation(data_path, N, model):
	# userId='113447232' user_bookId='17310690'
	test_c  = consumption(ratings_path=data_path+'test/test_N'+str(N)+'.data', rel_thresh=0, with_ratings=True)
	train_c = consumption(ratings_path=data_path+'eval_train_N'+str(N)+'.data', rel_thresh=0, with_ratings=False)
	MRRs   = []
	nDCGs  = []
	APs    = []
	Rprecs = []
	flat_docs   = np.load('./w2v-tmp/flattened_docs_fea05b2.npy').item()
	num_to_grId = np.load('./w2v-tmp/num_to_grId.npy').item()
	grId_to_num = np.load('./w2v-tmp/grId_to_num.npy').item()
	t = AnnoyIndex(300)
	t.load('./w2v-tmp/doc_vecs_t100.tree')	
	num_best = 20

	i = 1
	for userId in test_c:
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

		book_recs     = []
		book_recs_cos = []
		for user_bookId in train_c[userId]:#for user_doc in docs:
			try:
				docs = t.get_nns_by_item(grId_to_num[user_bookId], 4)
				book_recs_cos += [ str(num_to_grId[doc_num]) for doc_num in docs ]
			except KeyError as e:
				logging.info("{} ES UNO DE LOS LIBROS CUYO HTML NO PUDO SER DESCARGADO. PROSIGUIENDO CON EL SIGUIENTE LIBRO..".format(bookId))
				continue

		# Removemos de la primera lista los items consumidos, dado que get_nns_by_items() los incluye
		book_recs_cos = [bookId for bookId in book_recs_cos if bookId not in train_c[userId]]

		wmd_corpus = []
		num_to_grId_wmd = {}
		j = 0
		for grId in book_recs_cos:
			wmd_corpus.append( flat_docs[grId] )
			num_to_grId_wmd[j] = grId
			j += 1	
		grId_to_num_wmd = {v: k for k, v in num_to_grId_wmd.items()}

		index = WmdSimilarity(wmd_corpus, model, num_best= num_best, normalize_w2v_and_replace=False)

		for user_bookId in train_c[userId]:
			r = index[flat_docs[user_bookId]]
			book_recs.append( [ num_to_grId_wmd[id] for id,score in r ] )

			# wmds = dict((bookId, 0.0) for bookId in flat_docs)
			# user_bookId = str(user_doc['goodreadsId'][0]) #id de libro consumido por user
			
			# for bookId in flat_docs: #ids de libros en la DB
				# if bookId == user_bookId: continue
				# wmds[bookId] = model.wmdistance(flat_docs[bookId], flat_docs[user_bookId]) #1 - dist = similarity
			
			# sorted_sims = sorted(wmds.items(), key=operator.itemgetter(1), reverse=False) #[(<grId>, MAYOR sim), ..., (<grId>, menor sim)]
			# book_recs.append( [ bookId for bookId, sim in sorted_sims ] )


		book_recs = flatten_list(list_of_lists=book_recs, rows=len(book_recs[0]))
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

	with open('TwitterRatings/word2vec/option1_protocol_wmd.txt', 'a') as file:
		file.write( "N=%s, normal nDCG=%s, MAP=%s, MRR=%s, R-precision=%s\n" % \
				(N, mean(nDCGs), mean(APs), mean(MRRs), mean(Rprecs)) )

def option2_protocol_evaluation(data_path, N, model):
	test_c  = consumption(ratings_path=data_path+'test/test_N'+str(N)+'.data', rel_thresh=0, with_ratings=True)
	train_c = consumption(ratings_path=data_path+'eval_train_N'+str(N)+'.data', rel_thresh=0, with_ratings=False)
	MRRs   = []
	nDCGs  = []
	APs    = []
	Rprecs = []
	flat_docs  = np.load('./w2v-tmp/flattened_docs_fea05b2.npy').item()
	flat_users = np.load('./w2v-tmp/flattened_users_fea05b2.npy').item()
	docs2vec   = np.load('./w2v-tmp/docs2vec.npy').item()
	users2vec  = np.load('./w2v-tmp/users2vec.npy').item()
	num_best = 20
	
	i = 1		
	for userId in test_c:
		logging.info("MODO 2. {0} de {1}. User ID: {2}".format(i, len(test_c), userId))
		i += 1

		# wmds = dict((bookId, 0.0) for bookId in flat_docs)
		# for bookId in flat_docs:
			# wmds[bookId] = model.wmdistance(flat_users[userId], flat_docs[bookId])

		cosines = dict((bookId, 0.0) for bookId in docs2vec)
		for bookId in docs2vec:
			cosines[bookId] = 1 - spatial.distance.cosine(users2vec[userId], docs2vec[bookId]) #1 - dist = similarity

		sorted_sims   = sorted(cosines.items(), key=operator.itemgetter(1), reverse=True) #[(<grId>, MAYOR sim), ..., (<grId>, menor sim)]
		book_recs_cos = [ bookId for bookId, sim in sorted_sims ]
		book_recs_cos = remove_consumed(user_consumption=train_c[userId], rec_list=book_recs)


		wmd_corpus = []
		num_to_grId_wmd = {}
		j = 0
		for grId in book_recs_cos[:50]:
			wmd_corpus.append( flat_docs[grId] )
			num_to_grId_wmd[j] = grId
			j += 1
		grId_to_num_wmd = {v: k for k, v in num_to_grId_wmd.items()}
		# Creamos índice WMD con un subset de (50) ítems recomendados al usuario por cossim
		index = WmdSimilarity(wmd_corpus, model, num_best= num_best, normalize_w2v_and_replace=False)
		r = index[flat_users[userId]]

		book_recs = [ num_to_grId_wmd[id] for id,score in r ]
		# sorted_sims = sorted(wmds.items(), key=operator.itemgetter(1), reverse=False) #[(<grId>, MAYOR sim), ..., (<grId>, menor sim)]
		# book_recs   = [ bookId for bookId, sim in sorted_sims ]
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

	with open('TwitterRatings/word2vec/option2_protocol_wmd.txt', 'a') as file:
		file.write( "N=%s, normal nDCG=%s, MAP=%s, MRR=%s, R-precision=%s\n" % \
				(N, mean(nDCGs), mean(APs), mean(MRRs), mean(Rprecs)) )


def main():
	data_path = 'TwitterRatings/funkSVD/data/'
	solr = 'http://localhost:8983/solr/grrecsys'
	## Modelo w2v Google 300 ##
	# model = KeyedVectors.load_word2vec_format('/home/jschellman/gensim-data/word2vec-google-news-300/word2vec-google-news-300', binary=True)
	
	## Modelo glove (convertido a w2v) Twitter 200 ##
	# model = KeyedVectors.load_word2vec_format('/home/jschellman/gensim-data/glove-twitter-200/glove-twitter-200.txt')

	## Sólo por ahora para guardar el diccionario de vectores:
	# dict_docs =	flatten_all_docs(solr= solr, model= model, filter_extremes= True)
	# np.save('./w2v-tmp/flattened_docs_fea075b1.npy', dict_docs)
	
	# dict_users = flatten_all_users(data_path= data_path, model= model, as_tweets= True, filter_extremes= True)
	dict_users = mix_user_flattening(data_path= data_path)
	np.save('./w2v-tmp/flattened_users_mix.npy', dict_users)

	# dict_users = mix_user_flattening(data_path= data_path)

	# model.init_sims(replace=True)
	# for N in [5, 10, 15, 20]:
	# 	option1_protocol_evaluation(data_path= data_path, N=N, model= model)
	# 	option2_protocol_evaluation(data_path= data_path, N=N, model= model)


if __name__ == '__main__':
	main()

# AVG y STDEV LENGTH OF FLATTENED DOCS #
# lens=[]
# for id, flat_doc in flat_docs.items():
# 	lens.append( len(flat_doc) )
# mean(lens)
# stdev(lens)

