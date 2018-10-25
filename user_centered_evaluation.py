# coding=utf-8
import sys  

reload(sys)  
sys.setdefaultencoding('utf8')

import os
import json
from urllib import urlencode, quote_plus
from urllib2 import urlopen

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix
import numpy as np
import pandas as pd
from pyfm import pylibfm

from svd_evaluation import mean, stdev, MRR, rel_div, DCG, iDCG, nDCG, P_at_N, AP_at_N, R_precision, consumption, user_ranked_recs, opt_value
from solr_evaluation import remove_consumed, flatten_list
from pyFM_evaluation import loadData
from implicit_evaluation import IdCoder, get_data
from hybrid_evaluation import hybridize_recs
import implicit

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

#--------------------------------#
def diversity(l1, l2, N):
	# Todos los de l2 que no están en l1
	relative_complement = set( l2 ) - set( l1 )
	diversity = len( relative_complement ) / float( N )

	return diversity
#--------------------------------#

def solr_recs(solr, params, items):
	recs = []
	consumpt = [ str(itemId) for itemId, rating, auth1, auth2, auth3 in items ]

	for itemId in consumpt:
		encoded_params = urlencode(params)
		url            = solr + '/mlt?q=goodreadsId:'+ itemId + "&" + encoded_params
		response       = json.loads( urlopen(url).read().decode('utf8') )
		try:
			docs         = response['response']['docs']
		except:
			continue
		recs.append( [ str(doc['goodreadsId'][0]) for doc in docs ] )
	recs = flatten_list(list_of_lists=recs, rows=params['rows'])
	recs = remove_consumed(user_consumption= consumpt, rec_list= recs)
	recs = recs[:20]
	return recs

def implicit_recs(data_path, params, items):
	all_c   = consumption(ratings_path=data_path+'eval_all_N20.data', rel_thresh=0, with_ratings=True)

	# Agregamos el nuevo usuario a los datos
	consumpt = [ str(itemId) for itemId, rating, auth1, auth2, auth3 in items ]
	new_user = ('0', dict((str(itemId), str(rating)) for itemId, rating, auth1, auth2, auth3 in items) ) #mock userId: '0'
	all_c[ new_user[0] ] = new_user[1]
	items_ids = list(set( [ itemId for userId, itemsDict in all_c.items() for itemId in itemsDict ] ))
	idcoder   = IdCoder(items_ids, all_c.keys())

	# get_data() de implicit_eval, pero usando TODA la data, no sólo el set de train
	arrays  = {'items':[], 'users':[], 'data':[]}
	for userId in all_c:
		r_u = mean( map( int, all_c[userId].values() ) )
		for itemId in all_c[userId]:
			if int(all_c[userId][itemId]) >= r_u:
				arrays['items'].append(int( idcoder.coder('item', itemId) ))
				arrays['users'].append(int( idcoder.coder('user', userId) ))
				arrays['data'].append(1)
			else:
				arrays['items'].append(int( idcoder.coder('item', itemId) ))
				arrays['users'].append(int( idcoder.coder('user', userId) ))
				arrays['data'].append(0)
	ones = np.array( arrays['data'] )
	row = arrays['items']
	col = arrays['users']

	matrix         = csr_matrix((ones, (row, col)), dtype=np.float64 )
	user_items     = matrix.T.tocsr()
	model          = implicit.als.AlternatingLeastSquares(factors= params['f'], regularization= params['lamb'], iterations= params['mi'], dtype= np.float64)
	model.fit(matrix)

	recs = model.recommend(userid= int(idcoder.coder('user', new_user[0])), user_items= user_items, N= 200)
	recs = [ idcoder.decoder('item', tupl[0]) for tupl in recs ]
	recs = remove_consumed(user_consumption= consumpt, rec_list= recs)
	recs = recs[:20]

	return recs

def pyFM_recs(data_path, params, user):
	all_data, y_all, items = loadData("eval_all_N20.data", data_path=data_path, with_timestamps=False, with_authors=True)
	
	consumpt = [ str(itemId) for itemId, rating, auth1, auth2, auth3 in items ]

	for itemId, rating, auth1, auth2, auth3 in user:
		all_data.append( { 'item_id': str(itemId), 'author1_id': str(auth1), 'author3_id': str(auth3)+'\n', 'user_id': '0', 'author2_id': str(auth2) } )
		y_all = np.append(y_all, rating)
		items.add( str(itemId) )

	v = DictVectorizer()
	X_all = v.fit_transform(all_data)

	fm   = pylibfm.FM(num_factors=params['f'], num_iter=params['mi'], k0=params['bias'], k1=params['oneway'], init_stdev=params['init_stdev'], \
									validation_size=params['val_size'], learning_rate_schedule=params['lr_s'], initial_learning_rate=params['lr'], \
									power_t=params['invscale_pow'], t0=params['optimal_denom'], shuffle_training=params['shuffle'], seed=params['seed'], \
									task='regression', verbose=True)
	fm.fit(X_all, y_tr)

	user_rows = [ {'user_id': '0', 'item_id': str(itemId)} for itemId in items ]
	X_te      = v.transform(user_rows)
	preds     = fm.predict(X_te)
	recs      = [itemId for _, itemId in sorted(zip(preds, items), reverse=True)]
	recs      = remove_consumed(user_consumption= consumpt, rec_list= recs)
	recs      = recs[:20]

	return recs

## SOLO EN PYTHON 3.X
# from gensim.models import KeyedVectors
# from word2vec_evaluation import doc2vec, docs2vecs
# from wmd_evaluation import flat_doc, flat_user, get_extremes, flatten_all_docs
# from urllib.request import urlopen
# def w2v_recs(data_path, which_model, items, userId):
# 	test_c  = consumption(ratings_path=data_path+'test/test_N20.data', rel_thresh=0, with_ratings=True)
# 	train_c = consumption(ratings_path=data_path+'eval_train_N20.data', rel_thresh=0, with_ratings=False)
# 	consumpt = [ str(itemId) for itemId, rating, auth1, auth2, auth3 in items ]
	
# 	model = KeyedVectors.load_word2vec_format('/home/jschellman/gensim-data/word2vec-google-news-300/word2vec-google-news-300', binary=True)
# 	flat_docs = np.load('./w2v-tmp/flattened_docs.npy').item()
# 	extremes = get_extremes(flat_docs= flat_docs, n_below= 1, n_above= len(flat_docs) * 0.75)

# 	# Alt 1: en docs2vec & users2vec no están los libros nuevos y el usuario nuevo, entonces hago acá el embedding manualmente y luego los guardo 
# 	# flat_user_books = {}
# 	# for itemId, rating, auth1, auth2, auth3 in items:
# 	# 	logging.info("Flattening item {}".format(itemId))
# 	# 	url      = solr + '/query?q=goodreadsId:' + str(itemId)
# 	# 	response = json.loads( urlopen(url).read().decode('utf8') )
# 	# 	doc      = response['response']['docs']
# 	# 	flat_user_books[itemId] = flat_doc(document= doc[0], model= model, extremes= extremes)
# 	# flat_user = flat_user(flat_docs= flat_user_books, consumption= consumpt)
# 	# embd_user = doc2vec(list_document= flat_user, model= model)

# 	# embd_user_books = {}
# 	# for itemId, flat_doc in flat_user_books.items():
# 	# 	embd_user_books[itemId] = doc2vec(list_document= flat_doc, model= model)

# 	# docs2vec  = np.load('./w2v-tmp/'+which_model+'/docs2vec_'+which_model+'.npy').item()
# 	# for itemId in embd_user_books:
# 	# 	docs2vec[itemId] = embd_user_books[itemId]
# 	# np.save('./w2v-tmp/'+which_model+'/docs2vec_'+which_model+'.npy', docs2vec) #

# 	# distances = dict((bookId, 0.0) for bookId in docs2vec)
# 	# for bookId in docs2vec:
# 	# 	distances[bookId] = spatial.distance.cosine(embd_user, docs2vec[bookId])


# 	# Alt 2: en docs2vec y flattened_docs ya están los libros nuevos, dado que corrí el pipeline para hacer el embedding de todos los libros del index (incluídos los nuevos)
# 	# (sería lo adecuado si es que descargo libros adicionales de GR a parte de los libros de los usuarios de prueba)
# 	flat_docs = np.load('./w2v-tmp/flattened_docs_fea075b1.npy').item()
# 	flat_user_books = dict( (itemId, flat_docs[itemId]) for itemId, rating, auth1, auth2, auth3 in items )
# 	flat_user = flat_user(flat_docs= flat_user_books, consumption= consumpt)
# 	embd_user = doc2vec(list_document= flat_user, model= model)

# 	docs2vec  = np.load('./w2v-tmp/'+which_model+'/docs2vec_'+which_model+'.npy').item()

# 	distances = dict((bookId, 0.0) for bookId in docs2vec)
# 	for bookId in docs2vec:
# 		distances[bookId] = spatial.distance.cosine(embd_user, docs2vec[bookId])

# 	# LA RECOMENDACIÓN
# 	sorted_sims = sorted(distances.items(), key=operator.itemgetter(1), reverse=False) #[(<grId>, MENOR dist), ..., (<grId>, MAYOR dist)]
# 	recs   = [ bookId for bookId, sim in sorted_sims ]
# 	recs   = remove_consumed(user_consumption=consumpt, rec_list=recs)

# 	# Por si necesito usar el usuario para subsiguientes recomendaciones con w2v 
# 	users2vec = np.load('./w2v-tmp/'+which_model+'/users2vec_books_'+which_model+'.npy').item()
# 	# Si es que es user de GR: ID de GR
# 	# Si no: mock ID ("A0", "A1", etc..)
# 	users2vec[userId] = embd_user
# 	np.save('./w2v-tmp/'+which_model+'/users2vec_books_'+which_model+'.npy', users2vec) 

# 	return recs

def hybrid_recommendation(data_path, solr, params_cb, params_cf, params_hy, items):
	consumpt = [ str(itemId) for itemId, rating, auth1, auth2, auth3 in items ]

	recs_cb = solr_recs(solr= solr, params= params_cb, items= items)
	recs_cf = implicit_recs(data_path= data_path, params= params_cf, items= items)

	recs_hy = hybridize_recs(recs_cb=recs_cb, recs_cf=recs_cf, weight_cb=params_hy['weight_cb'], weight_cf=params_hy['weight_cf'])
	recs_hy = remove_consumed(user_consumption= consumpt, rec_list= recs_hy)
	recs_hy = recs_hy[:20]

	return recs_hy


def diversity_calculation(data_path, solr, params_cb, params_cf, params_hy):
	diversity_hy_cb = []
	diversity_hy_cf = []
	diversity_cb_cf = []

	all_c   = consumption(ratings_path= data_path+'eval_all_N20.data', rel_thresh= 0, with_ratings= True)
	items_ids = list(set( [ itemId for userId, itemsDict in all_c.items() for itemId in itemsDict ] ))
	idcoder   = IdCoder(items_ids, all_c.keys())

	ones, row, col = get_data(data_path= data_path, all_c= all_c, idcoder= idcoder, fold= 0, N= 20, mode= "testing")
	matrix         = csr_matrix((ones, (row, col)), dtype=np.float64 )
	user_items     = matrix.T.tocsr()
	model          = implicit.als.AlternatingLeastSquares(factors= params_cf['f'], regularization= params_cf['lamb'], iterations= params_cf['mi'], dtype= np.float64)
	model.fit(matrix)

	i = 0
	for userId in all_c: 
		i += 1
		logging.info("user {0} of {1}".format(i, len(all_c)))
		recommends = model.recommend(userid= int(idcoder.coder('user', userId)), user_items= user_items, N= 200)
		recs_cf    = [ idcoder.decoder('item', tupl[0]) for tupl in recommends ]
		recs_cf    = remove_consumed(user_consumption= all_c[userId], rec_list= recs_cf)
		recs_cf    = recs_cf[:20]

		recs_cb = []
		for itemId in all_c[userId]:
			encoded_params = urlencode(params_cb)
			url            = solr + '/mlt?q=goodreadsId:'+ itemId + "&" + encoded_params
			response       = json.loads( urlopen(url).read().decode('utf8') )
			try:
				docs           = response['response']['docs']
			except:
				continue
			recs_cb.append( [ str(doc['goodreadsId'][0]) for doc in docs ] )
		recs_cb = flatten_list(list_of_lists=recs_cb, rows=params_cb['rows'])
		recs_cb = remove_consumed(user_consumption= all_c[userId], rec_list= recs_cb)
		recs_cb = recs_cb[:20]

		recs_hy = hybridize_recs(recs_cb= recs_cb, recs_cf= recs_cf, weight_cb= params_hy['weight_cb'], weight_cf= params_hy['weight_cf'])
		recs_hy = remove_consumed(user_consumption= all_c[userId], rec_list= recs_hy)
		recs_hy = recs_hy[:20]

		diversity_hy_cb.append( diversity(l1= recs_hy, l2= recs_cb, N= 20) )
		diversity_hy_cf.append( diversity(l1= recs_hy, l2= recs_cf, N= 20) )
		diversity_cb_cf.append( diversity(l1= recs_cb, l2= recs_cf, N= 20) )

	logging.info("Diversities")
	logging.info("L1: HY. L2: CB \t {}".format( mean(diversity_hy_cb) ))
	logging.info("L1: HY. L2: CF \t {}".format( mean(diversity_hy_cf) ))
	logging.info("L1: CB. L2: CF \t {}".format( mean(diversity_cb_cf) ))

def main():
	data_path_context = 'TwitterRatings/funkSVD/data_with_authors/'
	data_path = 'TwitterRatings/funkSVD/data/'
	solr = "http://localhost:8983/solr/grrecsys"
	params_solr = {'echoParams' : 'none',
							'fl' : 'goodreadsId,description,title.titleOfficial,genres.genreName,author.authors.authorName,quotes.quoteText,author.authorBio,title.titleGreytext',
							'rows' : 100,
							'mlt.fl' : 'description,title.titleOfficial,genres.genreName,author.authors.authorName,quotes.quoteText',
							'mlt.boost' : 'false', #def: false
							'mlt.mintf' : 1, #def: 2
							'mlt.mindf' : 2, #def: 5
							'mlt.minwl' : 1, #def: 0
							'mlt.maxdf' : 25431, #docs*0.5, # en realidad no especificado
							'mlt.maxwl' : 8, #def: 0
							'mlt.maxqt' : 90, #def: 25
							'mlt.maxntp' : 150000} #def: 5000}
	params_imp = {'f': 20, 'lamb': 0.3, 'mi': 15}
	params_pyfm = {  'lr_s':'invscaling',
							'val_size':0.01,
							'shuffle':True,
							'bias':True,
							'invscale_pow':0.05,
							'f':20,
							'mi':10,
							'seed':20,
							'lr':0.001,
							'oneway':True,
							'optimal_denom':0.01,
							'init_stdev':0.0001}
	params_hy = {'weight_cb': 0.9, 'weight_cf': 0.1}

	denis = [('8267287', 0, '5650904', '83881', '0'), 
					 ('158014', 0, '1113469', '0', '0'), 
					 ('230514', 0, '134770', '0', '0'),
					 ('19508', 0, '909675', '0', '0'),
					 ('21031', 0, '909675', '0', '0'), 
					 ('123632', 0, '38285', '100950', '574570'),
					 ('17690', 0, '5223', '0', '0'),
					 ('485894', 0, '5223', '0', '0'),
					 ('56919', 0, '30195', '0', '0'),
					 ('191373', 0, '25824', '0', '0'),
					 ('4662', 0, '3190', '0', '0'),
					 ('16207545', 0, '128195', '0', '0'),
					 ('46170', 0, '1455', '0', '0'),
					 ('9646', 0, '3706', '0', '0'),
					 ('7144', 0, '3137322', '0', '0'),
					 ('4934', 0, '3137322', '0', '0'),
					 ('63034', 0, '72039', '0', '0'),
					 ('820183', 0, '500', '0', '0'),
					 ('426504', 0, '500', '0', '0'),
					 ('338798', 0, '5144', '0', '0'),
					 ('7588', 0, '5144', '0', '0'),
					 ('847635', 0, '201906', '0', '0'),
					 ('13031462', 0, '25591', '0', '0'),
					 ('214319', 0, '51808', '0', '0')] #[ (itemId, rating, authId1, authId2, authId3) ]


	lista_hyb = hybrid_recommendation(data_path= data_path, solr= solr, params_cb= params_solr, params_cf= params_imp, params_hy= params_hy, items= denis)
	
	lista_imp = implicit_recs(data_path= data_path, params= params_imp, items= denis)

	# SOLO PYTHON 3.x
	# model = KeyedVectors.load_word2vec_format('/home/jschellman/gensim-data/word2vec-google-news-300/word2vec-google-news-300', binary=True)
	# which_model = 'google'
	# # 1. Flatten todos los docs del index en Solr
	# dict_docs =	flatten_all_docs(solr= solr, model= model, filter_extremes= True)
	# # 2. Guarda flatten docs
	# np.save('./w2v-tmp/flattened_docs_fea075b1.npy', dict_docs)
	# # 3. Embedding de los flattened docs
	# dict_docs =	docs2vecs(model= model) # (por dentro carga "./w2v-tmp/flattened_docs_fea075b1.npy")
	# # 4. Guarda los embeddings
	# np.save('./w2v-tmp/'+which_model+'/docs2vec_'+which_model+'.npy', dict_docs)
	# # 5. Genera las recomendaciones
	# lista_w2v = w2v_recs(data_path= data_path, which_model= which_model, items= denis, userId="denis")
	

	i = 0
	logging.info("RECOMENDADOR A")
	for item in lista_hyb:
		i += 1
		url      = solr + '/select?q=goodreadsId:'+item+'&wt=json' # ex. item: 7144 (Crime and Punishment)
		response = json.loads( urlopen(url).read().decode('utf8') )
		doc      = response['response']['docs'][0]
		logging.info(i)
		logging.info("Título: " + doc['title.titleOfficial'][0])
		logging.info("Autor: " + doc['author.authors.authorName'][0])
		logging.info("URL: " + doc['href'][0])
		logging.info("- RELEVANTE / IRRELEVANTE")
		logging.info("- NOVEDOSO / NO NOVEDOSO")
		logging.info(" ")
		if i==10: break
	logging.info("Con esta lista en general me siento:")
	logging.info("- SATISFECHO / NO SATISFECHO")

	i = 0
	logging.info("RECOMENDADOR B")
	for item in lista_imp:
		i += 1
		url      = solr + '/select?q=goodreadsId:'+item+'&wt=json'
		response = json.loads( urlopen(url).read().decode('utf8') )
		doc      = response['response']['docs'][0]
		logging.info(i)
		logging.info("Título: " + doc['title.titleOfficial'][0])
		logging.info("Autor: " + doc['author.authors.authorName'][0])
		logging.info("URL: " + doc['href'][0])
		logging.info("- RELEVANTE / IRRELEVANTE")
		logging.info("- NOVEDOSO / NO NOVEDOSO")
		logging.info(" ")
		if i==10: break
	logging.info("Con esta lista en general me siento:")
	logging.info("- SATISFECHO / NO SATISFECHO")

	# i = 0
	# logging.info("RECOMENDADOR C")
	# for item in lista_w2v:
	# 	i += 1
	# 	url      = solr + '/select?q=goodreadsId:'+item+'&wt=json'
	# 	response = json.loads( urlopen(url).read().decode('utf8') )
	# 	doc      = response['response']['docs'][0]
	# 	logging.info(i)
	# 	logging.info("Título: " + doc['title.titleOfficial'][0])
	# 	logging.info("Autor: " + doc['author.authors.authorName'][0])
	# 	logging.info("URL: " + doc['href'][0])
	# 	logging.info("- RELEVANTE / IRRELEVANTE")
	# 	logging.info("- NOVEDOSO / NO NOVEDOSO")
	# 	logging.info(" ")
	# 	if i==10: break
	# logging.info("Con esta lista en general me siento:")
	# logging.info("- SATISFECHO / NO SATISFECHO")	

	# diversity_calculation(data_path= data_path, solr= solr, params_cb= params_solr, params_cf= params_imp, params_hy= params_hy)

if __name__ == '__main__':
	main()