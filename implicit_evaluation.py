# coding=utf-8

from utils_py2 import *
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
import implicit

#-----"PRIVATE" METHODS----------#
def get_ndcg(data_path, idcoder, fold, N, model, matrix_T):
	solr = "http://localhost:8983/solr/grrecsys"
	users_nDCGs = []
	val_c = consumption(ratings_path=data_path+'val/val_N'+str(N)+'.'+str(fold), rel_thresh=0, with_ratings=True)
	train_c = consumption(ratings_path=data_path+'train/train_N'+str(N)+'.'+str(fold), rel_thresh=0, with_ratings=False)
	for userId in val_c:
		recommends = model.recommend(userid= int(idcoder.coder('user', userId)), user_items= matrix_T, N= N)
		book_recs  = [ idcoder.decoder('item', tupl[0]) for tupl in recommends ]
		book_recs  = recs_cleaner(solr= solr, consumpt= train_c[userId], recs= book_recs[:100])
		recs       = user_ranked_recs(user_recs= book_recs, user_consumpt= val_c[userId])
		users_nDCGs.append( nDCG(recs=recs, alt_form=False, rel_thresh=False) )
	return mean(users_nDCGs)

def implicitJob(data_path, all_c, idcoder, params, N):
	nDCGs = []
	logging.info("Evaluando con params: {0}".format(params))
	for i in range(1, 4+1):
		# train_data, y_tr, _ = loadData('train/train_N'+str(N)+'.'+str(i))
		# X_tr = vectorizer.transform(train_data)
		ones, row, col = get_data(data_path= data_path, all_c= all_c, idcoder= idcoder, fold= i, N= N, mode= "tuning")
		matrix         = csr_matrix((ones, (row, col)), dtype=np.float64 )
		user_items     = matrix.T.tocsr()
		model          = implicit.als.AlternatingLeastSquares(factors= params['f'], regularization= params['lamb'], iterations= params['mi'], dtype= np.float64)
		model.fit(matrix)
		# val_data, y_va, _ = loadData('val/val_N'+str(N)+'.'+str(i))
		# X_va = vectorizer.transform(val_data)
		ndcg           = get_ndcg(data_path= data_path, idcoder= idcoder, fold= i, N= N, model= model, matrix_T= user_items)
		nDCGs.append( ndcg )
	return mean(nDCGs)
#--------------------------------#



def ALS_tuning(data_path, N):
	all_c     = consumption(ratings_path= data_path+'eval_all_N'+str(N)+'.data', rel_thresh= 0, with_ratings= True)
	items_ids = list(set( [ itemId for userId, itemsDict in all_c.items() for itemId in itemsDict ] ))
	idcoder   = IdCoder(items_ids, all_c.keys())

	defaults = {'f': 100, 'lamb': 0.01, 'mi': 15}
	results  = {'f': {}, 'lamb': {}, 'mi': {}}

	for param in ['f', 'mi', 'lamb']:
		
		if param=='f':
			for i in range(20, 2020, 20):
				defaults['f'] = i
				logging.info("Evaluando con params: {}".format(defaults))
				results['f'][i] = implicitJob(data_path= data_path, all_c= all_c, idcoder= idcoder, params= defaults, N= N)
			defaults['f'] = opt_value(results= results['f'], metric= 'ndcg')

		elif param=='mi':
			for i in [5, 10, 15, 20, 30, 45, 70, 100]:
				defaults['mi'] = i
				logging.info("Evaluando con params: {}".format(defaults))
				results['mi'][i] = implicitJob(data_path= data_path, all_c= all_c, idcoder= idcoder, params= defaults, N= N)
			defaults['mi'] = opt_value(results= results['mi'], metric= 'ndcg')

		elif param=='lamb':
			for i in [0.001, 0.002, 0.003, 0.004, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
				defaults['lamb'] = i
				logging.info("Evaluando con params: {}".format(defaults))
				results['lamb'][i] = implicitJob(data_path= data_path, all_c= all_c, idcoder= idcoder, params= defaults, N= N)
			defaults['lamb'] = opt_value(results= results['lamb'], metric= 'ndcg')

	with open('TwitterRatings/implicit/clean/opt_params.txt', 'w') as f:
		for param in defaults:
			f.write( "{param}:{value}\n".format(param=param, value=defaults[param]) )
		f.write( "nDCG:{nDCG}".format(nDCG=results['lamb'][ defaults['lamb'] ]) )

	with open('TwitterRatings/implicit/clean/params_ndcgs.txt', 'w') as f:
		for param in results:
			for value in results[param]:
				f.write( "{param}={value}\t : {nDCG}\n".format(param=param, value=value, nDCG=results[param][value]) )

	return defaults


def ALS_protocol_evaluation(data_path, params):
	# all_data, y_all, items = loadData("eval_all_N"+str(N)+".data")
	# v = DictVectorizer()
	# X_all = v.fit_transform(all_data)
	solr = "http://localhost:8983/solr/grrecsys"

	test_c  = consumption(ratings_path= data_path+'test/test_N20.data', rel_thresh= 0, with_ratings= True)
	train_c = consumption(ratings_path= data_path+'eval_train_N20.data', rel_thresh= 0, with_ratings= False)
	all_c   = consumption(ratings_path= data_path+'eval_all_N20.data', rel_thresh= 0, with_ratings= True)
	items_ids = list(set( [ itemId for userId, itemsDict in all_c.items() for itemId in itemsDict ] ))
	idcoder   = IdCoder(items_ids, all_c.keys())
	MRRs   = dict((N, []) for N in [5, 10, 15, 20])
	nDCGs  = dict((N, []) for N in [5, 10, 15, 20])
	APs    = dict((N, []) for N in [5, 10, 15, 20])
	Rprecs = dict((N, []) for N in [5, 10, 15, 20])

	# train_data, y_tr, _ = loadData('eval_train_N'+str(N)+'.data')
	# X_tr = v.transform(train_data)

	ones, row, col = get_data(data_path= data_path, all_c= all_c, idcoder= idcoder, fold= 0, N= 20, mode= "testing")
	matrix         = csr_matrix((ones, (row, col)), dtype=np.float64 )
	user_items     = matrix.T.tocsr()
	model          = implicit.als.AlternatingLeastSquares(factors= params['f'], regularization= params['lamb'], iterations= params['mi'], dtype= np.float64)
	model.fit(matrix)

	for userId in test_c: 
		recommends = model.recommend(userid= int(idcoder.coder('user', userId)), user_items= user_items, N= 200)
		book_recs  = [ idcoder.decoder('item', tupl[0]) for tupl in recommends ]
		book_recs  = remove_consumed(user_consumption= train_c[userId], rec_list= book_recs)
		book_recs  = recs_cleaner(solr= solr, consumpt= train_c[userId], recs= book_recs[:100])		
		recs       = user_ranked_recs(user_recs= book_recs, user_consumpt= test_c[userId])	

		for N in [5, 10, 15, 20]:
			mini_recs = dict((k, recs[k]) for k in recs.keys()[:N])
			MRRs[N].append( MRR(recs=mini_recs, rel_thresh=1) )
			nDCGs[N].append( nDCG(recs=mini_recs, alt_form=False, rel_thresh=False) )		
			APs[N].append( AP_at_N(n=N, recs=recs, rel_thresh=1) )
			Rprecs[N].append( R_precision(n_relevants=N, recs=mini_recs) )


	for N in [5, 10, 15, 20]:
		with open('TwitterRatings/implicit/protocol.txt', 'a') as file:
			file.write( "N=%s, nDCG=%s, MAP=%s, MRR=%s, R-precision=%s\n" % \
				(N, mean(nDCGs[N]), mean(APs[N]), mean(MRRs[N]), mean(Rprecs[N])) )	

def save_testing_recommendations(data_path, params):
	solr = "http://localhost:8983/solr/grrecsys"
	recommendations = {}

	test_c  = consumption(ratings_path= data_path+'test/test_N20.data', rel_thresh= 0, with_ratings= True)
	train_c = consumption(ratings_path= data_path+'eval_train_N20.data', rel_thresh= 0, with_ratings= False)
	all_c   = consumption(ratings_path= data_path+'eval_all_N20.data', rel_thresh= 0, with_ratings= True)
	items_ids = list(set( [ itemId for userId, itemsDict in all_c.items() for itemId in itemsDict ] ))
	idcoder   = IdCoder(items_ids, all_c.keys())

	ones, row, col = get_data(data_path= data_path, all_c= all_c, idcoder= idcoder, fold= 0, N= 20, mode= "testing")
	matrix         = csr_matrix((ones, (row, col)), dtype=np.float64 )
	user_items     = matrix.T.tocsr()
	model          = implicit.als.AlternatingLeastSquares(factors= params['f'], regularization= params['lamb'], iterations= params['mi'], dtype= np.float64)
	model.fit(matrix)

	for userId in test_c: 
		recommends = model.recommend(userid= int(idcoder.coder('user', userId)), user_items= user_items, N= 200)
		book_recs  = [ idcoder.decoder('item', tupl[0]) for tupl in recommends ]
		book_recs  = remove_consumed(user_consumption= train_c[userId], rec_list= book_recs)
		book_recs  = recs_cleaner(solr= solr, consumpt= train_c[userId], recs= book_recs[:100])		
		recommendations[userId] = book_recs
	
	np.save('TwitterRatings/recommended_items/implicit.npy', recommendations)


	
def main():
	data_path = 'TwitterRatings/funkSVD/data/'
	# opt_params = ALS_tuning(data_path= data_path, N= 20)
	opt_params = {'f': 20, 'lamb': 0.3, 'mi': 15}
	# for N in [5, 10, 15, 20]:
	# ALS_protocol_evaluation(data_path= data_path, params= opt_params)
	save_testing_recommendations(data_path= data_path, params= opt_params)

if __name__ == '__main__':
	main()



# """DEBUGGING"""
# import sqlite3
# import re, json
# import os
# from random import sample
# from jojFunkSvd import mean, stdev, MRR, rel_div, DCG, iDCG, nDCG, P_at_N, AP_at_N, R_precision, consumption, user_ranked_recs, opt_value
# from solr_evaluation import remove_consumed
# import logging
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
# import implicit
# import numpy as np
# from scipy.sparse import coo_matrix, csr_matrix
# import pandas as pd

# class IdCoder(object):
# 	def __init__(self, items_ids, users_ids):
# 		self.item_table = { str(i): items_ids[i] for i in range(0, len(items_ids)) }
# 		self.user_table = { str(i): users_ids[i] for i in range(0, len(users_ids)) }
# 		self.item_inverted = { v: k for k, v in self.item_table.iteritems() }
# 		self.user_inverted = { v: k for k, v in self.user_table.iteritems() }
# 	def coder(self, category, ID):
# 		if category=="item":
# 			return self.item_inverted[str(ID)]
# 		elif category=="user":
# 			return self.user_inverted[str(ID)]
# 	def decoder(self, category, ID):
# 		if category=="item":
# 			return self.item_table[str(ID)]
# 		elif category=="user":
# 			return self.user_table[str(ID)]


# data_path = 'TwitterRatings/funkSVD/data/'
# train_c   = consumption(ratings_path=data_path+'eval_train_N20.data', rel_thresh=0, with_ratings=True)
# all_c     = consumption(ratings_path=data_path+'eval_all_N20.data', rel_thresh=0, with_ratings=True)
# items_ids = list(set( [ itemId for userId, itemsDict in all_c.items() for itemId in itemsDict ] ))
# idcoder   = IdCoder(items_ids, all_c.keys())
# arrays    = {'items':[], 'users':[], 'data':[]}
# for userId in train_c:
# 	r_u = mean( map( int, all_c[userId].values() ) )
# 	for itemId in train_c[userId]:
# 		if int(train_c[userId][itemId]) >= r_u:
# 			arrays['items'].append(int( idcoder.coder('item', itemId) ))
# 			arrays['users'].append(int( idcoder.coder('user', userId) ))
# 			arrays['data'].append(1)
# 		else:
# 			arrays['items'].append(int( idcoder.coder('item', itemId) ))
# 			arrays['users'].append(int( idcoder.coder('user', userId) ))
# 			arrays['data'].append(0)


# coords = pd.DataFrame()
# coords['users'] = arrays['users']
# coords['items'] = arrays['items']
# coords['users'] = coords['users'].astype('category')
# coords['items'] = coords['items'].astype('category')
# data = np.array([1, 1, 1, 1, 1, 1, 1])
# row  = np.array([0, 0, 1, 3, 1, 0, 0])
# col  = np.array([0, 2, 1, 3, 1, 0, 0])
# ones = np.array( arrays['data'] )
# # row  = coords['items'].cat.codes.copy()
# row = arrays['items']
# # col  = coords['users'].cat.codes.copy()
# col = arrays['users']
# model  = implicit.als.AlternatingLeastSquares(factors= 100, regularization= 0.01, iterations= 15, dtype= np.float64)
# matrix = csr_matrix((ones, (row, col)), dtype=np.float64 )
# model.fit(matrix)
# user_items = matrix.T.tocsr()
# model.recommend(2, user_items)