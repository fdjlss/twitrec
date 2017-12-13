# coding=utf-8

import sqlite3
import re, json
import os
from random import sample
from jojFunkSvd import mean, stdev, MRR, rel_div, DCG, iDCG, nDCG, P_at_N, AP_at_N, R_precision, consumption, user_ranked_recs, opt_value
from solr_evaluation import remove_consumed
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import implicit
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
import pandas as pd

#-----"PRIVATE" METHODS----------#
class IdCoder(object):
	def __init__(self, items_ids, users_ids):
		self.item_table = { str(i): items_ids[i] for i in range(0, len(items_ids)) }
		self.user_table = { str(i): users_ids[i] for i in range(0, len(users_ids)) }
		self.item_inverted = { v: k for k, v in self.item_table.iteritems() }
		self.user_inverted = { v: k for k, v in self.user_table.iteritems() }
	def coder(self, category, ID):
		if category=="item":
			return self.item_inverted[str(ID)]
		elif category=="user":
			return self.user_inverted[str(ID)]
	def decoder(self, category, ID):
		if category=="item":
			return self.item_table[str(ID)]
		elif category=="user":
			return self.user_table[str(ID)]

def get_data(data_path, all_c, idcoder, fold, N):
	train_c = consumption(ratings_path=data_path+'train/train_N'+str(N)+'.'+str(fold), rel_thresh=0, with_ratings=True)
	arrays  = {'items':[], 'users':[], 'data':[]}
	for userId in train_c:
		r_u = mean( map( int, all_c[userId].values() ) )
		for itemId in train_c[userId]:
			if int(train_c[userId][itemId]) >= r_u:
				arrays['items'].append(int( idcoder.coder('item', itemId) ))
				arrays['users'].append(int( idcoder.coder('user', userId) ))
				arrays['data'].append(1)
			else:
				arrays['items'].append(int( idcoder.coder('item', itemId) ))
				arrays['users'].append(int( idcoder.coder('user', userId) ))
				arrays['data'].append(0)
	# coords = pd.DataFrame()
	# coords['users'] = arrays['users']
	# coords['items'] = arrays['items']
	# coords['users'] = coords['users'].astype('category')
	# coords['items'] = coords['items'].astype('category')
	ones = np.array( arrays['data'] )
	# row  = coords['items'].cat.codes.copy()
	# col  = coords['users'].cat.codes.copy()
	return ones, arrays['items'], arrays['users']

def get_ndcg(data_path, idcoder, model, matrix_T, N):
	users_nDCGs = []
	val_c = consumption(ratings_path=data_path+'val/val_N'+str(N)+'.'+str(i), rel_thresh=0, with_ratings=True)
	for userId in val_c:
		recommends = model.recommend(userid= userId, user_items= matrix_T, N= N)
		book_recs  = [ str(tupl[0]) for tupl in recommends ]
		recs       = user_ranked_recs(user_recs= book_recs, user_consumpt= val_c[userId])
		users_nDCGs.append( nDCG(recs=recs, alt_form=False, rel_thresh=False) )
	return mean(users_nDCGs)

def implicitJob(data_path, all_c, idcoder, params, N):
	nDCGs = []
	for i in range(1, 4+1):
		ones, row, col = get_data(data_path= data_path, all_c= all_c, idcoder= idcoder, fold= i, N= N)
		matrix         = csr_matrix((ones, (row, col)), dtype=np.float64 )
		user_items     = matrix.T.tocsr()
		model          = implicit.als.AlternatingLeastSquares(factors= params['f'], regularization= params['lamb'], iterations= params['mi'], dtype= np.float64)
		model.fit(matrix)
		ndcg = get_ndcg(data_path= data_path, idcoder= idcoder, model= model, matrix_T= user_items, N= N)
		nDCGs.append( ndcg )
	return mean(nDCGs)
#--------------------------------#



def ALS_tuning(data_path):
	all_c     = consumption(ratings_path= data_path+'eval_all_N'+str(N)+'.data', rel_thresh= 0, with_ratings= True)
	items_ids = list(set( [ itemId for userId, itemsDict in all_c.items() for itemId in itemsDict ] ))
	idcoder   = IdCoder(items_ids, all_c.keys())

	defaults = {'f': 100, 'lamb': 0.01, 'mi': 15}
	results  = {'f': {}, 'lamb': {}, 'mi': {}}

	for param in ['f', 'lamb', 'mi']:
		
		if param=='f':
			defaults['f'] = list(range(20, 1520, 20))
			for i in defaults['f']:
				defaults['f'] = i
				logging.info("Evaluando con params: {}".format(defaults))
				results['f'][i] = implicitJob(data_path= data_path, all_c= all_c, idcoder= idcoder, params= defaults, N= 20)
			defaults['f'] = opt_value(results= results['f'], metric= 'ndcg')

		elif param=='lamb':
			defaults['lamb'] = [0.001, 0.002, 0.003, 0.004, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
			for i in defaults['lamb']:
				defaults['lamb'] = i
				logging.info("Evaluando con params: {}".format(defaults))
				results['lamb'][i] = implicitJob(data_path= data_path, all_c= all_c, idcoder= idcoder, params= defaults, N= 20)
			defaults['lamb'] = opt_value(results= results['lamb'], metric= 'ndcg')

		elif param=='mi':
			defaults['mi'] = list(range(5, 515, 15))
			for i in defaults['mi']:
				defaults['mi'] = i
				logging.info("Evaluando con params: {}".format(defaults))
				results['mi'][i] = implicitJob(data_path= data_path, all_c= all_c, idcoder= idcoder, params= defaults, N= 20)
			defaults['mi'] = opt_value(results= results['mi'], metric= 'ndcg')

	with open('TwitterRatings/implicit/opt_params.txt', 'w') as f:
		for param in defaults:
			f.write( "{param}:{value}\n".format(param=param, value=defaults[param]) )

	with open('TwitterRatings/implicit/params_ndcgs.txt', 'w') as f:
		for param in results:
			for value in results[param]:
				f.write( "{param}={value}\t : {nDCG}\n".format(param=param, value=value, nDCG=results[param][value]) )

	return defaults


def main():
	data_path = 'TwitterRatings/funkSVD/data/'
	opt_params = ALS_tuning(data_path= data_path)


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