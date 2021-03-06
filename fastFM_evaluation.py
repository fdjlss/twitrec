# coding=utf-8

from utils_py2 import *
from fastFM.datasets import make_user_item_regression
from fastFM import als, sgd, bpr, mcmc
from sklearn.model_selection import train_test_split
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
from random import sample
import re

#-----"PRIVATE" METHODS----------#
def loadData_bpr(filename, data_path='TwitterRatings/funkSVD/data/', test=False, userId_va='0'):
	data = []
	y = []
	items=set()
	users=set()
	with open(data_path+'eval_train_N20.data', 'r') as f: #antes +filename
		for line in f:
			(userId,itemId,rating,timestamp,authorId1,authorId2,authorId3)=line.split(',')
			items.add(itemId)
			users.add(userId)
	if not test:
		with open(data_path+filename, 'r') as f:
			#Agregamos users con items consumidos
			for line in f:
				(userId,itemId,rating,timestamp,authorId1,authorId2,authorId3)=line.split(',')
				data.append({ "user_id": str(userId), "item_id": str(itemId) })
				y.append(float(rating))
			#Agregamos users con items no consumidos (random sampling)
			for line in f:
				(userId,itemId,rating,timestamp,authorId1,authorId2,authorId3)=line.split(',')
				random_itemId = sample(items, 1)[0]
				while { "user_id": str(userId), "item_id": str(random_itemId) } in data:
					random_itemId = sample(items, 1)[0]
				else:
					data.append({ "user_id": str(userId), "item_id": str(random_itemId) })
					y.append(float(1)) #Si user no consumió item, rating=1
	else:
		all_train_c = consumption(ratings_path= 'TwitterRatings/funkSVD/data/eval_train_N5.data', rel_thresh= 0, with_ratings= True)
		for itemId in items:
			data.append({ "user_id": str(userId_va), "item_id": str(itemId)})
			if itemId in all_train_c[userId_va]:
				y.append(float(all_train_c[userId_va][itemId]))
			else:
				y.append(float(1))
	return data, np.array(y), items

def user_items_pairing(ratings, ind_left, ind_right):
	user_pairs = []
	for i in xrange(ind_left, ind_right+1):
		for j in xrange(i+1, ind_right+1):
			if ratings[i] - ratings[j] > 0:
				user_pairs.append( [i, j] )
			elif ratings[i] - ratings[j] < 0:
				user_pairs.append( [j, i] )
	return user_pairs

def make_pairs(sparse_matrix, ratings):
	pairs = []
	first_user_idx, last_user_idx = 0,0
	prev_user_idx = sparse_matrix[0,:].nonzero()[1][1]
	for i in xrange(sparse_matrix.shape[0]): #recorremos por filas
		item_idx, user_idx = sparse_matrix[i,:].nonzero()[1] #item_index, user_index
		if i == xrange(sparse_matrix.shape[0])[-1]:
			last_user_idx = i
			user_pairs = user_items_pairing(ratings, first_user_idx, last_user_idx)
			pairs += user_pairs
			continue
		if user_idx != prev_user_idx:
			last_user_idx = i-1
			user_pairs = user_items_pairing(ratings, first_user_idx, last_user_idx)
			first_user_idx = i
			prev_user_idx = user_idx
			pairs += user_pairs
	return np.array(pairs)

def ndcg_bpr(preds, vectorizer, matrix, user_data, user_val, N):
	solr = "http://localhost:8983/solr/grrecsys"
	book_recs = []
	for i in range(len(preds)):
		pred_row = preds[i]
		l = vectorizer.inverse_transform( matrix[pred_row,:] )[0].keys()
		pred_itemId = [s for s in l if "item" in s][0].split('=')[-1]
		book_recs.append(pred_itemId)
		if i==100: break
	book_recs = remove_consumed(user_consumption=user_data, rec_list=book_recs)
	book_recs = recs_cleaner(solr= solr, consumpt= user_data, recs= book_recs[:100])		
	recs      = user_ranked_recs(user_recs=book_recs, user_consumpt=user_val)
	mini_recs = dict((k, recs[k]) for k in recs.keys()[:N])
	ndcg      = nDCG(recs=mini_recs, alt_form=False, rel_thresh=False)
	return ndcg

def fastFMJob_bpr(data_path, params, N, vectorizer, train_sets, pairs, items):
	ndcgs = []
	logging.info("Evaluando con params: {0}".format(params))
	for i in range(1, 4+1):
		fm = bpr.FMRecommender(n_iter=params['mi'], init_stdev=params['init_stdev'], rank=params['f'], random_state=123, \
													 l2_reg_w=params['l2_reg_w'], l2_reg_V=params['l2_reg_V'], l2_reg=params['l2_reg'], step_size=params['step_size'])
		X_tr = vectorizer.transform(train_sets[i])
		fm.fit(X_tr, pairs[i])
		val_c = consumption(ratings_path= 'TwitterRatings/funkSVD/data/val/val_N'+str(N)+'.'+str(i), rel_thresh= 0, with_ratings= True)
		train_c = consumption(ratings_path= 'TwitterRatings/funkSVD/data/train/train_N'+str(N)+'.'+str(i), rel_thresh= 0, with_ratings= True)
		users_ndcgs = []
		for userId in val_c:
			val_data, y_va, _  = loadData_bpr('val/val_N'+str(N)+'.'+str(i), data_path=data_path, test=True, userId_va=userId)
			X_va = vectorizer.transform(val_data)
			preds = fm.predict(X_va)
			preds = np.argsort(-preds)
			users_ndcgs.append(ndcg_bpr(preds=preds, vectorizer=vectorizer, matrix=X_va, user_data=train_c[userId], user_val=val_c[userId], N=N) )
		fold_ndcg = mean(users_ndcgs)
		logging.info("FM fold {0} nDCG: {1}. Solver: BPR".format(i, fold_ndcg) )
		ndcgs.append(fold_ndcg)
	return mean(ndcgs)

def fastFMJob(data_path, params, N, vectorizer, solver):
	rmses = []
	logging.info("Evaluando con params: {0}".format(params))
	for i in range(1, 4+1):
		train_data, y_tr, _ = loadData('train/train_N'+str(N)+'.'+str(i), data_path=data_path, with_timestamps=False, with_authors=False)
		val_data, y_va, _   = loadData('val/val_N'+str(N)+'.'+str(i), data_path=data_path, with_timestamps=False, with_authors=False)
		X_tr = vectorizer.transform(train_data)
		X_va = vectorizer.transform(val_data)
		if solver=="mcmc":
			fm = mcmc.FMRegression(n_iter=params['mi'], init_stdev=params['init_stdev'], rank=params['f'], random_state=123, copy_X=True)
			preds = fm.fit_predict(X_tr, y_tr, X_va)
			rmse  = sqrt( mean_squared_error(y_va, preds) )
			logging.info("FM RMSE: {0}. Solver: {1}".format(rmse, solver) )
			rmses.append(rmse)
		elif solver=="als":
			fm = als.FMRegression(n_iter=params['mi'], init_stdev=params['init_stdev'], rank=params['f'], random_state=123, \
														l2_reg_w=params['l2_reg_w'], l2_reg_V=params['l2_reg_V'], l2_reg=params['l2_reg'])
			fm.fit(X_tr, y_tr)
			preds = fm.predict(X_va)
			rmse  = sqrt( mean_squared_error(y_va, preds) )
			logging.info("FM RMSE: {0}. Solver: {1}".format(rmse, solver) )
			rmses.append(rmse)
		elif solver=="sgd":
			fm = sgd.FMRegression(n_iter=params['mi'], init_stdev=params['init_stdev'], rank=params['f'], random_state=123, \
														l2_reg_w=params['l2_reg_w'], l2_reg_V=params['l2_reg_V'], l2_reg=params['l2_reg'], step_size=params['step_size'])
			fm.fit(X_tr, y_tr)
			preds = fm.predict(X_va)
			rmse  = sqrt( mean_squared_error(y_va, preds) )
			logging.info("FM RMSE: {0}. Solver: {1}".format(rmse, solver) )
			rmses.append(rmse)
	return mean(rmses)
#--------------------------------#	 


def fastFM_tuning_bpr(data_path, N):
	all_data, y_all, _ = loadData("eval_all_N"+str(N)+".data", data_path=data_path, with_timestamps=False, with_authors=False)
	v = DictVectorizer()
	X_all = v.fit_transform(all_data)

	## Train sets caching ## 
	train_sets = dict((k, []) for k in range(1,5))
	pairs = dict((k, []) for k in range(1,5))
	for j in range(1, 5):
		print(j)
		train_sets[j], y_tr, items = loadData_bpr('train/train_N'+str(N)+'.'+str(j), data_path=data_path)
		X_tr = v.transform(train_sets[j])
		pairs[j] = make_pairs(X_tr, y_tr)
	########################

	defaults = {'mi':100, 'init_stdev':0.1, 'f':8, 'l2_reg_w':0.1, 'l2_reg_V':0.1, 'l2_reg':0, 'step_size':0.1}

	results  = dict((param, {}) for param in defaults.keys())

	for param in ['mi', 'f', 'init_stdev', 'l2_reg_w', 'l2_reg_V', 'l2_reg', 'step_size']:

		if param=='mi':
			for i in [1, 5, 10, 20, 50, 100, 150, 200]: 
				defaults['mi'] = i
				results['mi'][i] = fastFMJob_bpr(data_path= data_path, params= defaults, N=N, vectorizer= v, train_sets= train_sets, pairs= pairs, items= items)
			defaults['mi'] = opt_value(results= results['mi'], metric= 'ndcg')

		elif param=='f': 
			for i in [1, 5, 8, 10] + range(20, 1020, 20):
				defaults['f'] = i
				results['f'][i] = fastFMJob_bpr(data_path= data_path, params= defaults, N=N, vectorizer= v, train_sets= train_sets, pairs= pairs, items= items)
			defaults['f'] = opt_value(results= results['f'], metric= 'ndcg')

		elif param=='init_stdev':
			for i in [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]: 
				defaults['init_stdev'] = i
				results['init_stdev'][i] = fastFMJob_bpr(data_path= data_path, params= defaults, N=N, vectorizer= v, train_sets= train_sets, pairs= pairs, items= items)
			defaults['init_stdev'] = opt_value(results= results['init_stdev'], metric= 'ndcg')

		elif param=='l2_reg_w':
			for i in [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]: 
				defaults['l2_reg_w'] = i
				results['l2_reg_w'][i] = fastFMJob_bpr(data_path= data_path, params= defaults, N=N, vectorizer= v, train_sets= train_sets, pairs= pairs, items= items)
			defaults['l2_reg_w'] = opt_value(results= results['l2_reg_w'], metric= 'ndcg')

		elif param=='l2_reg_V':
			for i in [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]: 
				defaults['l2_reg_V'] = i
				results['l2_reg_V'][i] = fastFMJob_bpr(data_path= data_path, params= defaults, N=N, vectorizer= v, train_sets= train_sets, pairs= pairs, items= items)
			defaults['l2_reg_V'] = opt_value(results= results['l2_reg_V'], metric= 'ndcg')

		elif param=='l2_reg':
			for i in [0.0, 0.001, 0.003, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.08, 0.1]: 
				defaults['l2_reg'] = i
				results['l2_reg'][i] = fastFMJob_bpr(data_path= data_path, params= defaults, N=N, vectorizer= v, train_sets= train_sets, pairs= pairs, items= items)
			defaults['l2_reg'] = opt_value(results= results['l2_reg'], metric= 'ndcg')

		elif param=='step_size':
			for i in [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.5]: 
				defaults['step_size'] = i
				results['step_size'][i] = fastFMJob_bpr(data_path= data_path, params= defaults, N=N, vectorizer= v, train_sets= train_sets, pairs= pairs, items= items)
			defaults['step_size'] = opt_value(results= results['step_size'], metric= 'ndcg')

	with open('TwitterRatings/fastFM/bpr/clean/opt_params.txt', 'w') as f:
		for param in defaults:
			f.write( "{param}:{value}\n".format(param=param, value=defaults[param]) )

	with open('TwitterRatings/fastFM/bpr/clean/params_ndcgs.txt', 'w') as f:
		for param in results:
			for value in results[param]:
				f.write( "{param}={value}\t : {nDCG}\n".format(param=param, value=value, nDCG=results[param][value]) )

	return defaults

def fastFM_tuning(data_path, N, solver):
	all_data, y_all, _ = loadData("eval_all_N"+str(N)+".data", data_path=data_path, with_timestamps=False, with_authors=False)
	v = DictVectorizer()
	X_all = v.fit_transform(all_data)

	if solver=="mcmc":
		defaults = {'mi':100, 'init_stdev':0.1, 'f':8}
	elif solver=="als":
		defaults = {'mi':100, 'init_stdev':0.1, 'f':8, 'l2_reg_w':0.1, 'l2_reg_V':0.1, 'l2_reg':0}
	elif solver=="sgd":
		defaults = {'mi':100, 'init_stdev':0.1, 'f':8, 'l2_reg_w':0.1, 'l2_reg_V':0.1, 'l2_reg':0, 'step_size':0.1}

	results  = dict((param, {}) for param in defaults.keys())

	for param in ['mi', 'f', 'init_stdev']:

		if param=='mi':
			for i in [1, 5, 10, 20, 50, 100, 150, 200]: 
				defaults['mi'] = i
				results['mi'][i] = fastFMJob(data_path= data_path, params= defaults, N=N, vectorizer= v, solver= solver)
			defaults['mi'] = opt_value(results= results['mi'], metric= 'rmse')

		elif param=='f': 
			for i in [1, 5, 8, 10] + range(20, 2020, 20):
				defaults['f'] = i
				results['f'][i] = fastFMJob(data_path= data_path, params= defaults, N=N, vectorizer= v, solver= solver)
			defaults['f'] = opt_value(results= results['f'], metric= 'rmse')

		elif param=='init_stdev':
			for i in [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]: 
				defaults['init_stdev'] = i
				results['init_stdev'][i] = fastFMJob(data_path= data_path, params= defaults, N=N, vectorizer= v, solver= solver)
			defaults['init_stdev'] = opt_value(results= results['init_stdev'], metric= 'rmse')

	if solver!="mcmc":
		for param in ['l2_reg_w', 'l2_reg_V', 'l2_reg']:

			if param=='l2_reg_w':
				for i in [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]: 
					defaults['l2_reg_w'] = i
					results['l2_reg_w'][i] = fastFMJob(data_path= data_path, params= defaults, N=N, vectorizer= v, solver= solver)
				defaults['l2_reg_w'] = opt_value(results= results['l2_reg_w'], metric= 'rmse')

			elif param=='l2_reg_V':
				for i in [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]: 
					defaults['l2_reg_V'] = i
					results['l2_reg_V'][i] = fastFMJob(data_path= data_path, params= defaults, N=N, vectorizer= v, solver= solver)
				defaults['l2_reg_V'] = opt_value(results= results['l2_reg_V'], metric= 'rmse')

			elif param=='l2_reg':
				for i in [0.0, 0.001, 0.003, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.08, 0.1]: 
					defaults['l2_reg'] = i
					results['l2_reg'][i] = fastFMJob(data_path= data_path, params= defaults, N=N, vectorizer= v, solver= solver)
				defaults['l2_reg'] = opt_value(results= results['l2_reg'], metric= 'rmse')

	if solver=="sgd":
		for i in [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.5]: 
			defaults['step_size'] = i
			results['step_size'][i] = fastFMJob(data_path= data_path, params= defaults, N=N, vectorizer= v, solver= solver)
		defaults['step_size'] = opt_value(results= results['step_size'], metric= 'rmse')


	# Real testing
	train_data, y_tr, _ = loadData('eval_train_N'+str(N)+'.data', data_path=data_path, with_timestamps=False, with_authors=False)
	test_data, y_te, _  = loadData('test/test_N'+str(N)+'.data', data_path=data_path, with_timestamps=False, with_authors=False)
	X_tr = v.transform(train_data)
	X_te = v.transform(test_data)

	if solver=="mcmc":
		fm = mcmc.FMRegression(n_iter=defaults['mi'], init_stdev=defaults['init_stdev'], rank=defaults['f'], random_state=123, copy_X=True)
		preds = fm.fit_predict(X_tr, y_tr, X_te)
		rmse  = sqrt( mean_squared_error(y_te, preds) )
		logging.info("FM RMSE: {0}. Solver: {1}".format(rmse, solver) )
		with open('TwitterRatings/fastFM/mcmc/clean/opt_params.txt', 'w') as f:
			for param in defaults:
				f.write( "{param}:{value}\n".format(param=param, value=defaults[param]) )
			f.write( "RMSE:{rmse}".format(rmse= rmse) )
		with open('TwitterRatings/fastFM/mcmc/clean/params_rmses.txt', 'w') as f:
			for param in results:
				for value in results[param]:
					f.write( "{param}={value}\t : {RMSE}\n".format(param=param, value=value, RMSE=results[param][value]) )

	elif solver=="als":
		fm = als.FMRegression(n_iter=defaults['mi'], init_stdev=defaults['init_stdev'], rank=defaults['f'], random_state=123, \
													l2_reg_w=defaults['l2_reg_w'], l2_reg_V=defaults['l2_reg_V'], l2_reg=defaults['l2_reg'])
		fm.fit(X_tr, y_tr)
		preds = fm.predict(X_te)
		rmse  = sqrt( mean_squared_error(y_te, preds) )
		logging.info("FM RMSE: {0}. Solver: {1}".format(rmse, solver) )
		with open('TwitterRatings/fastFM/als/clean/opt_params.txt', 'w') as f:
			for param in defaults:
				f.write( "{param}:{value}\n".format(param=param, value=defaults[param]) )
			f.write( "RMSE:{rmse}".format(rmse= rmse) )
		with open('TwitterRatings/fastFM/als/clean/params_rmses.txt', 'w') as f:
			for param in results:
				for value in results[param]:
					f.write( "{param}={value}\t : {RMSE}\n".format(param=param, value=value, RMSE=results[param][value]) )

	elif solver=="sgd":
		fm = sgd.FMRegression(n_iter=defaults['mi'], init_stdev=defaults['init_stdev'], rank=defaults['f'], random_state=123, \
													l2_reg_w=defaults['l2_reg_w'], l2_reg_V=defaults['l2_reg_V'], l2_reg=defaults['l2_reg'], step_size=defaults['step_size'])
		fm.fit(X_tr, y_tr)
		preds = fm.predict(X_te)
		rmse  = sqrt( mean_squared_error(y_te, preds) )
		logging.info("FM RMSE: {0}. Solver: {1}".format(rmse, solver) )
		with open('TwitterRatings/fastFM/sgd/clean/opt_params.txt', 'w') as f:
			for param in defaults:
				f.write( "{param}:{value}\n".format(param=param, value=defaults[param]) )
			f.write( "RMSE:{rmse}".format(rmse= rmse) )
		with open('TwitterRatings/fastFM/sgd/clean/params_rmses.txt', 'w') as f:
			for param in results:
				for value in results[param]:
					f.write( "{param}={value}\t : {RMSE}\n".format(param=param, value=value, RMSE=results[param][value]) )

	return defaults

def fastFM_protocol_evaluation(data_path, params):
	solr = "http://localhost:8983/solr/grrecsys"
	all_data, y_all, items = loadData("eval_all_N20.data", data_path=data_path, with_timestamps=False, with_authors=False)
	v = DictVectorizer()
	X_all = v.fit_transform(all_data)

	test_c  = consumption(ratings_path= 'TwitterRatings/funkSVD/data/test/test_N20.data', rel_thresh= 0, with_ratings= True)
	train_c = consumption(ratings_path= 'TwitterRatings/funkSVD/data/eval_train_N20.data', rel_thresh= 0, with_ratings= False)
	all_c   = consumption(ratings_path= 'TwitterRatings/funkSVD/data/eval_all_N20.data', rel_thresh= 0, with_ratings= True)
	MRRs   = dict((N, []) for N in [5, 10, 15, 20])
	nDCGs  = dict((N, []) for N in [5, 10, 15, 20])
	APs    = dict((N, []) for N in [5, 10, 15, 20])
	Rprecs = dict((N, []) for N in [5, 10, 15, 20])

	train_data, y_tr, _ = loadData('eval_train_N20.data', data_path=data_path, with_timestamps=False, with_authors=False)
	X_tr = v.transform(train_data)
	fm = sgd.FMRegression(n_iter=params['mi'], init_stdev=params['init_stdev'], rank=params['f'], random_state=123, \
												l2_reg_w=params['l2_reg_w'], l2_reg_V=params['l2_reg_V'], l2_reg=params['l2_reg'], step_size=params['step_size'])
	fm.fit(X_tr, y_tr)



	p=0
	for userId in test_c:
		logging.info("#u: {0}/{1}".format(p, len(test_c)))
		p+=1
		user_rows = [ {'user_id': str(userId), 'item_id': str(itemId)} for itemId in items ]
		X_te      = v.transform(user_rows)
		preds     = fm.predict(X_te) #lista de ratings

		# book_recs = []
		# for i in range(len(preds)):
		# 	l = v.inverse_transform( X_te[i,:] )[0].keys()
		# 	pred_itemId = [s for s in l if "item" in s][0].split('=')[-1]
		# 	book_recs.append( [preds[i], pred_itemId] )
		# 	# if i==100: break #no necesitamos una lista de recomendaciones más larga que 100
		# book_recs = [ itemId for rating, itemId in sorted(book_recs, reverse=True) ] 

		# Son equivalentes, dado que items->user_rows->X_te->(preds)->inverse_transform
		# mantiene el orden de items
		# Esto es más rápido:
		book_recs = [itemId for _, itemId in sorted(zip(preds, items), reverse=True)]
		book_recs = remove_consumed(user_consumption= train_c[userId], rec_list= book_recs)
		book_recs = recs_cleaner(solr= solr, consumpt= train_c[userId], recs= book_recs[:100])		
		recs      = user_ranked_recs(user_recs= book_recs, user_consumpt= test_c[userId])	

		for N in [5, 10, 15, 20]:
			mini_recs = dict((k, recs[k]) for k in recs.keys()[:N])
			MRRs[N].append( MRR(recs=mini_recs, rel_thresh=1) )
			nDCGs[N].append( nDCG(recs=mini_recs, alt_form=False, rel_thresh=False) )		
			APs[N].append( AP_at_N(n=N, recs=recs, rel_thresh=1) )
			Rprecs[N].append( R_precision(n_relevants=N, recs=mini_recs) )


	for N in [5, 10, 15, 20]:
		with open('TwitterRatings/fastFM/sgd/clean/protocol.txt', 'a') as file:
			file.write( "N=%s, nDCG=%s, MAP=%s, MRR=%s, R-precision=%s\n" % \
				(N, mean(nDCGs[N]), mean(APs[N]), mean(MRRs[N]), mean(Rprecs[N])) )	




def fastFM_protocol_evaluation_bpr(data_path, params):
	# userId = '33120270'
	solr = "http://localhost:8983/solr/grrecsys"
	all_data, y_all, items = loadData_bpr("eval_all_N20.data", data_path=data_path)
	v = DictVectorizer()
	X_all = v.fit_transform(all_data)

	test_c  = consumption(ratings_path= 'TwitterRatings/funkSVD/data/test/test_N20.data', rel_thresh= 0, with_ratings= True)
	train_c = consumption(ratings_path= 'TwitterRatings/funkSVD/data/eval_train_N20.data', rel_thresh= 0, with_ratings= False)
	all_c   = consumption(ratings_path= 'TwitterRatings/funkSVD/data/eval_all_N20.data', rel_thresh= 0, with_ratings= True)
	MRRs   = dict((N, []) for N in [5, 10, 15, 20])
	nDCGs  = dict((N, []) for N in [5, 10, 15, 20])
	APs    = dict((N, []) for N in [5, 10, 15, 20])
	Rprecs = dict((N, []) for N in [5, 10, 15, 20])

	train_data, y_tr, _ = loadData_bpr('eval_train_N20.data', data_path=data_path)
	X_tr = v.transform(train_data)
	fm = bpr.FMRecommender(n_iter=params['mi'], init_stdev=params['init_stdev'], rank=params['f'], random_state=123, \
												 l2_reg_w=params['l2_reg_w'], l2_reg_V=params['l2_reg_V'], l2_reg=params['l2_reg'], step_size=params['step_size'])
	pairs_tr = make_pairs(X_tr, y_tr)
	fm.fit(X_tr, pairs_tr)

	p=0
	for userId in test_c:
		logging.info("#u: {0}/{1}".format(p, len(test_c)))
		p+=1
		user_rows = [ {'user_id': str(userId), 'item_id': str(itemId)} for itemId in items ]
		X_te      = v.transform(user_rows)
		preds     = fm.predict(X_te)
		preds     = np.argsort(-preds) # ordenamos las predicciones
		book_recs = []
		for i in range(len(preds)):
			print("i={}".format(i))
			pred_row = preds[i]
			l = v.inverse_transform( X_te[pred_row,:] )[0].keys()
			pred_itemId = [s for s in l if "item" in s][0].split('=')[-1]
			book_recs.append(pred_itemId)
			if i==100: break # no necesitamos más de 100

		book_recs = remove_consumed(user_consumption=train_c[userId], rec_list=book_recs)
		book_recs = recs_cleaner(solr= solr, consumpt= train_c[userId], recs= book_recs[:100])		
		recs      = user_ranked_recs(user_recs= book_recs, user_consumpt= test_c[userId])	

		for N in [5, 10, 15, 20]:
			mini_recs = dict((k, recs[k]) for k in recs.keys()[:N])
			MRRs[N].append( MRR(recs=mini_recs, rel_thresh=1) )
			nDCGs[N].append( nDCG(recs=mini_recs, alt_form=False, rel_thresh=False) )		
			APs[N].append( AP_at_N(n=N, recs=recs, rel_thresh=1) )
			Rprecs[N].append( R_precision(n_relevants=N, recs=mini_recs) )


	for N in [5, 10, 15, 20]:
		with open('TwitterRatings/fastFM/bpr/clean/protocol.txt', 'a') as file:
			file.write( "N=%s, nDCG=%s, MAP=%s, MRR=%s, R-precision=%s\n" % \
				(N, mean(nDCGs[N]), mean(APs[N]), mean(MRRs[N]), mean(Rprecs[N])) )	



def main():
	data_path = 'TwitterRatings/funkSVD/data_with_authors/'

	# opt_params_sgd = fastFM_tuning(data_path=data_path, N=20, solver="sgd")
	opt_params_bpr = fastFM_tuning_bpr(data_path=data_path, N=20)
	opt_params_sgd = {'mi':150, 'init_stdev':0.1, 'f':20, 'l2_reg_w':0.5, 'l2_reg_V':0.0001, 'l2_reg':0.5, 'step_size':0.1}
	# opt_params_bpr = {'mi':10, 'init_stdev':0.1, 'f':880, 'l2_reg_w':0.01, 'l2_reg_V':0.01, 'l2_reg':0.02, 'step_size':0.005}
	# for N in [5, 10, 15, 20]:
	fastFM_protocol_evaluation(data_path=data_path, params=opt_params_sgd)
	fastFM_protocol_evaluation_bpr(data_path=data_path, params=opt_params_bpr)

if __name__ == '__main__':
	main()

# userId='33120270'

# data_path = 'TwitterRatings/funkSVD/data/'
# N=20
# i=1
# train_c = consumption(ratings_path= data_path+'eval_train_N'+str(N)+'.data', rel_thresh= 0, with_ratings= True)
# all_data, y_all, items = loadData("eval_all_N"+str(N)+".data")
# v = DictVectorizer()
# X_all = v.fit_transform(all_data)

# train_data, y_tr, _ = loadData('train/train_N'+str(N)+'.'+str(i))
# val_data, y_va, _   = loadData('val/val_N'+str(N)+'.'+str(i))
# X_tr = v.transform(train_data)
# X_va = v.transform(val_data)

# # fm = mcmc.FMRegression(n_iter=100, init_stdev=0.1, rank=8, random_state=123, copy_X=True) #Illegal instruction (core dumped)
# # preds = fm.fit_predict(X_tr, y_tr, X_va)
# # fm  = als.FMRegression(n_iter=100, init_stdev=0.1, rank=8, random_state=123, l2_reg_w=0.1, l2_reg_V=0.1, l2_reg=0) #Illegal instruction (core dumped)
# # preds = fm.fit(X_tr, y_tr)
# fm        = bpr.FMRecommender(n_iter=100, init_stdev=0.1, rank=8, random_state=123, l2_reg_w=0.1, l2_reg_V=0.1, l2_reg=0, step_size=0.1)
# pairs_tr  = make_pairs(X_tr, y_tr)
# fm.fit(X_tr, pairs_tr)
# preds     = fm.predict(X_va)
# pred_rank = np.argsort(-preds)
# pairs_va  = make_pairs(X_va, y_va)
# real_rank = np.argsort()

# fm  = sgd.FMRegression(n_iter=100, init_stdev=0.1, rank=8, random_state=123, l2_reg_w=0.1, l2_reg_V=0.1, l2_reg=0, step_size=0.1)
# fm.fit(X_tr, y_tr)
# preds = fm.predict(X_va)
# rmse = sqrt( mean_squared_error(y_va, preds) )




# ratings = open(data_path+'train/train_N5.1','r')
# df = pd.read_csv(ratings, names=['User ID', 'Item ID', 'Rating'], dtype={'User ID': 'str', 'Item ID': 'str', 'Rating': 'float32'})
# y_tr= np.array(df['Rating'].as_matrix(), dtype=np.float64)
# df  = df.drop('Rating', 1)
# t_r = pd.get_dummies(df)
# X_tr= csc_matrix(t_r,  dtype=np.float64)

# ratings = open(data_path+'val/val_N5.1', 'r')
# df   = pd.read_csv(ratings, names=['User ID', 'Item ID', 'Rating'], dtype={'User ID': 'str', 'Item ID': 'str', 'Rating': 'float32'})
# y_va = np.array(df['Rating'].as_matrix(), dtype=np.float64)
# df   = df.drop('Rating', 1)
# t_r  = pd.get_dummies(df)
# X_va = csc_matrix(t_r, dtype=np.float64)


# ##############################
# # Intento 1 de agregación: NO funciona
# data_path = 'TwitterRatings/funkSVD/data/'
# ratings = open(data_path+'test/test_N20.data','r')
# initial_ratings_df = pd.read_csv(ratings, names=['User ID', 'Item ID', 'Rating'], dtype={'User ID': 'str', 'Item ID': 'str', 'Rating': 'float32'})
# initial_ratings_df.set_index('Session ID', inplace=True)
# t_r = pd.get_dummies(initial_ratings_df)
# f_r = t_r.filter(regex="Item.*")
# h_r_d = f_r.groupby(f_r.index).sum()

# merged = pd.merge(t_r, h_r_d, left_index=True, right_index=True)
# y = np.array(merged['Rating'].as_matrix())
# # merged.drop(['Rating'], 1, inplace=True)
# X = np.array(merged)
# # Para ver el tamaño de los arrays
# i=0
# for num in list(X[5]):
# 	if num!=0: 
# 		i+=1
# 		print(num) 

# ##############################
# # Intento 2 de agregación: funciona
# df = pd.DataFrame( np.array([['a', 'x', 5], ['a', 'y', 5], ['a', 'z', 4],['b', 'x', 1],['c', 'y', 2], ['c', 'z', 3]]), columns=['User ID', 'Item ID', 'Rating'])
# df['Rating'] = pd.to_numeric(df['Rating'])
# df['_User ID'] = df['User ID']
# df['_Rating'] = df['Rating']
# df.set_index(['User ID','Rating'], inplace=True)

# t_r = pd.get_dummies(df)
# f_r = t_r.filter(regex="Item.*")
# h_r_d = f_r.groupby(f_r.index).sum()
# merged = pd.merge(t_r, h_r_d, left_index=True, right_index=True)

# ##############################
