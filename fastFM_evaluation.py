# coding=utf-8

from fastFM.datasets import make_user_item_regression
from fastFM import als, sgd, bpr, mcmc
from sklearn.model_selection import train_test_split
from jojFunkSvd import mean, stdev, MRR, rel_div, DCG, iDCG, nDCG, P_at_N, AP_at_N, R_precision, consumption, user_ranked_recs, opt_value
from solr_evaluation import remove_consumed
from pyFM_evaluation import loadData
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt

#-----"PRIVATE" METHODS----------#
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
	first_u_ind, last_u_ind = 0,0
	prev_u_ind = sparse_matrix[0,:].nonzero()[1][1]
	for i in xrange(sparse_matrix.shape[0]):
		i_ind, u_ind = sparse_matrix[i,:].nonzero()[1]	
		if i == xrange(sparse_matrix.shape[0])[-1]:
			last_u_ind = i
			user_pairs = user_items_pairing(ratings, first_u_ind, last_u_ind)
			pairs += user_pairs
			continue
		if u_ind != prev_u_ind:
			last_u_ind = i-1
			user_pairs = user_items_pairing(ratings, first_u_ind, last_u_ind)
			first_u_ind = i
			prev_u_ind = u_ind
			pairs += user_pairs
	return np.array(pairs)

#TODO: CAMBIAR DE MÉTRICA
def fastFMJob_bpr(data_path, params, N, vectorizer):
	rmses = []
	logging.info("Evaluando con params: {0}".format(params))
	for i in range(1, 4+1):
		train_data, y_tr, _ = loadData('train/train_N'+str(N)+'.'+str(i))
		val_data, y_va, _   = loadData('val/val_N'+str(N)+'.'+str(i))
		X_tr = vectorizer.transform(train_data)
		X_va = vectorizer.transform(val_data)

		fm = bpr.FMRecommender(n_iter=params['mi'], init_stdev=params['init_stdev'], rank=params['f'], random_state=123, \
													 l2_reg_w=params['l2_reg_w'], l2_reg_V=params['l2_reg_V'], l2_reg=params['l2_reg'], step_size=params['step_size'])
		pairs_tr = make_pairs(X_tr, y_tr)
		fm.fit(X_tr, y_tr)
		preds = fm.predict(X_va)
		order = np.argsort(-preds)
		#metric = wea(order)


		logging.info("FM RMSE: {0}. Solver: {1}".format(rmse, solver) )
		rmses.append(rmse)
	return mean(rmses)

def fastFMJob(data_path, params, N, vectorizer, solver):
	rmses = []
	logging.info("Evaluando con params: {0}".format(params))
	for i in range(1, 4+1):
		train_data, y_tr, _ = loadData('train/train_N'+str(N)+'.'+str(i))
		val_data, y_va, _   = loadData('val/val_N'+str(N)+'.'+str(i))
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


#TODO: LA MÉTRICA QLA
def fastFM_tuning_bpr(data_path, N):
	all_data, y_all, _ = loadData("eval_all_N"+str(N)+".data")
	v = DictVectorizer()
	X_all = v.fit_transform(all_data)

	defaults = {'mi':100, 'init_stdev':0.1, 'f':8, 'l2_reg_w':0.1, 'l2_reg_V':0.1, 'l2_reg':0, 'step_size':0.1}

	results  = dict((param, {}) for param in defaults.keys())

	for param in ['mi', 'f', 'init_stdev', 'l2_reg_w', 'l2_reg_V', 'l2_reg', 'step_size']:

		if param=='mi':
			for i in [1, 5, 10, 20, 50, 100, 150, 200]: 
				defaults['mi'] = i
				results['mi'][i] = fastFMJob(data_path= data_path, params= defaults, N=N, vectorizer= v)
			defaults['mi'] = opt_value(results= results['mi'], metric= 'ndcg')

		elif param=='f': 
			for i in [1, 5, 8, 10] + range(20, 2020, 20):
				defaults['f'] = i
				results['f'][i] = fastFMJob(data_path= data_path, params= defaults, N=N, vectorizer= v)
			defaults['f'] = opt_value(results= results['f'], metric= 'ndcg')

		elif param=='init_stdev':
			for i in [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]: 
				defaults['init_stdev'] = i
				results['init_stdev'][i] = fastFMJob(data_path= data_path, params= defaults, N=N, vectorizer= v)
			defaults['init_stdev'] = opt_value(results= results['init_stdev'], metric= 'ndcg')

		elif param=='l2_reg_w':
			for i in [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]: 
				defaults['l2_reg_w'] = i
				results['l2_reg_w'][i] = fastFMJob(data_path= data_path, params= defaults, N=N, vectorizer= v)
			defaults['l2_reg_w'] = opt_value(results= results['l2_reg_w'], metric= 'ndcg')

		elif param=='l2_reg_V':
			for i in [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]: 
				defaults['l2_reg_V'] = i
				results['l2_reg_V'][i] = fastFMJob(data_path= data_path, params= defaults, N=N, vectorizer= v)
			defaults['l2_reg_V'] = opt_value(results= results['l2_reg_V'], metric= 'ndcg')

		elif param=='l2_reg':
			for i in [0.0, 0.001, 0.003, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.08, 0.1]: 
				defaults['l2_reg'] = i
				results['l2_reg'][i] = fastFMJob(data_path= data_path, params= defaults, N=N, vectorizer= v)
			defaults['l2_reg'] = opt_value(results= results['l2_reg'], metric= 'ndcg')

		elif param=='step_size':
			for i in [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.5]: 
				defaults['step_size'] = i
				results['step_size'][i] = fastFMJob(data_path= data_path, params= defaults, N=N, vectorizer= v)
			defaults['step_size'] = opt_value(results= results['step_size'], metric= 'ndcg')


	# # Real testing
	# train_data, y_tr, _ = loadData('eval_train_N'+str(N)+'.data')
	# X_tr = v.transform(train_data)
	# fm   = pylibfm.FM(num_factors=defaults['f'], num_iter=defaults['mi'], k0=defaults['bias'], k1=defaults['oneway'], init_stdev=defaults['init_stdev'], \
	# 								validation_size=defaults['val_size'], learning_rate_schedule=defaults['lr_s'], initial_learning_rate=defaults['lr'], \
	# 								power_t=defaults['invscale_pow'], t0=defaults['optimal_denom'], shuffle_training=defaults['shuffle'], seed=defaults['seed'], \
	# 								task='regression', verbose=True)
	# fm.fit(X_tr, y_tr)

	# test_data, y_te, _ = loadData('test/test_N'+str(N)+'.data')
	# X_te  = v.transform(test_data)
	# preds = fm.predict(X_te)
	# rmse  = sqrt( mean_squared_error(y_te, preds) )
	# print("FM RMSE: %.4f" % rmse)


	# with open('TwitterRatings/pyFM/opt_params.txt', 'w') as f:
	# 	for param in defaults:
	# 		f.write( "{param}:{value}\n".format(param=param, value=defaults[param]) )
	# 	f.write( "RMSE:{rmse}".format(rmse= rmse) )

	# with open('TwitterRatings/pyFM/params_rmses.txt', 'w') as f:
	# 	for param in results:
	# 		for value in results[param]:
	# 			f.write( "{param}={value}\t : {RMSE}\n".format(param=param, value=value, RMSE=results[param][value]) )

	# return defaults

def fastFM_tuning(data_path, N, solver):
	all_data, y_all, _ = loadData("eval_all_N"+str(N)+".data")
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
	train_data, y_tr, _ = loadData('eval_train_N'+str(N)+'.data')
	test_data, y_te, _  = loadData('test/test_N'+str(N)+'.data')
	X_tr = v.transform(train_data)
	X_te = v.transform(test_data)

	if solver=="mcmc":
		fm = mcmc.FMRegression(n_iter=defaults['mi'], init_stdev=defaults['init_stdev'], rank=defaults['f'], random_state=123, copy_X=True)
		preds = fm.fit_predict(X_tr, y_tr, X_te)
		rmse  = sqrt( mean_squared_error(y_te, preds) )
		logging.info("FM RMSE: {0}. Solver: {1}".format(rmse, solver) )
		with open('TwitterRatings/fastFM/mcmc/opt_params.txt', 'w') as f:
			for param in defaults:
				f.write( "{param}:{value}\n".format(param=param, value=defaults[param]) )
			f.write( "RMSE:{rmse}".format(rmse= rmse) )
		with open('TwitterRatings/fastFM/mcmc/params_rmses.txt', 'w') as f:
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
		with open('TwitterRatings/fastFM/als/opt_params.txt', 'w') as f:
			for param in defaults:
				f.write( "{param}:{value}\n".format(param=param, value=defaults[param]) )
			f.write( "RMSE:{rmse}".format(rmse= rmse) )
		with open('TwitterRatings/fastFM/als/params_rmses.txt', 'w') as f:
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
		with open('TwitterRatings/fastFM/sgd/opt_params.txt', 'w') as f:
			for param in defaults:
				f.write( "{param}:{value}\n".format(param=param, value=defaults[param]) )
			f.write( "RMSE:{rmse}".format(rmse= rmse) )
		with open('TwitterRatings/fastFM/sgd/params_rmses.txt', 'w') as f:
			for param in results:
				for value in results[param]:
					f.write( "{param}={value}\t : {RMSE}\n".format(param=param, value=value, RMSE=results[param][value]) )


	return defaults




def main():
	data_path = 'TwitterRatings/funkSVD/data/'
	opt_params_sgd = fastFM_tuning(data_path=data_path, N=20, solver="sgd")
	opt_params_bpr = fastFM_bpr_tuning(data_path=data_path, N=20, solver="bpr")
	for N in [5, 10, 15, 20]:
		fastFM_protocol_evaluation(data_path=data_path, params=opt_params, N=N)


if __name__ == '__main__':
	main()




# data_path = 'TwitterRatings/funkSVD/data/'
# N=5
# i=1
# train_c = consumption(ratings_path= data_path+'eval_train_N'+str(N)+'.data', rel_thresh= 0, with_ratings= True)
# all_data, y_all, _ = loadData("eval_all_N"+str(N)+".data")
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
