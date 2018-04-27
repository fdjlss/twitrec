# coding=utf-8

from sklearn.model_selection import train_test_split
from svd_evaluation import mean, stdev, MRR, rel_div, DCG, iDCG, nDCG, P_at_N, AP_at_N, R_precision, consumption, user_ranked_recs, opt_value
from solr_evaluation import remove_consumed
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix
import pandas as pd
from pyfm import pylibfm
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt

#-----"PRIVATE" METHODS----------#
def loadData(filename, data_path='TwitterRatings/funkSVD/data/'):
  data = []
  y = []
  items=set()
  with open(data_path+filename, 'r') as f:
    for line in f:
      (userId,itemId,rating)=line.split(',')
      data.append({ "user_id": str(userId), "item_id": str(itemId)})
      y.append(float(rating))
      items.add(itemId)
  return data, np.array(y), items

def pyFMJob(data_path, params, N, vectorizer):
	rmses = []
	logging.info("Evaluando con params: {0}".format(params))
	for i in range(1, 4+1):
		train_data, y_tr, _ = loadData('train/train_N'+str(N)+'.'+str(i))
		X_tr = vectorizer.transform(train_data)

		fm = pylibfm.FM(num_factors=params['f'], num_iter=params['mi'], k0=params['bias'], k1=params['oneway'], init_stdev=params['init_stdev'], \
										validation_size=params['val_size'], learning_rate_schedule=params['lr_s'], initial_learning_rate=params['lr'], \
										power_t=params['invscale_pow'], t0=params['optimal_denom'], shuffle_training=params['shuffle'], seed=params['seed'], \
										task='regression', verbose=True)
		fm.fit(X_tr, y_tr)

		val_data, y_va, _ = loadData('val/val_N'+str(N)+'.'+str(i))
		X_va = vectorizer.transform(val_data)

		preds = fm.predict(X_va)
		rmse  = sqrt( mean_squared_error(y_va, preds) )
		print("FM RMSE: %.4f" % rmse)
		rmses.append(rmse)
	return mean(rmses)
#--------------------------------#

def pyFM_tuning(data_path, N):

	all_data, y_all, _ = loadData("eval_all_N"+str(N)+".data")
	v = DictVectorizer()
	X_all = v.fit_transform(all_data)

	defaults = {'f': 100, 'mi': 20, 'bias': True, 'oneway': True , 'init_stdev': 0.1, 'val_size': 0.01, 'lr_s': 'optimal', 'lr': 0.01, \
							'invscale_pow': 0.5, 'optimal_denom': 0.001, 'shuffle': True, 'seed': 28} #cambio del original: f:20, mi:1
	results  = dict((param, {}) for param in defaults.keys())

	for param in ['mi', 'f', 'bias', 'oneway', 'init_stdev', 'val_size', 'lr_s', 'lr', 'invscale_pow', 'optimal_denom', 'shuffle', 'seed']:

		if param=='mi':
			for i in [1, 5, 10, 20, 50, 100, 150, 200]: 
				defaults['mi'] = i
				results['mi'][i] = pyFMJob(data_path= data_path, params= defaults, N=N, vectorizer= v)
			defaults['mi'] = opt_value(results= results['mi'], metric= 'rmse')

		elif param=='f': 
			for i in range(20, 2020, 20):
				defaults['f'] = i
				results['f'][i] = pyFMJob(data_path= data_path, params= defaults, N=N, vectorizer= v)
			defaults['f'] = opt_value(results= results['f'], metric= 'rmse')

		elif param=='bias':
			for i in [True, False]: 
				defaults['bias'] = i
				results['bias'][i] = pyFMJob(data_path= data_path, params= defaults, N=N, vectorizer= v)
			defaults['bias'] = opt_value(results= results['bias'], metric= 'rmse')

		elif param=='oneway':
			for i in [True, False]: 
				defaults['oneway'] = i
				results['oneway'][i] = pyFMJob(data_path= data_path, params= defaults, N=N, vectorizer= v)
			defaults['oneway'] = opt_value(results= results['oneway'], metric= 'rmse')

		elif param=='init_stdev':
			for i in [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]: 
				defaults['init_stdev'] = i
				results['init_stdev'][i] = pyFMJob(data_path= data_path, params= defaults, N=N, vectorizer= v)
			defaults['init_stdev'] = opt_value(results= results['init_stdev'], metric= 'rmse')

		elif param=='val_size':
			for i in [0.001, 0.01, 0.1, 0.5, 0.8, 0.9]: 
				defaults['val_size'] = i
				results['val_size'][i] = pyFMJob(data_path= data_path, params= defaults, N=N, vectorizer= v)
			defaults['val_size'] = opt_value(results= results['val_size'], metric= 'rmse')

		elif param=='lr_s':
			for i in ['constant', 'optimal', 'invscaling']: 
				defaults['lr_s'] = i

				if i=='optimal':
					for j in [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]:
						defaults['optimal_denom'] = j
						results['optimal_denom'][j] = pyFMJob(data_path= data_path, params= defaults, N=N, vectorizer= v)
					defaults['optimal_denom'] = opt_value(results= results['optimal_denom'], metric= 'rmse')
					results['lr_s'][i] = results['optimal_denom'][ defaults['optimal_denom'] ]

				elif i=='invscaling':
					for j in [0.001, 0.05, 0.1, 0.5, 0.8, 1.0]:
						defaults['invscale_pow'] = j
						results['invscale_pow'][j] = pyFMJob(data_path= data_path, params= defaults, N=N, vectorizer= v)
					defaults['invscale_pow'] = opt_value(results= results['invscale_pow'], metric= 'rmse')
					results['lr_s'][i] = results['invscale_pow'][ defaults['invscale_pow'] ]

				elif i=='constant':
					results['lr_s'][i] = pyFMJob(data_path= data_path, params= defaults, N=N, vectorizer= v)

			defaults['lr_s'] = opt_value(results= results['lr_s'], metric= 'rmse')

		elif param=='lr':
			for i in [0.001, 0.003, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.08, 0.1]: 
				defaults['lr'] = i
				results['lr'][i] = pyFMJob(data_path= data_path, params= defaults, N=N, vectorizer= v)
			defaults['lr'] = opt_value(results= results['lr'], metric= 'rmse')

		elif param=='shuffle':
			for i in [True, False]: 
				defaults['shuffle'] = i
				results['shuffle'][i] = pyFMJob(data_path= data_path, params= defaults, N=N, vectorizer= v)
			defaults['shuffle'] = opt_value(results= results['shuffle'], metric= 'rmse')

		elif param=='seed':
			for i in [10, 20, 28, 30, 50]: 
				defaults['seed'] = i
				results['seed'][i] = pyFMJob(data_path= data_path, params= defaults, N=N, vectorizer= v)
			defaults['seed'] = opt_value(results= results['seed'], metric= 'rmse')


	# Real testing
	train_data, y_tr, _ = loadData('eval_train_N'+str(N)+'.data')
	X_tr = v.transform(train_data)
	fm   = pylibfm.FM(num_factors=defaults['f'], num_iter=defaults['mi'], k0=defaults['bias'], k1=defaults['oneway'], init_stdev=defaults['init_stdev'], \
									validation_size=defaults['val_size'], learning_rate_schedule=defaults['lr_s'], initial_learning_rate=defaults['lr'], \
									power_t=defaults['invscale_pow'], t0=defaults['optimal_denom'], shuffle_training=defaults['shuffle'], seed=defaults['seed'], \
									task='regression', verbose=True)
	fm.fit(X_tr, y_tr)

	test_data, y_te, _ = loadData('test/test_N'+str(N)+'.data')
	X_te  = v.transform(test_data)
	preds = fm.predict(X_te)
	rmse  = sqrt( mean_squared_error(y_te, preds) )
	print("FM RMSE: %.4f" % rmse)


	with open('TwitterRatings/pyFM/opt_params.txt', 'w') as f:
		for param in defaults:
			f.write( "{param}:{value}\n".format(param=param, value=defaults[param]) )
		f.write( "RMSE:{rmse}".format(rmse= rmse) )

	with open('TwitterRatings/pyFM/params_rmses.txt', 'w') as f:
		for param in results:
			for value in results[param]:
				f.write( "{param}={value}\t : {RMSE}\n".format(param=param, value=value, RMSE=results[param][value]) )

	return defaults

def pyFM_protocol_evaluation(data_path, params, N):
	all_data, y_all, items = loadData("eval_all_N"+str(N)+".data")
	v = DictVectorizer()
	X_all = v.fit_transform(all_data)

	test_c  = consumption(ratings_path= data_path+'test/test_N'+str(N)+'.data', rel_thresh= 0, with_ratings= True)
	train_c = consumption(ratings_path= data_path+'eval_train_N'+str(N)+'.data', rel_thresh= 0, with_ratings= False)
	all_c   = consumption(ratings_path= data_path+'eval_all_N'+str(N)+'.data', rel_thresh= 0, with_ratings= True)
	MRRs          = []
	nDCGs_normal  = []
	APs           = []
	Rprecs        = []

	train_data, y_tr, _ = loadData('eval_train_N'+str(N)+'.data')
	X_tr = v.transform(train_data)
	fm   = pylibfm.FM(num_factors=params['f'], num_iter=params['mi'], k0=params['bias'], k1=params['oneway'], init_stdev=params['init_stdev'], \
									validation_size=params['val_size'], learning_rate_schedule=params['lr_s'], initial_learning_rate=params['lr'], \
									power_t=params['invscale_pow'], t0=params['optimal_denom'], shuffle_training=params['shuffle'], seed=params['seed'], \
									task='regression', verbose=True)
	fm.fit(X_tr, y_tr)

	p=0
	for userId in test_c:
		logging.info("#u: {0}/{1}".format(p, len(test_c)))
		p=+1
		user_rows = [ {'user_id': str(userId), 'item_id': str(itemId)} for itemId in items ]
		X_te      = v.transform(user_rows)
		preds     = fm.predict(X_te)
		book_recs = [itemId for _, itemId in sorted(zip(preds, items), reverse=True)]
		book_recs = remove_consumed(user_consumption= train_c[userId], rec_list= book_recs)
		recs      = user_ranked_recs(user_recs= book_recs, user_consumpt= test_c[userId])	

	####################################
		mini_recs = dict((k, recs[k]) for k in recs.keys()[:N]) #DEVUELVO SÓLO N RECOMENDACIONES
		nDCGs_normal.append( nDCG(recs=mini_recs, alt_form=False, rel_thresh=False) )
		APs.append( AP_at_N(n=N, recs=mini_recs, rel_thresh=1) )
		MRRs.append( MRR(recs=mini_recs, rel_thresh=1) )
		Rprecs.append( R_precision(n_relevants=N, recs=mini_recs) )

	with open('TwitterRatings/pyFM/protocol.txt', 'a') as file:
		file.write( "N=%s, normal nDCG=%s, MAP=%s, MRR=%s, R-precision=%s\n" % \
				(N, mean(nDCGs_normal), mean(APs), mean(MRRs), mean(Rprecs)) )	
	####################################

def main():
	data_path = 'TwitterRatings/funkSVD/data/'
	opt_params = pyFM_tuning(data_path=data_path, N=20)
	for N in [5, 10, 15, 20]:
		pyFM_protocol_evaluation(data_path=data_path, params=opt_params, N=N)

if __name__ == '__main__':
	main()

# """DEBUGGING"""
# userId = '322121021'


# import numpy as np
# from sklearn.feature_extraction import DictVectorizer
# from pyfm import pylibfm
# from sklearn.metrics import mean_squared_error

# # Read in data
# def loadData(filename,path="ml-100k/"):
#   data = []
#   y = []
#   users=set()
#   items=set()
#   with open(path+filename) as f:
#     for line in f:
#       (user,movieid,rating,ts)=line.split('\t')
#       data.append({ "user_id": str(user), "movie_id": str(movieid)})
#       y.append(float(rating))
#       users.add(user)
#       items.add(movieid)
#   return (data, np.array(y), users, items)

# (train_data, y_train, train_users, train_items) = loadData("ua.base")
# (test_data, y_test, test_users, test_items) = loadData("ua.test")
# v = DictVectorizer()
# X_train = v.fit_transform(train_data)
# X_test = v.transform(test_data)

# # Build and train a Factorization Machine
# fm = pylibfm.FM(num_factors=10, num_iter=20, verbose=True, task="regression", initial_learning_rate=0.001, learning_rate_schedule="optimal")
# fm.fit(X_train,y_train)

# # Evaluate
# preds = fm.predict(X_test)
# print("FM MSE: %.4f" % mean_squared_error(y_test,preds))