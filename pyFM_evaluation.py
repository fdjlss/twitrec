from sklearn.model_selection import train_test_split
from jojFunkSvd import mean, stdev, MRR, rel_div, DCG, iDCG, nDCG, P_at_N, AP_at_N, R_precision, consumption, user_ranked_recs, opt_value
from solr_evaluation import remove_consumed
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix
import pandas as pd
from implicit_evaluation import IdCoder
from pyfm import pylibfm
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_squared_error, mean_absolute_error

#-----"PRIVATE" METHODS----------#
def FMJob(data_path, params, N):
	maes = []
	for i in range(1, 4+1):
		ratings = open(data_path+'train/train_N'+str(N)+'.'+str(i), 'r')
		df  = pd.read_csv(ratings, names=['User ID', 'Item ID', 'Rating'], dtype={'User ID': 'str', 'Item ID': 'str', 'Rating': 'float32'})
		y_t = np.array(df['Rating'].as_matrix(), dtype=np.float64)
		df  = df.drop('Rating', 1)
		t_r = pd.get_dummies(df)
		X_t = np.array(t_r, dtype=np.float64)
		X_t = csr_matrix(X_t)
		fm  = pylibfm.FM(num_factors=params['f'], num_iter=params['mi'], k0=params['bias'], k1=params['oneway'], init_stdev=params['init_stdev'], \
										validation_size=params['val_size'], learning_rate_schedule=params['lr_s'], initial_learning_rate=params['lr'], \
										power_t=params['invscale_pow'], t0=params['optimal_denom'], shuffle_training=params['shuffle'], seed=params['seed'], \
										task='regression', verbose=True)
		fm.fit(X,y)
		ratings = open(data_path+'val/val_N'+str(N)+'.'+str(i), 'r')
		df  = pd.read_csv(ratings, names=['User ID', 'Item ID', 'Rating'], dtype={'User ID': 'str', 'Item ID': 'str', 'Rating': 'float32'})
		y_v = np.array(df['Rating'].as_matrix(), dtype=np.float64)
		df  = df.drop('Rating', 1)
		t_r = pd.get_dummies(df)
		X_v = np.array( t_r, dtype=np.float64 )
		X_v = csr_matrix(X)
		preds = fm.predict(X_v)
		mae = mean_absolute_error(y_v, preds)
		print("FM MAE: %.4f" % mean_absolute_error(y_v, preds))
		maes.append(mae)
	return mean(maes)
#--------------------------------#

def pyFM_tuning(data_path, N):

	defaults = {'f': 10, 'mi': 1, 'bias': True, 'oneway': True , 'init_stdev': 0.1, 'val_size': 0.01, 'lr_s': 'optimal', 'lr': 0.01, \
							'invscale_pow': 0.5, 'optimal_denom': 0.001, 'shuffle': True, 'seed': 28}
	results  = dict((param, {}) for param in defaults.keys())

	for param in ['f', 'mi', 'bias', 'oneway', 'init_stdev', 'val_size', 'lr_s', 'lr', 'invscale_pow', 'optimal_denom', 'shuffle', 'seed']:

		if param=='f': 
			for i in range(20, 2020, 20):
				defaults['f'] = i
				logging.info("Evaluando con params: {}".format(defaults))
				results['f'][i] = FMJob(data_path= data_path, params= defaults, N=N)
			defaults['f'] = opt_value(results= results['f'], metric= 'rmse')

		elif param=='mi':
			for i in [1, 5, 10, 20, 50, 100, 150, 200]: 
				defaults['mi'] = i
				logging.info("Evaluando con params: {}".format(defaults))
				results['mi'][i] = FMJob(data_path= data_path, params= defaults, N=N)
			defaults['mi'] = opt_value(results= results['mi'], metric= 'rmse')

		elif param=='bias':
			for i in [True, False]: 
				defaults['bias'] = i
				logging.info("Evaluando con params: {}".format(defaults))
				results['bias'][i] = FMJob(data_path= data_path, params= defaults, N=N)
			defaults['bias'] = opt_value(results= results['bias'], metric= 'rmse')

		elif param=='oneway':
			for i in [True, False]: 
				defaults['oneway'] = i
				logging.info("Evaluando con params: {}".format(defaults))
				results['oneway'][i] = FMJob(data_path= data_path, params= defaults, N=N)
			defaults['oneway'] = opt_value(results= results['oneway'], metric= 'rmse')

		elif param=='init_stdev':
			for i in [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0, 1]: 
				defaults['init_stdev'] = i
				logging.info("Evaluando con params: {}".format(defaults))
				results['init_stdev'][i] = FMJob(data_path= data_path, params= defaults, N=N)
			defaults['init_stdev'] = opt_value(results= results['init_stdev'], metric= 'rmse')

		elif param=='val_size':
			for i in [0.001, 0.01, 0.1, 0.5, 0.8, 0.9]: 
				defaults['val_size'] = i
				logging.info("Evaluando con params: {}".format(defaults))
				results['val_size'][i] = FMJob(data_path= data_path, params= defaults, N=N)
			defaults['val_size'] = opt_value(results= results['val_size'], metric= 'rmse')

		elif param=='lr_s':
			for i in ['constant', 'optimal', 'invscaling']: 
				defaults['lr_s'] = i

				if i=='optimal':
					for j in [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]:
						defaults['optimal_denom'] = j
						logging.info("Evaluando con params: {}".format(defaults))
						results['optimal_denom'][i] = FMJob(data_path= data_path, params= defaults, N=N)
					defaults['optimal_denom'] = opt_value(results= results['optimal_denom'], metric= 'rmse')
					results['lr_s'][i] = results['optimal_denom'][ defaults['optimal_denom'] ]
				elif i=='invscaling':
					for j in [0.001, 0.05, 0.1, 0.5, 0.8, 1]:
						defaults['invscale_pow'] = j
						logging.info("Evaluando con params: {}".format(defaults))
						results['invscale_pow'][i] = FMJob(data_path= data_path, params= defaults, N=N)
					defaults['invscale_pow'] = opt_value(results= results['invscale_pow'], metric= 'rmse')
					results['lr_s'][i] = results['invscale_pow'][ defaults['invscale_pow'] ]
				elif i=='constant':
					results['lr_s'][i] = FMJob(data_path= data_path, params= defaults, N=N)

			defaults['lr_s'] = opt_value(results= results['lr_s'], metric= 'rmse')

		elif param=='lr':
			for i in [0.001, 0.003, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.08, 0.1]: 
				defaults['lr'] = i
				logging.info("Evaluando con params: {}".format(defaults))
				results['lr'][i] = FMJob(data_path= data_path, params= defaults, N=N)
			defaults['lr'] = opt_value(results= results['lr'], metric= 'rmse')

		elif param=='shuffle':
			for i in [True, False]: 
				defaults['shuffle'] = i
				logging.info("Evaluando con params: {}".format(defaults))
				results['shuffle'][i] = FMJob(data_path= data_path, params= defaults, N=N)
			defaults['shuffle'] = opt_value(results= results['shuffle'], metric= 'rmse')

		elif param=='seed':
			for i in [10, 20, 28, 30, 50]: 
				defaults['seed'] = i
				logging.info("Evaluando con params: {}".format(defaults))
				results['seed'][i] = FMJob(data_path= data_path, params= defaults, N=N)
			defaults['seed'] = opt_value(results= results['seed'], metric= 'rmse')

	with open('TwitterRatings/pyFM/opt_params.txt', 'w') as f:
		for param in defaults:
			f.write( "{param}:{value}\n".format(param=param, value=defaults[param]) )

	with open('TwitterRatings/pyFM/params_maes.txt', 'w') as f:
		for param in results:
			for value in results[param]:
				f.write( "{param}={value}\t : {MAE}\n".format(param=param, value=value, MAE=results[param][value]) )

	return defaults


def main():
	data_path = 'TwitterRatings/funkSVD/data/'
	opt_params = pyFM_tuning(data_path=data_path, N=20)

if __name__ == '__main__':
	main()

# """DEBUGGING"""
# #####################################
# data_path = 'TwitterRatings/funkSVD/data/'
# ratings   = open(data_path+'train/train_N20.4','r')
# df  = pd.read_csv(ratings, names=['User ID', 'Item ID', 'Rating'], dtype={'User ID': 'str', 'Item ID': 'str', 'Rating': 'float32'})
# y   = np.array(df['Rating'].as_matrix(), dtype=np.float64)
# df  = df.drop('Rating', 1)
# t_r = pd.get_dummies(df)
# X   = np.array( t_r, dtype=np.float64 )
# X   = csr_matrix(X)

# fm = pylibfm.FM(num_factors=10, num_iter=20, verbose=True, task='regression',  initial_learning_rate=0.01, learning_rate_schedule='optimal' )
# fm.fit(X,y)

# data_path = 'TwitterRatings/funkSVD/data/'
# ratings   = open(data_path+'val/val_N20.4','r')
# df  = pd.read_csv(ratings, names=['User ID', 'Item ID', 'Rating'], dtype={'User ID': 'str', 'Item ID': 'str', 'Rating': 'float32'})
# y   = np.array(df['Rating'].as_matrix(), dtype=np.float64)
# df  = df.drop('Rating', 1)
# t_r = pd.get_dummies(df)
# X   = np.array( t_r, dtype=np.float64 )
# X   = csr_matrix(X)
# preds = fm.predict(X)
# print("FM MSE: %.4f" % mean_squared_error(y, preds))
# print("FM MAE: %.4f" % mean_absolute_error(y, preds))

# ###########################################
# # Gran ej.
# import numpy as np
# from sklearn.feature_extraction import DictVectorizer
# from pyfm import pylibfm

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