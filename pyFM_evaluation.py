# coding=utf-8

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
from math import sqrt

#-----"PRIVATE" METHODS----------#
def FMJob(data_path, params, N):
	rmses = []
	for i in range(1, 4+1):
		ratings = open(data_path+'train/train_N'+str(N)+'.'+str(i), 'r')
		df   = pd.read_csv(ratings, names=['User ID', 'Item ID', 'Rating'], dtype={'User ID': 'str', 'Item ID': 'str', 'Rating': 'float32'})
		y_tr = np.array(df['Rating'].as_matrix(), dtype=np.float64)
		df   = df.drop('Rating', 1)
		t_r  = pd.get_dummies(df)
		X_tr = np.array(t_r, dtype=np.float64)
		X_tr = csr_matrix(X_tr)
		
		logging.info("Evaluando con params: {0}\nFold {1}".format(params, i))

		fm   = pylibfm.FM(num_factors=params['f'], num_iter=params['mi'], k0=params['bias'], k1=params['oneway'], init_stdev=params['init_stdev'], \
										validation_size=params['val_size'], learning_rate_schedule=params['lr_s'], initial_learning_rate=params['lr'], \
										power_t=params['invscale_pow'], t0=params['optimal_denom'], shuffle_training=params['shuffle'], seed=params['seed'], \
										task='regression', verbose=True)

		fm.fit(X_tr, y_tr)
		ratings = open(data_path+'val/val_N'+str(N)+'.'+str(i), 'r')
		df   = pd.read_csv(ratings, names=['User ID', 'Item ID', 'Rating'], dtype={'User ID': 'str', 'Item ID': 'str', 'Rating': 'float32'})
		y_va = np.array(df['Rating'].as_matrix(), dtype=np.float64)
		df   = df.drop('Rating', 1)
		t_r  = pd.get_dummies(df)
		X_va = np.array(t_r, dtype=np.float64)
		X_va = csr_matrix(X_va)
		preds = fm.predict(X_va)
		rmse = sqrt( mean_squared_error(y_va, preds) )
		print("FM RMSE: %.4f" % rmse)
		rmses.append(rmse)
	return mean(rmses)
#--------------------------------#

def pyFM_tuning(data_path, N):

	defaults = {'f': 10, 'mi': 1, 'bias': True, 'oneway': True , 'init_stdev': 0.1, 'val_size': 0.01, 'lr_s': 'optimal', 'lr': 0.01, \
							'invscale_pow': 0.5, 'optimal_denom': 0.001, 'shuffle': True, 'seed': 28}
	results  = dict((param, {}) for param in defaults.keys())

	for param in ['f', 'mi', 'bias', 'oneway', 'init_stdev', 'val_size', 'lr_s', 'lr', 'invscale_pow', 'optimal_denom', 'shuffle', 'seed']:

		if param=='f': 
			for i in range(20, 2020, 20):
				defaults['f'] = i
				results['f'][i] = FMJob(data_path= data_path, params= defaults, N=N)
			defaults['f'] = opt_value(results= results['f'], metric= 'rmse')

		elif param=='lr_s':
			for i in ['constant', 'optimal', 'invscaling']: 
				defaults['lr_s'] = i

				if i=='optimal':
					for j in [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]:
						defaults['optimal_denom'] = j
						results['optimal_denom'][j] = FMJob(data_path= data_path, params= defaults, N=N)
					defaults['optimal_denom'] = opt_value(results= results['optimal_denom'], metric= 'rmse')
					results['lr_s'][i] = results['optimal_denom'][ defaults['optimal_denom'] ]

				elif i=='invscaling':
					for j in [0.001, 0.05, 0.1, 0.5, 0.8, 1.0]:
						defaults['invscale_pow'] = j
						results['invscale_pow'][j] = FMJob(data_path= data_path, params= defaults, N=N)
					defaults['invscale_pow'] = opt_value(results= results['invscale_pow'], metric= 'rmse')
					results['lr_s'][i] = results['invscale_pow'][ defaults['invscale_pow'] ]
					
				elif i=='constant':
					results['lr_s'][i] = FMJob(data_path= data_path, params= defaults, N=N)

			defaults['lr_s'] = opt_value(results= results['lr_s'], metric= 'rmse')

		elif param=='mi':
			for i in [1, 5, 10, 20, 50, 100, 150, 200]: 
				defaults['mi'] = i
				results['mi'][i] = FMJob(data_path= data_path, params= defaults, N=N)
			defaults['mi'] = opt_value(results= results['mi'], metric= 'rmse')

		elif param=='bias':
			for i in [True, False]: 
				defaults['bias'] = i
				results['bias'][i] = FMJob(data_path= data_path, params= defaults, N=N)
			defaults['bias'] = opt_value(results= results['bias'], metric= 'rmse')

		elif param=='oneway':
			for i in [True, False]: 
				defaults['oneway'] = i
				results['oneway'][i] = FMJob(data_path= data_path, params= defaults, N=N)
			defaults['oneway'] = opt_value(results= results['oneway'], metric= 'rmse')

		elif param=='init_stdev':
			for i in [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]: 
				defaults['init_stdev'] = i
				results['init_stdev'][i] = FMJob(data_path= data_path, params= defaults, N=N)
			defaults['init_stdev'] = opt_value(results= results['init_stdev'], metric= 'rmse')

		elif param=='val_size':
			for i in [0.001, 0.01, 0.1, 0.5, 0.8, 0.9]: 
				defaults['val_size'] = i
				results['val_size'][i] = FMJob(data_path= data_path, params= defaults, N=N)
			defaults['val_size'] = opt_value(results= results['val_size'], metric= 'rmse')

		elif param=='lr':
			for i in [0.001, 0.003, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.08, 0.1]: 
				defaults['lr'] = i
				results['lr'][i] = FMJob(data_path= data_path, params= defaults, N=N)
			defaults['lr'] = opt_value(results= results['lr'], metric= 'rmse')

		elif param=='shuffle':
			for i in [True, False]: 
				defaults['shuffle'] = i
				results['shuffle'][i] = FMJob(data_path= data_path, params= defaults, N=N)
			defaults['shuffle'] = opt_value(results= results['shuffle'], metric= 'rmse')

		elif param=='seed':
			for i in [10, 20, 28, 30, 50]: 
				defaults['seed'] = i
				results['seed'][i] = FMJob(data_path= data_path, params= defaults, N=N)
			defaults['seed'] = opt_value(results= results['seed'], metric= 'rmse')


	# Real testing
	ratings = open(data_path+'eval_train_N20.data', 'r')
	df   = pd.read_csv(ratings, names=['User ID', 'Item ID', 'Rating'], dtype={'User ID': 'str', 'Item ID': 'str', 'Rating': 'float32'})
	y_tr = np.array(df['Rating'].as_matrix(), dtype=np.float64)
	df   = df.drop('Rating', 1)
	t_r  = pd.get_dummies(df)
	X_tr = np.array(t_r, dtype=np.float64)
	X_tr = csr_matrix(X_tr)
	fm   = pylibfm.FM(num_factors=defaults['f'], num_iter=defaults['mi'], k0=defaults['bias'], k1=defaults['oneway'], init_stdev=defaults['init_stdev'], \
									validation_size=defaults['val_size'], learning_rate_schedule=defaults['lr_s'], initial_learning_rate=defaults['lr'], \
									power_t=defaults['invscale_pow'], t0=defaults['optimal_denom'], shuffle_training=defaults['shuffle'], seed=defaults['seed'], \
									task='regression', verbose=True)
	fm.fit(X_tr, y_tr)
	ratings = open(data_path+'test/test_N20.data', 'r')
	df   = pd.read_csv(ratings, names=['User ID', 'Item ID', 'Rating'], dtype={'User ID': 'str', 'Item ID': 'str', 'Rating': 'float32'})
	y_va = np.array(df['Rating'].as_matrix(), dtype=np.float64)
	df   = df.drop('Rating', 1)
	t_r  = pd.get_dummies(df)
	X_va = np.array(t_r, dtype=np.float64)
	X_va = csr_matrix(X_va)
	preds = fm.predict(X_va)
	rmse = sqrt( mean_squared_error(y_va, preds) )
	print("FM RMSE: %.4f" % rmse)

	with open('TwitterRatings/pyFM/opt_params.txt', 'w') as f:
		for param in defaults:
			f.write( "{param}:{value}\nRMSE:{RMSE}".format(param=param, value=defaults[param], RMSE=results['seed'][ defaults['seed'] ]) )

	with open('TwitterRatings/pyFM/params_maes.txt', 'w') as f:
		for param in results:
			for value in results[param]:
				f.write( "{param}={value}\t : {RMSE}\n".format(param=param, value=value, RMSE=results[param][value]) )

	return defaults

def pyFM_protocol_evaluation(data_path, params, N):
	test_c  = consumption(ratings_path=data_path+'test/test_N'+str(N)+'.data', rel_thresh=0, with_ratings=True)
	train_c = consumption(ratings_path= data_path+'eval_train_N'+str(N)+'.data', rel_thresh= 0, with_ratings= False)
	all_c   = consumption(ratings_path= data_path+'eval_all_N'+str(N)+'.data', rel_thresh= 0, with_ratings= True)
	MRRs          = []
	nDCGs_bin     = []
	nDCGs_normal  = []
	nDCGs_altform = []
	APs           = []
	Rprecs        = []

	ratings = open(data_path+'eval_train_N'+str(N)+'.data', 'r')
	df   = pd.read_csv(ratings, names=['User ID', 'Item ID', 'Rating'], dtype={'User ID': 'str', 'Item ID': 'str', 'Rating': 'float32'})
	y_tr = np.array(df['Rating'].as_matrix(), dtype=np.float64)
	df   = df.drop('Rating', 1)
	t_r  = pd.get_dummies(df)
	# X_tr = np.array(t_r, dtype=np.float64) #Parece que no es necesario
	X_tr = csr_matrix(t_r, dtype=np.float64)
	fm   = pylibfm.FM(num_factors=params['f'], num_iter=params['mi'], k0=params['bias'], k1=params['oneway'], init_stdev=params['init_stdev'], \
									validation_size=params['val_size'], learning_rate_schedule=params['lr_s'], initial_learning_rate=params['lr'], \
									power_t=params['invscale_pow'], t0=params['optimal_denom'], shuffle_training=params['shuffle'], seed=params['seed'], \
									task='regression', verbose=True)
	fm.fit(X_tr, y_tr)

	ratings_all = open(data_path+'eval_all_N'+str(N)+'.data', 'r')
	df   = pd.read_csv(ratings_all, names=['User ID', 'Item ID', 'Rating'], dtype={'User ID': 'str', 'Item ID': 'str', 'Rating': 'float32'})
	y_te = np.array(df['Rating'].as_matrix(), dtype=np.float64)
	df   = df.drop('Rating', 1)
	t_r  = pd.get_dummies(df)

	item_matrix = t_r.filter(regex="Item ID.*")
	item_cols = item_matrix.shape[1]
	# item_matrix.iloc[:item_cols, :] = np.diag( np.repeat(1, item_cols) )

	for i in range(0, item_cols):
		for j in range(0, item_cols):
			if i==j:
				item_matrix.iloc[i,j] = 1
			else:
				item_matrix.iloc[i,j] = 0

	# ratings_test = open(data_path+'test/test_N'+str(N)+'.data', 'r')
	# df   = pd.read_csv(ratings_test, names=['User ID', 'Item ID', 'Rating'], dtype={'User ID': 'str', 'Item ID': 'str', 'Rating': 'float32'})
	# y_te = np.array(df['Rating'].as_matrix(), dtype=np.float64)
	# df   = df.drop('Rating', 1)
	# t_r  = pd.get_dummies(df)

	# Separacion matrices
	for userId in test_c:
		t_r_u  = t_r.filter(regex="Item ID.*|User ID_"+str(userId)) 	#dejamos como df la col del AU y la de los ítems 
		t_r_u  = t_r_u.iloc[:t_r_u.shape[1]-1, :] 										#sliceamos el df para que quede una matrix (n°items) x (n°items + 1)
		t_r_u["User ID_"+str(userId)] = np.repeat(1, t_r_u.shape[0]) 	#llenamos de unos la col del AU
		t_r_u  = t_r_u.reset_index(drop=True)

		np.diag( np.repeat(1, t_r_u.shape[0]) )
		t_r_u.iloc[:, 1:] = np.diag( np.repeat(1, t_r_u.shape[0]) )

		X_te_u = csr_matrix(t_r_u, dtype=np.float64)
		preds  = fm.predict(X_te_u)

		# t_r_u  = t_r.loc[t_r['User ID_'+str(userId)] == 1] #t_r.loc[t_r['User ID_422541286'] == 1]
		# t_r_u  = t_r_u.reset_index(drop=True) #prevenir que el viejo index se agregue como nueva columna
		# X_te_u = np.array(t_r_u, dtype=np.float64)
		# X_te_u = csr_matrix(X_te_u)
		# preds  = fm.predict(X_te_u)

		#AHORA SÍ TENGO LAS PREDICCIONES.
		#SEGUIR POR ACÁ 2017/12/27
		for i in range(0, len(preds)):
			item_rows = t_r_u.filter(regex="Item ID.*")
			col_name  = item_rows.columns[ (item_rows==1).iloc[i] ][0]
			itemId    = col_name.split('_')[-1]
			rating    = preds[i]

		# en alguna parte...
		book_recs  = [ str(tupl[0]) for tupl in recommends ]
		book_recs  = remove_consumed(user_consumption= train_c[userId], rec_list= book_recs)
		recs       = user_ranked_recs(user_recs= book_recs, user_consumpt= test_c[userId])	



	####################################
	for userId in recommendationList[0]:
		recs      = user_ranked_recs(user_recs=recommendationList[0][userId], user_consumpt=user_consumption[userId])
		mini_recs = dict((k, recs[k]) for k in recs.keys()[:N]) #DEVUELVO SÓLO N RECOMENDACIONES
		nDCGs_normal.append( nDCG(recs=mini_recs, alt_form=False, rel_thresh=False) )
		nDCGs_bin.append( nDCG(recs=mini_recs, alt_form=False, rel_thresh=1) )
		nDCGs_altform.append( nDCG(recs=mini_recs, alt_form=True, rel_thresh=False) )			
		APs.append( AP_at_N(n=N, recs=recs, rel_thresh=1) )
		MRRs.append( MRR(recs=recs, rel_thresh=1) )
		Rprecs.append( R_precision(n_relevants=N, recs=mini_recs) )

	with open('TwitterRatings/funkSVD/'+output_filename, 'a') as file:
		file.write( "N=%s, normal nDCG=%s, alternative nDCG=%s, bin nDCG=%s, MAP=%s, MRR=%s, R-precision=%s\n" % \
				(N, mean(nDCGs_normal), mean(nDCGs_altform), mean(nDCGs_bin), mean(APs), mean(MRRs), mean(Rprecs)) )	
	####################################

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