# coding=utf-8

#--------------------------------#
import time
import pyreclab
from random import sample
import gc
from time import sleep
import os
from math import sqrt

# Logging
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
#--------------------------------#

def mean(lst):
		return float(sum(lst)) / len(lst)

def stddev(lst):
    m = mean(lst)
    return sqrt(float(reduce(lambda x, y: x + y, map(lambda x: (x - m) ** 2, lst))) / len(lst))

def opt_value(results):
	for val in results:
		if results[val] == min( list(results.values()) ): 
			opt_value = val
	return opt_value

def ratingsSampler(rats, fout, n):

	l = len(rats)
	K = l*n
	ratings_sampled = sample(rats, k=int(K))

	with open(fout, 'w') as f:
		f.write( '\n'.join('%s' % x for x in ratings_sampled) )

	del ratings_sampled

def SVDJob(iterator=[], param=""):

	rmses = []
	maes  = []

	for i in iterator:

		svd = pyreclab.SVD( dataset   = 'TwitterRatings/funkSVD/ratings.train',
												dlmchar   = b',',
												header    = False,
												usercol   = 0,
												itemcol   = 1,
												ratingcol = 2 )

		# Pésima decisión de diseño, sí sé, pero bueno..
		
		if param=="factors":
			f    = i
			mi   = 100
			lr   = 0.01
			lamb = 0.1
			fn   = 'metrics_f'
		elif param=="maxiter":
			f    = 1000
			mi   = i
			lr   = 0.01
			lamb = 0.1
			fn   = 'metrics_mi'
		elif param=="lr":
			f    = 1000
			mi   = 100
			lr   = i/200
			lamb = 0.1
			fn   = 'metrics_lr'
		elif param=="lamb":
			f    = 1000
			mi   = 100
			lr   = 0.01
			lamb = i/20
			fn   = 'metrics_lamb'

		logging.info( "-> Entrenando modelo.." )
		logging.info( "N° Factores: {0}; maxiter: {1}; learning rate: {2}; lambda: {3} ".format(f, mi, lr, lamb) )

		start = time.clock()
		svd.train( factors= f, maxiter= mi, lr= lr, lamb= lamb )
		end = time.clock()

		logging.info( "training time: " + str(end - start) )

		logging.info( "-> Test de Predicción.." )
		start = time.clock()
		predlist, mae, rmse = svd.test( input_file  = 'TwitterRatings/funkSVD/ratings.test',
		                                dlmchar     = b',',
		                                header      = False,
		                                usercol     = 0,
		                                itemcol     = 1,
		                                ratingcol   = 2,
																		output_file = 'TwitterRatings/funkSVD/predictions_'+str(f)+'.csv' )
		end = time.clock()
		logging.info( "prediction time: " + str(end - start) )

		logging.info( 'MAE: ' + str(mae) )
		logging.info( 'RMSE: ' + str(rmse) )

		rmses.append(rmse)
		maes.append(mae)

	with open('TwitterRatings/funkSVD/'+fn+'.txt', 'w') as f:
		for i in range(0, len(iterator)):
			f.write( "%s\t%s\t%s\n" % (iterator[i], rmses[i], maes[i] ) )

def generate_recommends(params):

	svd = pyreclab.SVD( dataset   = 'TwitterRatings/funkSVD/ratings.train',
											dlmchar   = b',',
											header    = False,
											usercol   = 0,
											itemcol   = 1,
											ratingcol = 2 )

	logging.info( "-> Entrenando modelo.." )
	logging.info( "N° Factores: {0}; maxiter: {1}; learning rate: {2}; lambda: {3} ".format(f, mi, lr, lamb) )

	start = time.clock()
	svd.train( factors= params['f'], maxiter= params['mi'], lr= params['lr'], lamb= params['lamb'] )
	end = time.clock()

	logging.info( "training time: " + str(end - start) )

	logging.info( "-> Test de Recomendación.." )
	start = time.clock()
	recommendationList = svd.testrec( input_file    = 'TwitterRatings/funkSVD/ratings.test',
                                      dlmchar     = b',',
                                      header      = False,
                                      usercol     = 0,
                                      itemcol     = 1,
                                      ratingcol   = 2,
                                      topn        = 10,
                                      output_file = 'TwitterRatings/funkSVD/ranking.json' )
	end = time.clock()
	logging.info( 'recommendation time: ' + str( end - start ) )


def boosting(folds):

	ratings_train, ratings_test = [], []
	ratings_train_path, ratings_test_path = 'TwitterRatings/funkSVD/ratings.train', 'TwitterRatings/funkSVD/ratings.test'
	with open(ratings_train_path, 'r') as f:
		for line in f:
			ratings_train.append( line.strip() )

	with open(ratings_test_path, 'r') as f:
		for line in f:
			ratings_test.append( line.strip() )

	defaults = {'f': 1000, 'mi': 100, 'lr': 0.01, 'lamb': 0.1}
	results = {'f': {}, 'mi': {}, 'lr': {}, 'lamb': {}}

	for param in defaults:

		if param=='f':
			defaults['f'] = list(range(100, 1525, 25))

			for i in defaults['f']:

				rmses = []
				maes  = []
				for _ in range(0, folds):
					ratingsSampler(ratings_train, 'TwitterRatings/funkSVD/ratings_temp.train', 0.8)
					svd = pyreclab.SVD( dataset   = 'TwitterRatings/funkSVD/ratings_temp.train',
															dlmchar   = b',',
															header    = False,
															usercol   = 0,
															itemcol   = 1,
															ratingcol = 2 )
					svd.train( factors= i, maxiter= defaults['mi'], lr= defaults['lr'], lamb= defaults['lamb'] )
					ratingsSampler(ratings_test, 'TwitterRatings/funkSVD/ratings_temp.test', 0.8)
					predlist, mae, rmse = svd.test( input_file  = 'TwitterRatings/funkSVD/ratings_temp.test',
					                                dlmchar     = b',',
					                                header      = False,
					                                usercol     = 0,
					                                itemcol     = 1,
					                                ratingcol   = 2)
					rmses.append(rmse)
					maes.append(mae)

				# Escribe 1 archivo por cada valor de cada parámetro
				with open('TwitterRatings/funkSVD/params/'+param+'/'+str(i)+'.txt', 'w') as f:
					for j in range(0, folds):
						f.write( "%s\t%s\n" % (rmses[j], maes[j]) )

				results['f'][i] = mean(rmses)

			defaults['f']  = opt_value(results['f'])

		elif param=='lamb':
			defaults['lamb'] = [x / 200.0 for x in list(range(2, 201, 22))]
		
			for i in defaults['lamb']:

				rmses = []
				maes  = []
				for _ in range(0, folds):
					ratingsSampler(ratings_train, 'TwitterRatings/funkSVD/ratings_temp.train', 0.8)
					svd = pyreclab.SVD( dataset   = 'TwitterRatings/funkSVD/ratings_temp.train',
															dlmchar   = b',',
															header    = False,
															usercol   = 0,
															itemcol   = 1,
															ratingcol = 2 )
					svd.train( factors= defaults['f'], maxiter= defaults['mi'], lr= defaults['lr'], lamb= i )
					ratingsSampler(ratings_test, 'TwitterRatings/funkSVD/ratings_temp.test', 0.8)
					predlist, mae, rmse = svd.test( input_file  = 'TwitterRatings/funkSVD/ratings_temp.test',
					                                dlmchar     = b',',
					                                header      = False,
					                                usercol     = 0,
					                                itemcol     = 1,
					                                ratingcol   = 2)
					rmses.append(rmse)
					maes.append(mae)

				# Escribe 1 archivo por cada valor de cada parámetro
				with open('TwitterRatings/funkSVD/params/'+param+'/'+str(i)+'.txt', 'w') as f:
					for j in range(0, folds):
						f.write( "%s\t%s\n" % (rmses[j], maes[j]) )

				results['lamb'][i] = mean(rmses)

			defaults['lamb'] = opt_value(results['lamb'])

		elif param=='lr':
			defaults['lr']   = [x / 2000.0 for x in list(range(2, 201, 11))]

			for i in defaults['lr']:

				rmses = []
				maes  = []
				for _ in range(0, folds):
					ratingsSampler(ratings_train, 'TwitterRatings/funkSVD/ratings_temp.train', 0.8)
					svd = pyreclab.SVD( dataset   = 'TwitterRatings/funkSVD/ratings_temp.train',
															dlmchar   = b',',
															header    = False,
															usercol   = 0,
															itemcol   = 1,
															ratingcol = 2 )
					svd.train( factors= defaults['f'], maxiter= defaults['mi'], lr= i, lamb= defaults['lamb'] )
					ratingsSampler(ratings_test, 'TwitterRatings/funkSVD/ratings_temp.test', 0.8)
					predlist, mae, rmse = svd.test( input_file  = 'TwitterRatings/funkSVD/ratings_temp.test',
					                                dlmchar     = b',',
					                                header      = False,
					                                usercol     = 0,
					                                itemcol     = 1,
					                                ratingcol   = 2)
					rmses.append(rmse)
					maes.append(mae)

				# Escribe 1 archivo por cada valor de cada parámetro
				with open('TwitterRatings/funkSVD/params/'+param+'/'+str(i)+'.txt', 'w') as f:
					for j in range(0, folds):
						f.write( "%s\t%s\n" % (rmses[j], maes[j]) )

				results['lr'][i] = mean(rmses)

			defaults['lr'] = opt_value(results['lr'])

		elif param=='mi':
			defaults['mi'] = list(range(10, 520, 20))

			for i in defaults['mi']:

				rmses = []
				maes  = []
				for _ in range(0, folds):
					ratingsSampler(ratings_train, 'TwitterRatings/funkSVD/ratings_temp.train', 0.8)
					svd = pyreclab.SVD( dataset   = 'TwitterRatings/funkSVD/ratings_temp.train',
															dlmchar   = b',',
															header    = False,
															usercol   = 0,
															itemcol   = 1,
															ratingcol = 2 )
					svd.train( factors= defaults['f'], maxiter= i, lr= defaults['lr'], lamb= defaults['lamb'] )
					ratingsSampler(ratings_test, 'TwitterRatings/funkSVD/ratings_temp.test', 0.8)
					predlist, mae, rmse = svd.test( input_file  = 'TwitterRatings/funkSVD/ratings_temp.test',
					                                dlmchar     = b',',
					                                header      = False,
					                                usercol     = 0,
					                                itemcol     = 1,
					                                ratingcol   = 2)
					rmses.append(rmse)
					maes.append(mae)

				# Escribe 1 archivo por cada valor de cada parámetro
				with open('TwitterRatings/funkSVD/params/'+param+'/'+str(i)+'.txt', 'w') as f:
					for j in range(0, folds):
						f.write( "%s\t%s\n" % (rmses[j], maes[j]) )

				results['mi'][i] = mean(rmses)

			defaults['mi'] = opt_value(results['mi'])

	return defaults


def RMSEMAE_distr():
	path = "TwitterRatings/funkSVD/params/"
	datos = {}

	for param in os.listdir( path ):

		datos[param] = {}
		
		for value in os.listdir( path + param ):
			
			rmses = []
			maes = []

			with open(path + param + '/' + value, 'r') as f:

				for line in f:
					rmse, mae = line.strip().split('\t')
					rmses.append( float(rmse) )
					maes.append( float(mae) )

			rmse_mean, rmse_stddev = mean( rmses ), stddev( rmses )
			mae_mean, mae_stddev = mean( maes ), stddev( maes )

			datos[param][value[:-4]] = [ [rmse_mean, rmse_stddev], [mae_mean, mae_stddev] ]

	with open("TwitterRatings/funkSVD/resumen3.txt", 'w') as f:
		for param in datos:
			f.write("%s\n" % param)
			for v in sorted(datos[param].items()):
				#<value>  <RMSE_mean>,<RMSE_stddev>  <MAE_mean>,<MAE_stddev>
				f.write("%s\t%s,%s\t%s,%s\n" % ( v[0], v[1][0][0], v[1][0][1], v[1][1][0], v[1][1][1] ) )


def PRF_calculator(params, folds, topN):

	ratings_train, ratings_test = [], []

	with open('TwitterRatings/funkSVD/ratings.train', 'r') as f:
		for line in f:
			ratings_train.append( line.strip() )

	with open('TwitterRatings/funkSVD/ratings.test', 'r') as f:
		for line in f:
			ratings_test.append( line.strip() )

	for n in topN: 
		precision_folds, recall_folds = [], []
		for _ in range(0, folds):
			# ratingsSampler(ratings_train, 'TwitterRatings/funkSVD/ratings_temp.train', 0.8)
			# ratingsSampler(ratings_test, 'TwitterRatings/funkSVD/ratings_temp.test', 0.8)

			svd = pyreclab.SVD( dataset   = 'TwitterRatings/funkSVD/ratings.total', #o ratings_temp.train
													dlmchar   = b',',
													header    = False,
													usercol   = 0,
													itemcol   = 1,
													ratingcol = 2 )

			svd.train( factors= params['f'], maxiter= params['mi'], lr= params['lr'], lamb= params['lamb'] )

			recommendationList = svd.testrec( input_file    = 'TwitterRatings/funkSVD/ratings.test', #o ratings_temp.test
			                                    dlmchar     = b',',
			                                    header      = False,
			                                    usercol     = 0,
			                                    itemcol     = 1,
			                                    ratingcol   = 2,
			                                    topn        = n,
			                                    includeRated= True)
			                                    # output_file = 'TwitterRatings/funkSVD/ranking_temp.json' )

			preferred_consumption = {}
			with open('TwitterRatings/funkSVD/ratings.test', 'r') as f:
				for line in f:
					userId, itemId, rating = line.strip().split(',')
					if userId not in preferred_consumption:
						preferred_consumption[userId] = []
					if int( rating ) >= 4: # preferencia: los que leyó y marcó con rating >= 4
						preferred_consumption[userId].append(itemId) 

			users_precisions, users_recalls = [], []
			for userId in recommendationList[0]:
				recs = set(recommendationList[0][userId])
				cons = set(preferred_consumption[userId])
				tp = len(recs & cons)
				fp = len(recs - cons)
				fn = len(cons - recs)
				users_precisions.append( float(tp) / (tp + fp) )
				users_recalls.append( float(tp) / (tp + fn) )

			precision_folds.append( mean(users_precisions) )
			recall_folds.append( mean(users_recalls) )

		p = mean( precision_folds )
		r = mean( recall_folds )
		f = 2*p*r / (p + r)

		with open('TwitterRatings/funkSVD/recall.txt', 'a') as file:
			file.write( "%s,%s,%s,%s\n" % (n, p, r, f) )





factores = range(300, 1025, 25) # [300, 325, .., 1000]
max_iters = range(100, 520, 20) # [100, 120, .., 500]
lrn_rates = range(2, 21, 1) # [2, 3, .., 20] / 200 = [0.01, 0.015, .., 0.1]
reg_params = range(2, 21, 1) # [2, 3, .., 20] / 20 = [0.1, 0.15, .., 1]

opt_params = boosting(folds=15)
RMSEMAE_distr()
# generate_recommends(params=opt_params)
PRF_calculator(params=opt_params, folds=5, topN=[10, 20, 50])