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
    mean = mean(lst)
    return sqrt(float(reduce(lambda x, y: x + y, map(lambda x: (x - mean) ** 2, lst))) / len(lst))


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

def SVDTesting():

	svd = pyreclab.SVD( dataset   = 'TwitterRatings/funkSVD/ratings.train',
											dlmchar   = b',',
											header    = False,
											usercol   = 0,
											itemcol   = 1,
											ratingcol = 2 )

	f    = 10000
	mi   = 10
	lr   = 0.0001
	lamb = 0.001

	logging.info( "-> Entrenando modelo.." )
	logging.info( "N° Factores: {0}; maxiter: {1}; learning rate: {2}; lambda: {3} ".format(f, mi, lr, lamb) )

	start = time.clock()
	svd.train( factors= f, maxiter= mi, lr= lr, lamb= lamb )
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




def boosting(iterator, param, folds):

	ratings_train, ratings_test = [], []
	ratings_train_path, ratings_test_path = 'TwitterRatings/funkSVD/ratings.train', 'TwitterRatings/funkSVD/ratings.test'
	with open(ratings_train_path, 'r') as f:
		for line in f:
			ratings_train.append( line.strip() )

	with open(ratings_test_path, 'r') as f:
		for line in f:
			ratings_test.append( line.strip() )


	for i in iterator:
		rmses = []
		maes  = []

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
		
		for _ in range(0, folds):

			ratingsSampler(ratings_train, 'TwitterRatings/funkSVD/ratings_temp.train', 0.8)
			svd = pyreclab.SVD( dataset   = 'TwitterRatings/funkSVD/ratings_temp.train',
													dlmchar   = b',',
													header    = False,
													usercol   = 0,
													itemcol   = 1,
													ratingcol = 2 )

			svd.train( factors= f, maxiter= mi, lr= lr, lamb= lamb )

			ratingsSampler(ratings_test, 'TwitterRatings/funkSVD/ratings_temp.test', 0.8)
			predlist, mae, rmse = svd.test( input_file  = 'TwitterRatings/funkSVD/ratings_temp.test',
			                                dlmchar     = b',',
			                                header      = False,
			                                usercol     = 0,
			                                itemcol     = 1,
			                                ratingcol   = 2,
																			output_file = 'TwitterRatings/funkSVD/predictions_'+str(f)+'.csv' )
			
			rmses.append(rmse)
			maes.append(mae)

			del svd
			del predlist

		# Escribe 1 archivo por cada valor de cada parámetro
		with open('TwitterRatings/funkSVD/params/'+param+'/'+str(i)+'.txt', 'w') as f:
			for j in range(0, folds):
				f.write( "%s\t%s\n" % (rmses[j], maes[j]) )

		del rmses
		del maes

		gc.collect()



def RMSEMAEdistr():
	path = "TwitterRatings/funkSVD/params/"
	datos = {}

	for param in os.listdir( path ):

		datos[param] = {}
		
		for value in os.listdir( path + param ):
			
			rmses = []
			maes  = []

			with open(path + param + '/' + value, 'r') as f:

				for line in f:
					rmse, mae = line.strip().split('\t')
					rmses.append( float(rmse) )
					maes.append( float(mae) )

			rmse_mean, rmse_stddev = mean( rmses ), stddev( rmses )
			mae_mean, mae_stddev = mean( maes ), stddev( maes )

			datos[param][value[:-4]] = [ [rmse_mean, rmse_stddev], [mae_mean, mae_stddev] ]

	with open(path+"resumen.txt", 'w') as f:
		for param in datos:
			f.write("%s\n" % param)
			for d in datos[param]:
				for value in d:
					f.write("%s\t%s,%s\t%s,%s\n" % (value, d[value][0][0], d[value][0][1], d[value][1][0], d[value][1][1]) )


factores = range(300, 1025, 25) # [300, 325, .., 1000]
max_iters = range(460, 520, 20) # [100, 120, .., 500]
lrn_rates = range(2, 21,1) # [2, 3, .., 20] / 200 = [0.01, 0.015, .., 0.1]
reg_params = range(2, 21, 1) # [2, 3, .., 20] / 20 = [0.1, 0.15, .., 1]

# boosting(iterator=factores, param="factors", folds=15)
boosting(iterator=max_iters, param="maxiter", folds=15)
boosting(iterator=lrn_rates, param="lr", folds=15)
boosting(iterator=reg_params, param="lamb", folds=15)
RMSEMAEdistr()