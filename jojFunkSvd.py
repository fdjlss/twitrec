# coding=utf-8

#--------------------------------#
import time
import pyreclab
from random import sample
import gc
from time import sleep

# Logging
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
#--------------------------------#


def ratingsSampler(rats, fin, fout, n):

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

	ratings = []
	with open(fin, 'r') as f:
		for line in f:
			ratings.append( line.strip() )


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

			ratingsSampler(ratings, 'TwitterRatings/funkSVD/ratings.train', 'TwitterRatings/funkSVD/ratings_temp.train', 0.8)
			svd = pyreclab.SVD( dataset   = 'TwitterRatings/funkSVD/ratings_temp.train',
													dlmchar   = b',',
													header    = False,
													usercol   = 0,
													itemcol   = 1,
													ratingcol = 2 )

			svd.train( factors= f, maxiter= mi, lr= lr, lamb= lamb )

			ratingsSampler(ratings, 'TwitterRatings/funkSVD/ratings.test', 'TwitterRatings/funkSVD/ratings_temp.test', 0.8)
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

			sleep(5)

		# Escribe 1 archivo por cada valor de cada parámetro
		with open('TwitterRatings/funkSVD/params/'+param+'/'+str(i)+'.txt', 'w') as f:
			for j in range(0, folds):
				f.write( "%s\t%s\n" % (rmses[j], maes[j]) )

		del rmses
		del maes

		gc.collect()



factores = range(500, 1025, 25) # [300, 325, .., 1000]
max_iters = range(100, 520, 20) # [100, 120, .., 500]
lrn_rates = range(2, 21,1) # [2, 3, .., 20] / 200 = [0.01, 0.015, .., 0.1]
reg_params = range(2, 21, 1) # [2, 3, .., 20] / 20 = [0.1, 0.15, .., 1]

boosting(iterator=factores, param="factors", folds=15)
boosting(iterator=max_iters, param="maxiter", folds=15)
boosting(iterator=lrn_rates, param="lr", folds=15)
boosting(iterator=reg_params, param="lamb", folds=15)
# SVDTesting()
