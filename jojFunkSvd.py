# coding=utf-8

#--------------------------------#
import time
import pyreclab

# Logging
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
#--------------------------------#

factores = [50, 100, 150, 200, 250, 300, 400, 500, 1000, 1500, 2000, 5000, 10000]
max_iters = [10, 50, 100, 200, 300, 500, 1000]
lrn_rates = [0.0001, 0.001, 0.01, 0.1, 1]
reg_params = [0.001, 0.01, 0.1, 1]


def SVDJob(iterator=[], f=1000, mi=100, lr=0.01, lamb=0.1):

	rmses = []
	maes =  []

	for i in iterator:

		svd = pyreclab.SVD( dataset   = 'TwitterRatings/funkSVD/ratings.train',
												dlmchar   = b',',
												header    = False,
												usercol   = 0,
												itemcol   = 1,
												ratingcol = 2 )

		# Pésima decisión de diseño, sí sé, pero bueno..
		if f==0:
			f=i
			fn = 'metrics_f.txt'
		if mi==0:
			mi=i
			fn = 'metrics_mi.txt'
		if lr==0:
			lr=i
			fn = 'metrics_lr.txt'
		if lamb==0:
			lamb=i
			fn = 'metrics_lamb.txt'

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

	with open('TwitterRatings/funkSVD/'+fn, 'w') as f:
		for i in range(0, len(iterator)):
			f.write( "%s\t%s\t%s\n" % (iterator[i], rmses[i], maes[i] ) )



SVDJob(iterator=factores, f=0)
SVDJob(iterator=max_iters, mi=0)
SVDJob(iterator=lrn_rates, lr=0)
SVDJob(iterator=reg_params, lamb=0)

