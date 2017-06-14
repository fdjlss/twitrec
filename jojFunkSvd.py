# coding=utf-8

#--------------------------------#
import time
import pyreclab

# Logging
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
#--------------------------------#

factores = [50, 100, 200, 500, 1000, 1500, 2000, 5000, 10000]
rmses = []
maes =  []

for f in factores:
	svd = pyreclab.SVD( dataset   = 'TwitterRatings/funkSVD/ratings.train',
											dlmchar   = b',',
											header    = False,
											usercol   = 0,
											itemcol   = 1,
											ratingcol = 2 )

	mi = 100
	lr = 0.01
	lamb = 0.1

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


with open('TwitterRatings/funkSVD/metrics.txt', 'w') as f:
	for i in range(0, len(factores)):
		f.write( "%s\t%s\t,%s\n" % (factores[i], rmses[i], maes[i] ) )
