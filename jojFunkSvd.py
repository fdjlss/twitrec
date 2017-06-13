# coding=utf-8

#--------------------------------#
import time
import pyreclab

# Logging
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
#--------------------------------#

svd = pyreclab.SVD( dataset   = 'TwitterRatings/funkSVD/ratings.train',
										dlmchar   = b',',
										header    = False,
										usercol   = 0,
										itemcol   = 1,
										ratingcol = 2 )

f = 1000
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
																output_file = 'TwitterRatings/funkSVD/predictions_1000.csv' )
end = time.clock()
logging.info( "prediction time: " + str(end - start) )


logging.info( 'MAE: ' + str(mae) )
logging.info( 'RMSE: ' + str(rmse) )