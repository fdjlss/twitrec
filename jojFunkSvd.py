# coding=utf-8

import time
import pyreclab
from random import sample
import gc
from time import sleep
import os
from math import sqrt, log
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


#-----"PRIVATE" METHODS----------#
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


def rel_div(relevance_i, i):
	return (2.0**relevance_i - 1) / log(1 + i, 2)
def DCG(recs):
	s = 0.0
	for place, relevance in recs:
		s += rel_div(relevance, place)
	return s
def iDCG(recs):
	place = 0
	i_recs = {}
	for relevance in sorted( recs.values(), reverse=True ):
		place += 1
		i_recs[i] = relevance
	return DCG(i_recs)
def nDCG(recs):
	return DCG(recs) / iDCG(recs)

def P_at_N(n, recs, rel_thresh):
	s = 0.0
	for place, relevance in recs:
		while place <= n:
			if relevance >= rel_thresh:
				s += 1
	return s / n

def AP_at_N(n, recs, rel_thresh):
	s = 0.0
	relevants_count = 0
	for place, relevance in recs:

		if relevance >= rel_thresh:
			rel_k = 1
			relevants_count += 1
		else:
			rel_k = 0
		s += P_at_N(place, recs, rel_thresh) * rel_k

	try:
		return s / min(n, relevants_count) 
	except ZeroDivisionError as e:
		return 0.0
#--------------------------------#


def SVDJob(train_path, test_path, f, mi, lr, lamb):
	svd = pyreclab.SVD( dataset   = train_path,
											dlmchar   = b',',
											header    = False,
											usercol   = 0,
											itemcol   = 1,
											ratingcol = 2 )
	svd.train( factors= f, maxiter= mi, lr= lr, lamb= lamb )
	predlist, mae, rmse = svd.test( input_file  = test_path,
	                                dlmchar     = b',',
	                                header      = False,
	                                usercol     = 0,
	                                itemcol     = 1,
	                                ratingcol   = 2)
	return predlist, mae, rmse

def boosting(folds):

	ratings_train, ratings_test = [], []
	with open('TwitterRatings/funkSVD/ratings.train', 'r') as f:
		for line in f:
			ratings_train.append( line.strip() )

	with open('TwitterRatings/funkSVD/ratings.test', 'r') as f:
		for line in f:
			ratings_test.append( line.strip() )

	defaults = {'f': 1000, 'mi': 100, 'lr': 0.01, 'lamb': 0.1}
	results = {'f': {}, 'mi': {}, 'lr': {}, 'lamb': {}}
	train_path = 'TwitterRatings/funkSVD/ratings_temp.train'
	test_path = 'TwitterRatings/funkSVD/ratings_temp.test'

	for param in defaults:

		if param=='f':
			defaults['f'] = list(range(100, 1525, 25))

			for i in defaults['f']:

				rmses = []
				maes  = []
				for _ in range(0, folds):
					ratingsSampler(ratings_train, train_path, 0.8)
					ratingsSampler(ratings_test, test_path, 0.8)
					predlist, mae, rmse = SVDJob(train_path=train_path, test_path=test_path, f= i, mi= defaults['mi'], lr= defaults['lr'], lamb= defaults['lamb'])
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
					ratingsSampler(ratings_train, train_path, 0.8)
					ratingsSampler(ratings_test, test_path, 0.8)
					predlist, mae, rmse = SVDJob(train_path=train_path, test_path=test_path, f= defaults['f'], mi= defaults['mi'], lr= defaults['lr'], lamb= i)
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
					ratingsSampler(ratings_train, train_path, 0.8)
					ratingsSampler(ratings_test, test_path, 0.8)
					predlist, mae, rmse = SVDJob(train_path=train_path, test_path=test_path, f= defaults['f'], mi= defaults['mi'], lr= i, lamb= defaults['lamb'])
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
					ratingsSampler(ratings_train, train_path, 0.8)
					ratingsSampler(ratings_test, test_path, 0.8)
					predlist, mae, rmse = SVDJob(train_path=train_path, test_path=test_path, f= defaults['f'], mi= i, lr= defaults['lr'], lamb= defaults['lamb'])
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

			recommendationList = svd.testrec( input_file  = 'TwitterRatings/funkSVD/ratings.test', #o ratings_temp.test
		                                    dlmchar     = b',',
		                                    header      = False,
		                                    usercol     = 0,
		                                    itemcol     = 1,
		                                    ratingcol   = 2,
		                                    topn        = n,
		                                    includeRated= True)

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

def nDCGMAP_calculator(params, folds, topN):
	pass
	# for n in topN:

	# 	svd = pyreclab.SVD( dataset   = 'TwitterRatings/funkSVD/ratings.train',
	# 											dlmchar   = b',',
	# 											header    = False,
	# 											usercol   = 0,
	# 											itemcol   = 1,
	# 											ratingcol = 2 )
	# 	svd.train( factors= params['f'], maxiter= params['mi'], lr= params['lr'], lamb= params['lamb'] )
	# 	recommendationList = svd.testrec( input_file    = 'TwitterRatings/funkSVD/ratings.test',
	#                                       dlmchar     = b',',
	#                                       header      = False,
	#                                       usercol     = 0,
	#                                       itemcol     = 1,
	#                                       ratingcol   = 2,
	#                                       topn        = n,
	#                                       includeRated= True )
	# 	# quiero..
	# 	nDCGs = []
	# 	APs = []
	# 	for user in users:
	# 		nDCGs.append( nDCG(recs(user)) )
	# 		APs.append( AP_at_N(n, recs(user), 4) )






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





factores = range(300, 1025, 25) # [300, 325, .., 1000]
max_iters = range(100, 520, 20) # [100, 120, .., 500]
lrn_rates = range(2, 21, 1) # [2, 3, .., 20] / 200 = [0.01, 0.015, .., 0.1]
reg_params = range(2, 21, 1) # [2, 3, .., 20] / 20 = [0.1, 0.15, .., 1]

opt_params = boosting(folds=15)
RMSEMAE_distr()
PRF_calculator(params=opt_params, folds=5, topN=[10, 20, 50])
# generate_recommends(params=opt_params)