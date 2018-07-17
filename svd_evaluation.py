# coding=utf-8

import time
import pyreclab
from random import sample
import gc
from time import sleep
import os
from os.path import isfile, join
from math import sqrt, log
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


#-----"PRIVATE" METHODS----------#
def remove_consumed(user_consumption, rec_list):
	l = rec_list
	for itemId in rec_list:
		if itemId in user_consumption: l.remove(itemId)
	return l
def mean(lst):
	return float(sum(lst)) / len(lst)

def stdev(lst):
  m = mean(lst)
  return sqrt(float(reduce(lambda x, y: x + y, map(lambda x: (x - m) ** 2, lst))) / len(lst))

def opt_value(results, metric):
	for val in results:
		if metric=='rmse':
			if results[val] == min( list(results.values()) ): 
				opt_value = val
		if metric=='ndcg':
			if results[val] == max( list(results.values()) ): 
				opt_value = val
	return opt_value

def ratingsSampler(rats, fout, n):
	l = len(rats)
	K = l*n
	ratings_sampled = sample(rats, k=int(K))
	with open(fout, 'w') as f:
		f.write( '\n'.join('%s' % x for x in ratings_sampled) )
	del ratings_sampled

def MRR(recs, rel_thresh):
	res = 0.0
	for place in recs:
		if int(recs[place]) >= rel_thresh:
			res = 1.0/int(place)
			break
	return res

def rel_div(relevance_i, i, alt_form):
	if alt_form:
		return ( 2.0**relevance_i - 1) / log(1.0 + i, 2)
	else:
		return relevance_i / log(1.0 + i, 2)

def DCG(recs, alt_form, rel_thresh):
	s = 0.0
	if rel_thresh==False:
		for place in recs:
			s += rel_div(recs[place], place, alt_form)
	else:
		for place in recs:
			if recs[place] >= rel_thresh:
				s += rel_div(1, place, alt_form)
			else:
				s += rel_div(0, place, alt_form)
	return s

def iDCG(recs, alt_form, rel_thresh):
	place = 0
	i_recs = {}
	for relevance in sorted( recs.values(), reverse=True ):
		place += 1
		i_recs[place] = relevance
	return DCG(i_recs, alt_form, rel_thresh)

def nDCG(recs, alt_form, rel_thresh):
	try:
		return DCG(recs, alt_form, rel_thresh) / iDCG(recs, alt_form, rel_thresh)
	except ZeroDivisionError as e:
		return 0.0

def P_at_N(n, recs, rel_thresh):
	s = 0.0
	for place in recs:
		if place <= n:
			if recs[place] >= rel_thresh:
				s += 1
		else:
			break
	return s / n

def AP_at_N(n, recs, rel_thresh):
	s = 0.0
	relevants_count = 0
	for place in recs:
		if recs[place] >= rel_thresh:
			rel_k = 1
			relevants_count += 1
		else:
			rel_k = 0
		s += P_at_N(place, recs, rel_thresh) * rel_k
	try:
		return s / min(n, relevants_count) 
	except ZeroDivisionError as e:
		return 0.0

def R_precision(n_relevants, recs):
	s = 0.0
	for place in recs:
		if recs[place] != 0:
			s += 1
	return s/n_relevants

def consumption(ratings_path, rel_thresh, with_ratings):
	c = {}
	with open(ratings_path, 'r') as f:
		if with_ratings:
			for line in f:
				userId, itemId, rating = line.strip().split(',')
				if userId not in c:
					c[userId] = {}
				if int( rating ) >= rel_thresh:
					c[userId][itemId] = rating
		else:
			for line in f:
				userId, itemId, rating = line.strip().split(',')
				if userId not in c:
					c[userId] = []
				if int( rating ) >= rel_thresh:
					c[userId].append(itemId)
	return c

def user_ranked_recs(user_recs, user_consumpt):
	recs = {}
	place = 1
	for itemId in user_recs:
		if itemId in user_consumpt:
			rating = int( user_consumpt[itemId] )
		else:
			rating = 0
		recs[place] = rating
		place += 1
	return recs

def relevance(user, q):
	ratings = [ int(r) for r in user.values() ]
	if q>=10:
		return mean(ratings)
	return ((0.5**q) * stdev(ratings)) + mean(ratings)

def SVDJob(data_path, params):
	# test = [] 
	# with open(data_path+'test/test.fold', 'r') as f:
	# 	for line in f:
	# 		test.append( line.strip() )
	val_folds   = os.listdir(data_path+'val/')
	maes, rmses = [], []
	"""HARDCODED AS FUCK: 4+1, N20""" #Rationale: N20 por ser folds más chicos. Así, ninguno de estos folds se interlapa con alguno de testing.
	for i in range(1, 4+1):
		svd = pyreclab.SVD( dataset   = data_path+'train/train_N20.'+str(i),
												dlmchar   = b',',
												header    = False,
												usercol   = 0,
												itemcol   = 1,
												ratingcol = 2 )
		svd.train( factors= params['f'], maxiter= params['mi'], lr= params['lr'], lamb= params['lamb'] )
		predlist, mae, rmse = svd.test( input_file  = data_path+'val/val_N20.'+str(i),
		                                dlmchar     = b',',
		                                header      = False,
		                                usercol     = 0,
		                                itemcol     = 1,
		                                ratingcol   = 2 )
		maes.append(mae)
		rmses.append(rmse)
	return mean(maes), mean(rmses)
#--------------------------------#

def svd_tuning(data_path):

	defaults = {'f': 1000, 'mi': 100, 'lr': 0.01, 'lamb': 0.1}
	results  = {'f': {}, 'mi': {}, 'lr': {}, 'lamb': {}}

	for param in ['f', 'lamb', 'lr', 'mi']:

		if param=='f':
			for i in range(100, 1525, 25):
				defaults['f'] = i
				logging.info("Entrenando con f={f}, lamb={lamb}, lr={lr}, mi={mi}".format(f=defaults['f'], lamb=defaults['lamb'], lr=defaults['lr'], mi=defaults['mi']) )
				mae, rmse = SVDJob(data_path= data_path, params= defaults)
				results['f'][i] = rmse
			defaults['f']  = opt_value(results=results['f'], metric='rmse')

		elif param=='mi':
			for i in range(10, 520, 20):
				defaults['mi'] = i
				logging.info("Entrenando con f={f}, lamb={lamb}, lr={lr}, mi={mi}".format(f=defaults['f'], lamb=defaults['lamb'], lr=defaults['lr'], mi=defaults['mi']) )
				mae, rmse = SVDJob(data_path= data_path, params= defaults)
				results['mi'][i] = rmse
			defaults['mi'] = opt_value(results=results['mi'], metric='rmse')

		elif param=='lamb':
			for i in [0.001, 0.005, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
				defaults['lamb'] = i		
				logging.info("Entrenando con f={f}, lamb={lamb}, lr={lr}, mi={mi}".format(f=defaults['f'], lamb=defaults['lamb'], lr=defaults['lr'], mi=defaults['mi']) )
				mae, rmse = SVDJob(data_path= data_path, params= defaults)
				results['lamb'][i] = rmse
			defaults['lamb'] = opt_value(results=results['lamb'], metric='rmse')

		elif param=='lr':
			for i in [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]:
				defaults['lr'] = i
				logging.info("Entrenando con f={f}, lamb={lamb}, lr={lr}, mi={mi}".format(f=defaults['f'], lamb=defaults['lamb'], lr=defaults['lr'], mi=defaults['mi']) )
				mae, rmse = SVDJob(data_path= data_path, params= defaults)
				results['lr'][i] = rmse
			defaults['lr'] = opt_value(results=results['lr'], metric='rmse')

	# Real testing
	svd = pyreclab.SVD( dataset   = data_path+'eval_train_N20.data',
											dlmchar   = b',',
											header    = False,
											usercol   = 0,
											itemcol   = 1,
											ratingcol = 2 )
	svd.train( factors= defaults['f'], maxiter= defaults['mi'], lr= defaults['lr'], lamb= defaults['lamb'] )
	predlist, mae, rmse = svd.test( input_file  = data_path+'test/test_N20.data',
	                                dlmchar     = b',',
	                                header      = False,
	                                usercol     = 0,
	                                itemcol     = 1,
	                                ratingcol   = 2 )

	with open('TwitterRatings/funkSVD/opt_params_ptcdssplit.txt', 'w') as f:
		for param in defaults:
			f.write( "{param}:{value}\n".format(param=param, value=defaults[param]) )
		f.write( "RMSE:{rmse}, MAE:{mae}".format(rmse=rmse, mae=mae) )

	return defaults

def RMSEMAE_distr(output_filename):
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

			rmse_mean, rmse_stdev = mean( rmses ), stdev( rmses )
			mae_mean, mae_stdev = mean( maes ), stdev( maes )

			datos[param][value[:-4]] = [ [rmse_mean, rmse_stdev], [mae_mean, mae_stdev] ]

	with open("TwitterRatings/funkSVD/"+output_filename, 'w') as f:
		for param in datos:
			f.write("%s\n" % param)
			for v in sorted(datos[param].items()):
				#<value>  <RMSE_mean>,<RMSE_stdev>  <MAE_mean>,<MAE_stdev>
				f.write("%s\t%s,%s\t%s,%s\n" % ( v[0], v[1][0][0], v[1][0][1], v[1][1][0], v[1][1][1] ) )

def PRF_calculator(params, folds, topN):

	ratings_train, ratings_test = [], []
	with open('TwitterRatings/funkSVD/ratings.train', 'r') as f:
		for line in f:
			ratings_train.append( line.strip() )

	with open('TwitterRatings/funkSVD/ratings.test', 'r') as f:
		for line in f:
			ratings_test.append( line.strip() )

	preferred_consumption = consumption(ratings_path='TwitterRatings/funkSVD/ratings.test', rel_thresh=4, with_ratings=False)

	for n in topN: 
		precision_folds, recall_folds = [], []
	# for _ in range(0, folds):
		# ratingsSampler(ratings_train, 'TwitterRatings/funkSVD/ratings_temp.train', 0.8)
		# ratingsSampler(ratings_test, 'TwitterRatings/funkSVD/ratings_temp.test', 0.8)

		svd = pyreclab.SVD( dataset   = 'TwitterRatings/funkSVD/ratings.train', #o ratings_temp.train
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
	                                    includeRated= False)

		users_precisions, users_recalls = [], []
		for userId in recommendationList[0]:
			recs = set(recommendationList[0][userId])
			cons = set(preferred_consumption[userId])
			tp = len(recs & cons)
			fp = len(recs - cons)
			fn = len(cons - recs)
			users_precisions.append( float(tp) / (tp + fp) )
			try:
				users_recalls.append( float(tp) / (tp + fn) )
			except ZeroDivisionError as e:
				continue
				
		precision_folds.append( mean(users_precisions) )
		recall_folds.append( mean(users_recalls) )

		p = mean( precision_folds )
		r = mean( recall_folds )
		f = 2*p*r / (p + r)

		with open('TwitterRatings/funkSVD/recall.txt', 'a') as file:
			file.write( "N=%s, P=%s, R=%s, F=%s\n" % (n, p, r, f) )

def nDCGMAP_calculator(data_path, params, topN, output_filename):

	user_consumption = consumption(ratings_path=data_path+'ratings.total', rel_thresh=0, with_ratings=True)
	svd = pyreclab.SVD( dataset   = data_path+'ratings.train',#data_path+'train/train.'+str(i),
											dlmchar   = b',',
											header    = False,
											usercol   = 0,
											itemcol   = 1,
											ratingcol = 2 )
	svd.train( factors= params['f'], maxiter= params['mi'], lr= params['lr'], lamb= params['lamb'] )
	recommendationList = svd.testrec( input_file    = data_path+'test/'+os.listdir(data_path+'test/')[0],#data_path+'val/val.'+str(i),
                                      dlmchar     = b',',
                                      header      = False,
                                      usercol     = 0,
                                      itemcol     = 1,
                                      ratingcol   = 2,
                                      topn        = 100,
                                      includeRated= False )
	MRR_thresh4   = []
	MRR_thresh3   = []
	nDCGs_bin_thresh4 = dict((n, []) for n in topN)
	nDCGs_bin_thresh3 = dict((n, []) for n in topN)
	nDCGs_normal  = dict((n, []) for n in topN)
	nDCGs_altform = dict((n, []) for n in topN)
	APs_thresh4   = dict((n, []) for n in topN)
	APs_thresh3   = dict((n, []) for n in topN)
	APs_thresh2   = dict((n, []) for n in topN)

	for userId in recommendationList[0]:
		recs = user_ranked_recs(user_recs=recommendationList[0][userId], user_consumpt=user_consumption[userId])

		MRR_thresh4.append( MRR(recs=recs, rel_thresh=4) )
		MRR_thresh3.append( MRR(recs=recs, rel_thresh=3) )
		for n in topN:
			mini_recs = dict((k, recs[k]) for k in recs.keys()[:n])
			nDCGs_bin_thresh4[n].append( nDCG(recs=mini_recs, alt_form=False, rel_thresh=4) )
			nDCGs_bin_thresh3[n].append( nDCG(recs=mini_recs, alt_form=False, rel_thresh=3) )
			nDCGs_normal[n].append( nDCG(recs=mini_recs, alt_form=False, rel_thresh=False) )
			nDCGs_altform[n].append( nDCG(recs=mini_recs, alt_form=True, rel_thresh=False) )			
			APs_thresh4[n].append( AP_at_N(n=n, recs=recs, rel_thresh=4) )
			APs_thresh3[n].append( AP_at_N(n=n, recs=recs, rel_thresh=3) )
			APs_thresh2[n].append( AP_at_N(n=n, recs=recs, rel_thresh=2) )


	with open('TwitterRatings/funkSVD/'+output_filename, 'a') as file:
		for n in topN:
			file.write( "N=%s, normal nDCG=%s, alternative nDCG=%s, bin nDCG(rel_thresh=4)=%s, bin nDCG(rel_thresh=3)=%s, MAP(rel_thresh=4)=%s, MAP(rel_thresh=3)=%s, MAP(rel_thresh=2)=%s, MRR(rel_thresh=4)=%s, MRR(rel_thresh=3)=%s\n" % \
				(n, mean(nDCGs_normal[n]), mean(nDCGs_altform[n]), mean(nDCGs_bin_thresh4[n]), mean(nDCGs_bin_thresh3[n]), mean(APs_thresh4[n]), mean(APs_thresh3[n]), mean(APs_thresh2[n]), mean(MRR_thresh4), mean(MRR_thresh3)) )	


def svd_protocol_evaluation(data_path, params):
	test_c  = consumption(ratings_path=data_path+'test/test_N20.data', rel_thresh=0, with_ratings=True) #debiera ser el test_c, pero como includeRated=False, da lo mismo
	train_c = consumption(ratings_path=data_path+'eval_train_N20.data', rel_thresh=0, with_ratings=False)
	svd = pyreclab.SVD( dataset   = data_path+'eval_train_N20.data',
											dlmchar   = b',',
											header    = False,
											usercol   = 0,
											itemcol   = 1,
											ratingcol = 2 )
	svd.train( factors= params['f'], maxiter= params['mi'], lr= params['lr'], lamb= params['lamb'] )
	recommendationList = svd.testrec( input_file    = data_path+'test/test_N20.data',
                                      dlmchar     = b',',
                                      header      = False,
                                      usercol     = 0,
                                      itemcol     = 1,
                                      ratingcol   = 2,
                                      topn        = 100,
                                      includeRated= False )

	MRRs   = dict((N, []) for N in [5, 10, 15, 20])
	nDCGs  = dict((N, []) for N in [5, 10, 15, 20])
	APs    = dict((N, []) for N in [5, 10, 15, 20])
	Rprecs = dict((N, []) for N in [5, 10, 15, 20])

	for userId in recommendationList[0]:
		book_recs  = remove_consumed(user_consumption= train_c[userId], rec_list= recommendationList[0][userId]) #da lo mismo este paso, según Gabriel el testrec no devuelve items consumidos
		recs       = user_ranked_recs(user_recs=book_recs, user_consumpt=test_c[userId])

		for N in [5, 10, 15, 20]:
			mini_recs = dict((k, recs[k]) for k in recs.keys()[:N])
			MRRs[N].append( MRR(recs=mini_recs, rel_thresh=1) )
			nDCGs[N].append( nDCG(recs=mini_recs, alt_form=False, rel_thresh=False) )		
			APs[N].append( AP_at_N(n=N, recs=recs, rel_thresh=1) )
			Rprecs[N].append( R_precision(n_relevants=N, recs=mini_recs) )

	for N in [5, 10, 15, 20]:
		with open('TwitterRatings/funkSVD/protocol.txt', 'a') as file:
			file.write( "N=%s, nDCG=%s, MAP=%s, MRR=%s, R-precision=%s\n" % \
				(N, mean(nDCGs[N]), mean(APs[N]), mean(MRRs[N]), mean(Rprecs[N])) )	


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



def main():
	data_path = 'TwitterRatings/funkSVD/data/'
	# opt_params = svd_tuning(data_path= data_path)
	# RMSEMAE_distr(output_filename="results_8020.txt")
	opt_params = {'f': 675, 'mi': 110, 'lr': 0.009, 'lamb': 0.05}
	# PRF_calculator(params=opt_params, folds=5, topN=[10, 20, 50])
	# nDCGMAP_calculator(data_path= data_path, params=opt_params, topN=[10, 15, 20, 50], output_filename="nDCGMAP.txt")
	# for N in [5, 10, 15, 20]:
	svd_protocol_evaluation(data_path=data_path, params=opt_params)
	# generate_recommends(params=opt_params)


if __name__ == '__main__':
	main()