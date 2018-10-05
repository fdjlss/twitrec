# coding=utf-8

import os
import json
from urllib import urlencode, quote_plus
from urllib2 import urlopen

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix
import numpy as np
import pandas as pd
from math import sqrt
import random
from pyfm import pylibfm

from svd_evaluation import mean, stdev, MRR, rel_div, DCG, iDCG, nDCG, P_at_N, AP_at_N, R_precision, consumption, user_ranked_recs, opt_value
from solr_evaluation import remove_consumed, flatten_list
from pyFM_evaluation import loadData
from implicit_evaluation import IdCoder
import implicit

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

###################################
def hybrid_recs_naive(ALPHA, book_recs_cb, book_recs_cf):
	methods = {'CB': int(ALPHA*100), 'CF': int((1-ALPHA)*100)}
	book_recs_mix = dict((i, '') for i in range(40))
	book_recs_cb = book_recs_cb[:20]
	book_recs_cf = book_recs_cf[:20]
	nice_items = {}

	# Priorizo poner los ítems recomendados tanto en R1 como en R2 y que están en el mismo puesto
	for idx in range(len(book_recs_cb)): #cb o cf, sólo buscamos ítems en el mismo puesto, además que tienen mismo largo
		if book_recs_cb[idx] == book_recs_cf[idx]:
			book_recs_mix[idx] = book_recs_cb[idx]
			nice_items[idx] = book_recs_cb[idx] #cb o cf, es el mismo ítem

	# Meter ítems en el mix recomendados en ambas listas que no estén en el mismo puesto
	for idx_cb in range(len(book_recs_cb)):
		if idx_cb in nice_items: continue #continúo si es un ítem que ya vi
		for idx_cf in range(len(book_recs_cf)):
			if idx_cf in nice_items: continue #continúo si es un ítem que ya vi
			if book_recs_cb[idx_cb] == book_recs_cf[idx_cf]:
				method = random.choice([x for x in methods for y in range(methods[x])]) #escojo random entre sistemas, con preferencia ALPHA de uno sobre otro	
				if method == 'CB':
					if book_recs_mix[idx_cb] == '': #si el puesto está libre, ponerlo ahí
						book_recs_mix[idx_cb] = book_recs_cb[idx_cb]
					else: #..de lo contrario ponerlo en el puesto del otro RS
						book_recs_mix[idx_cf] = book_recs_cf[idx_cf] 
				elif method == 'CF':
					if book_recs_mix[idx_cf] == '':
						book_recs_mix[idx_cf] = book_recs_cf[idx_cf] 
					else:
						book_recs_mix[idx_cb] = book_recs_cb[idx_cb]
				
	# Remover ítems de las listas originales que ya metí en el mix
	for place in book_recs_mix:
		if book_recs_mix[place] != '':
			book_recs_cb.remove(book_recs_mix[place])
			book_recs_cf.remove(book_recs_mix[place])

	# Meter ítems en el mix recomendados en una de las dos listas, dependiendo de la probabilidad ALPHA
	for idx in range(len(book_recs_cb)): #cb o cf, debieran tener el mismo largo
		for place in range(len(book_recs_mix)): #necsito buscar ordenadamente de arriba hacia abajo puestos vacíos
			if book_recs_mix[place] != '': continue #busca un puesto donde no haya metido ningún ítem hasta ahora y ponlo ahí
			method = random.choice([x for x in methods for y in range(methods[x])])
			if method == 'CB':
				book_recs_mix[place] = book_recs_cb[idx]
			elif method == 'CF':
				book_recs_mix[place] = book_recs_cf[idx]
			break

	return book_recs_mix

def hybrid_recs(recs_cb, recs_cf, weight_cb, weight_cf):
	concat = recs_cb + recs_cf
	all_items = list( set(recs_cb + recs_cf) )
	scores = dict((itemId, 0) for itemId in all_items )
	for itemId in scores:
		score_cb = 0
		score_cf = 0
		if itemId in recs_cb: score_cb = weight_cb / (recs_cb.index(itemId) + 1) #pq index parten desde 0
		if itemId in recs_cf: score_cf = weight_cf / (recs_cf.index(itemId) + 1)
		occurs = concat.count(itemId)
		item_score = (score_cb + score_cf) * occurs
		scores[itemId] = item_score
	return sorted(scores, key=scores.get, reverse=True)

def cf_models(cf_lib, N, data_path, params):
	#caching, ya que son siempre los mismos params del CF
	cf_models = {}
	
	all_data, y_all, items = loadData("eval_all_N"+str(N)+".data", data_path='TwitterRatings/funkSVD/data_with_authors/', with_timestamps=False, with_authors=True) #hardcoded bc fuck it
	if cf_lib == "pyFM":
		v = DictVectorizer()
		X_all = v.fit_transform(all_data)
		for i in range(1, 4+1):
			train_data, y_tr, _ = loadData('train/train_N'+str(N)+'.'+str(i), data_path=data_path, with_timestamps=False, with_authors=True)
			X_tr = v.transform(train_data)
			fm = pylibfm.FM(num_factors=params['f'], num_iter=params['mi'], k0=params['bias'], k1=params['oneway'], init_stdev=params['init_stdev'], \
											validation_size=params['val_size'], learning_rate_schedule=params['lr_s'], initial_learning_rate=params['lr'], \
											power_t=params['invscale_pow'], t0=params['optimal_denom'], shuffle_training=params['shuffle'], seed=params['seed'], \
											task='regression', verbose=True)
			fm.fit(X_tr, y_tr)
			cf_models[i] = fm 

		return cf_models, v, items

	elif cf_lib == "implicit":
		all_c     = consumption(ratings_path= data_path+'eval_all_N'+str(N)+'.data', rel_thresh= 0, with_ratings= True)
		items_ids = list(set( [ itemId for userId, itemsDict in all_c.items() for itemId in itemsDict ] ))
		idcoder   = IdCoder(items_ids, all_c.keys())
		for i in range(1, 4+1):
			ones, row, col = get_data(data_path= data_path, all_c= all_c, idcoder= idcoder, fold= i, N= N, mode= "tuning")
			matrix         = csr_matrix((ones, (row, col)), dtype=np.float64 )
			user_items     = matrix.T.tocsr()
			model          = implicit.als.AlternatingLeastSquares(factors= params['f'], regularization= params['lamb'], iterations= params['mi'], dtype= np.float64)
			model.fit(matrix)
			cf_models[i] = model

		return cf_models, idcoder, items


def hybridJob(data_path, solr, cf_models, cf_lib, transformer, items, params_cb, params_hy, N):
	nDCGs = []
	for i in range(1, 4+1):
		logging.info("fold {}".format(i))
		users_nDCGs = []
		# CB
		train_c = consumption(ratings_path=data_path+'train/train_N'+str(N)+'.'+str(i), rel_thresh=0, with_ratings=False)
		val_c   = consumption(ratings_path=data_path+'val/val_N'+str(N)+'.'+str(i), rel_thresh=0, with_ratings=True)

		# CF
		model = cf_models[i]

		for userId in val_c: #en solr_evaluation aparece "for userId in train_c", pero debiera ser lo mismo, ya que val_c y train_c debieran tener los mismos users
			logging.info("user {}".format(userId))
			if cf_lib=="pyFM":
				# CF
				user_rows = [ {'user_id': str(userId), 'item_id': str(itemId)} for itemId in items ]
				X_va      = transformer.transform(user_rows)
				preds     = model.predict(X_va)
				recs_cf   = [itemId for _, itemId in sorted(zip(preds, items), reverse=True)]
			elif cf_lib=="implicit":
				recommends = model.recommend(userid= int(transformer.coder('user', userId)), user_items= user_items, N= 200)
				recs_cf  = [ transformer.decoder('item', tupl[0]) for tupl in recommends ]
			recs_cf = remove_consumed(user_consumption= train_c[userId], rec_list= recs_cf)
			recs_cf = recs_cf[:200]

			# CB
			recs_cb = []
			for itemId in train_c[userId]:
				encoded_params = urlencode(params_cb)
				url            = solr + '/mlt?q=goodreadsId:'+ itemId + "&" + encoded_params
				response       = json.loads( urlopen(url).read().decode('utf8') )
				try:
					docs         = response['response']['docs']
				except TypeError as e:
					continue
				recs_cb.append( [ str(doc['goodreadsId'][0]) for doc in docs ] )

			recs_cb = flatten_list(list_of_lists=recs_cb, rows=params_cb['rows'])
			recs_cb = remove_consumed(user_consumption= train_c[userId], rec_list= recs_cb)
			recs_cb = recs_cb[:200]

			# HYBRID
			recs_hy = hybrid_recs(recs_cb=recs_cb, recs_cf=recs_cf, weight_cb=params_hy['weight_cb'], weight_cf=params_hy['weight_cf'])
			recs_hy = remove_consumed(user_consumption= train_c[userId], rec_list= recs_hy)
			recs_hy = user_ranked_recs(user_recs= recs_hy, user_consumpt= val_c[userId])	
			mini_recs = dict((k, recs_hy[k]) for k in recs_hy.keys()[:N])
			users_nDCGs.append( nDCG(recs=mini_recs, alt_form=False, rel_thresh=False) )
		
		nDCGs.append( mean(users_nDCGs) )
	return mean(nDCGs)
###################################

def hybrid_tuning(data_path, cf_lib, solr, params_cb, params_cf, N):

	cf_mods, transformer, items = cf_models(cf_lib=cf_lib, N=N, data_path=data_path, params=params_cf) #transformer: "pyFM"->vectorizer, "implicit"->idcoder

	defaults = {'weight_cb': 0.5, 'weight_cf': 0.5}
	results = dict((param, {}) for param in defaults.keys())
	for param in ['weight_cb', 'weight_cf']:

		if param=='weight_cb':
			for i in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]: 
				logging.info("param:{}, i={}".format(param, i))
				defaults[param] = i
				results[param][i] = hybridJob(data_path= data_path, solr=solr, cf_models=cf_mods, transformer=transformer, items=items, params_cb=params_cb, params_hy=defaults, N=N)
			defaults[param] = opt_value(results= results[param], metric= 'ndcg')

		elif param=='weight_cf':
			for i in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]: 
				logging.info("param:{}, i={}".format(param, i))
				defaults[param] = i
				results[param][i] = hybridJob(data_path= data_path, solr=solr, cf_models=cf_mods, transformer=transformer, items=items, params_cb=params_cb, params_hy=defaults, N=N)
			defaults[param] = opt_value(results= results[param], metric= 'ndcg')

	with open('TwitterRatings/hybrid/opt_params_CB-CF'+str(cf_lib)+'.txt', 'w') as f:
		for param in defaults:
			f.write( "{param}:{value}\n".format(param=param, value=defaults[param]) )

	with open('TwitterRatings/hybrid/params_ndcgs_CB-CF'+str(cf_lib)+'.txt', 'w') as f:
		for param in defaults:
			for value in results[param]:
				f.write( "{param}={value}\t : {nDCG}\n".format(param=param, value=value, nDCG=results[param][value]) )

	defaultsCBCF=defaults

	# Es muy tonto hacerlo así, pero p...
	defaults = {'weight_cb': 0.5, 'weight_cf': 0.5}
	results = dict((param, {}) for param in defaults.keys())
	for param in ['weight_cf', 'weight_cb']:

		if param=='weight_cb':
			for i in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]: 
				logging.info("param:{}, i={}".format(param, i))
				defaults[param] = i
				results[param][i] = hybridJob(data_path= data_path, solr=solr, cf_models=cf_mods, transformer=transformer, items=items, params_cb=params_cb, params_hy=defaults, N=N)
			defaults[param] = opt_value(results= results[param], metric= 'ndcg')

		elif param=='weight_cf':
			for i in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]: 
				logging.info("param:{}, i={}".format(param, i))
				defaults[param] = i
				results[param][i] = hybridJob(data_path= data_path, solr=solr, cf_models=cf_mods, transformer=transformer, items=items, params_cb=params_cb, params_hy=defaults, N=N)
			defaults[param] = opt_value(results= results[param], metric= 'ndcg')

	with open('TwitterRatings/hybrid/opt_params_CF'+str(cf_lib)+'-CB.txt', 'w') as f:
		for param in defaults:
			f.write( "{param}:{value}\n".format(param=param, value=defaults[param]) )

	with open('TwitterRatings/hybrid/params_ndcgs_CF'+str(cf_lib)+'-CB.txt', 'w') as f:
		for param in defaults:
			for value in results[param]:
				f.write( "{param}={value}\t : {nDCG}\n".format(param=param, value=value, nDCG=results[param][value]) )

	defaultsCFCB=defaults

	return defaultsCBCF, defaultsCFCB


def hybrid_protocol_evaluation(data_path, data_path_context, cf_lib, solr, params_cb, params_cf, params_hy, N):
	test_c  = consumption(ratings_path=data_path+'test/test_N20.data', rel_thresh=0, with_ratings=True)
	train_c = consumption(ratings_path=data_path+'eval_train_N20.data', rel_thresh=0, with_ratings=False)
	all_c   = consumption(ratings_path= data_path+'eval_all_N20.data', rel_thresh= 0, with_ratings= True)
	MRRs   = dict((N, []) for N in [5, 10, 15, 20])
	nDCGs  = dict((N, []) for N in [5, 10, 15, 20])
	APs    = dict((N, []) for N in [5, 10, 15, 20])
	Rprecs = dict((N, []) for N in [5, 10, 15, 20])

	if cf_lib=="pyFM":
		all_data, y_all, items = loadData("eval_all_N20.data", data_path=data_path_context, with_timestamps=False, with_authors=True)
		v = DictVectorizer()
		X_all = v.fit_transform(all_data)

		train_data, y_tr, _ = loadData('eval_train_N20.data', data_path=data_path_context, with_timestamps=False, with_authors=True)
		X_tr = v.transform(train_data)
		fm   = pylibfm.FM(num_factors=params_cf['f'], num_iter=params_cf['mi'], k0=params_cf['bias'], k1=params_cf['oneway'], init_stdev=params_cf['init_stdev'], \
										validation_size=params_cf['val_size'], learning_rate_schedule=params_cf['lr_s'], initial_learning_rate=params_cf['lr'], \
										power_t=params_cf['invscale_pow'], t0=params_cf['optimal_denom'], shuffle_training=params_cf['shuffle'], seed=params_cf['seed'], \
										task='regression', verbose=True)
		fm.fit(X_tr, y_tr)

	elif cf_lib=="implicit":
		items_ids = list(set( [ itemId for userId, itemsDict in all_c.items() for itemId in itemsDict ] ))
		idcoder   = IdCoder(items_ids, all_c.keys())
		ones, row, col = get_data(data_path= data_path, all_c= all_c, idcoder= idcoder, fold= 0, N= 20, mode= "testing")
		matrix         = csr_matrix((ones, (row, col)), dtype=np.float64 )
		user_items     = matrix.T.tocsr()
		model          = implicit.als.AlternatingLeastSquares(factors= params_cf['f'], regularization= params_cf['lamb'], iterations= params_cf['mi'], dtype= np.float64)
		model.fit(matrix)


	p=0
	for userId in test_c:
		logging.info("#u: {0}/{1}".format(p, len(test_c)))
		p+=1
		if cf_lib=="pyFM":
			user_rows = [ {'user_id': str(userId), 'item_id': str(itemId)} for itemId in items ]
			X_te      = v.transform(user_rows)
			preds     = fm.predict(X_te)
			recs_cf = [itemId for _, itemId in sorted(zip(preds, items), reverse=True)]
		elif cf_lib=="implicit":
			recommends = model.recommend(userid= int(idcoder.coder('user', userId)), user_items= user_items, N= 200)
			recs_cf = [ idcoder.decoder('item', tupl[0]) for tupl in recommends ]

		recs_cb = []
		for itemId in train_c[userId]:
			encoded_params = urlencode(params_cb)
			url            = solr + '/mlt?q=goodreadsId:'+ itemId + "&" + encoded_params
			response       = json.loads( urlopen(url).read().decode('utf8') )
			try:
				docs         = response['response']['docs']
			except TypeError as e:
				continue
			recs_cb.append( [ str(doc['goodreadsId'][0]) for doc in docs ] )

		recs_cb = flatten_list(list_of_lists=recs_cb, rows=params_cb['rows'])

		recs_cf = remove_consumed(user_consumption= train_c[userId], rec_list= recs_cf)
		recs_cf = recs_cf[:200]
		recs_cb = remove_consumed(user_consumption= train_c[userId], rec_list= recs_cb)
		recs_cb = recs_cb[:200]

		recs_hy = hybrid_recs(recs_cb=recs_cb, recs_cf=recs_cf, weight_cb=params_hy['weight_cb'], weight_cf=params_hy['weight_cf'])
		recs_hy = remove_consumed(user_consumption= train_c[userId], rec_list= recs_hy)
		recs_hy = user_ranked_recs(user_recs= recs_hy, user_consumpt= test_c[userId])	

		for N in [5, 10, 15, 20]:
			mini_recs = dict((k, recs_hy[k]) for k in recs_hy.keys()[:N])
			MRRs[N].append( MRR(recs=mini_recs, rel_thresh=1) )
			nDCGs[N].append( nDCG(recs=mini_recs, alt_form=False, rel_thresh=False) )		
			APs[N].append( AP_at_N(n=N, recs=recs_hy, rel_thresh=1) )
			Rprecs[N].append( R_precision(n_relevants=N, recs=mini_recs) )

	for N in [5, 10, 15, 20]:
		with open('TwitterRatings/hybrid/protocol.txt', 'a') as file:
			file.write( "N=%s, nDCG=%s, MAP=%s, MRR=%s, R-precision=%s\n" % \
				(N, mean(nDCGs[N]), mean(APs[N]), mean(MRRs[N]), mean(Rprecs[N])) )	



def main():
	data_path_context = 'TwitterRatings/funkSVD/data_with_authors/'
	data_path = 'TwitterRatings/funkSVD/data/'
	solr = "http://localhost:8983/solr/grrecsys"
	#solr modo 1
	params_cb = {'echoParams' : 'none',
							'fl' : 'goodreadsId,description,title.titleOfficial,genres.genreName,author.authors.authorName,quotes.quoteText,author.authorBio,title.titleGreytext',
							'rows' : 100,
							'mlt.fl' : 'description,title.titleOfficial,genres.genreName,author.authors.authorName,quotes.quoteText',
							'mlt.boost' : 'false', #def: false
							'mlt.mintf' : 1, #def: 2
							'mlt.mindf' : 2, #def: 5
							'mlt.minwl' : 1, #def: 0
							'mlt.maxdf' : 25431, #docs*0.5, # en realidad no especificado
							'mlt.maxwl' : 8, #def: 0
							'mlt.maxqt' : 90, #def: 25
							'mlt.maxntp' : 150000} #def: 5000}
	#pyFM w/ authors
	params_pyfm = {  'lr_s':'invscaling',
							'val_size':0.01,
							'shuffle':True,
							'bias':True,
							'invscale_pow':0.05,
							'f':20,
							'mi':10,
							'seed':20,
							'lr':0.001,
							'oneway':True,
							'optimal_denom':0.01,
							'init_stdev':0.0001}
	params_imp = {'f': 20, 'lamb': 0.3, 'mi': 15}


	opt_params_cbcf, opt_params_cfcb = hybrid_tuning(data_path=data_path, cf_lib='implicit', solr=solr, params_cb=params_cb, params_cf=params_imp, N=20) #"pyFM"->data_path_context, "implicit"->data_path

	hybrid_protocol_evaluation(data_path=data_path, data_path_context=data_path_context, cf_lib='implicit', solr=solr, params_cb=params_cb, params_cf=params_imp, params_hy=opt_params_cbcf, N=20)
	hybrid_protocol_evaluation(data_path=data_path, data_path_context=data_path_context, cf_lib='implicit', solr=solr, params_cb=params_cb, params_cf=params_imp, params_hy=opt_params_cfcb, N=20)

if __name__ == '__main__':
	main()