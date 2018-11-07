# coding=utf-8

import os
import re, json
from urllib import urlencode, quote_plus
from urllib2 import urlopen
from svd_evaluation import mean, stdev, MRR, rel_div, DCG, iDCG, nDCG, P_at_N, AP_at_N, R_precision, consumption, user_ranked_recs, opt_value
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


#-----"PRIVATE" METHODS----------#
def recs_cleaner(solr, consumpt, recs):
	# Ve los canonical hrefs de los items consumidos
	consumpt_hrefs = []
	for itemId in consumpt:
		url      = solr + '/select?q=goodreadsId:' + itemId + '&wt=json' 
		response = json.loads( urlopen(url).read().decode('utf8') )
		try:
			doc      = response['response']['docs'][0]
		except:
			logging.info(itemId)
		consumpt_hrefs.append( doc['href'][0] )

	# Saca todos los items cuyos hrefs ya los tenga el usuario
	for item in reversed(recs):
		url      = solr + '/select?q=goodreadsId:' + item + '&wt=json' 
		response = json.loads( urlopen(url).read().decode('utf8') )
		doc      = response['response']['docs'][0]
		rec_href = doc['href'][0]
		if rec_href in consumpt_hrefs: recs.remove(item)

	# Saca todos los ítems con hrefs iguales
	lista_dict = {}
	for item in recs:
		url      = solr + '/select?q=goodreadsId:' + item + '&wt=json' 
		response = json.loads( urlopen(url).read().decode('utf8') )
		doc      = response['response']['docs'][0]
		rec_href = doc['href'][0]		
		if rec_href not in lista_dict:
			lista_dict[rec_href] = []
			lista_dict[rec_href].append( item )
		else:
			lista_dict[rec_href].append( item )
		
	clean_recs = recs
	rep_hrefs = []
	for href in lista_dict: lista_dict[href] = lista_dict[href][:-1]
	for href in lista_dict: rep_hrefs += lista_dict[href]

	for rep_href in rep_hrefs: clean_recs.remove(rep_href)

	return clean_recs

def encoded_itemIds(item_list):
	ids_string = '('
	for itemId in item_list: ids_string += itemId + '%2520OR%2520'
	ids_string = ids_string[:-12] # para borrar el último "%2520OR%2520"
	ids_string += ')'
	return ids_string
def flatten_list(list_of_lists, rows):
	"""Eliminamos duplicados manteniendo orden"""
	flattened = []
	for i in range(0, rows): #asumimos que todas las listas tienen largo "rows"
		for j in range(0, len(list_of_lists)):
			try:
				flattened.append( list_of_lists[j][i] )
			except IndexError as e:
				continue
	return sorted(set(flattened), key=lambda x: flattened.index(x))
def remove_consumed(user_consumption, rec_list):
	l = rec_list
	for itemId in rec_list:
		if itemId in user_consumption: l.remove(itemId)
	return l
def option1Job(data_path, solr, params, N):
	val_folds = os.listdir(data_path+'val/')
	nDCGs = []
	"""HARDCODED AS FUCK: 4+1"""
	for i in range(1, 4+1):
		users_nDCGs = []
		train_c = consumption(ratings_path=data_path+'train/train_N'+str(N)+'.'+str(i), rel_thresh=0, with_ratings=False)
		val_c   = consumption(ratings_path=data_path+'val/val_N'+str(N)+'.'+str(i), rel_thresh=0, with_ratings=True)
		for userId in train_c:
			book_recs = []
			for itemId in train_c[userId]:
				encoded_params = urlencode(params)
				url            = solr + '/mlt?q=goodreadsId:'+ itemId + "&" + encoded_params
				response       = json.loads( urlopen(url).read().decode('utf8') )
				try:
					docs         = response['response']['docs']
				except TypeError as e:
					continue
				book_recs.append( [ str(doc['goodreadsId'][0]) for doc in docs ] )

			book_recs = flatten_list(list_of_lists=book_recs, rows=params['rows'])
			book_recs = remove_consumed(user_consumption=train_c[userId], rec_list=book_recs)
			book_recs = recs_cleaner(solr= solr, consumpt= train_c[userId], recs= book_recs[:50])
			try:
				recs = user_ranked_recs(user_recs=book_recs, user_consumpt=val_c[userId]) #...puede que en val. no esté el mismo usuario...
			except KeyError as e:
				logging.info("Usuario {0} del fold de train {1} no encontrado en fold de val.".format(userId, i))
				continue

			mini_recs = dict((k, recs[k]) for k in recs.keys()[:N]) # Metric for tuning: nDCG @ N
			users_nDCGs.append( nDCG(recs=mini_recs, alt_form=False, rel_thresh=False) )
		
		nDCGs.append( mean(users_nDCGs) )

	return mean(nDCGs)
def option2Job(data_path, solr, params, N):
	"""Genera recomendaciones para filas en <validation>.
	Calcula las métricas comparando con <training>"""
	val_folds = os.listdir(data_path+'val/')
	nDCGs = []
	"""HARDCODED AS FUCK: 4+1"""
	for i in range(1, 4+1):
		users_nDCGs = []
		train_c = consumption(ratings_path=data_path+'train/train_N'+str(N)+'.'+str(i), rel_thresh=0, with_ratings=False)
		val_c   = consumption(ratings_path=data_path+'val/val_N'+str(N)+'.'+str(i), rel_thresh=0, with_ratings=True)
		for userId in train_c:
			stream_url     = solr + '/query?q=goodreadsId:{ids}'
			ids_string     = encoded_itemIds(item_list=train_c[userId])
			encoded_params = urlencode(params)
			url            = solr + '/mlt?stream.url=' + stream_url.format(ids=ids_string) + "&" + encoded_params
			response       = json.loads( urlopen(url).read().decode('utf8') )
			try:
				docs         = response['response']['docs']
			except TypeError as e:
				continue 
			book_recs      = [ str(doc['goodreadsId'][0]) for doc in docs] 
			book_recs      = remove_consumed(user_consumption=train_c[userId], rec_list=book_recs)
			book_recs      = recs_cleaner(solr= solr, consumpt= train_c[userId], recs= book_recs[:50])
			try:
				recs         = user_ranked_recs(user_recs=book_recs, user_consumpt=val_c[userId]) #...puede que en val. no esté el mismo usuario...
			except KeyError as e:
				logging.info("Usuario {0} del fold de train {1} no encontrado en fold de val.".format(userId, i))
				continue

			mini_recs = dict((k, recs[k]) for k in recs.keys()[:N]) # Metric for tuning: nDCG @ N
			users_nDCGs.append( nDCG(recs=mini_recs, alt_form=False, rel_thresh=False) )

		nDCGs.append( mean(users_nDCGs) )

	return mean(nDCGs)
#--------------------------------#
def option1_tuning(data_path, solr, N):
	param_names = ['mlt.fl', 'mlt.boost', 'mlt.mintf', 'mlt.mindf', 'mlt.minwl', 'mlt.maxdf', 'mlt.maxwl', 'mlt.maxqt', 'mlt.maxntp']
	solr_fields = ['goodreadsId', 'description', 'title.titleOfficial', 'genres.genreName', 'author.authors.authorName', 'quotes.quoteText', 'author.authorBio', 'title.titleGreytext']
	mlt_fields  = {1:'description', 2:'title.titleOfficial', 3:'genres.genreName', 4:'author.authors.authorName', 5:'author.authors.authorName,description', 6:'author.authors.authorName,description,title.titleOfficial', 7:'quotes.quoteText', 8:'description,title.titleOfficial,genres.genreName,author.authors.authorName,quotes.quoteText'}
	defaults = {'echoParams' : 'none',
							'fl' : ','.join(solr_fields),#'goodreadsId,'+ mlt_fields[1],
							'rows' : 100,
							'mlt.fl' : mlt_fields[1],
							'mlt.boost' : 'false', #def: false
							'mlt.mintf' : 2, #def: 2
							'mlt.mindf' : 5, #def: 5
							'mlt.minwl' : 0, 
							'mlt.maxdf' : 50000, # en realidad no especificado
							'mlt.maxwl' : 0,
							'mlt.maxqt' : 25, #def: 25
							'mlt.maxntp' : 5000 }

	url = solr + '/query?q=*:*&rows=100000' #n docs: 50,862 < 100,000
	docs = json.loads( urlopen(url).read().decode('utf8') )
	docs = docs['response']['docs']
	docs_num = len(docs)
	del docs

	results = dict((param, {}) for param in param_names)
	for param in param_names: 
		
		if param=='mlt.fl':
			for i in mlt_fields.values():
				defaults['mlt.fl'] = i
				logging.info("Evaluando con params: {}".format(defaults))
				results['mlt.fl'][i] = option1Job(data_path=data_path, solr=solr, params=defaults, N=N)
			defaults['mlt.fl']  = opt_value(results=results['mlt.fl'], metric='ndcg')

		if param=='mlt.minwl':
			for i in range(0, 11):
				defaults['mlt.minwl'] = i
				logging.info("Evaluando con params: {}".format(defaults))
				results['mlt.minwl'][i] = option1Job(data_path=data_path, solr=solr, params=defaults, N=N)
			defaults['mlt.minwl']  = opt_value(results=results['mlt.minwl'], metric='ndcg')		

		if param=='mlt.mintf':
			for i in range(0, 21):
				defaults['mlt.mintf'] = i
				logging.info("Evaluando con params: {}".format(defaults))
				results['mlt.mintf'][i] = option1Job(data_path=data_path, solr=solr, params=defaults, N=N)
			defaults['mlt.mintf']  = opt_value(results=results['mlt.mintf'], metric='ndcg')

		if param=='mlt.mindf':
			for i in range(0, 21):
				defaults['mlt.mindf'] = i
				logging.info("Evaluando con params: {}".format(defaults))
				results['mlt.mindf'][i] = option1Job(data_path=data_path, solr=solr, params=defaults, N=N)
			defaults['mlt.mindf']  = opt_value(results=results['mlt.mindf'], metric='ndcg')		

		if param=='mlt.maxdf':
			for i in [int(docs_num*0.2), int(docs_num*0.35), int(docs_num*0.5), int(docs_num*0.75), int(docs_num*0.9)]:
				defaults['mlt.maxdf'] = i
				logging.info("Evaluando con params: {}".format(defaults))
				results['mlt.maxdf'][i] = option1Job(data_path=data_path, solr=solr, params=defaults, N=N)
			defaults['mlt.maxdf']  = opt_value(results=results['mlt.maxdf'], metric='ndcg')	

		if param=='mlt.maxwl':
			for i in range(0, 25, 2):
				defaults['mlt.maxwl'] = i
				logging.info("Evaluando con params: {}".format(defaults))
				results['mlt.maxwl'][i] = option1Job(data_path=data_path, solr=solr, params=defaults, N=N)
			defaults['mlt.maxwl']  = opt_value(results=results['mlt.maxwl'], metric='ndcg')	

		if param=='mlt.maxqt':
			for i in [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]: # no se pueden tener 0 query terms
				defaults['mlt.maxqt'] = i
				logging.info("Evaluando con params: {}".format(defaults))
				results['mlt.maxqt'][i] = option1Job(data_path=data_path, solr=solr, params=defaults, N=N)
			defaults['mlt.maxqt']  = opt_value(results=results['mlt.maxqt'], metric='ndcg')	

		if param=='mlt.maxntp':
			for i in [500, 1000, 5000, 10000, 50000, 100000, 150000, 200000]: #agregados 3 desde últimos resultados
				defaults['mlt.maxntp'] = i
				logging.info("Evaluando con params: {}".format(defaults))
				results['mlt.maxntp'][i] = option1Job(data_path=data_path, solr=solr, params=defaults, N=N)
			defaults['mlt.maxntp']  = opt_value(results=results['mlt.maxntp'], metric='ndcg')	

		if param=='mlt.boost':
			for i in ['true', 'false']:
				defaults['mlt.boost'] = i
				logging.info("Evaluando con params: {}".format(defaults))
				results['mlt.boost'][i] = option1Job(data_path=data_path, solr=solr, params=defaults, N=N)
			defaults['mlt.boost']  = opt_value(results=results['mlt.boost'], metric='ndcg')

	with open('TwitterRatings/CB/clean/option1_opt_params.txt', 'w') as f:
		for param in defaults:
			f.write( "{param}:{value}\n".format(param=param, value=defaults[param]) )

	with open('TwitterRatings/CB/clean/option1_params_ndcgs.txt', 'w') as f:
		for param in param_names:
			for value in results[param]:
				f.write( "{param}={value}\t : {nDCG}\n".format(param=param, value=value, nDCG=results[param][value]) )

	return defaults

def option2_tuning(data_path, solr, N):

	param_names = ['mlt.fl', 'mlt.boost', 'mlt.mintf', 'mlt.mindf', 'mlt.minwl', 'mlt.maxdf', 'mlt.maxwl', 'mlt.maxqt', 'mlt.maxntp']
	solr_fields = ['goodreadsId', 'description', 'title.titleOfficial', 'genres.genreName', 'author.authors.authorName', 'quotes.quoteText', 'author.authorBio', 'title.titleGreytext']
	mlt_fields  = {1:'description', 2:'title.titleOfficial', 3:'genres.genreName', 4:'author.authors.authorName', 5:'quotes.quoteText'}
	defaults = {'echoParams' : 'none',
							'fl' : ','.join(solr_fields),#'goodreadsId,'+ mlt_fields[1],
							'rows' : 100,
							'mlt.fl' : mlt_fields[1],
							'mlt.boost' : 'false', #def: false
							'mlt.mintf' : 2, #def: 2
							'mlt.mindf' : 5, #def: 5
							'mlt.minwl' : 0, #def: 0
							'mlt.maxdf' : 50000, # en realidad no especificado
							'mlt.maxwl' : 0, #def: 0 
							'mlt.maxqt' : 25, #def: 25
							'mlt.maxntp' : 5000 } #def: 5000

	url = solr + '/query?q=*:*&rows=100000' #n docs: 50,862 < 100,000
	docs = json.loads( urlopen(url).read().decode('utf8') )
	docs = docs['response']['docs']
	docs_num = len(docs)
	del docs

	results = dict((param, {}) for param in param_names)
	for param in param_names: 
		
		if param=='mlt.fl':
			for i in mlt_fields.values():
				defaults['mlt.fl'] = i
				# defaults['fl'] = 'goodreadsId,'+ i
				logging.info("Evaluando con params: {}".format(defaults))
				results['mlt.fl'][i] = option2Job(data_path=data_path, solr=solr, params=defaults, N=N)
			defaults['mlt.fl']  = opt_value(results=results['mlt.fl'], metric='ndcg')
			# defaults['fl'] = 'goodreadsId,'+ defaults['mlt.fl']

		if param=='mlt.minwl':
			for i in range(0, 11):
				defaults['mlt.minwl'] = i
				logging.info("Evaluando con params: {}".format(defaults))
				results['mlt.minwl'][i] = option2Job(data_path=data_path, solr=solr, params=defaults, N=N)
			defaults['mlt.minwl']  = opt_value(results=results['mlt.minwl'], metric='ndcg')		

		if param=='mlt.mintf':
			for i in range(0, 21):
				defaults['mlt.mintf'] = i
				logging.info("Evaluando con params: {}".format(defaults))
				results['mlt.mintf'][i] = option2Job(data_path=data_path, solr=solr, params=defaults, N=N)
			defaults['mlt.mintf']  = opt_value(results=results['mlt.mintf'], metric='ndcg')

		if param=='mlt.mindf':
			for i in range(0, 21):
				defaults['mlt.mindf'] = i
				logging.info("Evaluando con params: {}".format(defaults))
				results['mlt.mindf'][i] = option2Job(data_path=data_path, solr=solr, params=defaults, N=N)
			defaults['mlt.mindf']  = opt_value(results=results['mlt.mindf'], metric='ndcg')		

		if param=='mlt.maxdf':
			for i in [int(docs_num*0.2), int(docs_num*0.35), int(docs_num*0.5), int(docs_num*0.75), int(docs_num*0.9)]:
				defaults['mlt.maxdf'] = i
				logging.info("Evaluando con params: {}".format(defaults))
				results['mlt.maxdf'][i] = option2Job(data_path=data_path, solr=solr, params=defaults, N=N)
			defaults['mlt.maxdf']  = opt_value(results=results['mlt.maxdf'], metric='ndcg')	

		if param=='mlt.maxwl':
			for i in range(0, 25, 2):
				defaults['mlt.maxwl'] = i
				logging.info("Evaluando con params: {}".format(defaults))
				results['mlt.maxwl'][i] = option2Job(data_path=data_path, solr=solr, params=defaults, N=N)
			defaults['mlt.maxwl']  = opt_value(results=results['mlt.maxwl'], metric='ndcg')	

		if param=='mlt.maxqt':
			for i in [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]: # no se pueden tener 0 query terms
				defaults['mlt.maxqt'] = i
				logging.info("Evaluando con params: {}".format(defaults))
				results['mlt.maxqt'][i] = option2Job(data_path=data_path, solr=solr, params=defaults, N=N)
			defaults['mlt.maxqt']  = opt_value(results=results['mlt.maxqt'], metric='ndcg')	

		if param=='mlt.maxntp':
			for i in [500, 1000, 5000, 10000, 50000, 100000, 150000, 200000]: #agregados 3 desde últimos resultados
				defaults['mlt.maxntp'] = i
				logging.info("Evaluando con params: {}".format(defaults))
				results['mlt.maxntp'][i] = option2Job(data_path=data_path, solr=solr, params=defaults, N=N)
			defaults['mlt.maxntp']  = opt_value(results=results['mlt.maxntp'], metric='ndcg')	

		if param=='mlt.boost':
			for i in ['true', 'false']:
				defaults['mlt.boost'] = i
				logging.info("Evaluando con params: {}".format(defaults))
				results['mlt.boost'][i] = option2Job(data_path=data_path, solr=solr, params=defaults, N=N)
			defaults['mlt.boost']  = opt_value(results=results['mlt.boost'], metric='ndcg')

	with open('TwitterRatings/CB/clean/option2_opt_params.txt', 'w') as f:
		for param in defaults:
			f.write( "{param}:{value}\n".format(param=param, value=defaults[param]) )

	with open('TwitterRatings/CB/clean/option2_params_ndcgs.txt', 'w') as f:
		for param in param_names:
			for value in results[param]:
				f.write( "{param}={value}\t : {nDCG}\n".format(param=param, value=value, nDCG=results[param][value]) )

	return defaults


def option1_protocol_evaluation(data_path, solr, params):
	test_c  = consumption(ratings_path=data_path+'test/test_N20.data', rel_thresh=0, with_ratings=True)
	train_c = consumption(ratings_path=data_path+'eval_train_N20.data', rel_thresh=0, with_ratings=False)
	MRRs   = dict((N, []) for N in [5, 10, 15, 20])
	nDCGs  = dict((N, []) for N in [5, 10, 15, 20])
	APs    = dict((N, []) for N in [5, 10, 15, 20])
	Rprecs = dict((N, []) for N in [5, 10, 15, 20])

	for userId in test_c:
		book_recs = []
		for itemId in train_c[userId]:
			encoded_params = urlencode(params)
			url            = solr + '/mlt?q=goodreadsId:'+ itemId + "&" + encoded_params
			response       = json.loads( urlopen(url).read().decode('utf8') )
			try:
				docs         = response['response']['docs']
			except TypeError as e:
				continue
			book_recs.append( [ str(doc['goodreadsId'][0]) for doc in docs ] )

		book_recs = flatten_list(list_of_lists=book_recs, rows=params['rows'])
		book_recs = remove_consumed(user_consumption=train_c[userId], rec_list=book_recs)
		book_recs = recs_cleaner(solr= solr, consumpt= train_c[userId], recs= book_recs[:50])
		try:
			recs = user_ranked_recs(user_recs=book_recs, user_consumpt=test_c[userId])
		except KeyError as e:
			logging.info("Usuario {0} del fold de train (total) no encontrado en fold de 'test'".format(userId))
			continue

		for N in [5, 10, 15, 20]:
			mini_recs = dict((k, recs[k]) for k in recs.keys()[:N])
			MRRs[N].append( MRR(recs=mini_recs, rel_thresh=1) )
			nDCGs[N].append( nDCG(recs=mini_recs, alt_form=False, rel_thresh=False) )		
			APs[N].append( AP_at_N(n=N, recs=recs, rel_thresh=1) )
			Rprecs[N].append( R_precision(n_relevants=N, recs=mini_recs) )

	for N in [5, 10, 15, 20]:
		with open('TwitterRatings/CB/clean/option1_protocol.txt', 'a') as file:
			file.write( "N=%s, nDCG=%s, MAP=%s, MRR=%s, R-precision=%s\n" % \
				(N, mean(nDCGs[N]), mean(APs[N]), mean(MRRs[N]), mean(Rprecs[N])) )	

def option2_protocol_evaluation(data_path, solr, params):
	test_c  = consumption(ratings_path=data_path+'test/test_N20.data', rel_thresh=0, with_ratings=True)
	train_c = consumption(ratings_path=data_path+'eval_train_N20.data', rel_thresh=0, with_ratings=False)
	MRRs   = dict((N, []) for N in [5, 10, 15, 20])
	nDCGs  = dict((N, []) for N in [5, 10, 15, 20])
	APs    = dict((N, []) for N in [5, 10, 15, 20])
	Rprecs = dict((N, []) for N in [5, 10, 15, 20])

	for userId in test_c:
		stream_url     = solr + '/query?q=goodreadsId:{ids}'
		ids_string     = encoded_itemIds(item_list=train_c[userId])
		encoded_params = urlencode(params)
		url            = solr + '/mlt?stream.url=' + stream_url.format(ids=ids_string) + "&" + encoded_params
		response       = json.loads( urlopen(url).read().decode('utf8') )
		try:
			docs         = response['response']['docs']
		except TypeError as e:
			continue
		book_recs      = [ str(doc['goodreadsId'][0]) for doc in docs] 
		book_recs      = remove_consumed(user_consumption=train_c[userId], rec_list=book_recs)
		book_recs      = recs_cleaner(solr= solr, consumpt= train_c[userId], recs= book_recs[:50])
		try:
			recs         = user_ranked_recs(user_recs=book_recs, user_consumpt=test_c[userId])
		except KeyError as e:
			logging.info("Usuario {0} del fold de train (total) no encontrado en fold de 'test'".format(userId))
			continue

		for N in [5, 10, 15, 20]:
			mini_recs = dict((k, recs[k]) for k in recs.keys()[:N])
			MRRs[N].append( MRR(recs=mini_recs, rel_thresh=1) )
			nDCGs[N].append( nDCG(recs=mini_recs, alt_form=False, rel_thresh=False) )		
			APs[N].append( AP_at_N(n=N, recs=recs, rel_thresh=1) )
			Rprecs[N].append( R_precision(n_relevants=N, recs=mini_recs) )


	for N in [5, 10, 15, 20]:
		with open('TwitterRatings/CB/clean/option2_protocol.txt', 'a') as file:
			file.write( "N=%s, nDCG=%s, MAP=%s, MRR=%s, R-precision=%s\n" % \
				(N, mean(nDCGs[N]), mean(APs[N]), mean(MRRs[N]), mean(Rprecs[N])) )	

def main():
	data_path = 'TwitterRatings/funkSVD/data/'
	solr = "http://localhost:8983/solr/grrecsys"
	params_o1 = option1_tuning(data_path=data_path, solr=solr, N=20)
	params_o2 = option2_tuning(data_path=data_path, solr=solr, N=20)
	# params_o1 = {'echoParams' : 'none',
	# 						'fl' : 'goodreadsId,description,title.titleOfficial,genres.genreName,author.authors.authorName,quotes.quoteText,author.authorBio,title.titleGreytext',
	# 						'rows' : 100,
	# 						'mlt.fl' : 'description,title.titleOfficial,genres.genreName,author.authors.authorName,quotes.quoteText',
	# 						'mlt.boost' : 'false', #def: false
	# 						'mlt.mintf' : 1, #def: 2
	# 						'mlt.mindf' : 2, #def: 5
	# 						'mlt.minwl' : 1, #def: 0
	# 						'mlt.maxdf' : 25431 #docs*0.5, # en realidad no especificado
	# 						'mlt.maxwl' : 8, #def: 0
	# 						'mlt.maxqt' : 90, #def: 25
	# 						'mlt.maxntp' : 150000 #def: 5000}
	# params_o2 = {'echoParams' : 'none',
	# 						'fl' : 'goodreadsId,description,title.titleOfficial,genres.genreName,author.authors.authorName,quotes.quoteText,author.authorBio,title.titleGreytext',
	# 						'rows' : 100,
	# 						'mlt.fl' : 'author.authors.authorName',
	# 						'mlt.boost' : 'false', #def: false
	# 						'mlt.mintf' : 2, #def: 2
	# 						'mlt.mindf' : 8, #def: 5
	# 						'mlt.minwl' : 3, 
	# 						'mlt.maxdf' : 25431, # en realidad no especificado
	# 						'mlt.maxwl' : 8,
	# 						'mlt.maxqt' : 40, #def: 25
	# 						'mlt.maxntp' : 150000 }
	# for N in [5, 10, 15, 20]:
	option1_protocol_evaluation(data_path=data_path, solr=solr, params=params_o1)
	option2_protocol_evaluation(data_path=data_path, solr=solr, params=params_o2)
	# option1_testing(data_path=data_path, solr=solr, topN=[5, 10, 15, 20, 50], params=params_o1)
	# option2_testing(data_path=data_path, solr=solr, topN=[5, 10, 15, 20, 50], params=params_o2)
	# option2(solr=solr, rows=rows, fl=fl, topN=[5, 10, 15, 20, 50], mlt_field='description')
	# option2(solr=solr, rows=rows, fl=fl, topN=[5, 10, 15, 20, 50], mlt_field='title.titleOfficial')
	# option2(solr=solr, rows=rows, fl=fl, topN=[5, 10, 15, 20, 50], mlt_field='genres.genreName')
	# option2(solr=solr, rows=rows, fl=fl, topN=[5, 10, 15, 20, 50], mlt_field='author.authors.authorName')
	# option2(solr=solr, rows=rows, fl=fl, topN=[5, 10, 15, 20, 50], mlt_field='quotes.quoteText')

if __name__ == '__main__':
	main()


# """DEBUGGING:"""
# from pprint import pprint
# import os
# import re, json
# from urllib import urlencode, quote_plus
# from urllib2 import urlopen
# from jojFunkSvd import mean, stddev, MRR, rel_div, DCG, iDCG, nDCG, P_at_N, AP_at_N, consumption, user_ranked_recs, opt_value
# import logging
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
# def encoded_itemIds(item_list):
# 	ids_string = '('
# 	for itemId in item_list: ids_string += itemId + '%2520OR%2520'
# 	ids_string = ids_string[:-12] # para borrar el último "%2520OR%2520"
# 	ids_string += ')'
# 	return ids_string

# def remove_consumed(user_consumption, rec_list):
# 	l = rec_list
# 	for itemId in rec_list:
# 		if itemId in user_consumption: l.remove(itemId)
# 	return l

# def flatten_list(list_of_lists, rows):
# 	# eliminamos duplicados manteniendo orden
# 	flattened = []
# 	for i in range(0, rows): #asumimos que todas las listas tienen largo "rows"
# 		for j in range(0, len(list_of_lists)):
# 			try:
# 				flattened.append( list_of_lists[j][i] )
# 			except IndexError as e:
# 				continue
# 	return sorted(set(flattened), key=lambda x: flattened.index(x))

# data_path = 'TwitterRatings/funkSVD/data/'
# solr = "http://localhost:8983/solr/grrecsys"
# val_folds = os.listdir(data_path+'val/')
# nDCGs = []
# train_c = consumption(ratings_path=data_path+'train/train.1', rel_thresh=0, with_ratings=True)
# val_c   = consumption(ratings_path=data_path+'val/val.1', rel_thresh=0, with_ratings=False)
# solr_fields = ['goodreadsId', 'description', 'title.titleOfficial', 'genres.genreName', 'author.authors.authorName', 'quotes.quoteText', 'author.authorBio', 'title.titleGreytext']
# param_names = ['mlt.fl', 'mlt.boost', 'mlt.mintf', 'mlt.mindf', 'mlt.minwl', 'mlt.maxdf', 'mlt.maxwl', 'mlt.maxqt', 'mlt.maxntp']
# mlt_fields  = {1:'description', 2:'title.titleOfficial', 3:'genres.genreName', 4:'author.authors.authorName', 5:'quotes.quoteText'}
# defaults = {'fl' : ','.join(solr_fields),
# 						'rows' : 100,
# 						'mlt.fl' : mlt_fields[1], #'description'
# 						'mlt.boost' : 'false',
# 						'mlt.mintf' : 2,
# 						'mlt.mindf' : 5,
# 						'mlt.minwl' : 0,
# 						'mlt.maxdf' : 10000, # en realidad no especificado
# 						'mlt.maxwl' : 0,
# 						'mlt.maxqt' : 25,
# 						'mlt.maxntp' : 5000}
# results = dict((param, {}) for param in param_names)
# stream_url     = solr + '/query?q=goodreadsId:{ids}'
# ids_string     = encoded_itemIds(item_list=val_c['113447232'])
# encoded_params = urlencode(defaults)
# url            = solr + '/mlt?stream.url=' + stream_url.format(ids=ids_string) + "&" + encoded_params
# response       = json.loads( urlopen(url).read().decode('utf8') )
# docs           = response['response']['docs']
# book_recs      = [ str(doc['goodreadsId'][0]) for doc in docs] 
# book_recs      = remove_consumed(user_consumption=val_c['113447232'], rec_list=book_recs)
# recs           = user_ranked_recs(user_recs=book_recs, user_consumpt=train_c['113447232']) #...puede que en train no esté el mismo usuario...

# mini_recs = dict((k, recs[k]) for k in recs.keys()[:10]) # Metric for tuning: nDCG at 10
# users_nDCGs.append( nDCG(recs=mini_recs, alt_form=False, rel_thresh=3) ) # relevant item if: rating>=3