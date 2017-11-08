# coding=utf-8

import re, json
from urllib import urlencode, quote_plus
from urllib2 import urlopen
from jojFunkSvd import mean, stddev, MRR, rel_div, DCG, iDCG, nDCG, P_at_N, AP_at_N, consumption, user_ranked_recs, opt_value
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


#-----"PRIVATE" METHODS----------#
def encoded_itemIds(item_list):
	ids_string = '('
	for itemId in item_list: ids_string += itemId + '%2520OR%2520'
	ids_string = ids_string[:-12] # para borrar el último "%2520OR%2520"
	ids_string += ')'
	return ids_string
def flatten_list(list_of_lists, rows):
	# eliminamos duplicados manteniendo orden
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


def option2Job(data_path, solr, params):
	"""Genera recomendaciones para filas en <validation>.
	Calcula las métricas comparando con <training>"""
	val_folds = os.listdir(data_path+'val/')
	nDCGs = []
	for i in range(0, len(val_folds)):
		users_nDCGs = []
		train_c = consumption(ratings_path=data_path+'train/train.'+str(i), rel_thresh=0, with_ratings=True)
		val_c   = consumption(ratings_path=data_path+'val/val.'+str(i), rel_thresh=0, with_ratings=False)
		for userId in train_c:
			stream_url     = solr + '/query?q=goodreadsId:{ids}'
			ids_string     = encoded_itemIds(item_list=val_c[userId])
			encoded_params = urlencode(params)
			url            = solr + '/mlt?stream.url=' + stream_url.format(ids=ids_string) + "&" + encoded_params
			response       = json.loads( urlopen(url).read().decode('utf8') )
			try:
				docs         = response['response']['docs']
			except TypeError as e:
				continue
			book_recs      = [ str(doc['goodreadsId'][0]) for doc in docs] 
			book_recs      = remove_consumed(user_consumption=val_c[userId], rec_list=book_recs)
			recs           = user_ranked_recs(user_recs=book_recs, user_consumpt=train_c[userId])

			mini_recs = dict((k, recs[k]) for k in recs.keys()[:10])
			users_nDCGs.append( nDCG(recs=mini_recs, alt_form=False, rel_thresh=3) )

		nDCGs.append( mean(users_nDCGs) )

	return mean(nDCGs)
#--------------------------------#

def option1(data_path, solr, q, rows, fl, topN):
	train_c = consumption(ratings_path=data_path+'ratings.train', rel_thresh=0, with_ratings=False)
	total_c = consumption(ratings_path=data_path+'ratings.total', rel_thresh=0, with_ratings=True)
	nDCGs_normal  = dict((n, []) for n in topN)
	nDCGs_altform = dict((n, []) for n in topN)
	APs_thresh4   = dict((n, []) for n in topN)
	APs_thresh3   = dict((n, []) for n in topN)
	APs_thresh2   = dict((n, []) for n in topN)
	for userId in train_c:
		logging.info("-> Option 1. Viendo usuario {0}/{1}".format(userId, len(train_c)) )
		book_recs = []
		for itemId in train_c[userId]:
			logging.info("Viendo item {0}".format(itemId))
			base_params = {'q' : q.format(goodreadsId=itemId),
										 'rows' : rows, 
										 'fl' : fl,
										 'omitHeader' : 'true',
										 'debugQuery' : 'on'}
			logging.info("Encoding params..")
			encoded_params = urlencode(base_params)
			url            = solr + '/mlt?' + encoded_params
			logging.info("Fetching response..")
			response       = json.loads( urlopen(url).read().decode('utf8') )
			try:
				docs           = response['response']['docs']
			except TypeError as e:
				continue
			parsed_query   = response['debug']['parsedquery']
			logging.info("Apending book IDs..")
			book_recs.append( [ str(doc['goodreadsId'][0]) for doc in docs ] )

		logging.info("Flattening list..")
		book_recs = flatten_list(list_of_lists=book_recs, rows=rows)
		logging.info("Removing consumed..")
		book_recs = remove_consumed(user_consumption=train_c[userId], rec_list=book_recs)
		recs = {}
		place = 1
		logging.info("Making recs..")
		for itemId in book_recs:
			if itemId in total_c[userId]:
				rating = int( total_c[userId][itemId] )
			else:
				rating = 0
			recs[place] = rating
			place += 1

		for n in topN: 
			mini_recs = dict((k, recs[k]) for k in recs.keys()[:n])
			logging.info("Appending result of individual nDCG..")
			logging.info("Appending result of individual AP..")
			nDCGs_normal[n].append( nDCG(recs=mini_recs, alt_form=False) )
			nDCGs_altform[n].append( nDCG(recs=mini_recs, alt_form=True) )			
			APs_thresh4[n].append( AP_at_N(n=n, recs=recs, rel_thresh=4) )
			APs_thresh3[n].append( AP_at_N(n=n, recs=recs, rel_thresh=3) )
			APs_thresh2[n].append( AP_at_N(n=n, recs=recs, rel_thresh=2) )

	with open('TwitterRatings/CB/option1_results.txt', 'a') as file:
		for n in topN:
			file.write( "N=%s, normal nDCG=%s, alternative nDCG=%s, MAP(rel_thresh=4)=%s, MAP(rel_thresh=3)=%s, MAP(rel_thresh=2)=%s\n" % \
				(n, mean(nDCGs_normal[n]), mean(nDCGs_altform[n]), mean(APs_thresh4[n]), mean(APs_thresh3[n]), mean(APs_thresh2[n]) ) )	

def option2_tuning(data_path, solr):

	param_names = ['mlt.fl', 'mlt.boost', 'mlt.mintf', 'mlt.mindf', 'mlt.minwl', 'mlt.maxdf', 'mlt.maxwl', 'mlt.maxqt', 'mlt.maxntp']
	solr_fields = ['description', 'title.titleOfficial', 'genres.genreName', 'author.authors.authorName', 'quotes.quoteText', 'author.authorBio', 'title.titleGreytext']
	mlt_fields  = {1:'description', 2:'title.titleOfficial', 3:'genres.genreName', 4:'author.authors.authorName', 5:'quotes.quoteText'}
	defaults = {'fl' : ','.join(solr_fields),
							'rows' : 100,
							'mlt.fl' : mlt_fields[1],
							'mlt.boost' : False,
							'mlt.mintf' : 2,
							'mlt.mindf' : 5,
							'mlt.minwl' : 0,
							'mlt.maxdf' : 10000, # en realidad no especificado
							'mlt.maxwl' : 0,
							'mlt.maxqt' : 25,
							'mlt.maxntp' : 5000,
							'mlt.qf' : mlt_fields[1] }

	results = dict((param, {}) for param in param_names)
	for param in param_names: 
		
		if param=='mlt.fl':
			for i in mlt_fields.values():
				defaults['mlt.fl'] = i
				logging.info("Evaluando con params: {}".format(defaults))
				results['mlt.fl'] = option2Job(data_path=data_path, solr=solr, params=defaults)
			defaults['mlt.fl']  = opt_value(results=results['mlt.fl'], metric='ndcg')

		if param=='mlt.boost':
			#TODO



	with open('TwitterRatings/CB/opt_params.txt', 'w') as f:
		for param in defaults:
			f.write( "{param}:{value}\n".format(param=param, value=defaults[param]) )
		f.write( "nDCG:{nDCG}".format(mean(nDCG)) )

	return defaults




def option2(data_path, solr, rows, fl, topN, mlt_field):
	train_c = consumption(ratings_path=data_path+'ratings.train', rel_thresh=0, with_ratings=False)
	total_c = consumption(ratings_path=data_path+'ratings.total', rel_thresh=0, with_ratings=True)
	MRR_thresh4   = []
	MRR_thresh3   = []
	nDCGs_bin_thresh4 = dict((n, []) for n in topN)
	nDCGs_bin_thresh3 = dict((n, []) for n in topN)
	nDCGs_normal  = dict((n, []) for n in topN)
	nDCGs_altform = dict((n, []) for n in topN)
	APs_thresh4   = dict((n, []) for n in topN)
	APs_thresh3   = dict((n, []) for n in topN)
	APs_thresh2   = dict((n, []) for n in topN)

	for userId in train_c:
		logging.info("-> Option 2. mlt.fl: {2}. Viendo usuario {0}/{1}".format(userId, len(train_c), mlt_field) )
		stream_url = solr + '/query?q=goodreadsId:{ids}'

		ids_string = '('
		for itemId in train_c[userId]:
			ids_string += itemId + '%2520OR%2520'
		ids_string = ids_string[:-12] # para borrar el último "%2520OR%2520"
		ids_string += ')'

		base_params = {'rows' : rows,
									 'fl' : fl,
									 'mlt.fl' : mlt_field,
									 'omitHeader' : 'true',
									 'debugQuery' : 'on'}
		encoded_params = urlencode(base_params)
		url            = solr + '/mlt?stream.url=' + stream_url.format(ids=ids_string) + "&" + encoded_params
		response       = json.loads( urlopen(url).read().decode('utf8') )
		try:
			docs           = response['response']['docs']
		except TypeError as e:
			continue
		parsed_query   = response['debug']['parsedquery']
		book_recs      = [ str(doc['goodreadsId'][0]) for doc in docs] 
		book_recs      = remove_consumed(user_consumption=train_c[userId], rec_list=book_recs)

		recs = {}
		place = 1
		for itemId in book_recs:
			if itemId in total_c[userId]:
				rating = int( total_c[userId][itemId] )
			else:
				rating = 0
			recs[place] = rating
			place += 1

		for n in topN: 
			mini_recs = dict((k, recs[k]) for k in recs.keys()[:n])
			nDCGs_bin_thresh4[n].append( nDCG(recs=mini_recs, alt_form=False, rel_thresh=4) )
			nDCGs_bin_thresh3[n].append( nDCG(recs=mini_recs, alt_form=False, rel_thresh=3) )
			nDCGs_normal[n].append( nDCG(recs=mini_recs, alt_form=False, rel_thresh=False) )
			nDCGs_altform[n].append( nDCG(recs=mini_recs, alt_form=True, rel_thresh=False) )			
			APs_thresh4[n].append( AP_at_N(n=n, recs=recs, rel_thresh=4) )
			APs_thresh3[n].append( AP_at_N(n=n, recs=recs, rel_thresh=3) )
			APs_thresh2[n].append( AP_at_N(n=n, recs=recs, rel_thresh=2) )

	with open('TwitterRatings/CB/option2_results_'+mlt_field+'.txt', 'a') as file:
		for n in topN:
			file.write( "N=%s, normal nDCG=%s, alternative nDCG=%s, bin nDCG(rel_thresh=4)=%s, bin nDCG(rel_thresh=3)=%s, MAP(rel_thresh=4)=%s, MAP(rel_thresh=3)=%s, MAP(rel_thresh=2)=%s\n" % \
				(n, mean(nDCGs_normal[n]), mean(nDCGs_altform[n]), mean(nDCGs_bin_thresh4[n]), mean(nDCGs_bin_thresh3[n]), mean(APs_thresh4[n]), mean(APs_thresh3[n]), mean(APs_thresh2[n])) )		

def main():
	data_path = 'TwitterRatings/funkSVD/data/'
	solr = "http://localhost:8983/solr/grrecsys"
	q = 'goodreadsId:{goodreadsId}'
	rows = 100
	fl = 'id,goodreadsId,title.titleOfficial,rating.ratingAvg,genres.genreName,description'
	option1(solr=solr, q=q, rows=rows, fl=fl, topN=[5, 10, 15, 20, 50])
	option2(solr=solr, rows=rows, fl=fl, topN=[5, 10, 15, 20, 50], mlt_field='description')
	option2(solr=solr, rows=rows, fl=fl, topN=[5, 10, 15, 20, 50], mlt_field='title.titleOfficial')
	option2(solr=solr, rows=rows, fl=fl, topN=[5, 10, 15, 20, 50], mlt_field='genres.genreName')
	option2(solr=solr, rows=rows, fl=fl, topN=[5, 10, 15, 20, 50], mlt_field='author.authors.authorName')
	option2(solr=solr, rows=rows, fl=fl, topN=[5, 10, 15, 20, 50], mlt_field='quotes.quoteText')

if __name__ == '__main__':
	main()


