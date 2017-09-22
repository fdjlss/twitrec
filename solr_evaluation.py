# coding=utf-8

import re, json
# from urllib.request import urlopen
# from urllib.parse import urlencode, quote_plus
from urllib import urlencode, quote_plus
from urllib2 import urlopen
from jojFunkSvd import mean, stddev, rel_div, DCG, iDCG, nDCG, P_at_N, AP_at_N, consumption
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


#-----"PRIVATE" METHODS----------#
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
#--------------------------------#

def option1(solr, q, rows, fl, topN):
	train_c = consumption(ratings_path='TwitterRatings/funkSVD/ratings.train', rel_thresh=0, with_ratings=False)
	total_c  = consumption(ratings_path='TwitterRatings/funkSVD/ratings.total', rel_thresh=0, with_ratings=True)
	nDCGs = dict((n, []) for n in topN)
	APs = dict((n, []) for n in topN)
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
			nDCGs[n].append( nDCG(recs=mini_recs, binary_relevance=False) )
			logging.info("Appending result of individual AP..")
			APs[n].append( AP_at_N(n=n, recs=recs, rel_thresh=4) )

	with open('TwitterRatings/CB/option1_results.txt', 'a') as file:
		for n in topN:
			file.write( "N=%s, nDCG=%s, MAP=%s\n" % (n, mean(nDCGs[n]), mean(APs[n])) )	

def option2(solr, rows, fl, topN, mlt_field):
	train_c = consumption(ratings_path='TwitterRatings/funkSVD/ratings.train', rel_thresh=0, with_ratings=False)
	total_c  = consumption(ratings_path='TwitterRatings/funkSVD/ratings.total', rel_thresh=0, with_ratings=True)
	nDCGs = dict((n, []) for n in topN)
	APs = dict((n, []) for n in topN)

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
			nDCGs[n].append( nDCG(recs=mini_recs, binary_relevance=False) )
			APs[n].append( AP_at_N(n=n, recs=recs, rel_thresh=4) )

	with open('TwitterRatings/CB/option2_results_'+mlt_field+'.txt', 'a') as file:
		for n in topN:
			file.write( "N=%s, nDCG=%s, MAP=%s\n" % (n, mean(nDCGs[n]), mean(APs[n])) )	


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