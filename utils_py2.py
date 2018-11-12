# coding=utf-8

from random import sample
from os.path import isfile, join
import numpy as np
from math import sqrt, log
import json
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# -- from SVD eval:
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

def consumption(ratings_path, rel_thresh, with_ratings, with_timestamps=False):
	c = {}
	with open(ratings_path, 'r') as f:
		for line in f:
			userId, itemId, rating, timestamp = line.strip().split(',')
			###########
			if userId not in c:
				if with_ratings:
					c[userId] = {}
				else:
					c[userId] = []
			if int( rating ) >= rel_thresh:
				if with_ratings:
					if with_timestamps:
						c[userId][itemId] = (rating,timestamp)
					else:
						c[userId][itemId] = rating
				else:
					if with_timestamps:
						c[userId].append( (itemId,timestamp) )
					else:
						c[userId].append( itemId )
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

# -- from Solr eval:
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

# -- from pyFM eval:
def loadData(filename, data_path='TwitterRatings/funkSVD/data/', with_timestamps=False, with_authors=False):
	data = []
	y = []
	items=set()
	with open(data_path+filename, 'r') as f:
		for line in f:
			(userId,itemId,rating,timestamp,authorId1,authorId2,authorId3)=line.split(',')
			if with_timestamps and with_authors:
				data.append({ "user_id": str(userId), "item_id": str(itemId), "timestamp": str(timestamp), "author1_id": str(authorId1), "author2_id": str(authorId2), "author3_id": str(authorId3) })
			if with_timestamps and not with_authors:
				data.append({ "user_id": str(userId), "item_id": str(itemId), "timestamp": str(timestamp) })
			if not with_timestamps and with_authors:
				data.append({ "user_id": str(userId), "item_id": str(itemId), "author1_id": str(authorId1), "author2_id": str(authorId2), "author3_id": str(authorId3) })
			if not with_timestamps and not with_authors:
				data.append({ "user_id": str(userId), "item_id": str(itemId) })
			y.append(float(rating))
			items.add(itemId)
	return data, np.array(y), items

# -- from implicit eval:
class IdCoder(object):
	def __init__(self, items_ids, users_ids):
		self.item_table = { str(i): items_ids[i] for i in range(0, len(items_ids)) }
		self.user_table = { str(i): users_ids[i] for i in range(0, len(users_ids)) }
		self.item_inverted = { v: k for k, v in self.item_table.iteritems() }
		self.user_inverted = { v: k for k, v in self.user_table.iteritems() }
	def coder(self, category, ID):
		if category=="item":
			return self.item_inverted[str(ID)]
		elif category=="user":
			return self.user_inverted[str(ID)]
	def decoder(self, category, ID):
		if category=="item":
			return self.item_table[str(ID)]
		elif category=="user":
			return self.user_table[str(ID)]

def get_data(data_path, all_c, idcoder, fold, N, mode):
	if mode=="tuning":
		train_c = consumption(ratings_path= data_path+'train/train_N'+str(N)+'.'+str(fold), rel_thresh=0, with_ratings=True)
	elif mode=="testing":
		train_c = consumption(ratings_path= data_path+'eval_train_N'+str(N)+'.data', rel_thresh= 0, with_ratings= True)
	arrays  = {'items':[], 'users':[], 'data':[]}
	for userId in train_c:
		r_u = mean( map( int, train_c[userId].values() ) )
		for itemId in train_c[userId]:
			if int(train_c[userId][itemId]) >= r_u:
				arrays['items'].append(int( idcoder.coder('item', itemId) ))
				arrays['users'].append(int( idcoder.coder('user', userId) ))
				arrays['data'].append(1)
			else:
				arrays['items'].append(int( idcoder.coder('item', itemId) ))
				arrays['users'].append(int( idcoder.coder('user', userId) ))
				arrays['data'].append(0)
	ones = np.array( arrays['data'] )
	return ones, arrays['items'], arrays['users'] # value, rows, cols

def get_ndcg(data_path, idcoder, fold, N, model, matrix_T):
	users_nDCGs = []
	val_c = consumption(ratings_path=data_path+'val/val_N'+str(N)+'.'+str(fold), rel_thresh=0, with_ratings=True)
	for userId in val_c:
		recommends = model.recommend(userid= int(idcoder.coder('user', userId)), user_items= matrix_T, N= N)

		book_recs  = [ idcoder.decoder('item', tupl[0]) for tupl in recommends ]
		recs       = user_ranked_recs(user_recs= book_recs, user_consumpt= val_c[userId])
		users_nDCGs.append( nDCG(recs=recs, alt_form=False, rel_thresh=False) )
	return mean(users_nDCGs)


# -- from Hybrid eval:
def hybridize_recs(recs_cb, recs_cf, weight_cb, weight_cf):
	concat = recs_cb + recs_cf
	all_items = list( set(recs_cb + recs_cf) )
	scores = dict((itemId, 0) for itemId in all_items )
	for itemId in scores:
		score_cb = 0
		score_cf = 0
		if itemId in recs_cb: score_cb = weight_cb / float(recs_cb.index(itemId) + 1) #pq index parten desde 0
		if itemId in recs_cf: score_cf = weight_cf / float(recs_cf.index(itemId) + 1)
		occurs = concat.count(itemId)
		item_score = (score_cb + score_cf) * occurs
		scores[itemId] = item_score
	return sorted(scores, key=scores.get, reverse=True)

# -- from UCE:
def recs_cleaner(solr, consumpt, recs):
	from urllib2 import urlopen
	# Ve los canonical hrefs de los items consumidos
	consumpt_hrefs = []
	for itemId in consumpt:
		url      = solr + '/select?q=goodreadsId:' + itemId + '&wt=json' 
		response = json.loads( urlopen(url).read().decode('utf8') )
		doc      = response['response']['docs'][0]
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


def main():
	pass

if __name__ == '__main__':
	main()