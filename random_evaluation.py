# coding=utf-8

import sqlite3
import re, json
import os
from random import sample
from jojFunkSvd import mean, stddev, rel_div, DCG, iDCG, nDCG, P_at_N, AP_at_N, consumption, ratingsSampler
from solr_evaluation import remove_consumed
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def book_list(db_conn):
	"""Obtiene una lista de IDs de todos los libros
	encontrados en la DB (SQLite)
	"""
	book_list  = []
	c          = db_conn.cursor()
	table_name = 'user_reviews'
	col_name   = 'url_book'
	c.execute( "SELECT DISTINCT {0} FROM {1}".format(col_name, table_name) )
	all_rows = c.fetchall()
	for tupl in all_rows:
		url_book = tupl[0]
		try:
		# Book ID es el número incluido en la URI del libro en GR
		# Hay veces que luego deĺ número le sigue un punto o un guión,
		# y luego el nombre del libro separado con guiones
			book_id = url_book.split('/')[-1].split('-')[0].split('.')[0]
		except AttributeError as e:
			logging.info( "url_book es NULL en la DB!" )
			continue
		book_list.append(book_id)

	return book_list

def random_eval(db_conn, topN, folds):
	total_c = consumption(ratings_path='TwitterRatings/funkSVD/ratings.total', rel_thresh=0, with_ratings=True)
	books   = book_list(db_conn=db_conn)
	a       = dict((metric, []) for metric in ['nDCG(1)', 'nDCG(2)', 'MAP4', 'MAP3', 'MAP2'])
	results = dict((n, a) for n in topN)

	ratings_train = []
	with open('TwitterRatings/funkSVD/ratings.train', 'r') as f:
		for line in f:
			ratings_train.append( line.strip() )

	for _ in folds:
		nDCGs_normal  = dict((n, []) for n in topN)
		nDCGs_altform = dict((n, []) for n in topN)
		APs_thresh4   = dict((n, []) for n in topN)
		APs_thresh3   = dict((n, []) for n in topN)
		APs_thresh2   = dict((n, []) for n in topN)
		ratingsSampler(rats=ratings_train, fout='TwitterRatings/funkSVD/ratings_temp.train', sample_fraction=0.8)
		train_c = consumption(ratings_path='TwitterRatings/funkSVD/ratings_temp.train', rel_thresh=0, with_ratings=False)
		for userId in train_c:
			book_recs = sample(books, k=200)
			book_recs = remove_consumed(user_consumption=train_c[userId], rec_list=book_recs)

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
				nDCGs_normal[n].append( nDCG(recs=mini_recs, alt_form=False) )
				nDCGs_altform[n].append( nDCG(recs=mini_recs, alt_form=True) )			
				APs_thresh4[n].append( AP_at_N(n=n, recs=recs, rel_thresh=4) )
				APs_thresh3[n].append( AP_at_N(n=n, recs=recs, rel_thresh=3) )
				APs_thresh2[n].append( AP_at_N(n=n, recs=recs, rel_thresh=2) )

		for n in topN:
			results[n]['nDCG(1)'].append( mean(nDCGs_normal[n]) )
			results[n]['nDCG(2)'].append( mean(nDCGs_altform[n]) ) 
			results[n]['MAP4'].append( mean(APs_thresh4[n]) ) 
			results[n]['MAP3'].append( mean(APs_thresh3[n]) ) 
			results[n]['MAP2'].append( mean(APs_thresh2[n]) ) 


	for n in topN:
		with open('TwitterRatings/random/results.txt', 'a') as file:
			file.write("\nn={}".format(n))
			# Orden: nDCG(1) - nDCG(2) - MAP4 - MAP3 - MAP2
			for metric in results[n]:
				file.write( ", {0}={1}".format(metric, mean(results[n][metric])) )		


def main():
	sqlite_file = 'db/goodreads.sqlite'
	conn = sqlite3.connect(sqlite_file)
	random_eval(db_conn=conn, topN=[5, 10, 15, 20, 50], folds=10)
	conn.close()

if __name__ == '__main__':
	main()