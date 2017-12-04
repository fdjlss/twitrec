# coding=utf-8

import sqlite3
import re, json
import os
from random import sample
from jojFunkSvd import mean, stdev, MRR, rel_div, DCG, iDCG, nDCG, P_at_N, AP_at_N, R_precision, consumption, user_ranked_recs, opt_value
from solr_evaluation import remove_consumed
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

#-----"PRIVATE" METHODS----------#
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
#--------------------------------#

def random_eval(data_path, db_conn, N):
	total_c = consumption(ratings_path=data_path+'eval_all_N'+str(N)+'.data', rel_thresh=0, with_ratings=True)
	test_c  = consumption(ratings_path=data_path+'test/test_N'+str(N)+'.data', rel_thresh=0, with_ratings=True)
	train_c = consumption(ratings_path=data_path+'eval_train_N'+str(N)+'.data', rel_thresh=0, with_ratings=False)
	books   = book_list(db_conn=db_conn)

	MRRs          = []
	nDCGs_bin     = []
	nDCGs_normal  = []
	nDCGs_altform = []
	APs           = []
	Rprecs        = []

	for userId in train_c:
		book_recs = sample(books, k=200)
		book_recs = remove_consumed(user_consumption=train_c[userId], rec_list=book_recs)
		recs      = user_ranked_recs(user_recs=book_recs, user_consumpt=test_c[userId])
		mini_recs = dict((k, recs[k]) for k in recs.keys()[:N])

		MRRs.append( MRR(recs=recs, rel_thresh=1) )
		nDCGs_bin.append( nDCG(recs=mini_recs, alt_form=False, rel_thresh=1) )
		nDCGs_normal.append( nDCG(recs=mini_recs, alt_form=False, rel_thresh=False) )
		nDCGs_altform.append( nDCG(recs=mini_recs, alt_form=True, rel_thresh=False) )			
		APs.append( AP_at_N(n=N, recs=recs, rel_thresh=1) )
		Rprecs.append( R_precision(n_relevants=N, recs=mini_recs) )

	with open('TwitterRatings/random/results.txt', 'a') as file:
		file.write( "N=%s, normal nDCG=%s, alternative nDCG=%s, bin nDCG=%s, MAP=%s, MRR=%s, R-precision=%s\n" % \
				(N, mean(nDCGs_normal), mean(nDCGs_altform), mean(nDCGs_bin), mean(APs), mean(MRRs), mean(Rprecs)) )

def main():
	sqlite_file = 'db/goodreads.sqlite'
	data_path = 'TwitterRatings/funkSVD/data/'
	conn = sqlite3.connect(sqlite_file)
	for N in [5, 10, 15, 20]:
		random_eval(data_path=data_path, db_conn=conn, N=N)
	conn.close()

if __name__ == '__main__':
	main()