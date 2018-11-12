# coding=utf-8

import sqlite3
import re, json
import os
from random import sample
from utils_py2 import *
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

def random_eval(data_path, db_conn):
	test_c  = consumption(ratings_path=data_path+'test/test_N20.data', rel_thresh=0, with_ratings=True)
	train_c = consumption(ratings_path=data_path+'eval_train_N20.data', rel_thresh=0, with_ratings=False)
	books   = book_list(db_conn=db_conn)

	MRRs   = dict((N, []) for N in [5, 10, 15, 20])
	nDCGs  = dict((N, []) for N in [5, 10, 15, 20])
	APs    = dict((N, []) for N in [5, 10, 15, 20])
	Rprecs = dict((N, []) for N in [5, 10, 15, 20])

	for userId in test_c:
		book_recs = sample(books, k=200)
		book_recs = remove_consumed(user_consumption=train_c[userId], rec_list=book_recs)
		recs      = user_ranked_recs(user_recs=book_recs, user_consumpt=test_c[userId])

		for N in [5, 10, 15, 20]:
			mini_recs = dict((k, recs[k]) for k in recs.keys()[:N])
			MRRs[N].append( MRR(recs=mini_recs, rel_thresh=1) )
			nDCGs[N].append( nDCG(recs=mini_recs, alt_form=False, rel_thresh=False) )		
			APs[N].append( AP_at_N(n=N, recs=recs, rel_thresh=1) )
			Rprecs[N].append( R_precision(n_relevants=N, recs=mini_recs) )


	for N in [5, 10, 15, 20]:
		with open('TwitterRatings/random/results.txt', 'a') as file:
			file.write( "N=%s, nDCG=%s, MAP=%s, MRR=%s, R-precision=%s\n" % \
				(N, mean(nDCGs[N]), mean(APs[N]), mean(MRRs[N]), mean(Rprecs[N])) )	




def main():
	sqlite_file = 'db/goodreads.sqlite'
	data_path = 'TwitterRatings/funkSVD/data/'
	conn = sqlite3.connect(sqlite_file)
	# for N in [5, 10, 15, 20]:
	random_eval(data_path=data_path, db_conn=conn)
	conn.close()

if __name__ == '__main__':
	main()