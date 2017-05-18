"""
Tareas de evaluación para el modelo w2v
"""

#--------------------------------#
# Logging
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
#--------------------------------#
# Gensim
from gensim import corpora, models, similarities, matutils, utils, summarization
from gensim.parsing import preprocessing
#--------------------------------#
# File tools
from os import listdir
from os.path import isfile, join, getmtime
#--------------------------------#
import random
from stop_words import get_stop_words
#--------------------------------#


######################################
def complete_paths(path, filenames):
	"""
	Devuelve lista con path completo de archivos
	"""
	return [ "{0}{1}".format(path, filenames[i]) for i in range(0, len(filenames)) ]

def tokenize_doc(filepath, stopwords):
	"""
	Convierte documento en una lista de tokens (porter-)stemmed,
	lowercased, striped, stop-word & special character filtered 
	"""
	with open(filepath, 'r', encoding='ISO-8859-1') as file:
		s = file.read()
	s = utils.simple_preprocess(s)#preprocessing.preprocess_string(s)
	return s#[w for w in s if w not in stopwords]

def get_ids(filepath):
	ids = {}
	with open(filepath, 'r', encoding='utf-8') as f:
		for line in f:
			splits = line.strip().split(',',1)
			id, name = splits[0], splits[1]
			ids[name] = id
	return ids

def PRF_calc(precs, recls):
	P = sum(precs) / float( len(precs) )
	R = sum(recls) / float( len(recls) )
	try:
		F = 2.0*P*R / (P + R)
	except ZeroDivisionError:
		F = 0
	return P, R, F
######################################

stoplist = set(get_stop_words('en') + get_stop_words('spanish') + get_stop_words('ar'))

consumo = {}
# Creamos diccionario en donde las keys son id de usuario que tiene como valor
# una lista de items (ids) consumidos por ese usuario: consumo[user1_id] = {item1_id:rating, item2_id:rating, ...}
with open("TwitterRatings/ratings.txt", 'r', encoding='utf-8') as f:
	for line in f:
		line = line.strip().split(',')
		if line[0] not in consumo:
			consumo[ line[0] ] = {}
		try:
			consumo[ line[0] ][ line[1] ] = int(line[2])
		except ValueError:
			consumo[ line[0] ][ line[1] ] = 0

path_books  = 'TwitterRatings/items_goodreads/'
path_tweets = 'TwitterRatings/users_goodreads/'
filenames_books  = [f for f in listdir(path_books) if isfile(join(path_books, f))]
filenames_tweets = [f for f in listdir(path_tweets) if isfile(join(path_tweets, f))]

books  = complete_paths(path_books, filenames_books)
tweets = complete_paths(path_tweets, filenames_tweets)

books.sort( key=lambda x: getmtime(x) )
tweets.sort()

ids_books = get_ids("books_ids.txt")
ids_users = get_ids("users_ids.txt")

model = models.Word2Vec.load('./tmp/min_count5_size_100.w2v')

topNs = [10, 50]

# tweets_rand = [ tweets[i] for i in sorted(random.sample(range(len(tweets)), 500)) ]
# books_rand = [ books[i] for i in sorted(random.sample(range(len(books)), 5000)) ]

for topN in topNs:
	i = 0
	precs_cos = []
	precs_wmd = []
	recls_cos = []
	recls_wmd = []
	for user in tweets:#_rand:#tweets:
		if i%500==0 or i==4023:
			logging.info("Viendo user: {0}".format(user))
		screen_name = user[31:-4]
		user_id     = ids_users[screen_name]
		if user_id not in consumo:
			continue
			
		query       = tokenize_doc(user, stoplist)

		user_cos_scores = {}
		user_wmd_scores = {}
		j = 0
		for book in books:#books_rand:#books:
			title    = book[31:-4]#.replace('__','_')
			book_str = tokenize_doc(book, stoplist)
			# if j%200 == 0:
			# 	logging.info("Viendo libro {0}: {1}".format(j, title))

			while True:
				try:
					score = model.n_similarity(query, book_str)
				except KeyError as e:
					err_key = e.args[0]
					# logging.info(err_key)
					if err_key in query:
						query.remove(err_key)
					else:
						book_str.remove(err_key)
					continue
				break

			# logging.info("Calc cos")
			user_cos_scores[ ids_books[title] ] = score
			# logging.info("Calc wmd")
			# score = model.wmdistance(query, book_str)
			# user_wmd_scores[ ids_books[title] ] = score
			j += 1

		# Obtenemos los IDs de los Top-N libros más similares
		topN_cos_recs = sorted(user_cos_scores, key=user_cos_scores.get, reverse=True)[:topN]
		# topN_wmd_recs = sorted(user_wmd_scores, key=user_wmd_scores, reverse=True)[:topN]

		hits_cos = len( set(topN_cos_recs).intersection(consumo[user_id]) )
		# hits_wmd = len( set(topN_wmd_recs).intersection(consumo[user_id]) )

		precs_cos.append( hits_cos / float(topN) )
		# precs_wmd.append( hits_wmd / float(topN) )
		recls_cos.append( hits_cos / float( len(consumo[user_id]) ) )
		# recls_wmd.append( hits_wmd / float( len(consumo[user_id]) ) )
		logging.info("Precision de usuario: {0}".format(precs_cos[i]))
		logging.info("Recall de usuario: {0}".format(recls_cos[i]))
		i += 1

	P_cos, R_cos, F_cos = PRF_calc(precs_cos, recls_cos)
	# P_wmd, R_wmd, F_wmd = PRF_calc(precs_wmd, recls_wmd)

	logging.info("w2v. topN={0}. (P,R,F)_cos=({1},{2},{3}).".format(topN, P_cos, R_cos, F_cos))# (P,R,F)_wmd=({4},{5},{6}).".format(topN, P_cos, R_cos, F_cos, P_wmd, R_wmd,F_wmd) )



