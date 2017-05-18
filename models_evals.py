# CORRER CUANDO dictionary-maker.py HAYA CORRIDO
"""
Tareas para evaluar modelos a base de Precision, Recall & F-Measure
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
# module 'gensim.matutils' no reconoce algunas funciones
# que aparecen en la documentación, así que copio el código fuente
# de jaccard, hellinger y kullback_leibler
import math

import numpy
import scipy.sparse
from scipy.stats import entropy
import scipy.linalg
from scipy.linalg.lapack import get_lapack_funcs

from six import iteritems, itervalues, string_types
from six.moves import xrange, zip as izip
#--------------------------------#



######################################
def complete_paths(path, filenames):
	"""
	Devuelve lista con path completo de archivos
	"""
	return [ "{0}{1}".format(path, filenames[i]) for i in range(0, len(filenames)) ]

def tokenize_doc(filepath):
	"""
	Convierte documento en una lista de tokens (porter-)stemmed,
	lowercased, striped, stop-word & special character filtered 
	"""
	with open(filepath, 'r', encoding='ISO-8859-1') as file:
		s = file.read()
	return preprocessing.preprocess_string(s)

def isbow(vec):
  """
  Checks if vector passed is in bag of words representation or not.
  Vec is considered to be in bag of words format if it is 2-tuple format.
  """
  if scipy.sparse.issparse(vec):
    vec = vec.todense().tolist()
  try:
    id_, val_ = vec[0] # checking first value to see if it is in bag of words format by unpacking
    id_, val_ = int(id_), float(val_)
  except IndexError:
    return True # this is to handle the empty input case
  except Exception:
  	return False
  return True

def jaccard(vec1, vec2):
  """
  A distance metric between bags of words representation.
  Returns 1 minus the intersection divided by union, where union is the sum of the size of the two bags.
  If it is not a bag of words representation, the union and intersection is calculated in the traditional manner.
  Returns a value in range <0,1> where values closer to 0 mean less distance and thus higher similarity.

  """

  # converting from sparse for easier manipulation
  if scipy.sparse.issparse(vec1):
    vec1 = vec1.toarray()
  if scipy.sparse.issparse(vec2):
    vec2 = vec2.toarray()
  if isbow(vec1) and isbow(vec2): 
    # if it's in bow format, we use the following definitions:
    # union = sum of the 'weights' of both the bags
    # intersection = lowest weight for a particular id; basically the number of common words or items 
    union = sum(weight for id_, weight in vec1) + sum(weight for id_, weight in vec2)
    vec1, vec2 = dict(vec1), dict(vec2)
    intersection = 0.0
    for feature_id, feature_weight in iteritems(vec1):
      intersection += min(feature_weight, vec2.get(feature_id, 0.0))
    return 1 - float(intersection) / float(union)
  else:
    # if it isn't in bag of words format, we can use sets to calculate intersection and union
    if isinstance(vec1, numpy.ndarray):
      vec1 = vec1.tolist()
    if isinstance(vec2, numpy.ndarray):
      vec2 = vec2.tolist()
    vec1 = set(vec1)
    vec2 = set(vec2)
    intersection = vec1 & vec2
    union = vec1 | vec2
    return 1 - float(len(intersection)) / float(len(union))

def hellinger(vec1, vec2):
  """
  Hellinger distance is a distance metric to quantify the similarity between two probability distributions.
  Distance between distributions will be a number between <0,1>, where 0 is minimum distance (maximum similarity) and 1 is maximum distance (minimum similarity).
  """
  if scipy.sparse.issparse(vec1):
    vec1 = vec1.toarray()
  if scipy.sparse.issparse(vec2):
    vec2 = vec2.toarray()
  if isbow(vec1) and isbow(vec2): 
    # if it is a bag of words format, instead of converting to dense we use dictionaries to calculate appropriate distance
    vec1, vec2 = dict(vec1), dict(vec2)
    if len(vec2) < len(vec1): 
      vec1, vec2 = vec2, vec1 # swap references so that we iterate over the shorter vector
    sim = numpy.sqrt(0.5*sum((numpy.sqrt(value) - numpy.sqrt(vec2.get(index, 0.0)))**2 for index, value in iteritems(vec1)))
    return sim
  else:
	  sim = numpy.sqrt(0.5 * ((numpy.sqrt(vec1) - numpy.sqrt(vec2))**2).sum())
	  return sim

def kullback_leibler(vec1, vec2, num_features=None):
  """
  A distance metric between two probability distributions.
  Returns a distance value in range <0,1> where values closer to 0 mean less distance (and a higher similarity)
  Uses the scipy.stats.entropy method to identify kullback_leibler convergence value.
  If the distribution draws from a certain number of docs, that value must be passed.
  """
  if scipy.sparse.issparse(vec1):
    vec1 = vec1.toarray()
  if scipy.sparse.issparse(vec2):
    vec2 = vec2.toarray() # converted both the vectors to dense in case they were in sparse matrix 
  if isbow(vec1) and isbow(vec2): # if they are in bag of words format we make it dense
    if num_features != None: # if not None, make as large as the documents drawing from
      dense1 = matutils.sparse2full(vec1, num_features)
      dense2 = matutils.sparse2full(vec2, num_features)
      return entropy(dense1, dense2)
    else:
      max_len = max(len(vec1), len(vec2))
      dense1 = matutils.sparse2full(vec1, max_len)
      dense2 = matutils.sparse2full(vec2, max_len)
      return entropy(dense1, dense2)
  else:
    # this conversion is made because if it is not in bow format, it might be a list within a list after conversion
    # the scipy implementation of Kullback fails in such a case so we pick up only the nested list.
    if len(vec1) == 1:
      vec1 = vec1[0]
    if len(vec2) == 1:
      vec2 = vec2[0]
    return scipy.stats.entropy(vec1, vec2)

def PR_appender(topN,  user, id_user, model=None, distances=[], ps_cos=[], ps_jacc=[], ps_bm25=[], ps_hell=[], ps_kl=[], rs_cos=[], rs_jacc=[], rs_bm25=[], rs_hell=[], rs_kl=[]):
	id_user = str(id_user)
	# Listas con forma [[score_b1, id_b1], [score_b2, id_b2], ...]
	scs_cos = []
	scs_jacc = []
	scs_bm25 = []
	scs_hell = []
	scs_kl = []

	# Obtenemos listas de scores:
	for book, id_book in corpus_books:#Si quiero truncar el corpus: utils.ClippedCorpus(corpus_books, 10000)
		# logging.info("Libro {0}".format(i))
		if model=="tf-idf":
			user = tfidf[ user ] # Asumiendo que el modelo está cargado y tiene nombre tfidf
			book = tfidf[ book ]
		elif model=="lsi":
			user = lsi[ tfidf[user] ]
			book = lsi[ tfidf[book] ]
		elif model=="lda":
			user = lda[ user ]
			book = lda[ book ]

		if "cos" in distances:
			scs_cos.append( [matutils.cossim(user, book), str(id_book)] )
		if "jacc" in distances:
			scs_jacc.append( [jaccard(user, book), str(id_book)] )
		if "bm25" in distances:
			# Esto posible ya que en el corpus de libros (BoW), index = id_book.
			# Llegadas de nuevos libros reciben id_book >= len(corpus_books)
			scs_bm25.append( [bm25.get_score(user, id_book, avg_idf), str(id_book)] ) # Uso de rankeo con bm25 hecho fuera el scope de esta función 
		if "hell" in distances:
			scs_hell.append( [hellinger(user, book), str(id_book)] )
		if "kl" in distances:
			scs_kl.append( [kullback_leibler(user, book), str(id_book)] )

	# Ordenamos por score
	scs_cos.sort( key=lambda x: x[0], reverse=True ) # higher: more sim
	scs_jacc.sort( key=lambda x: x[0], reverse=False ) # lower: more sim
	scs_bm25.sort( key=lambda x: x[0], reverse=True ) # higher: more sim
	scs_hell.sort( key=lambda x: x[0], reverse=False ) # lower: more sim
	scs_kl.sort( key=lambda x: x[0], reverse=False ) # lower: more sim

	hits_cos = 0 # #hits = #tp
	hits_jacc = 0
	hits_bm25 = 0
	hits_hell = 0
	hits_kl = 0
	for i in range(0, topN):
		if "cos" in distances:
			if scs_cos[i][1] in consumo[id_user]:
				hits_cos += 1
		if "jacc" in distances:
			if scs_jacc[i][1] in consumo[id_user]:
				hits_jacc += 1
		if "bm25" in distances:
			if scs_bm25[i][1] in consumo[id_user]:
				hits_bm25 += 1
		if "hell" in distances:
			if scs_hell[i][1] in consumo[id_user]:
				hits_hell += 1
		if "kl" in distances:
			if scs_kl[i][1] in consumo[id_user]:
				hits_kl += 1

	if "cos" in distances:
		ps_cos.append( hits_cos / float(topN) )
		rs_cos.append( hits_cos / float( len(consumo[id_user]) ) )
	if "jacc" in distances:
		ps_jacc.append( hits_jacc / float(topN) )
		rs_jacc.append( hits_jacc / float( len(consumo[id_user]) ) )
	if "bm25" in distances:
		ps_bm25.append( hits_bm25 / float(topN) )
		rs_bm25.append( hits_bm25 / float( len(consumo[id_user]) ) )
	if "hell" in distances:
		ps_hell.append( hits_hell / float(topN) )
		rs_hell.append( hits_hell / float( len(consumo[id_user]) ) )
	if "kl" in distances:
		ps_kl.append( hits_kl / float(topN) )
		rs_kl.append( hits_kl / float( len(consumo[id_user]) ) )

def PR_appender_rat_thresh(topN, user, id_user, model=None, distances=[], ps_cos=[], ps_jacc=[], ps_bm25=[], ps_hell=[], ps_kl=[], rs_cos=[], rs_jacc=[], rs_bm25=[], rs_hell=[], rs_kl=[]):
	id_user = str(id_user)
	# Listas con forma [[score_b1, id_b1], [score_b2, id_b2], ...]
	scs_cos = []
	scs_jacc = []
	scs_bm25 = []
	scs_hell = []
	scs_kl = []

	# Obtenemos listas de scores:
	for book, id_book in corpus_books:
		if model=="tf-idf":
			user = tfidf[ user ] # Asumiendo que el modelo está cargado y tiene nombre tfidf
			book = tfidf[ book ]
		elif model=="lsi":
			user = lsi[ tfidf[user] ]
			book = lsi[ tfidf[book] ]
		elif model=="lda":
			user = lda[ user ]
			book = lda[ book ]

		if "cos" in distances:
			scs_cos.append( [matutils.cossim(user, book), str(id_book)] )
		if "jacc" in distances:
			scs_jacc.append( [jaccard(user, book), str(id_book)] )
		if "bm25" in distances:
			# Esto posible ya que en el corpus de libros (BoW), index = id_book.
			# Llegadas de nuevos libros reciben id_book >= len(corpus_books)
			scs_bm25.append( [bm25.get_score(user, id_book, avg_idf), str(id_book)] ) # Uso de rankeo con bm25 hecho fuera el scope de esta función 
		if "hell" in distances:
			scs_hell.append( [hellinger(user, book), str(id_book)] )
		if "kl" in distances:
			scs_kl.append( [kullback_leibler(user, book), str(id_book)] )

	# Ordenamos por score
	scs_cos.sort( key=lambda x: x[0], reverse=True ) # higher: more sim
	scs_jacc.sort( key=lambda x: x[0], reverse=False ) # lower: more sim
	scs_bm25.sort( key=lambda x: x[0], reverse=True ) # higher: more sim
	scs_hell.sort( key=lambda x: x[0], reverse=False ) # lower: more sim
	scs_kl.sort( key=lambda x: x[0], reverse=False ) # lower: more sim

	hits_cos = 0 # #hits = #tp
	hits_jacc = 0
	hits_bm25 = 0
	hits_hell = 0
	hits_kl = 0
	for i in range(0, topN):
		if (scs_cos[i][1] in consumo[id_user]) and ( consumo[ id_user ][ scs_cos[i][1] ] >= 4):
			hits_cos += 1
		if (scs_jacc[i][1] in consumo[id_user]) and ( consumo[ id_user ][ scs_jacc[i][1] ] >= 4):
			hits_jacc += 1
		if (scs_bm25[i][1] in consumo[id_user]) and ( consumo[ id_user ][ scs_bm25[i][1] ] >= 4):
			hits_bm25 += 1
		if (scs_hell[i][1] in consumo[id_user]) and ( consumo[ id_user ][ scs_hell[i][1] ] >= 4):
			hits_hell += 1
		if (scs_kl[i][1] in consumo[id_user]) and ( consumo[ id_user ][ scs_kl[i][1] ] >= 4):
			hits_kl += 1

	if "cos" in distances:
		ps_cos.append( hits_cos / float(topN) )
		rs_cos.append( hits_cos / float( len(consumo[id_user]) ) )
	if "jacc" in distances:
		ps_jacc.append( hits_jacc / float(topN) )
		rs_jacc.append( hits_jacc / float( len(consumo[id_user]) ) )
	if "bm25" in distances:
		ps_bm25.append( hits_bm25 / float(topN) )
		rs_bm25.append( hits_bm25 / float( len(consumo[id_user]) ) )
	if "hell" in distances:
		ps_hell.append( hits_hell / float(topN) )
		rs_hell.append( hits_hell / float( len(consumo[id_user]) ) )
	if "kl" in distances:
		ps_kl.append( hits_kl / float(topN) )
		rs_kl.append( hits_kl / float( len(consumo[id_user]) ) )

def PRF_calc(precs, recls):
	P = sum(precs) / float( len(precs) )
	R = sum(recls) / float( len(recls) )
	try:
		F = 2.0*P*R / (P + R)
	except ZeroDivisionError:
		F = 0
	return P, R, F

def get_ids(filepath):
	ids = []
	with open(filepath, 'r', encoding='utf-8') as f:
		for line in f:
			id = line.strip().split(',')[0]
			ids.append(id)
	return ids
######################################



dictionary = corpora.Dictionary.load('./tmp/books_and_tweets.dict')

# corpus_tweets = corpora.MmCorpus('./tmp/corpus_tweets.mm')
# corpus_books  = corpora.MmCorpus('./tmp/corpus_books.mm')
corpus_all    = corpora.MmCorpus('./tmp/corpus_all_sampled.mm')

######################################
path_books  = 'TwitterRatings/items_goodreads_sampled/'
path_tweets = 'TwitterRatings/users_goodreads_sampled/'
filenames_books  = [f for f in listdir(path_books) if isfile(join(path_books, f))]
filenames_tweets = [f for f in listdir(path_tweets) if isfile(join(path_tweets, f))]

books  = complete_paths(path_books, filenames_books)
tweets = complete_paths(path_tweets, filenames_tweets)

books.sort( key=lambda x: getmtime(x) )
tweets.sort()

ids_books = get_ids("books_ids_sampled.txt")
ids_users = get_ids("users_ids_sampled.txt")


# tweets = [tweets[i] for i in [2, 6, 18, 19, 24, 25, 26, 27, 28, 29, 30]]
# ids_users = [ids_users[i] for i in [2, 6, 18, 19, 24, 25, 26, 27, 28, 29, 30]]

class MyTextCorpus(corpora.TextCorpus):
	def get_texts(self):
		if self.metadata:
			for filepath, metadata in self.input:
				yield tokenize_doc(filepath), metadata
		else:
			for filepath in self.input:
				yield tokenize_doc(filepath)


logging.info("Inicializamos corpus")
corpus_books            = MyTextCorpus()
corpus_books.metadata   = True
corpus_tweets           = MyTextCorpus()
corpus_tweets.metadata  = True
corpus_books.dictionary = dictionary
corpus_tweets.dictionary= dictionary
corpus_books.input      = list( zip( books, ids_books ) )
corpus_tweets.input     = list( zip( tweets, ids_users ) )
######################################


consumo = {}
# Creamos diccionario en donde las keys son id de usuario que tiene como valor
# una lista de items (ids) consumidos por ese usuario: consumo[user1_id] = {item1_id:rating, item2_id:rating, ...}
with open("TwitterRatings/ratings_sampled.txt", 'r', encoding='utf-8') as f:
	for line in f:
		line = line.strip().split(',')
		if line[0] not in consumo:
			consumo[ line[0] ] = {}
		try:
			consumo[ line[0] ][ line[1] ] = int(line[2])
		except ValueError:
			consumo[ line[0] ][ line[1] ] = 0


############### CREACIÓN DE MODELOS #######################
### BoW: 
# corpus_tweets & corpus_books
# ########

# ### TF-IDF:
# tfidf = models.TfidfModel(corpus_all, normalize=True)
# tfidf.save('./tmp/model.tfidf')
# # Para ser usado en LSI:
# corpus_all_tfidf = tfidf[corpus_all]
# corpora.MmCorpus.serialize('./tmp/corpus_all_tfidf.mm', corpus_all_tfidf)
# ########

# ### LSI:
# lsi_dims = 500
# lsi = models.LsiModel(corpus_all_tfidf, id2word=dictionary, num_topics=lsi_dims)
# lsi.save('./tmp/model200.lsi')
########

# ### LDA:
# # Se recomienda clippear corpus pues entrenamiento de LDA es lento
# clipped_corpus_all = utils.ClippedCorpus(corpus_all, 4000)
# lda_dims = 500
# lda = models.LdaModel(clipped_corpus_all, id2word=dictionary, num_topics=lda_dims)
# lda.save('./tmp/model200.lda')
########
############################################################


############### CARGANDO MODELOS #######################
logging.info("Cargando modelos..")

logging.info("Cargando modelo TFIDF..")
tfidf = models.TfidfModel.load('./tmp/model.tfidf')

# logging.info("Cargando modelo LSI..")
# lsi   = models.LsiModel.load('./tmp/model200.lsi')

# logging.info("Cargando modelo LDA..")
# lda   = models.LdaModel.load('./tmp/model200.lda')

# logging.info("Cargando modelo BM25..")
# bm25 = summarization.bm25.BM25( [doc for doc, _ in corpus_books] )
########################################################


################ EXPERIMENTOS #################
topNs = [5,10] # = #tp(hits) + #fp(recomendados && ¬consumidos)

# Para ser usado en Okapi BM25:
avg_idf = 0
for k in tfidf.idfs:
	avg_idf += tfidf.idfs[k]
avg_idf = avg_idf / len(tfidf.idfs)
logging.info("avg_idf={0}".format(avg_idf))

for topN in topNs:
	precs_cos_bow, recls_cos_bow  = [], []
	precs_jacc_bow, recls_jacc_bow = [], []
	precs_bm25_bow, recls_bm25_bow = [], []

	precs_cos_tfidf, recls_cos_tfidf = [], []
	precs_hell_tfidf, recls_hell_tfidf = [], []
	precs_kl_tfidf, recls_kl_tfidf = [], []

	precs_cos_lsi, recls_cos_lsi = [], []
	precs_hell_lsi, recls_hell_lsi = [], []
	precs_kl_lsi, recls_kl_lsi = [], []

	precs_cos_lda, recls_cos_lda = [], []
	precs_hell_lda, recls_hell_lda = [], []
	precs_kl_lda, recls_kl_lda = [], []

	n = 0
	# Sacamos score por cada usuario
	for user, id_user in corpus_tweets:#utils.SlicedCorpus(corpus_tweets, slice(1000,1010)):#corpus_tweets:
		if n%10==0 or n==99:####n%500==0 or n==4023: # Cada 500 usuarios imprimimos un mensaje
			logging.info("Viendo usuario #{0} de 4024".format(n+1))
		if id_user not in consumo: #REVISAR ESTO. POR QUÉ NO HABRÍA DE ESTAR EN CONSUMO EL USUARIO??
			continue
			
		n += 1
		##### BoW: cossim, jaccard, Okapi BM25
		PR_appender(topN=topN, user=user, id_user=id_user, distances=["cos", "jacc"], ps_cos=precs_cos_bow, rs_cos=recls_cos_bow, ps_jacc=precs_jacc_bow, rs_jacc=recls_jacc_bow)#, ps_bm25=precs_bm25_bow, rs_bm25=recls_bm25_bow  )
		##### TF-IDF: cossim, hellinger, kullback_leibler
		PR_appender(topN=topN, model="tf-idf", user=user, id_user=id_user, distances=["cos", "hell"], ps_cos=precs_cos_tfidf, rs_cos=recls_cos_tfidf, ps_hell=precs_hell_tfidf, rs_hell=recls_hell_tfidf)#, ps_kl=precs_kl_tfidf, rs_kl=recls_kl_tfidf )
		##### LSI: cossim, hellinger, kullback_leibler
		# PR_appender(topN=topN, model="lsi", user=user, id_user=id_user, distances=["cos", "hell"], ps_cos=precs_cos_lsi, rs_cos=recls_cos_lsi, ps_hell=precs_hell_lsi, rs_hell=recls_hell_lsi)#, ps_kl=precs_kl_lsi, rs_kl=recls_kl_lsi )
		##### LDA: cossim, hellinger, kullback_leibler
		# PR_appender(topN=topN, model="lda", user=user, id_user=id_user, distances=["cos", "hell"], ps_cos=precs_cos_lda, rs_cos=recls_cos_lda, ps_hell=precs_hell_lda, rs_hell=recls_hell_lda)#, ps_kl=precs_kl_lda, rs_kl=recls_kl_lda )

	P_cos_bow, R_cos_bow, F_cos_bow = PRF_calc(precs_cos_bow, recls_cos_bow)
	P_jacc_bow, R_jacc_bow, F_jacc_bow = PRF_calc(precs_jacc_bow, recls_jacc_bow)
	# P_bm25_bow, R_bm25_bow, F_bm25_bow = PRF_calc(precs_bm25_bow, recls_bm25_bow)
	logging.info("BoW con topN={0}. (P,R,F)_cossim=({1},{2},{3}). (P,R,F)_jacc=({4},{5},{6}). )".format(topN,P_cos_bow,R_cos_bow,F_cos_bow,P_jacc_bow,R_jacc_bow,F_jacc_bow))#,P_bm25_bow,R_bm25_bow,F_bm25_bow))

	P_cos_tfidf, R_cos_tfidf, F_cos_tfidf = PRF_calc(precs_cos_tfidf, recls_cos_tfidf)
	P_hell_tfidf, R_hell_tfidf, F_hell_tfidf = PRF_calc(precs_hell_tfidf, recls_hell_tfidf)
	# P_kl_tfidf, R_kl_tfidf, F_kl_tfidf = PRF_calc(precs_kl_tfidf, recls_kl_tfidf)
	logging.info("TF-IDF con topN={0}. (P,R,F)_cossim=({1},{2},{3}). (P,R,F)_hell=({4},{5},{6}) )".format(topN,P_cos_tfidf,R_cos_tfidf,F_cos_tfidf,P_hell_tfidf,R_hell_tfidf,F_hell_tfidf))#,P_kl_tfidf,R_kl_tfidf,F_kl_tfidf))

	# P_cos_lsi, R_cos_lsi, F_cos_lsi = PRF_calc(precs_cos_lsi, recls_cos_lsi)
	# P_hell_lsi, R_hell_lsi, F_hell_lsi = PRF_calc(precs_hell_lsi, recls_hell_lsi)
	# # P_kl_lsi, R_kl_lsi, F_kl_lsi = PRF_calc(precs_kl_lsi, recls_kl_lsi)
	# logging.info("LSI,topics={4} con topN={0}. (P,R,F)_cossim=({1},{2},{3}). (P,R,F)_hell=({5},{6},{7}) )".format(topN,P_cos_lsi,R_cos_lsi,F_cos_lsi, lsi_dims,P_hell_lsi,R_hell_lsi,F_hell_lsi))#,P_kl_lsi,R_kl_lsi,F_kl_lsi,lsi_dims))

	# P_cos_lda, R_cos_lda, F_cos_lda = PRF_calc(precs_cos_lda, recls_cos_lda)
	# P_hell_lda, R_hell_lda, F_hell_lda = PRF_calc(precs_hell_lda, recls_hell_lda)
	# # P_kl_lda, R_kl_lda, F_kl_lda = PRF_calc(precs_kl_lda, recls_kl_lda)
	# logging.info("LDA,topics={4} con topN={0}. (P,R,F)_cossim=({1},{2},{3}). (P,R,F)_hell=({5},{6},{7}) )".format(topN,P_cos_lda,R_cos_lda,F_cos_lda, lda_dims,P_hell_lda,R_hell_lda,F_hell_lda))#,P_kl_lda,R_kl_lda,F_kl_lda,lda_dims))
###############################################





















########################## EXAMPLE ####################################
# c            = MyTextCorpus()
# dictionary = corpora.Dictionary.load('./tmp/tut_ex.dict')
# c.dictionary = dictionary
# c.metadata   = False
# c.input      = documents # = Tutorial 1 gensim documentation
# c.metadata   = True
# ids = [1, 2, 3, 4, 55, 65, 123, 67, 444, 9990]
# c.input      = list(zip(documents, ids))

# new_doc = "Human computer interaction"
# new_vec = dictionary.doc2bow( preprocessing.preprocess_string(new_doc) )
# doc0_vec = dictionary.doc2bow( preprocessing.preprocess_string(documents[0]) )
# index   = similarities.MatrixSimilarity( [doc for doc, _ in c] )
# sims    = index[new_vec]
# logging.info(list(enumerate(sims)))
#######################################################################
# OK





