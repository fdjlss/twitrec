# coding=utf-8

from gensim.corpora import Dictionary
from gensim.models import KeyedVectors
from gensim.models.word2vec import Word2Vec
from gensim.parsing.preprocessing import preprocess_string, strip_tags, strip_punctuation, strip_multiple_whitespaces, strip_numeric
from gensim.similarities import WmdSimilarity
from nltk.corpus import stopwords
from textblob.blob import TextBlob
import os
import json
import numpy as np
from urllib.parse import urlencode, quote_plus
from urllib.request import urlopen
import operator
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
stop_words = set(stopwords.words('spanish') + stopwords.words('english') + stopwords.words('german') + stopwords.words('arabic') + \
								 stopwords.words('french') + stopwords.words('italian') + stopwords.words('portuguese') + ['goodreads', 'http', 'https', 'www', '"'])
CUSTOM_FILTERS = [lambda x: x.lower(), strip_tags, strip_punctuation, strip_multiple_whitespaces, strip_numeric]

# -- from WMD eval:
def get_extremes(flat_docs, n_below, n_above):
	texts    = [flat_doc for id, flat_doc in flat_docs.items() ]
	dct      = Dictionary(texts)
	word_fqs = dict((dct[i], dct.dfs[i]) for i in range(len(dct)) )
	sorted_x = sorted(word_fqs.items(), key=operator.itemgetter(1), reverse=True)
	extremes = []
	for word, freq in sorted_x:
		if freq >= n_above:
			extremes.append(word)
		if freq <= n_below:
			extremes.append(word)
	return extremes

def get_tweets_as_flat_docs(tweet_path, train_users):
	# Obtenemos sólo los tweets de los usuarios dentro del train set protocolar
	filenames = [ f for f in os.listdir(tweet_path) ]
	p_filenames = []
	for userId in train_users:
		p_filenames.append( [filename for filename in filenames if userId in filename][0] )

	# Leemos todos los tweets en strings preprocesados en una lista
	flat_docs = []
	for file in p_filenames:
		with open(tweet_path+file, 'r', encoding='ISO-8859-1') as f:
			flat_user = f.read().replace('\n', ' ')
		flat_docs.append( preprocess_string(flat_user) )

	# Dado que flat_user() recibe un dict, se debe crear un mock dict
	flat_docs = dict(enumerate(flat_docs))
	return flat_docs

def flat_doc(document, model, extremes=None):
	flat_doc = ""
	for field in document:
		if not isinstance(document[field], list): continue #No tomamos en cuenta los campos 'id' y '_version_': auto-generados por Solr
		for value in document[field]:
			## Detección y traducción ##
			if field=='author.authors.authorName' or field=='author.authorBio' or field=='description' or field=='quotes.quoteText':
				value_blob = TextBlob(value)
				try:
					if value_blob.detect_language() != 'en':
						try: 
							value = value_blob.translate(to='en')
						except Exception as e: 
							value = value #e = NotTranslated('Translation API returned the input string unchanged.',)
				except Exception as e:
					value = value #e = TranslatorError('Must provide a string with at least 3 characters.')
			############################
			flat_doc += str(value)+' ' #Se aplana el documento en un solo string
	flat_doc = preprocess_string(flat_doc, CUSTOM_FILTERS) #Preprocesa el string
	flat_doc = [w for w in flat_doc if w not in stop_words] #Remueve stop words
	if extremes:
		flat_doc = [w for w in flat_doc if w not in extremes]
	flat_doc = [w for w in flat_doc if w in model.vocab] #Deja sólo palabras del vocabulario
	if flat_doc == []:
		flat_doc = ['book'] #Si el libro queda vacío, agregarle un token para no tener problemas más adelante
	return flat_doc

def flatten_all_docs(solr, model, filter_extremes=False):
	dict_docs = {}
	url = solr + '/query?q=*:*&rows=100000' #n docs: 50,862 < 100,000
	docs = json.loads( urlopen(url).read().decode('utf8') )
	docs = docs['response']['docs']

	i = 0 
	if filter_extremes:	
		flat_docs = np.load('./w2v-tmp/flattened_docs.npy').item()		
		n_above = len(flat_docs) * 0.75
		n_below = 1
		extremes = get_extremes(flat_docs=flat_docs, n_below=n_below, n_above=n_above)
		for doc in docs:
			i+=1
			goodreadsId = str( doc['goodreadsId'][0] )
			logging.info("{0} de {1}. Doc: {2}".format(i, len(docs), goodreadsId))
			dict_docs[goodreadsId] = flat_doc(document= doc, model= model, extremes= extremes)
	
	else:
		for doc in docs:
			i+=1
			goodreadsId = str( doc['goodreadsId'][0] )
			logging.info("{0} de {1}. Doc: {2}".format(i, len(docs), goodreadsId))
			dict_docs[goodreadsId] = flat_doc(document= doc, model= model)
	del docs
	return dict_docs

def flat_user(flat_docs, consumption):
	flat_user = []
	for bookId in consumption:
		try:
			flat_user += flat_docs[bookId]
		except KeyError as e:
			# Esto pasa sólo con 2 libros hasta el momento..
			logging.info("{} ES UNO DE LOS LIBROS CUYO HTML NO PUDO SER DESCARGADO. PROSIGUIENDO CON EL SIGUIENTE LIBRO..".format(bookId))
			continue	
	return flat_user

def flat_user_as_tweets(tweets_file, model, extremes=None):
	with open(tweets_file, 'r', encoding='ISO-8859-1') as f:
		flat_user = f.read().replace('\n', ' ')
	flat_user = preprocess_string(flat_user, CUSTOM_FILTERS)
	# No hago trabajo de traducción con TextBlob, ya que tendría que tomar 
	# el string completo, lo que sería demasiado costoso
	flat_user = [w for w in flat_user if w not in stop_words] #Remueve stop words
	if extremes: 
		flat_user = [w for w in flat_user if w not in extremes]
	flat_user = [w for w in flat_user if w in model.vocab] #Deja sólo palabras del vocabulario
	return flat_user

def flatten_all_users(data_path, model, as_tweets=False, filter_extremes=False):
	train_c = consumption(ratings_path=data_path+'eval_train_N5.data', rel_thresh=0, with_ratings=False)
	dict_users = {}

	i = 0
	if as_tweets:
		tweet_path = "/home/jschellman/tesis/TwitterRatings/users_goodreads/"
		filenames  = [ f for f in os.listdir(tweet_path) ]
		flat_docs  = get_tweets_as_flat_docs(tweet_path=tweet_path, train_users= train_c)
		n_above    = len(flat_docs) * 0.75
		n_below    = 1
		extremes   = get_extremes(flat_docs= flat_docs, n_below= n_below, n_above= n_above)
		for userId in train_c:
			i+=1
			logging.info("USERS 2 VECS. AS TWEETS. {0} de {1}. User: {2}".format(i, len(train_c), userId))
			tweets_file = [filename for filename in filenames if userId in filename][0]
			dict_users[userId] = flat_user_as_tweets(tweets_file= tweet_path+tweets_file, model= model, extremes= extremes)

	else:
		if filter_extremes:	
			flat_docs = np.load('./w2v-tmp/flattened_docs_fea075b1.npy').item()
		else:
			flat_docs = np.load('./w2v-tmp/flattened_docs.npy').item()
		for userId in train_c:
			i+=1
			logging.info("USERS 2 VECS. {0} de {1}. User: {2}".format(i, len(train_c), userId))
			dict_users[userId] = flat_user(flat_docs= flat_docs, consumption= train_c[userId])

	return dict_users

def mix_user_flattening(data_path):
	train_c = consumption(ratings_path=data_path+'eval_train_N5.data', rel_thresh=0, with_ratings=False)
	flat_users_books = np.load('./w2v-tmp/flattened_users_fea075b1.npy').item() # Books
	flat_users_tweets = np.load('./w2v-tmp/flattened_users_tweets.npy').item()
	dict_users = {}

	for userId in train_c:
		dict_users[userId] = flat_users_books[userId] + flat_users_tweets[userId]

	return dict_users

# -- from w2v eval:
def doc2vec(list_document, model):
	# MAX POOLING
	matrix_doc = np.zeros((model.vector_size,), dtype=float)
	for token in list_document:
		# Debiera estar manejado en las funciones de wmd_evaluation,
		# pero habría que hacer este manejo si considero el vocab de Twitter
		# y convierto en vector con modelo Wiki, por ej..
		if token not in model.vocab: continue 
		matrix_doc = np.vstack((matrix_doc, model[token]))
	matrix_doc = np.delete(matrix_doc, 0, 0) #Elimina la primera fila de sólo ceros
	vec_doc = max_pool(np_matrix= matrix_doc)
	return vec_doc

def docs2vecs(model):
	ids2vec = {}
	flat_docs = np.load('./w2v-tmp/flattened_docs_fea075b1.npy').item()
	i = 0
	for bookId, flat_doc in flat_docs.items():
		i+=1
		logging.info("{0} de {1}. Doc: {2}".format(i, len(flat_docs), bookId))
		ids2vec[bookId] = doc2vec(list_document= flat_doc, model= model)
	return ids2vec

# -- from UCE w2v:
def recs_cleaner(solr, consumpt, recs):
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