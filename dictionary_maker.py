#--------------------------------#
# Logging
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
#--------------------------------#
# Gensim
from gensim import corpora
from gensim.parsing import preprocessing
#--------------------------------#
# Regex & word tools
# import re
from six import iteritems
from stop_words import get_stop_words
#--------------------------------#
# File tools
from os import listdir
from os.path import isfile, join
#--------------------------------#


######################################
#def doc_to_wordList(filepath):
	# """
	# Convierte un documento en una lista de tokens lowercased, striped
	# y con caracteres especiales removidos
	# """
	# with open(filepath, 'r', encoding='utf-8') as file:
	# 	# Regex monstruoso para no meter al diccionario ningún caracter raro
	# 	w = [ re.sub('[+\`\´\"._*¿¡!?\[\]\(\)\{\}#$%&/=\\\\:;,^~<>]', '', word) for line in file for word in line.strip().lower().split() ]
	# return w
def tokenize_doc(filepath):
	"""
	Convierte documento en una lista de tokens (porter-)stemmed,
	lowercased, striped, stop-word & special character filtered 
	"""
	with open(filepath, 'r', encoding='ISO-8859-1') as file:
		s = file.read()
	return preprocessing.preprocess_string(s)
	
# No la uso acá pero es para tener todas las funciones bien localizadas...

######################################


# Stop words
stoplist = set(get_stop_words('en') + get_stop_words('spanish') + get_stop_words('ar'))

path_books  = 'TwitterRatings/items_goodreads/'
path_tweets = 'TwitterRatings/users_goodreads/'
filenames_books  = [f for f in listdir(path_books) if isfile(join(path_books, f))]
filenames_tweets = [f for f in listdir(path_tweets) if isfile(join(path_tweets, f))]
filenames_books.sort()
filenames_tweets.sort()


#### Construyendo el diccionario #####
# Inicializamos diccionario
dictionary = corpora.Dictionary( [['hello', 'world'], ['second', 'document']] )

for i in range(0, len(filenames_books)):
	print("Extendiendo vocabulario con info de libros..{0}/{1}".format(i, len(filenames_books)) )
	# Extiendo con un documento a la vez
	dictionary.add_documents( [tokenize_doc( path_books+filenames_books[i] )], prune_at=None )

for i in range(0, len(filenames_tweets)):
	print("Extendiendo vocabulario con tweets..{0}/{1}".format(i, len(filenames_tweets)) )
	# Extiendo con un documento a la vez
	dictionary.add_documents( [tokenize_doc( path_tweets+filenames_tweets[i] )], prune_at=None )

# Remuevo stopwords y palabras que aparecen 1 sola vez
stop_ids = [dictionary.token2id[stopword] for stopword in stoplist if stopword in dictionary.token2id]
once_ids = [tokenid for tokenid, docfreq in iteritems(dictionary.dfs) if docfreq == 1]

print("Filtrando stop-words y once-words..")
dictionary.filter_tokens(stop_ids + once_ids) 
dictionary.compactify()  # remove gaps in id sequence after words that were removed
print(dictionary)

print("Guardando en disco..")
dictionary.save('./tmp/books_and_tweets.dict')
######################################
# 587,045 unique tokens