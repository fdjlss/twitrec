# CORRER CUANDO dictionary-maker.py y ids_assigner.py HAYAN CORRIDO
"""
Crea corpus para ser usado en tareas
TF-IDF, LSI, LDA, RP o HDP
"""

#--------------------------------#
# Logging
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
#--------------------------------#
# Gensim
from gensim import corpora
from gensim.parsing import preprocessing
#--------------------------------#
# File tools
from os import listdir
from os.path import isfile, join, getmtime
#--------------------------------#


######################################
#def doc_to_string(filepath):
	# """
	# Lee el documento y devuelve un string	striped 
	# y con caracteres especiales removidos
	# """
	# with open(filepath, 'r', encoding='utf-8') as file:
	# 	s = file.read().replace('\n', ' ')
	# s = re.sub('[+\`\´\"._*¿¡!?\[\]\(\)\{\}#$%&/=\\\\:;,^~<>]', '', s).strip()
	# return s
# def tokenize(text): #pasarle doc_to_string(filepath)
	# """
	# Lowercases, tokenizes, y remueve stop-words	de un string (documento)
	# """
	# # No es necesario remover palabras que aparecen 1 sola vez
	# # ya que por default allow_update=False: el diccionario
	# # no crea ids para nuevas palabras
	# stop_words = get_stop_words('en')
	# return [ token for token in simple_preprocess(text) if token not in stop_words ]

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

def get_ids(filepath):
	ids = []
	with open(filepath, 'r', encoding='utf-8') as f:
		for line in f:
			id = line.strip().split(',')[0]
			ids.append(id)
	return ids
######################################

path_books  = 'TwitterRatings/items_goodreads_sampled/'
path_tweets = 'TwitterRatings/users_goodreads_sampled/'
filenames_books  = [f for f in listdir(path_books) if isfile(join(path_books, f))]
filenames_tweets = [f for f in listdir(path_tweets) if isfile(join(path_tweets, f))]
# filenames_books.sort()
# filenames_tweets.sort()

books  = complete_paths(path_books, filenames_books)
tweets = complete_paths(path_tweets, filenames_tweets)

books.sort( key=lambda x: getmtime(x) )
tweets.sort()

ids_books = get_ids("books_ids_sampled.txt")
ids_users = get_ids("users_ids_sampled.txt")


######## Construyendo Corpus #########
class MyTextCorpus(corpora.TextCorpus):
	def get_texts(self):
		if self.metadata:
			for filepath, metadata in self.input:
				yield tokenize_doc(filepath), metadata
		else:
			for filepath in self.input:
				# yield tokenize( open(filename).read() )
				yield tokenize_doc(filepath)#tokenize( doc_to_string(filepath) )


print("Inicializamos corpus")
c_books           = MyTextCorpus()
c_books.metadata  = True
c_tweets          = MyTextCorpus()
c_tweets.metadata = True
c_all             = MyTextCorpus() # .metadata = False, es sólo para entrenar los modelos
# Cargamos diccionario creado en dictionary-maker.py
dictionary         = corpora.Dictionary.load('./tmp/books_and_tweets.dict')
c_books.dictionary = dictionary
c_tweets.dictionary= dictionary
c_all.dictionary   = dictionary
# Entregamos lista con fullpath de los archivos
c_books.input      = list( zip( books, ids_books ) )
c_tweets.input     = list( zip( tweets, ids_users ) )
c_all.input        = books+tweets
# Guardamos corpus, corpuses, corpuseses... corpi? :V
print("Guardando..")
corpora.MmCorpus.serialize('./tmp/corpus_books_sampled.mm', c_books, metadata=True)
corpora.MmCorpus.serialize('./tmp/corpus_tweets_sampled.mm', c_tweets, metadata=True)
corpora.MmCorpus.serialize('./tmp/corpus_all_sampled.mm', c_all)
######################################















########################## EXAMPLE ####################################
# # CORRER:
# filenames_js = []
# for filename in filenames_json:
# 	filename = filename.split('-')
# 	filenames_js.append(filename[0]+'.txt')

# filenames_ints = []
# for filename in filenames_interacs:
# 	filename = filename[2:]
# 	filenames_ints.append(filename)

# # LUEGO CONVERTIR A SETS filenames_tweets y filenames_js y ver 
# # qué cosa hay en uno que no está en el otro.
# # convendría borrar archivos en filenames_tweets ya que
# # la info en tweets ya está recolectada..



# TEST para REPL
# class MyTextCorpus(corpora.TextCorpus):
# 	def get_texts(self):
# 		if self.metadata:
# 			for s, m in self.input:
# 				yield preprocessing.preprocess_string(s), m
# 		else:
# 			for s in self.input:
# 				yield preprocessing.preprocess_string(s)
#######################################################################