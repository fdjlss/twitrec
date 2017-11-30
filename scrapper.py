# coding=utf-8

"""
1) Descarga los reviews (HTML) dado los datos de Hamid (JSON)
2) Descarga los libros (HTML) dado los reviews
3) Crea set de training y de test para el CF

X) Mientras hace todo, crea y guarda en DB (SQLite) todo
"""


#--------------------------------#
# Parsear los JSON (dataset de Hamid)
import json
# Ingresar en directorios
from os import listdir
from os.path import isfile, join
# Para scrapping
# import urllib.request
# import urlparse
# import httplib
# Logging
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
# Conexion con BD
import sqlite3
# Parsear HTML descargado
from bs4 import BeautifulSoup
# Random numbers para fechas desconocidas. Sample para selección random de ítems candidatos en caso que sobren
from random import randint, sample
from jojFunkSvd import consumption
from statistics import mean, stdev
import collections
#--------------------------------#

#-----"PRIVATE" METHODS----------#
# Esto no es estrictamente necesario, es sólo para
# que los nombres de archivos de los HTML guardados
# sean consistentes.
# También sirve para identificar 
def unshorten_url(url):
	"""
	Devuelve la URL expandida en caso que se le pase
	una URL acortada (bit.ly, goo.gl, etc..).
	Devuelve la URL en caso que no haya redirección.
	"""
	parsed = urlparse.urlparse(url)
	h = httplib.HTTPConnection(parsed.netloc)
	h.request('HEAD', parsed.path)
	response = h.getresponse()
	if response.status/100 == 3 and response.getheader('Location'):
		return response.getheader('Location')
	else:
		return url

def chunks(seq, num):
  avg = len(seq) / float(num)
  out = []
  last = 0.0
  while last < len(seq):
    out.append(seq[int(last):int(last + avg)])
    last += avg
  return out

def relevance(user, q):
	ratings = [ v[0] for k, v in user.items() ]
	if q>=10:
		return mean(ratings)
	return ((0.5**q) * stdev(ratings)) + mean(ratings)
#--------------------------------#

def reviews_wgetter(path_jsons, db_conn):
	"""
	Recibe direccion de directorio de documentos JSON 
	y el objeto de la conexión de la BD 
	"""

	c = db_conn.cursor()

	# Creacion de la tabla en la BD: user_reviews(user_id, url_review, rating)
	table_name = 'user_reviews'
	col_user_id = 'user_id'
	col_url = 'url_review'
	col_rating = 'rating'

	c.execute( 'CREATE TABLE IF NOT EXISTS {0} ({1} {2}, {3} {4} PRIMARY KEY, {5} {6})'\
	.format(table_name, \
					col_user_id, 'INTEGER', \
					col_url, 'TEXT', \
					col_rating, 'INTEGER') )


	# Listando el contenido del directorio <path_jsons>/
	json_titles = [ f for f in listdir(path_jsons) if isfile(join(path_jsons, f)) ]

	for i in range(0, len(json_titles)):

		with open(path_jsons+json_titles[i], 'r') as f:
			# Recuperando toda la info del documento
			data_json = json.load(f)

		for j in range(0, len(data_json)):
			# Guardando texto del tweet
			tweet  = data_json[j]['text']

			# Guardando URL de la opinion del usuario en GR 
			try:
				url_review = data_json[j]['entities']['urls'][-1]['expanded_url']
				url_review = unshorten_url( url_review )
			except Exception as e:
				logging.info("¡Tweet con contenido NO predefinido!")
				continue

			# Guardando username del usuario en Twitter
			screen_name = data_json[j]['user']['screen_name']

			# Guardando ID del usuario en Twitter
			user_id = data_json[j]['user']['id']

			logging.info( "Obteniendo HTML del Tweet {1}/{2}. Usuario: {0}, {3}/{4}.".format( screen_name, j, len(data_json), i, len(json_titles) ) )

			# Guardando en disco el HTML crawleado de url_review
			file_name = url_review.split('/')[-1] # Cortamos después del último '/' de la URL
			file_name = file_name.split('?')[0] # Cortamos después del primer '?' de la URI
			save_path = "/mnt/f90f82f4-c2c7-4e53-b6af-7acc6eb85058/crawling_data/goodreads_crawl/user_reviews/" + file_name + ".html"

			# Intentando ingresar a la URL
			# Si no es accesible o si no corresponde a ruta de GR, 
			# sigue con el próximo tweet
			if "goodreads.com/review" in url_review:
				try:
					urllib.request.urlretrieve( url_review, save_path )
				except Exception as e:
					logging.info("No se pudo ingresar al sitio!")
					continue
			else:
				logging.info("Enlace no es ruta de review de GR")
				continue

			# Abriedo HTML recién guardado para capturar el rating
			with open( save_path ) as fp:
				soup = BeautifulSoup(fp, 'html.parser')

			# Guardamos el rating
			# A veces en GR no se renderiza el HTML que incluye el rating (why? dunno), 
			# pero sí está el rating puesto en el Tweet ("1 out of 5 stars to [...]")..
			# ..en esos casos se usa un regex para capturar el rating desde el texto del tweet.
			# Si todo falla guardamos el rating como 0, sólo indicando que el usuario
			# consumió aquel item (presuponiendo de que si aparece la URL del review en el tweet
			# es porque el item fue consumido) 
			try:
				rating = int( soup.div(class_='rating')[0].find_all('span', class_='value-title')[0]['title'] )
			# En caso que no encuentre rating en la ruta del review (porque no puede encontrar 
			# la rewview o porque no hay estrellitas donde debiera estar el rating)...
			except Exception as e:
				try:
					#..lo capturo con un regex desde el tweet
					match  = re.search(r"(\d+) of (\d+) stars", tweet.lower())
					rating = int( match.group(1) )
					if rating > 5 or rating < 0: 
						rating = 0
				except Exception as er:
					rating = 0

			# Insertando tupla (user_id, url_review, rating) en la BD
			try:
				c.execute( "INSERT INTO {0} ({1}, {2}, {3}) VALUES (?, ?, ?)" \
					.format(table_name, col_user_id, col_url  , col_rating), \
								 						 (user_id    , file_name, rating) )
			except sqlite3.IntegrityError:
				logging.info( 'ERROR: URI de review ya existe: {}'.format(file_name) )

		# Manda los cambios al final de pasar por todos los tweets de cada usuario
		db_conn.commit()

def add_column_book_url(db_conn, alter_table=False):

	db_conn.row_factory = lambda cursor, row: row[0]

	c = db_conn.cursor()

	table_name = 'user_reviews'
	col_book = 'url_book'
	reviews_path = "/mnt/f90f82f4-c2c7-4e53-b6af-7acc6eb85058/crawling_data/goodreads_crawl/user_reviews/"

	# Creamos columna que contiene las URL de los libros en la tabla de consumos
	if alter_table:
		c.execute( "ALTER TABLE {0} ADD COLUMN {1} {2}".format(table_name, col_book, 'TEXT') )

	c.execute( "SELECT url_review FROM {0}".format(table_name) )
	all_rows = c.fetchall()

	i = 0
	for url_review in all_rows:
		logging.info("Viendo fila {0} de {1}".format(i, len(all_rows)) )
		i+=1

		with open( reviews_path+url_review+'.html', 'r') as fp:
			soup = BeautifulSoup(fp, 'html.parser')

		try:
			url_book = soup.div(class_='bookTitle')[0].get('href')
		except Exception as e:
			logging.info("URL DE LIBRO NO ENCONTRADO: {}".format(e))
			logging.info("Encontrado HTML conflictivo: {}".format(url_review))
			
			with open("non_user_reviews_htmls.txt", 'a+') as f:
				f.write( "{0}\n".format(url_review) )

			continue

		try:
			c.execute( "UPDATE {0} SET {1} = '{2}' WHERE url_review = '{3}'"\
				.format(table_name,
								col_book,
								 url_book,
								 url_review))
		except sqlite3.IntegrityError:
			logging.info( 'ERROR ACTUALIZANDO VALORES'.format(file_name) )


	db_conn.commit()

def books_wgetter(db_conn):
	"""
	Descarga HTML de libros a partir de los url_book de la DB
	"""
	db_conn.row_factory = lambda cursor, row: row[0]

	c = db_conn.cursor()

	table_name = 'user_reviews'
	col_book = 'url_book'
	goodreads_url = "https://www.goodreads.com"


	# Obtiene lista única de URL de libros consumidos por usuarios
	c.execute( "SELECT DISTINCT {0} FROM {1}".format(col_book, table_name) )
	books_urls = c.fetchall()

	i = 0
	for book_url in books_urls:

		logging.info( "VIENDO LIBRO {0}... {1} DE {2}".format(book_url, i, len(books_urls)) )
		i+=1

		# Algunas URL son NULL: no se pudo rescatar el URL del libro a partir de la URL de la review 
		try:
			url = goodreads_url+book_url
		except TypeError as e:
			logging.info( "TypeError: book_url es NULL" )
			continue

		# Intenta descargar HTML del libro dado el url_book de la tabla
		# OJO: aún así descarga el HTML redireccionado en caso de error al ingresar a la ruta
		try:
			file_name = book_url.split('/')[-1] 
			save_file = "/mnt/f90f82f4-c2c7-4e53-b6af-7acc6eb85058/crawling_data/goodreads_crawl/books_data/" + file_name + ".html"
			urllib.request.urlretrieve( url, save_file )
		except Exception as e:
			logging.info( "NO PUDO ACCEDERSE A LIBRO {0}, Error: {1}".format(book_url, e) )
			continue

def add_column_timestamp(db_conn, alter_table=False):
	"""
	Agrega columna timestamp a la tabla según la fecha
	de consumo parseada del url_review del usuario
	"""

	c = db_conn.cursor()

	table_name = 'user_reviews'
	col_timestamp = 'timestamp'
	reviews_path = "/mnt/f90f82f4-c2c7-4e53-b6af-7acc6eb85058/crawling_data/goodreads_crawl/user_reviews/"

	if alter_table:
		c.execute( "ALTER TABLE {0} ADD COLUMN {1} {2}".format(table_name, col_timestamp, 'INTEGER') )

	c.execute( "SELECT * FROM {0}".format(table_name) )
	all_rows = c.fetchall()

	i = 0
	for tupl in all_rows:
		logging.info( "-> Viendo tupla {0} de {1}. Usuario: {2}, Review: {3}".format(i, len(all_rows), tupl[0], tupl[1]) )
		i+=1

		try:
			with open(reviews_path+tupl[1]+'.html', 'r') as fp:
				soup = BeautifulSoup(fp, 'html.parser')
		except Exception as e:
			logging.info( "No se pudo abrir HTML {0}. Error: {1}".format(tupl[1], e) )
			continue

		try:
			date = int( soup.div(class_='dtreviewed')[0].find_all('span', class_='value-title')[0]['title'].replace('-', '') )
		except Exception as e:
			logging.info( "No se pudo parsear fecha" )
			continue

		try:
			c.execute( "UPDATE {0} SET {1} = '{2}' WHERE user_id = {3} AND url_review = '{4}'"\
				.format( table_name,
								 col_timestamp,
								 date,
								 tupl[0],
								 tupl[1] ))
		except sqlite3.IntegrityError:
			logging.info( 'ERROR ACTUALIZANDO VALORES'.format(file_name) )
			continue

	db_conn.commit()

def create_users_table(path_jsons, db_conn):
	c = db_conn.cursor()

	# Creacion de la tabla en la BD: users(id, screen_name)
	table_name = 'users'
	col_id = 'id'
	col_screen_name = 'screen_name' #The screen name, handle, or alias that this user identifies themselves with. screen_names are unique but subject to change
	col_followers = 'followers' #The number of followers this account currently has. Under certain conditions of duress, this field will temporarily indicate “0”.
	col_friends = 'friends' #The number of users this account is following (AKA their “followings”)
	col_location = 'location' #Nullable . The user-defined location for this account’s profile. Not necessarily a location,
	col_lang = 'lang' #The BCP 47 code for the user’s self-declared user interface language. May or may not have anything to do with the content of their Tweets.
	col_favourites = 'favourites' #The number of Tweets this user has liked in the account’s lifetime.
	col_tweets = 'tweets' #The number of Tweets (including retweets) issued by the user. 
	
	col_description = 'description' #Nullable . The user-defined UTF-8 string describing their account.
	col_listed = 'listed' #The number of public lists that this user is a member of.
	col_utc = 'utc_offset' #Nullable . The offset from GMT/UTC in seconds.
	col_name = 'name' #The name of the user, as they’ve defined it. Not necessarily a person’s name.
	col_time_zone = 'time_zone'


	c.execute( 'CREATE TABLE IF NOT EXISTS {0} ({1} {2} PRIMARY KEY, {3} {4}, {5} {6}, {7} {8}, {9} {10}, {11} {12}, {13} {14}, {15} {16}, {17} {18}, {19} {20}, {21} {22}, {23} {24}, {25} {26})'\
	.format(table_name, \
					col_id, 'INTEGER', \
					col_screen_name, 'TEXT', \
					col_followers, 'INTEGER', \
					col_friends, 'INTEGER', \
					col_location, 'TEXT', \
					col_lang, 'TEXT', \
					col_favourites,'INTEGER', \
					col_tweets, 'INTEGER', \
					col_description, 'TEXT', \
					col_listed, 'INTEGER', \
					col_utc, 'INTEGER', \
					col_name, 'TEXT', \
					col_time_zone, 'TEXT'))

	# Listando el contenido del directorio <path_jsons>/
	json_titles = [ f for f in listdir(path_jsons) if isfile(join(path_jsons, f)) ]

	for i in range(0, len(json_titles)):

		with open(path_jsons+json_titles[i], 'r') as f:
			# Recuperando toda la info del documento
			data_json = json.load(f)

		user_id          = data_json[-1]['user']['id']
		screen_name      = data_json[-1]['user']['screen_name']
		followers_count  = data_json[-1]['user']['followers_count']
		friends_count    = data_json[-1]['user']['friends_count']
		location         = data_json[-1]['user']['location']
		lang             = data_json[-1]['user']['lang']
		favourites_count = data_json[-1]['user']['favourites_count']
		tweets_count     = data_json[-1]['user']['statuses_count']
		description      = data_json[-1]['user']['description']
		listed_count     = data_json[-1]['user']['listed_count']
		utc_offset       = data_json[-1]['user']['utc_offset']
		name             = data_json[-1]['user']['name']
		time_zone        = data_json[-1]['user']['time_zone']

		# Insertando tupla en la BD:
		try:
			c.execute( "INSERT INTO {0} ({1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}, {10}, {11}, {12}, {13}) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)" \
				.format(table_name, col_id, col_screen_name, col_followers, col_friends, col_location, col_lang, col_favourites, col_tweets, col_description, col_listed, col_utc, col_name, col_time_zone), \
							 						 (user_id, screen_name, followers_count, friends_count, location, lang, favourites_count, tweets_count, description, listed_count, utc_offset, name, time_zone) )
		except sqlite3.IntegrityError:
			logging.info( 'ERROR: Usuario ya ingresado: {}'.format(userid) )

		# Manda los cambios al final de pasar por todos los tweets de cada usuario
		db_conn.commit()

def ratings_maker(db_conn, folds, out_path):
	"""
	Guarda un set de entrenamiento y un set de test a partir
	de datos de la DB
	"""
	c = db_conn.cursor()
	table_name = 'user_reviews'

	
	c.execute( "SELECT DISTINCT *\
							FROM {table_name}\
							WHERE timestamp IS NOT NULL\
							AND url_book IS NOT NULL\
							GROUP BY url_review\
							ORDER BY timestamp asc".format(table_name=table_name) ) 
	all_rows = c.fetchall()

	interactions = []
	logging.info("-> Iterando sobre resultado de la consulta..")
	for tupl in all_rows:
		user_id, url_review, rating, url_book, timestamp = tupl
		# Book ID es el número incluido en la URI del libro en GR
		# Hay veces que luego deĺ número le sigue un punto o un guión,
		# y luego el nombre del libro separado con guiones
		book_id = url_book.split('/')[-1].split('-')[0].split('.')[0]
		interactions.append( (user_id, book_id, rating, int(timestamp)) )
	
	lists = chunks(seq=interactions, num=folds)

	logging.info("Guardando validation folds y training aggregated folds..")
	for i in range(0, folds-1):
		with open(out_path+'val/val.'+str(i+1), 'w') as f:
			f.write( '\n'.join('%s,%s,%s' % x[:-1] for x in lists[i]) ) # x[:-1] : no guardamos el timestamp

		with open(out_path+'train/train.'+str(i+1), 'w') as f:
			f.write( '\n'.join('%s,%s,%s' % x[:-1] for l in lists[:i] + lists[i+1:] for x in l) )

	logging.info("Guardando test..")
	with open(out_path+'test/test.'+str(folds), 'w') as f:
		f.write( '\n'.join('%s,%s,%s' % x[:-1] for x in lists[-1]) )

	logging.info("Guardando train (total-test)..")
	with open(out_path+'ratings.train', 'w') as f:
		f.write( '\n'.join('%s,%s,%s' % x[:-1] for l in lists[:-1] for x in l) )

	logging.info("Guardando total..")
	with open(out_path+'ratings.total', 'w') as f:
		f.write( '\n'.join('%s,%s,%s' % x[:-1] for x in interactions) )


def evaluation_set(db_conn, M, N, folds, out_path):
	"""
	Guarda un set de entrenamiento y un set de test a partir
	de datos de la DB
	"""
	c = db_conn.cursor()
	table_name = 'user_reviews'
	c.execute( "SELECT DISTINCT *\
							FROM {table_name}\
							WHERE user_id IN (SELECT user_id\
											FROM user_reviews\
											GROUP BY user_id\
											HAVING COUNT(*) > {M})\
							AND timestamp IS NOT NULL\
							AND url_book IS NOT NULL\
							GROUP BY url_review\
							ORDER BY timestamp asc".format(table_name=table_name, M=M) )
	all_rows = c.fetchall()
	users = {}
	logging.info("-> Iterando sobre resultado de la consulta 1..")
	for tupl in all_rows:
		user_id, url_review, rating, url_book, timestamp = tupl
		book_id = url_book.split('/')[-1].split('-')[0].split('.')[0]
		if user_id not in users:
			users[user_id] = {}
		users[user_id][int(timestamp)] = ( rating, book_id )

	c.execute( "SELECT DISTINCT *\
							FROM {table_name}\
							WHERE timestamp IS NOT NULL\
							AND url_book IS NOT NULL\
							GROUP BY url_review\
							ORDER BY timestamp asc".format(table_name=table_name) )
	all_rows = c.fetchall()
	everything = {}
	for tupl in all_rows:
		user_id, url_review, rating, url_book, timestamp = tupl
		book_id = url_book.split('/')[-1].split('-')[0].split('.')[0]
		if user_id not in everything:
			everything[user_id] = {}
		everything[user_id][book_id] = rating		

	eval_test_set = {}
	eval_train_set = {}
	"""Construcción Test Set"""
	for user_id in users:
		user_train_set = {} 
		user_test_set = {}
		od = collections.OrderedDict(sorted(users[user_id].items(), reverse=True))
		user_ratings = { v[1] : v[0] for k, v in od.items() }

		step = 0
		while len(user_test_set) != N:

			last_items_selected = {}
			for timestamp, tupl in od.items():
				if (tupl[0] >= relevance(user= od, q= step)) and (tupl[1] not in user_test_set.keys()): 
					last_items_selected[tupl[1]] = tupl[0]

			i = 0
			while True:
				try:
					user_test_set.update( dict(sample( last_items_selected.items(), k= N-len(user_test_set)-i )) )
				except ValueError as e:
					i += 1
					print("Sample larger than population. Trying with k={}".format(N-len(user_test_set)-i))
				else:
					break

			step += 1
			if step == 11:
				break

		if len(user_test_set) != N:
			continue 

		eval_test_set[user_id] = user_test_set 
		# eval_train_set[user_id] = { item_id : user_ratings[item_id] for item_id in set(user_ratings) - set(user_test_set) }
	
	"""Construcción Train Set: chanto todo lo que no esté en el Test Set"""
	for user_id in everything:
		if user_id in eval_test_set:
			for item_id in everything[user_id]:
				if item_id in eval_test_set[user_id]:
					continue
				else:
					if user_id not in eval_train_set:
						eval_train_set[user_id] = {}
					eval_train_set[user_id][item_id] = everything[user_id][item_id]

	"""Construcción Folds de train/validation"""
	lists = [ [] for _ in range(0, folds-1) ]
	i = randint(0, folds-1)
	for user_id in eval_train_set:
		for item_id in eval_train_set[user_id]:
			i += 1
			lists[ i%(folds-1) ].append( (user_id, item_id, eval_train_set[user_id][item_id]) )

	logging.info("Guardando test..")
	with open(out_path+'test/test_N'+str(N)+'.data', 'w') as f:
		for user, d in eval_test_set.items():
			for item, rating in d.items():
				f.write( '{user},{item},{rating}\n'.format(user=user, item=item, rating=rating) )

	logging.info("Guardando train..")
	with open(out_path+'eval_train_N'+str(N)+'.data', 'w') as f:
		for user, d in eval_train_set.items():
			for item, rating in d.items():
				f.write( '{user},{item},{rating}\n'.format(user=user, item=item, rating=rating) )

	logging.info("Guardando validation folds y training aggregated folds..")
	for i in range(0, folds-1):
		with open(out_path+'val/val_N'+str(N)+'.'+str(i+1), 'w') as f:
			f.write( '\n'.join('%s,%s,%s' % x[:-1] for x in lists[i]) ) # x[:-1] : no guardamos el timestamp

		with open(out_path+'train/train_N'+str(N)+'.'+str(i+1), 'w') as f:
			f.write( '\n'.join('%s,%s,%s' % x[:-1] for l in lists[:i] + lists[i+1:] for x in l) )

	eval_all_set = {}
	eval_all_set.update(eval_test_set)
	eval_all_set.update(eval_train_set)
	logging.info("Guardando total..")
	with open(out_path+'eval_all_N'+str(N)+'.data', 'w') as f:
		for user, d in eval_all_set.items():
			for item, rating in d.items():
				f.write( '{user},{item},{rating}\n'.format(user=user, item=item, rating=rating) )

def users_wgetter(user_twitter_path):
	pass

def statistics(db_conn):
	"""Hay que correrlo en python 2.x"""
	c = db_conn.cursor()
	table_name = 'user_reviews'
	c.execute( "SELECT * FROM {0} ORDER BY timestamp asc".format(table_name) )
	all_rows = c.fetchall()
	interactions = []
	logging.info("-> Iterando sobre resultado de la consulta..")
	for tupl in all_rows:
		user_id, url_review, rating, url_book, timestamp = tupl
		try:
			book_id = url_book.split('/')[-1].split('-')[0].split('.')[0]
		except AttributeError as e:
			logging.info( "url_book es NULL en la DB! Tratado de obtener desde la review {0}".format(url_review) )
			continue
		interactions.append( (user_id, book_id, rating, int(timestamp)) )

	ratings = [ tuple[2] for tuple in interactions ]
	users = [ tuple[0] for tuple in interactions ]
	users = set(users)
	logging.info( "MEAN ratings:{avgrat}±{stddevrat}".format(avgrat=mean(ratings), stddevrat=stddev(ratings)) )

	c.execute( "SELECT user_id, COUNT(rating), timestamp FROM {} GROUP BY user_id ORDER BY timestamp asc".format(table_name) )
	all_rows = c.fetchall()
	results = []
	for tupl in all_rows:
		user_id, ratings, timestamp= tupl
		if timestamp != None: results.append( (user_id, ratings) )

	rats_usr = [ tuple[1] for tuple in results ]
	logging.info( "MEAN ratings per user:{avgratusr}±{stddevratusr}".format(avgratusr=mean(rats_usr), stddevratusr=stddev(rats_usr)) )
		
	c.execute( "SELECT url_book, COUNT(rating), timestamp FROM {} GROUP BY url_book ORDER BY timestamp asc".format(table_name) )
	all_rows = c.fetchall()
	results = []
	for tupl in all_rows:
		url_book, ratings, timestamp= tupl
		if timestamp != None: results.append( (user_id, ratings) )

	rats_book = [ tuple[1] for tuple in results ]
	logging.info( "MEAN ratings per book:{avgratbook}±{stddevratbook}".format(avgratbook=mean(rats_book), stddevratbook=stddev(rats_book)) )

  #La sparsity se puede obtener con las siguientes consultas:
  # SELECT DISTINCT COUNT(user_id) FROM user_reviews WHERE rating IS NOT 0;
  # SELECT DISTINCT COUNT(url_book) FROM user_reviews WHERE rating IS NOT 0;
  # SELECT COUNT(rating) FROM user_reviews WHERE rating IS NOT 0; 

# Creando la conexion a la BD
sqlite_file = 'db/goodreads.sqlite'
conn = sqlite3.connect(sqlite_file)

# Direccion de los archivos del dataset de Hamid
path_jsons = 'TwitterRatings/goodreads_renamed/'

# 1)
# reviews_wgetter(path_jsons, conn)
# 2)
# add_column_book_url(conn)
# 3)
# books_wgetter(conn)
# 4)
# add_column_timestamp(db_conn= conn, alter_table= True)
# 5)
# ratings_maker(db_conn= conn, folds= 5, out_path='TwitterRatings/funkSVD/data/')
# 5.1)
evaluation_set(db_conn=conn, M=10, N=5, folds=5, out_path='TwitterRatings/funkSVD/data/')
evaluation_set(db_conn=conn, M=20, N=10, folds=5, out_path='TwitterRatings/funkSVD/data/')
evaluation_set(db_conn=conn, M=30, N=15, folds=5, out_path='TwitterRatings/funkSVD/data/')
evaluation_set(db_conn=conn, M=40, N=20, folds=5, out_path='TwitterRatings/funkSVD/data/')
# 6)
# create_users_table(path_jsons= path_jsons, db_conn= conn)
# 7)
# statistics(db_conn= conn)
# Cerramos la conexion a la BD
conn.close()