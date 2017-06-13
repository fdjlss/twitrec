# coding=utf-8

#--------------------------------#
# Parsear los JSON (dataset de Hamid)
import json

# Ingresar en directorios
from os import listdir
from os.path import isfile, join

# Para scrapping
import urllib
# import urlparse
import httplib

# Logging
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Conexion con BD
import sqlite3

# Parsear HTML descargado
from bs4 import BeautifulSoup
#--------------------------------#

# Esto no es estrictamente necesario, es sólo para
# que los nombres de archivos de los HTML guardados
# sean consistentes.
# También sirve para identificar 
# def unshorten_url(url):
# 	"""
# 	Devuelve la URL expandida en caso que se le pase
# 	una URL acortada (bit.ly, goo.gl, etc..).
# 	Devuelve la URL en caso que no haya redirección.
# 	"""
# 	parsed = urlparse.urlparse(url)
# 	h = httplib.HTTPConnection(parsed.netloc)
# 	h.request('HEAD', parsed.path)
# 	response = h.getresponse()
# 	if response.status/100 == 3 and response.getheader('Location'):
# 		return response.getheader('Location')
# 	else:
# 		return url


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
					urllib.urlretrieve( url_review, save_path )
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
			urllib.urlretrieve( url, save_file )
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


def ratings_maker(db_conn, frac_train, frac_test, output_train, output_test):
	"""
	Guarda un set de entrenamiento y un set de test a partir
	de datos de la DB
	"""
	c = db_conn.cursor()
	table_name = 'user_reviews'

	c.execute( "SELECT * FROM {0}".format(table_name) )
	all_rows = c.fetchall()


	

def users_wgetter(user_twitter_path):
	pass


# Creando la conexion a la BD
sqlite_file = 'db/goodreads.sqlite'
conn = sqlite3.connect(sqlite_file)

# Direccion de los archivos del dataset de Hamid
path_jsons = 'TwitterRatings/goodreads_renamed/'


# reviews_wgetter(path_jsons, conn)
# add_column_book_url(conn)
# books_wgetter(conn)
add_column_timestamp(db_conn= conn, alter_table= True)
# ratings_maker(db_conn= conn, frac_train= 80, frac_test= 20, output_train= 'TwitterRatings/ratings.train', output_test= 'TwitterRatings/ratings.test')


# Cerramos la conexion a la BD
conn.close()