# coding=utf-8

#--------------------------------#
# Parsear los JSON (dataset de Hamid)
import json

# Ingresar en directorios
from os import listdir
from os.path import isfile, join

# Para scrapping
import urllib

# Logging
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Conexion con BD
import sqlite3

# Parsear HTML descargado
from bs4 import BeautifulSoup
#--------------------------------#


def reviews_wgetter(path_jsons, db_c):
	"""
	Recibe direccion de y cursor de la BD 
	"""

	# Creacion de la tabla en la BD: user_reviews(user_id, url_review, rating)
	table_name = 'user_reviews'
	col_user_id = 'user_id'
	col_url = 'url_review'
	col_rating = 'rating'
	db_c.execute( 'CREATE TABLE IF NOT EXISTS {0} ({1} {2}, {3} {4}, {5} {6})'\
	.format(table_name, \
					col_user_id, 'INTEGER', \
					col_url, 'TEXT', \
					col_rating, 'INTEGER') )


	# Listando el contenido del directorio <path_jsons>/
	json_titles = [ f for f in listdir(path_jsons) if isfile(join(path_jsons, f)) ]

	for i in range(0, len(json_titles)):

		with open(path_jsons+json_titles[i], 'r') as f:
			data_json = json.load(f)

		for j in range(0, len(data_json)):
			# Guardando URL de la opinion del usuario en GR 
			url_review = data_json[j]['entities']['urls'][-1]['expanded_url']

			# Guardando username del usuario en Twitter
			screen_name = data_json[j]['user']['screen_name']
			# Guardando ID del usuario en Twitter
			user_id = data_json[j]['user']['id']

			logging.info( "Obteniendo HTML del Tweet {1}/{2}. Usuario: {0}, {3}/{4}.".format( screen_name, j, len(data_json), i, len(json_titles) ) )

			# Guardando en disco el HTML crawleado de url_review
			file_name = url_review.split('/')[-1]
			save_path = "/mnt/f90f82f4-c2c7-4e53-b6af-7acc6eb85058/crawling_data/goodreads_crawl/" + file_name + ".html"

			# Intentando ingresar a la URL
			# Si no es accesible o si no corresponde a ruta de GR, 
			# sigue con el próximo tweet
			if "goodreads" in url_review:
				try:
					urllib.urlretrieve( url_review, save_path )
				except Exception as e:
					logging.info("No se pudo ingresar al sitio!")
					continue
			else:
				continue

			
			with open( save_path ) as fp:
				soup = BeautifulSoup(fp, 'html.parser')

			# Guardamos el rating
			rating = int( soup.div(class_='rating')[0].find_all('span', class_='value-title')[0]['title'] )

			try:
				db_c.execute( "INSERT INTO {0} ({1}, {2}, {3}) VALUES ({4}, {5}, {6})"\
				.format(table_name, \
								col_user_id, \
								col_url, \
								col_rating,\
								user_id, \
								file_name, \
								rating) )
			except sqlite3.IntegrityError:
				logging.info('ERROR: Hubo un error insertando datos en la tabla {}'.format(table_name))






def books_wgetter(book_path):
	pass

def users_wgetter(user_twitter_path):
	pass


# Creando la conexion a la BD
sqlite_file = 'db/goodreads.sqlite'
conn = sqlite3.connect(sqlite_file)
c = conn.cursor()

# Direccion de los archivos del dataset de Hamid
path_jsons = 'TwitterRatings/goodreads_renamed/'


reviews_wgetter(path_jsons, c)

# Cerramos la conexion a la BD
conn.close()