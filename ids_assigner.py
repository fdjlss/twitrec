# CORRER CUANDO ds-items-crawler.py Y interacs_renamer ESTÉN LISTOS
"""
Asigna IDs únicos para libros y para usuarios.
Los guarda en archivos de salida {books_ids, users_ids}.txt
con formato "id,{Book_Title__by__Author_Name, user_screen_name}"

Incluye función que asigna timestamps a ratings
basándose en lo hallado en el dataset original (Hamed)
"""

#--------------------------------#
# Logging
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
#--------------------------------#
import json
#--------------------------------#
import re
#--------------------------------#
# File tools
from os import listdir
from os.path import isfile, join, getmtime
#--------------------------------#


path_books = 'TwitterRatings/items_goodreads/'
filenames_books = [f for f in listdir(path_books) if isfile(join(path_books, f))]

path_json  = 'TwitterRatings/goodreads_renamed/'
filenames_json  = [f for f in listdir(path_json) if isfile(join(path_json, f))]
filenames_json.sort()

path_interacs = 'TwitterRatings/users_interactions/'
filenames_interacs = [f for f in listdir(path_interacs) if isfile(join(path_interacs, f))]
filenames_interacs.sort()


######################################
def date_format(date):
	date = date.split(" ")
	year, month, day = date[5], date[1], date[2]

	month_mapper = {
		'Jan': '01',
		'Feb': '02',
		'Mar': '03',
		'Apr': '04',
		'May': '05',
		'Jun': '06',
		'Jul': '07',
		'Aug': '08',
		'Sep': '09',
		'Oct': '10',
		'Nov': '11',
		'Dec': '12'	
	}

	month = month_mapper[month]
	date = "".join([year, month, day])
	date = int(date)

	return date

def complete_paths(path, filenames):
	"""
	Devuelve lista con path completo de archivos
	"""
	return [ "{0}{1}".format(path, filenames[i]) for i in range(0, len(filenames)) ]

def title2id(path, filenames, output):
	"""
	Asigna un ID único a cada libro. Archivo de salida: "output".
	ID: posición de documento con su info en directorio que lo contiene. 
	Asignación de ID según orden de llegada (para hacer al sistema
	fácilmente extensible con la llegada de nuevos documentos).
	"""
	paths = complete_paths(path, filenames)
	paths.sort( key=lambda x: getmtime(x) )
	ids_titles = []

	for i in range(0, len(paths)):
		# ID es único dado que existe 1 solo archivo para cada libro
		title = paths[i][31:-4]#.replace('__','_') # No comentado en mi PC. Comentado en niebla.
		ids_titles.append([i, title])

	with open(output, 'w+') as f:
		for tupl in ids_titles:
			f.write("{0},{1}\n".format( tupl[0], tupl[1] ) )

	return 0

def screenName2id(path, filenames, output):
	"""
	Asigna un ID único a cada usuario. Archivo de salida: "/output".
	ID: ID de la cuenta de usuario asignado por Twitter, encontrado
	en los documentos JSON en user.id.
	"""
	paths = complete_paths(path, filenames)
	ids_users = []

	for doc in paths:

		with open(doc, 'r') as f:
			data_json = json.load(f)

		ids_users.append([ data_json[0]['user']['id'], data_json[0]['user']['screen_name'] ])

	with open(output, 'w+') as f:
		for tupl in ids_users:
			f.write("{0},{1}\n".format( tupl[0], tupl[1] ) )

	return 0

def mini_renamer(path, filenames, items_ids):
	"""
	En los archivos de /TwitterRatings/users_interactions/
	cambia los títulos de los libros por los ids dentro de books_ids.txt.
	Para facilitar la tarea de crear un solo archivo con todas las interacciones
	de todos los usuarios, luego de correr esta función debiera correr interacs_maker.py
	"""

	ids_dict = {}
	with open(items_ids, 'r', encoding='utf-8') as f:
		for line in f:
			line = line.strip().split(',',1)
			id, title = line[0], line[1]
			ids_dict[title] = id

	paths = complete_paths(path, filenames)

	for doc in paths:
		with open(doc, 'r', encoding='utf-8') as f:
			filedata = f.read().split("\n")

		for i in range(0, len(filedata)):
			if filedata[i] == '':
				continue

			# Título textual del libro
			old_title   = filedata[i].split(',',1)[1]#[2:]

			# Índice de la posición del título en el mapeo id<->título
			if old_title in ids_dict:
				id = ids_dict[old_title]
				filedata[i] = filedata[i][:2] + filedata[i][2:].replace(old_title, id)
			else:
				filedata[i] = ""

		filedata = "\n".join(filedata)

		with open(doc, 'w') as f:
			f.write(filedata)

	return 0

def interacs_maker(path, filenames, users_ids, output):
	"""
	Crea archivo con ratings en formato user_id,item_id,rating a partir
	"""
	ids_dict = {}
	with open(users_ids, 'r', encoding='utf-8') as f:
		for line in f:
			line = line.strip().split(',',1)
			id, screen_name = line[0], line[1]
			ids_dict[screen_name] = id

	# with open(users_ids, 'r', encoding='utf-8') as f:
	# 	ids = f.read().split("\n")
	# enum_ids = enumerate(ids)

	paths = complete_paths(path, filenames)
	
	interactions = []

	for i in range(0, len(paths)):
		screen_name = paths[i][36:-4] # 乁〳 ❛ д ❛ 〵ㄏ
		id          = ids_dict[screen_name]
		# index       = [i for i, s in enum_ids if screen_name in s][0]
		# id          = re.search(r'\d+', ids[index]).group()
		
		with open(paths[i], 'r', encoding='utf-8') as f:
			print(paths[i])
			for line in f:
				print(line)

				if line=='\n' or line=='':
					continue

				line = line.strip().split(',') # Ya no hay temor: números enteros. Coma sólo separa rating de item_id
				interactions.append( [id, line[1], line[0]] ) # [user_id, item_id, rating]

	with open(output, 'w+') as f:
		for triple in interactions:
			f.write("{0},{1},{2}\n".format( triple[0], triple[1], triple[2] ) )

	return 0

def compare_interactions(path, filenames, ratings, distribution):
	"""
	Función auxiliar que corrobora que en las colecciones JSON, 
	los usuarios tengan la misma cantidad de tweets (=documentos)
	que interacciones (ratings) en ratings.txt (haciendo uso de
	TwitterRatings/books_distribution.txt creado en books_distribution.py) 
	"""
	paths = complete_paths(path, filenames)

	# { user_id : [ratings] } (ratings.txt)
	dict_rats_rats = {}
	with open(ratings, 'r') as f:

		for triple in f:

			triple = triple.strip().split(',')
			user_id, book_id, rating = triple[0], triple[1], triple[2]

			if user_id not in dict_rats_rats:
				dict_rats_rats[user_id] = []

			dict_rats_rats[user_id].append(rating)


	# { user_id : #ratings } (books_distribution.txt (ratings.txt) )
	dict_dist = {}
	with open(distribution, 'r') as f:
		for duple in f:
			duple = duple.strip().split(',')
			dict_dist[duple[0]] = int(duple[1])


	# { user_id : [ratings] } (JSONs)
	dict_rats_json = {}
	# { user_id : #tweets } (JSONs)
	dict_json = {}
	for collection in paths:

		with open(collection, 'r') as f:
			data_json = json.load(f)
			user_id   = str(data_json[0]['user']['id'])
		
		dict_json[ user_id ] = len(data_json)

		if user_id not in dict_rats_json:
			dict_rats_json[user_id] = []

		for i in range(0, len(data_json)):
			try:
				tweet  = data_json[i]['text']
				match  = re.search(r"(\d+) of (\d+) stars", tweet.lower())
				rating = match.group(1)
			except Exception:
				continue

			if not rating.isdigit() or int(rating)>5 or int(rating)<=0:
				continue

			dict_rats_json[user_id].append(rating)


	############################################################################################
	diff1 = set( dict_dist.keys() ) - set( dict_json.keys() )
	diff2 = set( dict_json.keys() ) - set( dict_dist.keys() )
	print("Los usuarios que están en ratings pero no están en JSONs:")
	print(diff1, len(list(diff1)))
	print("Los que usuarios están en JSONs pero no están en ratings:")
	print(diff2, len(list(diff2)))
	print("Sacando los usuarios problemáticos..")

	for user_id in list(diff2):
		dict_json.pop(user_id)

	diffs = []
	for user_id in dict_json:
		diff_values = dict_json[user_id] - dict_dist[user_id]
		if diff_values != 0:
			diffs.append(diff_values)

	print("Cantidad de users en ratings.txt:", len(list(dict_dist.keys())) )
	print("Cantidad de users en JSONs menos los problemáticos:", len(list(dict_json.keys())) )
	print("Cantidad de users que tienen diferencias:", len(diffs))
	############################################################################################

	print("############################################################")

	############################################################################################
	diff1 = set( dict_rats_rats.keys() ) - set( dict_rats_json.keys() )
	diff2 = set( dict_rats_json.keys() ) - set( dict_rats_rats.keys() )
	print("Los usuarios que están en ratings pero no están en JSONs:")
	print(diff1, len(list(diff1)))
	print("Los que usuarios están en JSONs pero no están en ratings:")
	print(diff2, len(list(diff2)))
	print("Sacando los usuarios problemáticos..")

	for user_id in list(diff2):
		dict_rats_json.pop(user_id)

	diffs = []
	for user_id in dict_rats_json:
		len_json = len(dict_rats_json[user_id])
		len_rats = len(dict_rats_rats[user_id])
		diff = len_json - len_rats
		if diff != 0:
			diffs.append( [ diff, user_id, dict_rats_json[user_id], dict_rats_rats[user_id] ] )

	print(len(diffs))

	for i in range(0, 5):
		print(diffs[i][0], diffs[i][1], diffs[i][2], diffs[i][3])
		print("------")
	############################################################################################

def date_adder(path, filenames, ratings, out_ratings=None):
	"""
	Agrega una columna de fechas con formato YYYYMMDD
	en archivo de ratings
	"""
	paths = complete_paths(path, filenames)

	# { user_id : [ratings] } (ratings.txt)
	dict_rats_rats = {}
	with open(ratings, 'r') as f:
		for triple in f:

			triple = triple.strip().split(',')
			user_id, book_id, rating = triple[0], triple[1], triple[2]

			if user_id not in dict_rats_rats:
				dict_rats_rats[user_id] = []

			dict_rats_rats[user_id].append( "{0},{1}".format(book_id, rating) )


	# { user_id : [ratings] } (JSONs)
	dict_rats_json = {}
	for collection in paths:

		with open(collection, 'r') as f:
			data_json = json.load(f)
			user_id   = str(data_json[0]['user']['id'])
		
		if user_id not in dict_rats_json:
			dict_rats_json[user_id] = []

		for i in range(0, len(data_json)):

			try:
				tweet  = data_json[i]['text']
				match  = re.search(r"(\d+) of (\d+) stars", tweet.lower())
				rating = match.group(1)
				date   = date_format( data_json[i]['created_at'] )
			except Exception:
				continue

			if not rating.isdigit() or int(rating)>5 or int(rating)<=0:
				continue

			dict_rats_json[user_id].append("{0},{1}".format(rating, date))


	diff2 = set( dict_rats_json.keys() ) - set( dict_rats_rats.keys() )

	print("Sacando los usuarios problemáticos..")
	for user_id in list(diff2):
		dict_rats_json.pop(user_id)


	for user_id in dict_rats_rats:
		for i in range(0, len(dict_rats_rats[user_id])):
			
			book_id, rating = dict_rats_rats[user_id][i].split(',')[0], dict_rats_rats[user_id][i].split(',')[1]

			for j in range(i, len(dict_rats_json[user_id])):

				if rating == dict_rats_json[user_id][j][0]:
					#TIMESTAMP
					dict_rats_rats[user_id][i] = book_id + "," + rating + dict_rats_json[user_id][j][1:] # = "<book_id>" + "," + "<rating>" + ",<timestamp>"
					break

				else:
					continue
	
	#####################			
	# Nos aseguramos de que todos los ratings tengan fecha:
	problematicos = []
	for user_id, ratings in dict_rats_rats.items():
		prob = False
		for duple in ratings:
			if len(duple) == 1:
				prob = True

		if prob:		
			problematicos.append(user_id)

	# for user in problematicos:
	# 	print("Problemático: {0}".format(user))

	print("Los usuarios con algún rating sin fecha:", len(problematicos))
	#####################

	with open(out_ratings, 'w+') as f:
		for user_id, ratings in dict_rats_rats.items():
			for line in ratings:
				book_id, rating, timestamp = line.split(',')[0], line.split(',')[1], line.split(',')[2]

				f.write( "{0},{1},{2},{3}\n".format(user_id, book_id, rating, timestamp) )
######################################


# title2id(path_books, filenames_books, "books_ids.txt")
# print("Books IDs Listos")

# screenName2id(path_json, filenames_json, "users_ids.txt")
# print("Users IDs Listos")

# mini_renamer(path_interacs, filenames_interacs, "books_ids.txt")
# print("Mini renamer Listo")

# # Correr sólo cuando mini_renamer() haya terminado!
# interacs_maker(path_interacs, filenames_interacs, "users_ids.txt", "TwitterRatings/ratings.txt")
# print("ratings.txt Listo")

# compare_interactions(path_json, filenames_json, 'TwitterRatings/ratings_old.txt' ,'TwitterRatings/books_distribution.txt')
date_adder(path_json, filenames_json, 'TwitterRatings/ratings_old.txt', 'TwitterRatings/ratings_old_with_timestamps.txt')