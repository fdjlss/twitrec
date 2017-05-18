# coding=utf-8
"""
Crawlea las rutas de los libros consumidos por los usuarios en la aplicación Goodreads.
Info scrappeada:
- Título
- Géneros (lista construida por los usuarios de GR)
- Descripción (reseña)
- Autor: nombre, biografía
- Fecha estreno
- Citas textuales
- Opiniones de los usuarios (máx. 10 usuarios)
Info persistida en archivos /items_goodreads/Título_Libro__by__Nombre_Autor.txt
"""

#--------------------------------#
import json
import re
import time
#--------------------------------#
from os import listdir
from os.path import isfile, join
#--------------------------------#
from selenium import webdriver
from selenium.webdriver.common.keys import Keys 
#--------------------------------#

path = "TwitterRatings/goodreads_renamed/"
filenames = [f for f in listdir(path) if isfile(join(path, f))]
filenames.sort()

path_books  = 'TwitterRatings/items_goodreads/'
book_titles  = [f[:-4] for f in listdir(path_books) if isfile(join(path_books, f))]

browser = webdriver.PhantomJS()

# count = 80054

# Recorremos cada usuario
for i in range(504,1317):#, 1317):# len(filenames)):
	with open(path+filenames[i]) as data_file:
		data_json = json.load(data_file)

	user_ratings = []
	screen_name  = data_json[0]['user']['screen_name']

	# Recorremos cada tweet del usuario
	for j in range(0, len(data_json)):
		# count += 1
		# For debugging purposes...
		print("Tweet {3}/{2} del usuario {0}: {1}".format(i, filenames[i][0:-4], len(data_json), j+1))
	



		book_info = []

		# Asumimos que la última URL del tweet es la URL del libro en goodreads
		# ..si es que el tweet es de contenido predefinido. De no ser así
		# la propiedad "urls" es un arreglo vacío.
		try:
			url = data_json[j]['entities']['urls'][-1]['expanded_url']
		except Exception as e:
			print("¡Tweet con contenido NO predefinido!")
			continue

		browser.get(url)
		try:
			title_review = browser.find_element_by_xpath("//div[@class='mainContentFloat']/h1")
			title_test   = title_review.text
			ind_gt       = title_test.find('>')
			title_test   = title_test[ ind_gt+2 : ]
			author_test  = browser.find_element_by_xpath("//a[@class='authorName']/span").text
			book_filename_test = title_test + "  by  " + author_test
			book_filename_test = book_filename_test.replace(' ', '_').replace('/', '-')

			if book_filename_test in book_titles:
				print(title_test)
				print("-- El libro ya existe! guardando sólo la interacción.. (me quedé en la review del usuario) --")
				try:
					tweet  = data_json[j]['text']
					match  = re.search(r"(\d+) of (\d+) stars", tweet.lower())
					rating = match.group(1) 
					user_ratings.append(rating+","+book_filename_test)
				except Exception:
					pass

				continue

			browser.find_element_by_class_name('bookTitle').click()
		except Exception as e:
			print("¡Hubo un error en ingresar al review del usuario o a la info. del libro!")
			continue

		#### Título
		try:
			# Muchos errores últimamente...
			title = browser.find_element_by_xpath("//h1[@id='bookTitle']").text
		except Exception:
			continue

		# For debugging purposes...
		print("{0}".format(title))
		try:
			greyed_title = browser.find_element_by_xpath("//h1[@id='bookTitle']/a").text
			title = title.replace(greyed_title, '')
		except Exception as e:
			print("-- No hay texto en gris --")
			pass
		# Remuevo leading y trailing whitespaces. 
		title = title.strip()
		book_info.append(title)

		#### Autor (1). Por propósitos de optimización
		author_name = browser.find_element_by_class_name('authorName').text
		###############################################################################
		book_filename = title + "  by  " + author_name + ".txt"
		# Remuevo "/", de lo contrario se confundiría con path nuevo
		book_filename = book_filename.replace(' ', '_').replace('/', '-')
		book_title = book_filename[:-4]
		if book_title in book_titles:
			print("-- El libro ya existe! guardando sólo la interacción.. --")
			try:
				tweet  = data_json[j]['text']
				match  = re.search(r"(\d+) of (\d+) stars", tweet.lower())
				rating = match.group(1) 
				user_ratings.append(rating+","+book_filename[:-4])
			except Exception:
				pass

			continue

		###############################################################################
		

		
		# clickeo todo los text links "more"
		try:
			more_text_links = browser.find_elements_by_link_text('...more')
			for more_text_link in more_text_links:
				more_text_link.click()
		except Exception:
			pass



		#### Géneros
		genre_containers = browser.find_elements_by_xpath("//a[@class='actionLinkLite bookPageGenreLink']")
		# genres = []
		for genre_container in genre_containers:
			# genres.append(genre_container.text)
			book_info.append(genre_container.text)

		#### Descripción
		try:
			description = browser.find_element_by_id('description').text.replace('(less)', ' ')
			book_info.append(description)		
		except Exception as e:
			print("-- No hay descripción --")
			pass
		
		### Autor (2)
		book_info.append(author_name)
		try:
			author_bio = browser.find_element_by_xpath("//div[@id='aboutAuthor']//div[@class='readable']").text.replace('(less)', '')
			book_info.append(author_bio)
		except Exception as e:
			print("-- No hay bio del autor --")
			pass

		#### Fecha
		try:
			date = browser.find_element_by_xpath("//div[@id='details']/div[2]").text
			book_info.append(date)
		except Exception as e:
			print("-- No hay fecha --")
			pass

		#### Quotes
		try:
			# quotes = []
			quote_containers = browser.find_elements_by_css_selector(".rightContainer div.bigBoxContent.containerWithHeaderContent .stacked span.readable ")
			for quote_container in quote_containers:
				# quotes.append(quote_container.text)
				book_info.append(quote_container.text)
		except Exception as e:
			print("-- No hay quotes --")
			pass

		#### Community Reviews
		try:
			review_containers = browser.find_elements_by_class_name('reviewText')
			# reviews = []
			k = 0
			for review_container in review_containers:
				# reviews.append( review_container.text.replace('(less)', '').replace('\n', ' ') )
				book_info.append( review_container.text.replace('(less)', '').replace('\n', ' ') )
				k += 1
				if k == 15:
					# Dejamos a lo más 15 reviews de la comunidad
					break
		except Exception as e:
			print("-- No hay reviews de la comunidad --")
			pass


		print("Escribiendo en archivo..")
		# Si no existe, crea. Si existe, sobreescribe para no apendizar la misma info en el mismo archivo
		with open("TwitterRatings/items_goodreads/"+book_filename, 'w+') as f:
			for info in book_info:
				f.write("%s\n" % info)

		# print(title)
		# print(description)
		# print(genres) #list
		# print(reviews) #list


		try:
			tweet  = data_json[j]['text']
			match  = re.search(r"(\d+) of (\d+) stars", tweet.lower())
			rating = match.group(1) 
			user_ratings.append(rating+","+book_filename[:-4])
		except Exception:
			pass

	user_ratings = "\n".join(user_ratings)
	print("Escribiendo interacciones de usuario {0}".format(screen_name))
	with open('TwitterRatings/users_interactions/i-' + screen_name + '.txt', 'w') as f:
		f.write(user_ratings)

browser.close()