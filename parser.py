# coding=utf-8

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import json
import os
from bs4 import BeautifulSoup
import re
from shutil import copyfile
import urllib
from io import open

def books_parse(save_path, DATA_PATH, BOOKS_PATH):
	data = []
	i=0
	num_books_parsed = 0
	leng=len(os.listdir(os.path.join(DATA_PATH, BOOKS_PATH)) )
	# """FOR DEBUGGING PURPOSES:"""
	# for j in range(len( os.listdir(os.path.join(DATA_PATH, BOOKS_PATH )) )-10, len( os.listdir(os.path.join(DATA_PATH, BOOKS_PATH )) )):
	# 	filename = os.listdir(os.path.join(DATA_PATH, BOOKS_PATH ))[j]
	for filename in os.listdir( os.path.join(DATA_PATH, BOOKS_PATH) ):
		i+=1
		logging.info("{0} de {1}. Parseando libro {2}..".format(i, leng, filename))
		book_data = {}
		
		with open( os.path.join(DATA_PATH, BOOKS_PATH, filename), 'r' , encoding="utf-8") as fp:
			soup = BeautifulSoup(fp, 'html.parser')

		"""href"""
		# TIENE que tener este elemento. De modo contrario el html actual es una error page
		try:
			href = soup.find('link', rel="canonical").get('href') # string
			num_books_parsed += 1 # For monitoring purposes
		except AttributeError as e:
			logging.info("Hit a bad link. Continuing..")
			continue

		"""Book ID"""
		goodreadsId = soup.find('input', id="book_id").get('value') # string
		
		"""Title"""
		logging.info("viendo título. ")
		title_el = soup.find('h1', id="bookTitle")
		titleOfficial = title_el.get_text().strip().split('\n')[0] # string
		try:
			titleGreytext = title_el.find('a', class_="greyText").get_text().strip()[1:-1] # string
			titleGreytextHref = title_el.find('a', class_="greyText").get('href') # string (href)
		except AttributeError as e:
			try:
				del titleGreytext
				del titleGreytextHref
			except Exception as e2:
				pass
		titleOg = soup.find('meta', {"property":'og:title'}).get('content') # string

		"""Authors"""
		logging.info("viendo autores. ")
		authors_element = soup.find_all("a", class_="authorName")
		nAuthors = len( authors_element ) # int
		authorMulti = False # bool
		if nAuthors > 1: authorMulti = True
		try:
			auth_el = soup.find("div", id="aboutAuthor").find("div", class_="readable")
			if auth_el.find("a") != None:#"more".lower() in auth_el.find("a").get_text().lower():
				authorBio = auth_el.find("span", style="display:none").get_text().strip() # string (text)
			else:
				authorBio = auth_el.get_text().strip()
		except Exception as e:
			try:
				del authorBio
			except Exception as e2:
				pass

		authors = []
		for el in authors_element:
			authorHref = el.get("href") # string (href)
			authorName = el.get_text() # string
			authorGoodreads = False
			try:			
				sibling_one = el.find_next_sibling()
				if sibling_one.name == 'span':
					if "goodreads author" in sibling_one.get_text().lower():
						authorGoodreads = True
					else:
						authorRole = sibling_one.get_text().strip('()')
			except AttributeError as e:
				try:
					del authorRole
				except Exception as e2:
					pass

			try:
				sibling_two = el.find_next_sibling().find_next_sibling()
				if sibling_two.name == 'span':
					authorRole = sibling_one.get_text().strip('()')
			except AttributeError as e:
				try:
					del authorRole
				except Exception as e2:
					pass

			author = {'authorGoodreads' : authorGoodreads,
								'authorName': authorName,
								# 'authorRole': authorRole,
								'authorHref': authorHref}

			try:
				author['authorRole'] = authorRole
			except NameError as e:
				pass

			authors.append(author) # array

		"""Ratings"""
		logging.info("viendo ratings. ")
		ratingStarscore = 0 # int
		for el in soup.find("span", class_="stars staticStars").contents:
			ratingStarscore += int( el.get("class")[1].strip('p') )
		try:
			ratingNum = int( soup.find("span", class_="value-title", itemprop="ratingCount").get("title") ) # int
			ratingRevNum = int( soup.find_all("span", class_="value-title")[-1].get("title") ) # int
		except AttributeError as e:
			ratingNum = int( soup.find_all("span", class_="value-title")[0].get("title") )
			ratingRevNum = int( soup.find_all("span", class_="value-title")[1].get("title") )
		ratingAvg = float( soup.find("span", class_="average", itemprop="ratingValue").get_text() ) # float
		
		try:
			graph_data = soup.find("span", id="rating_graph").get_text()
			rating5, rating4, rating3, rating2, rating1 = list( map(int, graph_data[graph_data.find("[")+1 : graph_data.find("]")].split(',')) ) # int x 5
		except AttributeError as e:
			s = soup.find("a", id="rating_details").find_next_sibling("script").string
			s1 = re.search(r"new Tip\(\$\('rating_details'\), \"\\n(.+)\\n\\n\", { style: 'goodreads'", s).group(1)
			soup_temp = BeautifulSoup(s1.replace("\\n","").replace("\\",""), 'html.parser')
			rats = []
			for tr_el in soup_temp.find("table", id="rating_distribution").find_all("tr"):
				if tr_el.find("div") is None:
					continue
				title = tr_el.find("div").get("title")
				rats.append( int(re.search(r"\d+", title).group(0)) )
			rating5, rating4, rating3, rating2, rating1 = rats

		s = soup.find("a", id="rating_details_tip").find_next_sibling("script").string
		try:
			ratingPctPplLiked = int ( re.search(r"(\d+)<\\/span>% of people liked it", s).group(1) )
		except AttributeError as e:
			try:
				del ratingPctPplLiked
			except Exception as e2:
				pass

		"""Description"""
		logging.info("viendo descripción. ")
		descr_el = soup.find("div", id="description")
		try:
			if descr_el.find("a") != None:
				descriptionText = descr_el.find("span", style="display:none").get_text().strip() # TEXT
			else:
				descriptionText = descr_el.find("span").get_text().strip()
		except AttributeError as e:
			try:
				del descriptionText
			except Exception as e2:
				pass

		"""Details"""
		logging.info("viendo detalles. ")
		try:
			detailBookFormatType = soup.find("span", itemprop="bookFormatType").get_text() # string
		except AttributeError as e:
			try:
				del detailBookFormatType
			except Exception as e2:
				pass
		try:
			s = soup.find("span", itemprop="numberOfPages").get_text()
			detailNoOfPages = int( re.search(r'\d+', s).group(0) ) # int
		except AttributeError as e:
			try:
				del detailNoOfPages
			except Exception as e2:
				pass

		"""Also Enjoyed By Readers"""
		logging.info("viendo AEBR. ")
		readersBookIds = []
		readers_el = soup.find("div", {"id" : re.compile("relatedWorks-*")} )
		try:		
			for el in related_el.find("div", class_="carouselRow").find("ul").find_all("li", class_="cover"):
				readersBookIds.append( re.search(r"\d+", el.get("id")).group(0) ) # array of strings
		except NameError as e:
			pass

		"""Books By Same Author"""
		logging.info("viendo BBSA. ")
		booksBySameAuthor = []
		for el in soup.find_all("div", class_="tooltipTrigger"):
			booksBySameAuthor.append( el.get("data-resource-id") ) # array of strings

		"""Genres"""
		logging.info("viendo géneros. ")
		genres = []
		try:
			genres_el = soup.find("div", class_="h2Container", text=re.compile("Genres")).find_next_sibling(class_="bigBoxBody")
			for el in genres_el.find("div", class_="bigBoxContent").find_all("div", class_="elementList"):
				if ">" in el.find("div", class_="left").get_text():
					genreName = el.find_all("a", class_="actionLinkLite bookPageGenreLink")[1].get_text() # string
				else:
					genreName = el.find("a", class_="actionLinkLite bookPageGenreLink").get_text()
				s = el.find("div", class_="right").get_text()
				genreVotes = int( re.search(r"\d+", s).group(0) ) # int

				genre = {'genreName': genreName,
								 'genreVotes': genreVotes}
				genres.append(genre) # array
		except AttributeError as e:
			pass

		"""Quotes"""
		logging.info("viendo quotes..")
		quotes = []
		try:
			quotes_el = soup.find("div", class_="h2Container", text=re.compile("Quotes")).find_next_sibling(class_="bigBoxBody")
			for el in quotes_el.find("div", class_="bigBoxContent").find_all("div", class_="stacked"):
				quoteText = el.find("span", class_="readable").get_text() # text
				s = el.find("nobr").find("a", class_="actionLinkLite").get_text()
				quoteVotes = int( re.search(r"\d+", s).group(0) ) # int
				quote = {'quoteText': quoteText,
								 'quoteVotes': quoteVotes}
				quotes.append(quote) # array 
		except AttributeError as e:
			pass

		logging.info("GENERANDO DICT..")
		book_data = {
			'href': href,
			'goodreadsId': goodreadsId,
			# 'description': descriptionText,
			'title': {
				'titleOfficial': titleOfficial,
				# 'titleGreytext': titleGreytext,
				# 'titleGreytextHref': titleGreytextHref,
				'titleOg': titleOg
			},
			'author': {
				'authorMulti': authorMulti,
				'authorNum': nAuthors,
				#'authorBio': authorBio,
				'authors': authors
			},
			'rating': {
				'ratingStarscore': ratingStarscore,
				'ratingNum': ratingNum,
				'ratingRevNum': ratingRevNum,
				'ratingAvg': ratingAvg,
				'ratingDetail': {
					'rating5': rating5,
					'rating4': rating4,
					'rating3': rating3,
					'rating2': rating2,
					'rating1': rating1,
					# 'ratingPctPplLiked': ratingPctPplLiked
				}
			},
			'detail': {
				# 'detailBookFormatType': detailBookFormatType,
				# 'detailNoOfPages': detailNoOfPages
			},
			# 'readersPreferences': readersBookIds,
			'booksBySameAuthor': booksBySameAuthor
			# 'genres': genres
			# 'quotes': [{quotes}]
		}

		try:
			book_data['description'] = descriptionText
		except NameError as e:
			pass

		try:
			book_data['title']['titleGreytext'] = titleGreytext
			book_data['title']['titleGreytextHref'] = titleGreytextHref
		except Exception as e:
			pass

		try:
			book_data['author']['authorBio'] = authorBio
		except NameError as e:
			pass

		try:
			book_data['readersPreferences'] = readersBookIds
		except NameError as e:
			pass

		try:
			book_data['rating']['ratingPctPplLiked'] = ratingPctPplLiked
		except NameError as e:
			pass


		try:
			book_data['detail']['detailNoOfPages'] = detailNoOfPages
		except NameError as e:
			pass

		try:
			book_data['detail']['detailBookFormatType'] = detailBookFormatType
		except NameError as e:
			pass

		try:
			book_data['quotes'] = quotes
		except NameError as e:
			pass

		try:
			book_data['genres'] = genres
		except NameError as e:
			pass

		data.append(book_data)


	# endfor
	logging.info("DUMPEANDO JSON CON {0} LIBROS PARSEADOS DE {1} DESCARGADOS..".format(num_books_parsed, leng))
	with open( os.path.join(save_path, "books.json" ), 'w', encoding="utf-8" ) as outfile:
		outfile.write(unicode( json.dumps(data, outfile, ensure_ascii=False) ))

def new_booker(new_ids_list, DATA_PATH, BOOKS_PATH, NEW_SAVES):
	# Remuevo los libros que ya descargué
	save_path      = os.path.join(DATA_PATH, BOOKS_PATH)
	saved_list     = os.listdir( save_path )
	saved_list     = [ book_url.split('.')[0].split('-')[0] for book_url in saved_list ] #capturamos el bookId desde la url ('3262719-more-than-meets-the-eye-official-guidebook-volume-2.html')
	save_path_temp = os.path.join(DATA_PATH, NEW_SAVES)

	copies = []
	for bookId in new_ids_list:
		if bookId in saved_list: 
			logging.info("{} ya está descargado".format(bookId))
			copies.append(bookId)

	for bookId in copies:
		new_ids_list.remove(bookId)

	# Descargar libros a save_path (BOOKS_PATH) y a save_path_temp (NEW_SAVES)
	prefix = 'https://www.goodreads.com/book/show/'

	i = 0
	for bookId in new_ids_list:
		i+=1
		logging.info( "DESCARGANDO LIBRO {0}... {1} DE {2}".format(bookId, i, len(new_ids_list)) )
		url = prefix+bookId

		# Intenta descargar HTML del libro dado el url_book de la tabla
		# OJO: aún así descarga el HTML redireccionado en caso de error al ingresar a la ruta
		file_name = url.split('/')[-1] 
		save_file = save_path + file_name + ".html"
		try:
			urllib.urlretrieve( url, save_file ) #Los nuevos no tendran la forma "bookId-book-name.html", sino "bookId.html"
			save_file_temp = save_path_temp + file_name + ".html"
			copyfile(save_file, save_file_temp)
		except Exception as e:
			logging.info( "URL {} es invalida..".format(url) )
			continue

	logging.info("PROCESO EXITOSO")

from goodreads import client
def get_books_from_gr_api(query, api_key, api_secret):
	gc = client.GoodreadsClient(api_key, api_secret)
	new_books = []
	for i in range(1,10):
		try:
			new_books += gc.search_books(q=query, page=i, search_field='all')
		except:
			continue

	new_books = [ book.gid for book in new_books ]
	new_books = list( set(new_books) )
	return new_books

from random import randint
def get_books_from_rng(list_len):
	new_books = []
	for _ in range(list_len):
		new_books.append( str(randint(0, 8000000)) )
	# Removemos rngs repetidos
	new_books = list( set(new_books) )

	return new_books

def main():
	DATA_PATH = "/mnt/f90f82f4-c2c7-4e53-b6af-7acc6eb85058/crawling_data/goodreads_crawl/"
	BOOKS_PATH = "books_data/"
	NEW_SAVES = "books_data_temp/"

	# books_parse(save_path= os.path.join(DATA_PATH, "books_data_parsed"), DATA_PATH= DATA_PATH, BOOKS_PATH= BOOKS_PATH)
	api_key = 'MNpblm5HetY1zSGowr0GXA'
	api_secret = 'pf5pU1MmQ8UiyIVvbdo2BbZUvK1pZhVSREYA2Ftrak'
	
	# Genera lista de bookIds desde busqueda en GR
	# ids_list = get_books_from_gr_api(query="hola", api_key= api_key, api_secret= api_secret)
	# Lista desde consumo de los usuarios del estudio
	ids_list = [
							'68226',
							'30315130',
							'34505172',
							'40801830',
							'32736921',
							'6942333',
							'40728451',
							'35215762',
							'4778447',
							'40786226',
							'36198346',
							'25803999',
							'19349634',
							'39073905',
							'2700720',
							'40788941',
							'38313362',
							'32882885',
							'40787080',
							'36647198',
							'4381',
							'65393',
							'7391371',
							'18599963',
							'19317918',
							'73785',
							'16035258',
							'35489679',
							'20815651',
							'2672483',
							'5907',
							'27284444',
							'2833978',
							'3517391',
							'2833969',
							'6117218',
							'1659167',
							'2139688',
							'2833972',
							'6944887',
							'63358',
							'984744',
							'17206506',
							'12994251',
							'8322565',
							'7776692',
							'6596839',
							'1836303',
							'36113530',
							'34666553',
							'40816809',
							'17730046',
							'36565389',
							'18001084',
							'43817',
							'13570408',
							'1173855',
							'39815443',
							'31574965',
							'32076168',
							'33151805',
							'12736350',
							'5463488',
							'29975824',
							'28068172',
							'13331099',
							'6404670',
							'35393824',
							'1950540',
							'33955945',
							'4292873',
							'9969571',
							'39093312',
							'13183688',
							'17833717',
							'31365875',
							'12951147',
							'33916599',
							'25851933',
							'30244626',
							'37752692',
							'119537',
							'25352808',
							'1412613',
							'3184678',
							'1246661',
							'1310812',
							'8479835',
							'16156537',
							'18125893',
							'10587120',
							'455592',
							'33699',
							'40268',
							'17453698',
							'17238220',
							'421379',
							'13575638',
							'13575648',
							'13575634',
							'7688227',
							'27812079',
							'5463650',
							'16067631',
							'7128363',
							'65517',
							'29794850',
							'34951841',
							'88432',
							'87819',
							'34574054',
							'4692',
							'71565',
							'1233548',
							'31842429',
							'7544706',
							'8428682',
							'34383053',
							'13518355',
							'20325243',
							'12816769',
							'894092',
							'147483',
							'1043247',
							'5470',
							'18741291',
							'15991572',
							'36383978',
							'62779',
							'17839962',
							'25227173',
							'12172',
							'1378736',
							'373556',
							'2544232',
							'52036',
							'18872806',
							'34272538',
							'23645312',
							'6745991',
							'87812',
							'25786338',
							'8958575',
							'6767951',
							'67006',
							'23002631',
							'12288158',
							'19183854',
							'686583',
							'54984'
							]

	# # Lo hacemos gradual en caso que hayan problemas
	# for i in range(10):
	# 	# Genera lista de bookIds con RNG (no todos seran goodreadsId validos)
	# 	ids_list = get_books_from_rng(list_len= 1000)

	# 	# Descarga HTMLs y ponlos en BOOKS_PATH y en NEW_SAVES
	new_booker(new_ids_list= ids_list, DATA_PATH=DATA_PATH, BOOKS_PATH=BOOKS_PATH, NEW_SAVES=NEW_SAVES)

	
	# Parsea lo del arg BOOKS_PATH y lo mete en un JSON en arg save_path 
	books_parse(save_path= os.path.join(DATA_PATH, "books_data_parsed_temp"), DATA_PATH= DATA_PATH, BOOKS_PATH= NEW_SAVES)


if __name__ == '__main__':
	main()


# Para la consola:
# import json
# import os
# from bs4 import BeautifulSoup
# filename='/mnt/f90f82f4-c2c7-4e53-b6af-7acc6eb85058/crawling_data/goodreads_crawl/books_data/10987.Voyager.html'
# with open(filename, 'r') as fp: soup = BeautifulSoup(fp, 'html.parser')

# import requests
# url = 'https://www.goodreads.com/book/show/77232.Legends'
# url = 'https://www.goodreads.com/book/show/20931395-xocal-soyg-r-m.html'
# page = requests.get(url).text
# soup2 = BeautifulSoup(page, 'html.parser')

# filename='/mnt/f90f82f4-c2c7-4e53-b6af-7acc6eb85058/crawling_data/goodreads_crawl/books_data/1028284.The_Servant_of_Two_Masters.html'
# with open(filename, 'r') as fp: soup3 = BeautifulSoup(fp, 'html.parser')


# [...]
