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
							'33861714',
							'12939011',
							'54016',
							'830512',
							'191373',
							'2506868',
							'29542527',
							'13159923',
							'1254890',
							'71543',
							'607046',
							'1342445',
							'212412',
							'63036',
							'2442',
							'33153950',
							'32178697',
							'30831296',
							'29283286',
							'29283284',
							'13496222',
							'15781696',
							'16070117',
							'33861718',
							'33861722',
							'21393921',
							'33861742',
							'33861740',
							'33861745',
							'33861755',
							'33861757',
							'33861761',
							'33861763',
							'33861766',
							'33861767',
							'33861768',
							'33861769',
							'33861770',
							'33861771',
							'191216',
							'282733',
							'191217',
							'282735',
							'282736',
							'201196',
							'144378',
							'545015',
							'545016',
							'191215',
							'287261',
							'287237',
							'262321',
							'262322',
							'287263',
							'191218',
							'262319',
							'287262',
							'262320',
							'287260',
							'287258',
							'287264',
							'287257',
							'191220',
							'287259',
							'35832083',
							'35229486',
							'31842429',
							'24962379',
							'35261686',
							'1063064',
							'11022',
							'27417',
							'29582',
							'191219',
							'10258481',
							'191213',
							'191212',
							'9639938',
							'527523',
							'616692',
							'4365071',
							'26804520',
							'251912',
							'26000953',
							'25999785',
							'25411428',
							'24203851',
							'23480770',
							'22839677',
							'22839669',
							'20564669',
							'20564662',
							'20564656',
							'17336139',
							'17336131',
							'17336146',
							'17336117',
							'8877365',
							'264384',
							'2274632',
							'16241133',
							'342827',
							'23819220',
							'1935834',
							'61794',
							'18386',
							'53061',
							'4900',
							'34510',
							'64218',
							'33',
							'64216',
							'64217',
							'34504',
							'34499',
							'68378',
							'23454',
							'259836',
							'16160797',
							'25735012',
							'31392867',
							'11588995',
							'33127189',
							'18214414',
							'853510',
							'11713386',
							'23848959',
							'55227',
							'378201',
							'12618',
							'20949792',
							'5805',
							'12868617',
							'16328',
							'11056011',
							'6533771',
							'13633818',
							'25982608',
							'22323227',
							'22323226',
							'9799626',
							'4043732',
							'13647304',
							'12955471',
							'5651056',
							'3550170',
							'23181736',
							'7622729',
							'2256729',
							'48654',
							'612143',
							'71464',
							'243685',
							'1896046',
							'2323476',
							'209594',
							'10638859',
							'390562',
							'18747430',
							'25913024',
							'956325',
							'46581',
							'472331',
							'63034',
							'4069',
							'20493532',
							'1725523',
							'87665',
							'9533',
							'509784',
							'41821',
							'2602148',
							'71299',
							'122418',
							'122401',
							'122438',
							'122447',
							'122408',
							'122437',
							'122421',
							'71300',
							'122441',
							'122420',
							'122444',
							'122449',
							'122448',
							'122443',
							'71294',
							'71301',
							'6544413',
							'1452958',
							'18406662',
							'96358',
							'15997',
							'1371',
							'1715',
							'18107518',
							'2052',
							'386372',
							'7723797',
							'7841672',
							'23277518',
							'23987967',
							'17286721',
							'10127019',
							'18743109',
							'1471198',
							'54479',
							'10664113',
							'1452955',
							'25098265',
							'30474',
							'25342024',
							'1034948',
							'21489251',
							'20498618',
							'7132211',
							'817992',
							'21332599',
							'13155775',
							'916114',
							'25307',
							'57401',
							'24653486',
							'11125',
							'960',
							'968',
							'23492220',
							'1345610',
							'23311937',
							'1178230',
							'23878',
							'9712',
							'23875',
							'11030',
							'11033',
							'785481',
							'87186',
							'54076',
							'90413',
							'11031',
							'11028',
							'937913',
							'63032',
							'49195',
							'246259',
							'1254347',
							'18736925',
							'320286',
							'15025688',
							'106590',
							'16121',
							'112469',
							'20820982',
							'7826309',
							'1089868',
							'6933892',
							'8562239',
							'9488716',
							'189761',
							'174709',
							'596354',
							'43105',
							'43106',
							'677840',
							'21147993',
							'7815360',
							'16285233',
							'55399',
							'5096',
							'1345609',
							'1782787',
							'66842',
							'15881',
							'16161133',
							'2825644',
							'2552',
							'29396',
							'71298',
							'6394645',
							'6931292',
							'2256766',
							'112760',
							'184419',
							'175516',
							'5129',
							'639787',
							'7613',
							'343',
							'76778',
							'608474',
							'5246',
							'1554',
							'43035',
							'28862',
							'24128',
							'15645',
							'102868',
							'13006',
							'12957',
							'46787',
							'49552',
							'12996',
							'1622',
							'8852',
							'1420',
							'7624',
							'18135',
							'5470',
							'189783',
							'119322',
							'15575',
							'4631',
							'263215',
							'759965',
							'145655',
							'202601',
							'1055588',
							'1446942',
							'977019',
							'977017',
							'105399',
							'1237300',
							'12617',
							'66841',
							'16117911',
							'5043',
							'16099176',
							'6957725',
							'332980',
							'1599538',
							'178799',
							'6402364',
							'1202',
							'6892870',
							'5060378',
							'2429135',
							'2166088',
							'7032076',
							'650901',
							'8078106',
							'89230',
							'12914',
							'7144',
							'12857',
							'17125',
							'196006',
							'7062170',
							'7282903',
							'64533',
							'1322523',
							'55148',
							'5599860',
							'5599865',
							'60437',
							'60438',
							'60432',
							'1381',
							'33514',
							'8518400',
							'16343',
							'99383',
							'91479',
							'91478',
							'91476',
							'91477',
							'47212',
							'10569',
							'6148028',
							'7260188',
							'2767052',
							'5094',
							'34084',
							'43615',
							'12158480',
							'6484128',
							'234225',
							'34507',
							'34506',
							'34497',
							'24213',
							'17245',
							'18490',
							'40395',
							'2349856',
							'377965',
							'11006684',
							'141270',
							'1219103',
							'13497',
							'62291',
							'10572',
							'13496',
							'76120',
							'7194',
							'30933',
							'31352',
							'9532',
							'7967',
							'375802',
							'229355',
							'19351',
							'151191',
							'4381',
							'5107',
							'90414',
							'1367415',
							'1334181',
							'7332',
							'5907',
							'26435',
							'17267',
							'11149',
							'76688',
							'76679',
							'41810',
							'30016',
							'76683',
							'30013',
							'41811',
							'29580',
							'29581',
							'29579',
							'3590',
							'8921'

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
