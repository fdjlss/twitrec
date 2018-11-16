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
						'8699986',
						'20136524',
						'25864535',
						'36505519',
						'40874032',
						'25667918',
						'38363799',
						'32758901',
						'36636727',
						'28675440',
						'18490567',
						'18046624',
						'42178541',
						'1296090',
						'7056969',
						'13261812',
						'21901765',
						'40901227',
						'32617610',
						'21531761',
						'5973711',
						'38644410',
						'38326698',
						'8695',
						'41003387',
						'6224605',
						'23534312',
						'386162',
						'32109569',
						'38464980',
						'10837197',
						'40040034',
						'40677464',
						'19054062',
						'245233',
						'9359913',
						'35495952',
						'15730101',
						'3392123',
						'34731718',
						'32708658',
						'4981',
						'10644930',
						'25735618',
						'23168817',
						'38205798',
						'20616776',
						'39664007',
						'6643770',
						'822338',
						'35850901',
						'25342750',
						'36316380',
						'30555488',
						'497499',
						'36376249',
						'28818849',
						'21327',
						'21324',
						'25875934',
						'1260901',
						'7274961',
						'38787',
						'9980049',
						'214608',
						'8507879',
						'156182',
						'21094608',
						'21792828',
						'15773662',
						'13330370',
						'13214',
						'1220491',
						'28204540',
						'25210549',
						'22358464',
						'18809235',
						'30301478',
						'34057427',
						'11289',
						'25893783',
						'17929637',
						'20706317',
						'30231743',
						'297673',
						'25352226',
						'9796633',
						'31392851',
						'22822858',
						'30631883',
						'12398221',
						'12041255',
						'3977',
						'116254',
						'20518872',
						'58832',
						'77565',
						'2075382',
						'13147230',
						'6050678',
						'7094569',
						'17262203',
						'20697410',
						'18966806',
						'18007564',
						'3158265',
						'6954438',
						'25785993',
						'33950022',
						'10221487',
						'23173542',
						'26836761',
						'29229326',
						'32861475',
						'28687036',
						'30419644',
						'32445731',
						'13629345',
						'635150',
						'141828',
						'27737',
						'1539711',
						'15639',
						'52318',
						'32926680',
						'751275',
						'32570518',
						'21330',
						'17910622',
						'791246',
						'31447858',
						'22078240',
						'636402',
						'41074',
						'4407',
						'60926',
						'233099',
						'18934851',
						'94486',
						'21937699',
						'1716643',
						'25330405',
						'27245657',
						'24700989',
						'63034',
						'18375252',
						'12084781',
						'31675270',
						'24666002',
						'1804388',
						'30348199',
						'730626',
						'18633212',
						'15775767',
						'27876506',
						'8810',
						'27181416',
						'816',
						'76171',
						'20441724',
						'7768400',
						'22733729',
						'77566',
						'34767693',
						'375802',
						'23093359',
						'87280',
						'5043',
						'88077',
						'21325',
						'17690',
						'192377',
						'33404423',
						'21329',
						'32178469',
						'3636',
						'167010',
						'26823982',
						'21326',
						'18601315',
						'10204484',
						'18373310',
						'19503292',
						'6324799',
						'10807515',
						'9721812',
						'8357481',
						'24107971',
						'6371258',
						'5727046',
						'12306359',
						'29478503',
						'20657611',
						'23923594',
						'29532211',
						'6681037',
						'7047824',
						'24037092',
						'5470',
						'29579',
						'76778',
						'13453029',
						'11169036',
						'9517',
						'29605141',
						'31116669',
						'26067601',
						'12489810',
						'8477057',
						'1690450',
						'29885370',
						'25320981',
						'12264214',
						'6845703',
						'4709237',
						'25760538',
						'92570',
						'92572',
						'20797851',
						'29881951',
						'24978609',
						'10642888',
						'19274013',
						'640311',
						'22444375',
						'6388990',
						'18682567',
						'25241787',
						'23012579',
						'1008335',
						'7619398',
						'17404026',
						'1220493',
						'6375845',
						'29881757',
						'16146156',
						'21532179',
						'45732',
						'17671909',
						'15798376',
						'27839513',
						'23306469',
						'33472',
						'24486333',
						'19239597',
						'13573235',
						'5805',
						'18631082',
						'15802147',
						'22808339',
						'21921122',
						'57948',
						'13037817',
						'16237451',
						'12104789',
						'17861416',
						'23149638',
						'65259',
						'59980',
						'13529854',
						'16270364',
						'7829082',
						'28946503',
						'3046901',
						'8908',
						'2709688',
						'216363',
						'25597560',
						'22182096',
						'2739296',
						'28454554',
						'26818036',
						'25707123',
						'24955358',
						'19022383',
						'9967113',
						'3908540',
						'28423615',
						'18386812',
						'605684',
						'28293',
						'1520869',
						'12418289',
						'3743385',
						'27566056',
						'780541',
						'230514',
						'13557008',
						'8301077',
						'18814258',
						'170448',
						'10808860',
						'27738424',
						'92644',
						'10210',
						'14506',
						'3392482',
						'769660',
						'4526',
						'9717',
						'87302',
						'103796',
						'896535',
						'63137',
						'7022969',
						'59151',
						'122528',
						'67006',
						'60990',
						'3867',
						'119001',
						'20320287',
						'6690845',
						'2258782',
						'9642879',
						'17137635',
						'23588759',
						'38333',
						'4422157',
						'382975',
						'418235',
						'17202179',
						'1228958',
						'527756',
						'33',
						'612963',
						'17898637',
						'60756',
						'241524',
						'2159845',
						'52828',
						'1075344',
						'349830',
						'30672',
						'31196',
						'14201',
						'415',
						'18806561',
						'25716374',
						'5305807',
						'418233',
						'225697',
						'465543',
						'6421982',
						'7178',
						'9516',
						'10762697',
						'25273399',
						'25487239',
						'9833206',
						'16078215',
						'1393700',
						'18630542',
						'7321911',
						'1782828',
						'1174976',
						'1626066',
						'1626065',
						'1626062',
						'982314',
						'16171259',
						'751608',
						'22808340',
						'17445120',
						'13265983',
						'10354233',
						'7939077',
						'5594873',
						'2189427',
						'17790667',
						'2059770',
						'2657',
						'21885391',
						'12726963',
						'9681679',
						'6945489',
						'1773295',
						'6139539',
						'24026290',
						'6362973',
						'2424202',
						'22328',
						'604635',
						'71565',
						'2189436',
						'17673034',
						'12118449',
						'10117252',
						'10117248',
						'14896896',
						'7815',
						'21797116',
						'12767075',
						'17186272',
						'12002427',
						'10797026',
						'17186357',
						'865',
						'3392149',
						'165556',
						'71300',
						'122449',
						'89321',
						'71292',
						'287505',
						'352383',
						'257213',
						'837812',
						'17961',
						'22911',
						'17689',
						'74256',
						'16156431',
						'1457234',
						'6950028',
						'15195',
						'27736',
						'13726787',
						'4250',
						'9714145',
						'54741',
						'5226295',
						'9781772',
						'18808936',
						'11386576',
						'2189426',
						'2166148',
						'7231703',
						'53932',
						'56867',
						'3320531',
						'10733001',
						'99805',
						'2678240',
						'6438735',
						'6076184',
						'847067',
						'20612891',
						'21450505',
						'6570742',
						'1511725',
						'1453311',
						'27779',
						'15698',
						'239154',
						'6319674',
						'21854344',
						'6591772',
						'1816869',
						'7167740',
						'3873116',
						'11386718',
						'6137789',
						'2429135',
						'6181',
						'1243154',
						'18490',
						'11428441',
						'10127019',
						'15822571',
						'6828896',
						'8980957',
						'17062785',
						'7935732',
						'9481362',
						'11251863',
						'19470303',
						'8545154',
						'18898065',
						'11301989',
						'18925136',
						'2195464',
						'23875',
						'18135',
						'1420',
						'8852',
						'12996',
						'2506868',
						'341644',
						'771941',
						'157993',
						'5297',
						'94351',
						'19689802',
						'7686895',
						'14059482',
						'333538',
						'916114',
						'59503',
						'293579',
						'17181247',
						'35740',
						'60030',
						'6213658',
						'771983',
						'1755359',
						'1043247',
						'6148028',
						'15881',
						'2',
						'5',
						'6',
						'2539',
						'2538',
						'47668',
						'53969',
						'11991',
						'11989',
						'110457',
						'426504',
						'9771553',
						'96358',
						'485894',
						'320',
						'191373',
						'60142',
						'6796708',
						'23529',
						'11870085',
						'1975322',
						'17855756',
						'17306293',
						'11297',
						'4929',
						'10357575',
						'91953',
						'49552',
						'77203',
						'118944',
						'472331',
						'17470674',
						'2767052',
						'34',
						'3',
						'7937843',
						'77727',
						'15196',
						'19030845',
						'17245',
						'4671'
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
