# coding=utf-8

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import json
import os
from bs4 import BeautifulSoup
import re

DATA_PATH = "/mnt/f90f82f4-c2c7-4e53-b6af-7acc6eb85058/crawling_data/goodreads_crawl/"
BOOKS_PATH = "books_data/"
REVIEWS_PATH = "user_reviews/"

def books_parse(save_path, DATA_PATH, BOOKS_PATH):
	data = []
	i=0
	leng=len(os.listdir(os.path.join(DATA_PATH, BOOKS_PATH)) )
	for filename in os.listdir( os.path.join(DATA_PATH, BOOKS_PATH) ):
		i+=1
		print("{0} de {1}. Parseando libro {2}..".format(i, leng, filename))
		book_data = {}
		
		with open( os.path.join(DATA_PATH, BOOKS_PATH, filename), 'r' , encoding="utf-8") as fp:
			soup = BeautifulSoup(fp, 'html.parser')

		"""Title"""
		print("viendo título..", end=' ')
		href = soup.find('link', rel="canonical").get('href') # string
		goodreadsId = soup.find('input', id="book_id").get('value') # string
		titleOfficial = soup.find('h1', id="bookTitle").get_text().strip().split('\n')[0] # string
		titleGreytext = soup.find('a', class_="greyText").get_text().strip() # string
		titleGreytextHref = soup.find('a', class_="greyText").get('href') # string (href)
		titleOg = soup.find('meta', {"property":'og:title'}).get('content') # string

		"""Authors"""
		print("viendo autores..", end=' ')
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
				pass

			try:
				sibling_two = el.find_next_sibling().find_next_sibling()
				if sibling_two.name == 'span':
					authorRole = sibling_one.get_text().strip('()')
			except AttributeError as e:
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
		print("viendo ratings..", end=' ')
		ratingStarscore = 0 # int
		for el in soup.find("span", class_="stars staticStars").contents:
			ratingStarscore += int( el.get("class")[1].strip('p') )
		ratingNum = int( soup.find("span", class_="value-title", itemprop="ratingCount").get("title")) # int
		ratingRevNum = int( soup.find_all("span", class_="value-title")[-1].get("title") ) # int
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
				title = tr_el.find("div").get("title")
				rats.append( int(re.search(r"\d+", title).group(0)) )
			rating5, rating4, rating3, rating2, rating1 = rats

		s = soup.find("a", id="rating_details_tip").find_next_sibling("script").string
		ratingPctPplLiked = int ( re.search(r"(\d+)<\\/span>% of people liked it", s).group(1) )
		
		"""Description"""
		print("viendo descripción..", end=' ')
		descr_el = soup.find("div", id="description")
		try:
			if descr_el.find("a") != None:
				descriptionText = descr_el.find("span", style="display:none").get_text().strip() # TEXT
			else:
				descriptionText = descr_el.find("span").get_text().strip()
		except AttributeError as e:
			pass

		"""Details"""
		print("viendo detalles..", end=' ')
		try:
			detailBookFormatType = soup.find("span", itemprop="bookFormatType").get_text() # string
		except AttributeError as e:
			pass
		try:
			s = soup.find("span", itemprop="numberOfPages").get_text()
			detailNoOfPages = int( re.search(r'\d+', s).group(0) ) # int
		except AttributeError as e:
			pass

		"""Also Enjoyed By Readers"""
		print("viendo AEBR..", end=' ')
		readersBookIds = []
		readers_el = soup.find("div", {"id" : re.compile("relatedWorks-*")} )
		try:		
			for el in related_el.find("div", class_="carouselRow").find("ul").find_all("li", class_="cover"):
				readersBookIds.append( re.search(r"\d+", el.get("id")).group(0) ) # array of strings
		except NameError as e:
			pass

		"""Books By Same Author"""
		print("viendo BBSA..", end=' ')
		booksBySameAuthor = []
		for el in soup.find_all("div", class_="tooltipTrigger"):
			booksBySameAuthor.append( el.get("data-resource-id") ) # array of strings

		"""Genres"""
		print("viendo géneros..", end=' ')
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
		print("viendo quotes..")
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

		print("GENERANDO DICT..")
		book_data = {
			'href': href,
			'goodreadsId': goodreadsId,
			# 'description': descriptionText,
			'title': {
				'titleOfficial': titleOfficial,
				'titleGreytext': titleGreytext,
				'titleGreytextHref': titleGreytextHref,
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
					'ratingPctPplLiked': ratingPctPplLiked
				}
			},
			'detail': {
				# 'detailBookFormatType': detailBookFormatType,
				# 'detailNoOfPages': detailNoOfPages
			},
			# 'readersPreferences': readersBookIds,
			'booksBySameAuthor': booksBySameAuthor
			# 'genres': genres
			# 'quotes': quotes
		}

		try:
			book_data['description'] = descriptionText
		except NameError as e:
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
	print("DUMPEANDO JSON..")
	json.dump( data, os.path.join(save_path, "books.json" ) )


books_parse(os.path.join(DATA_PATH, "books_data_parsed"), DATA_PATH, BOOKS_PATH)

# Para la consola:
# import json
# import os
# from bs4 import BeautifulSoup
# filename='/mnt/f90f82f4-c2c7-4e53-b6af-7acc6eb85058/crawling_data/goodreads_crawl/books_data/10987.Voyager.html'
# with open(filename, 'r') as fp: soup = BeautifulSoup(fp, 'html.parser')

# import requests
# url = 'https://www.goodreads.com/book/show/77232.Legends'
# page = requests.get(url).text
# soup2 = BeautifulSoup(page, 'html.parser')
# url = "https://www.goodreads.com/book/show/20960153-entre-las-sectas-y-el-fin-del-mundo-una-noche-que-murmura-esperanzas"
# page = requests.get(url).text
# soup3 = BeautifulSoup(page, 'html.parser')


# [...]
