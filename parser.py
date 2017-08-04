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
	for filename in os.listdir( os.path.join(DATA_PATH, BOOKS_PATH) ):
		print("Parseando libro {0}..".format(filename))
		book_data = {}
		
		with open( os.path.join(DATA_PATH, BOOKS_PATH, filename), 'r' , encoding="utf-8") as fp:
			soup = BeautifulSoup(fp, 'html.parser')

		"""Title"""
		print("viendo título..")
		href = soup.find('link', rel="canonical").get('href') # string
		goodreadsId = soup.find('input', id="book_id").get('value') # string
		titleOfficial = soup.find('h1', id="bookTitle").get_text().strip().split('\n')[0] # string
		titleGreytext = soup.find('a', class_="greyText").get_text().strip() # string
		titleGreytextHref = soup.find('a', class_="greyText").get('href') # string (href)
		titleOg = soup.find('meta', {"property":'og:title'}).get('content') # string

		"""Authors"""
		print("viendo autores..")
		authors_element = soup.find_all("a", class_="authorName")
		nAuthors = len( authors_element ) # int
		authorMulti = False # bool
		auth_el = soup.find("div", id="aboutAuthor").find("div", class_="readable")
		if auth_el.find("a") != None:#"more".lower() in auth_el.find("a").get_text().lower():
			authorBio = auth_el.find("span", style="display:none").get_text().strip() # string (text)
		else:
			authorBio = auth_el.get_text().strip()
		if nAuthors > 1: authorMulti = True

		authors = []
		for el in authors_element:
			authorHref = el.get("href") # string (href)
			authorName = el.get_text() # string
			authorGoodreads = False
			sibling_one = el.find_next_sibling()
			sibling_two = el.find_next_sibling().find_next_sibling()
			if sibling_one.name == 'span':
				if "goodreads author" in sibling_one.get_text().lower():
					authorGoodreads = True
				else:
					authorRole = sibling_one.get_text().strip('()')
			if sibling_two.name == 'span':
				authorRole = sibling_one.get_text().strip('()')

			author = {'authorGoodreads' : authorGoodreads,
								'authorName': authorName,
								'authorRole': authorRole,
								'authorHref': authorHref}
			authors.append(author) # array

		"""Ratings"""
		print("viendo ratings..")
		ratingStarscore = 0 # int
		for el in soup.find("span", class_="stars staticStars").contents:
			ratingStarscore += int( el.get("class")[1].strip('p') )
		ratingNum = int( soup.find("span", class_="value-title", itemprop="ratingCount").get("title")) # int
		ratingRevNum = int( soup.find_all("span", class_="value-title")[-1].get("title") ) # int
		ratingAvg = float( soup.find("span", class_="average", itemprop="ratingValue").get_text() ) # float
		data = soup.find("span", id="rating_graph").get_text()
		rating5, rating4, rating3, rating2, rating1 = list( map(int, data[data.find("[")+1 : data.find("]")].split(',')) ) # int x 5
		s = soup.find("a", id="rating_details_tip").find_next_sibling("script").string
		ratingPctPplLiked = int ( re.search(r"(\d+)<\\/span>% of people liked it", s).group(1) )
		
		"""Description"""
		print("viendo descripción..")
		descr_el = soup.find("div", id="description")
		if descr_el.find("a") != None:
			descriptionText = descr_el.find("span", style="display:none").get_text().strip() # TEXT
		else:
			descriptionText = descr_el.find("span").get_text().strip()

		"""Details"""
		print("viendo detalles..")
		detailBookFormatType = soup.find("span", itemprop="bookFormatType").get_text() # string
		s = soup.find("span", itemprop="numberOfPages").get_text()
		detailNoOfPages = int( re.search(r'\d+', s).group(0) ) # int

		"""Also Enjoyed By Readers"""
		print("viendo AEBR..")
		readersBookIds = []
		readers_el = soup.find("div", {"id" : re.compile("relatedWorks-*")} )
		for el in related_el.find("div", class_="carouselRow").find("ul").find_all("li", class_="cover"):
			readersBookIds.append( re.search("\d+", el.get("id")).group(0) ) # array of strings

		"""Books By Same Author"""
		print("viendo BBSA..")
		booksBySameAuthor = []
		for el in soup.find_all("div", class_="tooltipTrigger"):
			booksBySameAuthor.append( el.get("data-resource-id") ) # array of strings

		"""Genres"""
		print("viendo géneros..")
		genres = []
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

		"""Quotes"""
		print("viendo quotes..")
		quotes = []
		quotes_el = soup.find("div", class_="h2Container", text=re.compile("Quotes")).find_next_sibling(class_="bigBoxBody")
		for el in quotes_el.find("div", class_="bigBoxContent").find_all("div", class_="stacked"):
			quoteText = el.find("span", class_="readable").get_text() # text
			s = el.find("nobr").find("a", class_="actionLinkLite").get_text()
			quoteVotes = int( re.search("\d+", s).group(0) ) # int
			quote = {'quoteText': quoteText,
							 'quoteVotes': quoteVotes}
			quotes.append(quote) # array 

		print("GENERANDO DICT..")
		book_data = {
			'href': href,
			'goodreadsId': goodreadsId,
			'description': descriptionText,
			'title': {
				'titleOfficial': titleOfficial,
				'titleGreytext': titleGreytext,
				'titleGreytextHref': titleGreytextHref,
				'titleOg': titleOg
			},
			'author': {
				'authorMulti': authorMulti,
				'authorNum': nAuthors,
				'authorBio': authorBio,
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
				'detailBookFormatType': detailBookFormatType,
				'detailNoOfPages': detailNoOfPages
			},
			'readersPreferences': readersBookIds,
			'booksBySameAuthor': booksBySameAuthor,
			'genres': genres,
			'quotes': quotes
		}

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
# url = "https://www.goodreads.com/book/show/35200635"
# page = requests.get(url).text
# soup3 = BeautifulSoup(page, 'html.parser')


# [...]
