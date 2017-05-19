#--------------------------------#
import json
from os import listdir
from os.path import isfile, join

import urllib.request
#--------------------------------#


def reviews_wgetter(user_review_path):
	"""
	Recibe un usuario y 
	"""
	
	data_json = json.load(data)
	pass

def books_wgetter(book_path):
	pass

def users_wgetter(user_twitter_path):
	pass



path_jsons = 'TwitterRatings/goodreads_renamed/'
json_titles = [ f for f in listdir(path_jsons) if isfile(join(path_jsons, f)) ]

for i in range(0, len(json_titles)):

	with open(path_jsons+json_titles[i], 'r') as f:
		data_json = json.load(f)

	for j in range(0, len(data_json)):
		url_review = data_json[j]['entities']['urls'][-1]['expanded_url']
		screen_name = data_json[j]['user']['screen_name']

		print( "Obteniendo HTML del Tweet {1}/{2}. Usuario: {0}, {3}/{4}.".format( screen_name, j, len(data_json), i, len(json_titles) ) )

		file_name = url_review.split('/')[-1]
		urllib.request.urlretrieve( url_review, file_name )
