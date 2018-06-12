# coding=utf-8
"""
Renombra los archivos de los tweets de los usuarios (en tweets_dir)
por el nombre más id del usuario:

<screen_name>.txt --> <screen_name>-<userId>.txt

siguiendo el formado de los archivos de los JSON descargados (en jsons_dir).
Para representar al usuario por sus tweets y ser usado en evaluaciones word2vec.
"""

import os
import json

jsons_dir = "/home/jschellman/tesis/TwitterRatings/goodreads_renamed/"
tweets_dir = "/home/jschellman/tesis/TwitterRatings/users_goodreads/"

for filename in os.listdir(jsons_dir):
	screen_name = filename.split('-')[0]
	userId      = filename.split('-')[1].split('.')[0]
	os.rename(tweets_dir+screen_name+'.txt', tweets_dir+screen_name+"-"+userId+'.txt')