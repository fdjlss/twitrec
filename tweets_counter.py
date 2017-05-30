from os import listdir
from os.path import isfile, join
import numpy as np
import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import json


# Parámetros globales de los gráficos
plot_color = "#4285f4"
plt.rcParams["font.family"] = "Arial"
##################################################################
path = u"TwitterRatings/users_goodreads_sampled/"
filenames = [f for f in listdir(path) if isfile(join(path, f))]

user_tweet_counter = [0 for i in range(0, len(filenames))]

for i in range(0,len(filenames)):
	with open(path+filenames[i], 'r', encoding="ISO-8859-1") as f:
		user_tweet_counter[i] += sum(1 for _ in f)

print("Total de tweets:", sum(user_tweet_counter))
print("Cantidad de usuarios: ", len(filenames))
user_tweet_counter.sort()

plt.plot(user_tweet_counter, color=plot_color)
axes = plt.gca()
axes.set_xlim([0, 4024])
# plt.title("Histograma de tuits por usuario (dataset actual)")
# plt.ylabel("#tweets")
# plt.xlabel("")
# plt.show()
##################################################################



##################################################################
# path_json = "TwitterRatings/goodreads_renamed/"
# filenames_json = [f for f in listdir(path_json) if isfile(join(path_json, f))]
# filenames_json.sort()

# json_counter = [0 for i in range(0, len(filenames_json))]

# for i in range(0, len(filenames_json)):
# 	with open(path_json+filenames_json[i]) as data_file:
# 		data_json = json.load(data_file)

# 	json_counter[i] = len(data_json)

# print(sum(json_counter))
# print(len(filenames_json))
# json_counter.sort()

# plt.plot(json_counter, color=plot_color)
# axes = plt.gca()
# axes.set_xlim([min(json_counter), len(json_counter)])
# plt.title("Histograma de tuits por usuario (dataset Hamed et al.)")
# plt.ylabel("#tweets")
# plt.xlabel("")
# plt.show()

# json_counter.sort( reverse=True )
# print(sum(json_counter[:808]))
# ##################################################################



# Método antiguo
# ##################################################################
# ratings = [0, 0, 0, 0, 0, 0]
# with open("TwitterRatings/ratings_sampled.txt", 'r') as f:
# 	for line in f:
# 		line = line.strip().split(",")
# 		if line[2] == '0':
# 			ratings[0] += 1
# 		if line[2] == '1':
# 			ratings[1] += 1
# 		if line[2] == '2':
# 			ratings[2] += 1
# 		if line[2] == '3':
# 			ratings[3] += 1
# 		if line[2] == '4':
# 			ratings[4] += 1
# 		if line[2] == '5':
# 			ratings[5] += 1


# plt.bar([0, 1, 2, 3, 4, 5], ratings, align='center', edgecolor="none", color=plot_color)
# axes = plt.gca()
# axes.set_xlim([0.5,5.5])
# plt.title("Distribución de ratings (muestreo)")
# plt.ylabel("Cuentas")
# plt.xlabel("Rating")
# plt.show()

# print(sum(ratings[1:]))
# ##################################################################



# Método nuevo
# ##################################################################
# Conexion con BD
# import sqlite3
# counts = [0, 0, 0, 0, 0, 0]

# sqlite_file = 'db/goodreads.sqlite'
# conn = sqlite3.connect(sqlite_file)
# conn.row_factory = lambda cursor, row: row[0]
# c = conn.cursor()

# ratings = c.execute("SELECT rating FROM user_reviews").fetchall()
# print( "#ratings = {0}".format(len(ratings)) )

# with open("TwitterRatings/ratings.txt", 'r') as f:
# 	for rating in ratings:
# 		if rating == 0:
# 			counts[0] += 1
# 		if rating == 1:
# 			counts[1] += 1
# 		if rating == 2:
# 			counts[2] += 1
# 		if rating == 3:
# 			counts[3] += 1
# 		if rating == 4:
# 			counts[4] += 1
# 		if rating == 5:
# 			counts[5] += 1


# plt.bar([0, 1, 2, 3, 4, 5], counts, align='center', color=plot_color)
# axes = plt.gca()
# axes.set_xlim([-0.5,5.5])
# plt.title("Distribución de ratings ")
# plt.ylabel("Cuentas")
# plt.xlabel("Rating")
# # plt.savefig('db/h_ratings2.png')

# print( "#ratings (no 0) = {0}".format(sum(counts[1:])) )
# print( "#ratings 1 = {0}".format(counts[1]*100/79889) )
# print( "#ratings 2 = {0}".format(counts[2]*100/79889) )
# print( "#ratings 3 = {0}".format(counts[3]*100/79889) )
# print( "#ratings 4 = {0}".format(counts[4]*100/79889) )
# print( "#ratings 5 = {0}".format(counts[5]*100/79889) )
# ##################################################################