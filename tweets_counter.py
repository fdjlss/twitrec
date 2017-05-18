from os import listdir
from os.path import isfile, join
import numpy as np
import matplotlib.pyplot as plt
import json

##################################################################
path = u"TwitterRatings/users_goodreads/"
filenames = [f for f in listdir(path) if isfile(join(path, f))]


tot_tweet_counter = 0
user_tweet_counter = [0 for i in range(0, len(filenames))]

for i in range(0,len(filenames)):
	with open(path+filenames[i], 'r', encoding="ISO-8859-1") as f:
		user_tweet_counter[i] += sum(1 for _ in f)

print(sum(user_tweet_counter))
print(len(filenames))
user_tweet_counter.sort()

plt.plot(user_tweet_counter)
plt.title("Histograma de tweets (dataset complementado)")
plt.ylabel("#tweets")
plt.xlabel("Ids usuarios")
plt.show()
##################################################################



##################################################################
path_json = "TwitterRatings/goodreads_renamed/"
filenames_json = [f for f in listdir(path_json) if isfile(join(path_json, f))]
filenames_json.sort()

json_counter = [0 for i in range(0, len(filenames_json))]

for i in range(0, len(filenames_json)):
	with open(path_json+filenames_json[i]) as data_file:
		data_json = json.load(data_file)

	json_counter[i] = len(data_json)

print(sum(json_counter))
print(len(filenames_json))
json_counter.sort()

plt.plot(json_counter)
plt.title("Histograma de tweets (dataset original)")
plt.ylabel("#tweets")
plt.xlabel("Ids usuarios")
plt.show()

json_counter.sort( reverse=True )
print(sum(json_counter[:808]))
##################################################################




##################################################################
ratings = [0, 0, 0, 0, 0, 0]
with open("TwitterRatings/ratings.txt", 'r') as f:
	for line in f:
		line = line.strip().split(",")
		if line[2] == '0':
			ratings[0] += 1
		if line[2] == '1':
			ratings[1] += 1
		if line[2] == '2':
			ratings[2] += 1
		if line[2] == '3':
			ratings[3] += 1
		if line[2] == '4':
			ratings[4] += 1
		if line[2] == '5':
			ratings[5] += 1


plt.bar([0, 1, 2, 3, 4, 5], ratings, align='center')
plt.title()
plt.ylabel("#ratings")
plt.xlabel("Rating")
plt.show()

print(sum(ratings[1:]))
##################################################################