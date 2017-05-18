"""
Calcula distribución de consumo de libros por usuario
a partir de ratings.txt en books_distribution.txt
"""



ratings = []

with open('TwitterRatings/ratings.txt', 'r') as f:
	for line in f:
		ratings.append(line)
	
ratings.sort()


with open('TwitterRatings/books_distribution.txt', 'w+') as f:

	for i in range(0, len(ratings)):
		user = ratings[i].split(',')[0]

		if i == 0:
			current_user = user
			relative_sum = 1
			continue

		if current_user == user:
			relative_sum += 1

		if current_user != user or i == len(ratings)-1:
			f.write( "{0},{1}\n".format(current_user, str(relative_sum)) )
			relative_sum = 1
			current_user = user

