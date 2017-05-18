"""
Crea un ratings_sampled.txt, creado a partir de selección aleatoria de usuarios.
Se seleccionan libros sólo de estos usuarios aleatorios.
"""

from random import sample
import shutil

"""
Creación ratings_sampled.txt
"""
users = []
with open('TwitterRatings/books_distribution.txt', 'r') as f:
	for line in f:
		users.append( line.strip().split(',')[0] )

ratings = []
with open('TwitterRatings/ratings.txt', 'r') as f:
	for line in f:
		ratings.append( line.strip().split(',') )

users_sampled = sample(users, k=100)

books_sampled = []
ratings_sampled = []

with open('TwitterRatings/ratings_sampled.txt', 'w+') as f:

	for triple in ratings:
		if triple[0] in users_sampled:
			books_sampled.append(triple[1])
			ratings_sampled.append(triple[2])

			f.write( "{0},{1},{2}\n".format(triple[0], triple[1], triple[2]) )

# Dejamos sólo los items únicos
books_sampled = list(set(books_sampled))


"""
Creación {books,users}_ids_sampled.txt
"""
with open('books_ids_sampled.txt', 'w+', encoding='utf-8') as bisf, open('books_ids.txt', 'r', encoding='utf-8') as bif:
	for duple in bif:
		duple = duple.strip().split(',', 1) # .split(',', 1) ya que hay libros con "," en el título

		if duple[0] in books_sampled:
			bisf.write( "{0},{1}\n".format(duple[0], duple[1]) )

with open('users_ids_sampled.txt', 'w+', encoding='utf-8') as uisf, open('users_ids.txt', 'r', encoding='utf-8') as uif:
	for duple in uif:
		duple = duple.strip().split(',', 1)

		if duple[0] in users_sampled:
			uisf.write( "{0},{1}\n".format(duple[0], duple[1]) )


"""
Copia de archivos sampleados a TwitterRatings/{items,users}_goodreads_sampled/
"""
usernames_sampled = []
with open('users_ids_sampled.txt', 'r', encoding='utf-8') as f:
	for line in f:
		usernames_sampled.append( line.strip().split(',', 1)[1] )

		shutil.copy2('TwitterRatings/users_goodreads/' + line.strip().split(',')[1] + '.txt', 'TwitterRatings/users_goodreads_sampled')

with open('books_ids_sampled.txt', 'r', encoding='utf-8') as f:
	for line in f:
		# Cambiar "_by_" por "__by__" (sólo el último "_by_") en mi PC. No es aśi en niebla
		book_title = line.strip().split(',', 1)[1]
		# book_title = "__by__".join(book_title.rsplit('_by_', 1))
		shutil.copy2('TwitterRatings/items_goodreads/' + book_title + '.txt', 'TwitterRatings/items_goodreads_sampled')		