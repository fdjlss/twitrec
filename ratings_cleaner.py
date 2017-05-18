"""
- Borra ratings repetidos
- Borra ratings = 0
- Borra ratings > 5
"""

def invalid_ratings(oldf, newf):
	"""
	Se copian en un nuevo archivo de texto las líneas
	que contengan necesariamente las 3: user_id, book_id y rating.
	No se copian ratings =0 ó >5
	"""

	with open(oldf, 'r') as f, open(newf, 'w') as out:

		for line in f:
			line = line.split(',')
			if len(line) == 3 and int(line[2]) != 0 and int(line[2]) <= 5:
				out.write(",".join(line))


def duped_lines(oldf, newf):
	"""
	Se copian en un nuevo archivo de texto las líneas
	únicas del archivo creado por invalid_ratings()
	"""

	lines_seen = set()

	with open(oldf, 'r') as f, open(newf, 'w') as out:

		for line in f:
			splitted_line = line.split(',')[:-1]
			ratingless_line = ",".join(splitted_line)

			if ratingless_line not in lines_seen:
				out.write(line)
				lines_seen.add(ratingless_line)



invalid_ratings('TwitterRatings/ratings.txt', 'TwitterRatings/ratings_good.txt')
duped_lines('TwitterRatings/ratings_good.txt', 'TwitterRatings/ratings_better.txt')
print("DONE!")