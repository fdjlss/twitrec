"""
Grafica distribución de libros por usuario
a partir de books_distribution.txt
"""

from os import listdir
from os.path import isfile, join
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

distribution = []

with open('TwitterRatings/books_distribution_sampled.txt', 'r') as f:
	for line in f:
		distribution.append( line.strip().split(',') )

# Convierte la cantidad de libros consumidos en int
distribution = list(map(lambda x: [x[0], int(x[1])], distribution))

distribution.sort( key=lambda x: x[1] )

x = [duple[0] for duple in distribution]
y = [duple[1] for duple in distribution]

# plt.xlabel("Users IDs")
# plt.ylabel("# Consumed Books")

distribution = pd.DataFrame({ "User ID": x,
															"Consumed Books": y })

distribution.sort(columns='Consumed Books', inplace=True, ascending=False)

ax = distribution.plot(y='Consumed Books', kind='bar')
# _ = ax.set_xticklabels(distribution['User ID'])
plt.show()