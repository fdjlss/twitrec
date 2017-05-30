# coding=utf-8
"""
Grafica distribución de libros por usuario
a partir de books_distribution.txt
"""

from os import listdir
from os.path import isfile, join
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


plot_color = "#4285f4"
plt.rcParams["font.family"] = "Arial"

distribution = []

with open('TwitterRatings/books_distribution_sampled.txt', 'r') as f:
	for line in f:
		distribution.append( line.strip().split(',') )

# Convierte la cantidad de libros consumidos en int
distribution = list(map(lambda x: [x[0], int(x[1])], distribution))

distribution.sort( key=lambda x: x[1] )

x = [duple[0] for duple in distribution]
y = [duple[1] for duple in distribution]
x2 = [i for i in range(0,len(y))]
# y.sort()
# # distribution = pd.DataFrame({ "User ID": x2,
# # 															"Consumed Books": y })

# distribution = pd.DataFrame(data=y, index=x2)

# # distribution.sort(columns='Consumed Books', inplace=True, ascending=True)


# ax = distribution.plot(kind='bar', stacked=False, width=1, title="Distribución de libros por usuarios", legend=False, rot=0, color=plot_color)
# ax.set_xlabel("")
# ax.set_ylabel("#Libros consumidos")

# n = 10

# ticks = ax.xaxis.get_ticklocs()
# ticklabels = [l.get_text() for l in ax.xaxis.get_ticklabels()]
# ax.xaxis.set_ticks(ticks[::n])
# ax.xaxis.set_ticklabels(ticklabels[::n])

# plt.setp(ax.get_xticklabels(), visible=True)

# plt.show()

# H_ books / users
####################
plt.plot(y, color=plot_color)
axes = plt.gca()
axes.set_xlim([min(y), len(y)])
plt.title("Distribución de libros por usuario (muestreo)")
plt.ylabel("#Libros consumidos")
plt.xlabel("")
plt.show()
#####################