"""
Filtra líneas problemáticas o no útiles de 
los archivos i-*.txt en TwitterRatings/users_interactions/
"""

#--------------------------------#
# Logging
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
#--------------------------------#
import re
#--------------------------------#
# File tools
from os import listdir
from os.path import isfile, join, getmtime
#--------------------------------#

path_interacs = 'TwitterRatings/users_interactions/'
filenames_interacs = [f for f in listdir(path_interacs) if isfile(join(path_interacs, f))]
filenames_interacs.sort()

for i in range(0, len(filenames_interacs)):
	with open(path_interacs+filenames_interacs[i], 'r', encoding='utf-8') as interacs:
		filedata = interacs.read().split("\n")

		for line in list(filedata):
			splitted = line.strip().split(',',1)

			"""
			Se remueven líneas no compuestas de <rating>,<book_title>
			o que el primer caracter no sea dígito
			"""
			if len(splitted) != 2 or not splitted[0].isdigit() :
				filedata.remove(line)

			"""
			Se remueven líneas en las que el primer número
			sea mayor a 5
			"""
			if splitted[0].isdigit():
				if int(splitted[0]) > 5:
					filedata.remove(line)


	filedata = "\n".join(filedata)

	with open(path_interacs+filenames_interacs[i], 'w') as f:
		f.write(filedata)
