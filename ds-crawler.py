import time
#--------------------------------#
from os import listdir
from os.path import isfile, join
#--------------------------------#
from selenium import webdriver
from selenium.webdriver.common.keys import Keys 
#--------------------------------#



###---------------- Ejemplo ---------------###
# # base_url = u'https://twitter.com/search?q='
# # query = u'communism will prevail'
# browser = webdriver.PhantomJS()
# url = u'https://twitter.com/TSM_Leffen'
# browser.get(url)
# # time.sleep(5)

# # body = browser.find_element_by_tag_name('body')

# ## Funciona para todos los otros browsers:
# # for _ in range(1):
# # 	body.send_keys(Keys.PAGE_DOWN)
# # 	time.sleep(0.2)

# ## Funciona para PhantomJS:
# for _ in range(200):
# 	browser.execute_script("window.scrollTo(0, document.body.scrollHeight);")
# 	time.sleep(0.2)

# tweets = browser.find_elements_by_class_name('tweet-text')
# tweets_list = []
# for tweet in tweets:
# 	tweets_list.append(tweet.text)
# browser.quit()

# print(len(tweets_list))
# # browser.close()
# ######################################################
# # webdriver.quit(): cierra ventana y apaga el driver #
# # webdriver.close(): cierra sólo la ventana          #
# ######################################################

# # for tweet in tweets:
# # 	print(tweet.text)
###-----------------------------------------###


browser = webdriver.PhantomJS()
base_url = u'https://twitter.com/'
n_scrolls = 200

path = "TwitterRatings/users_goodreads/"
filenames = [f for f in listdir(path) if isfile(join(path, f))]
filenames.sort()

# Iteración sobre todos los documentos guardados
for i in range(0,len(filenames)):
	# For debugging purposes...
	username = filenames[i][0:-4]
	print("{0}/{2}, usuario: {1}".format(i, username, len(filenames)) )

		
	### Scrappeamos tweets del infinite scrolling
	# Ingresamos a la ruta del usuario
	url = base_url+username

	# Atrapamos errores de conexión:
	k = 0
	max_tries = 5
	while k < max_tries:

		try:
			browser.get(url)

			# Automatic infinite scrolling
			for _ in range(n_scrolls):
				browser.execute_script("window.scrollTo(0, document.body.scrollHeight);")
				time.sleep(0.1)

			# Guardamos textos de tweets en lista
			# (Objeto WebElement (tweets) sólo accesible
			# cuando hay conexión y driver está en proceso)
			scrapped_tweets = browser.find_elements_by_class_name('tweet-text')
			scrapped_tweet_list = []
			for tweet in scrapped_tweets:
				scrapped_tweet_list.append(tweet.text.replace('\n', ' '))
			
		except Exception as e:
			k += 1
			if k==max_tries:
				print("{1} intentos. Error. Añadiendo {0} a conflictivos".format(username, max_tries))
				with open("problematicos.txt", "a") as fprobs:
					fprobs.write(username+"\n")

			# print("Error. Intento {0}/10".format(k))
			continue

		break


	# Obtenemos lista de tweets del documento
	if k < max_tries:
		with open(path+filenames[i], 'a+') as f:
			doc_tweet_list = f.read().splitlines() #readlines() para mantener los \n

			# Guardamos los tweets recuperados que no están en el documento	
			diff_tweets = list(set(scrapped_tweet_list) - set(doc_tweet_list))

			# Escribimos en screen_name.txt la info. nueva:
			print( "Escribiendo {0} tweets".format(len(diff_tweets)) )
			for tweet in diff_tweets:
				f.write("%s\n" % tweet)


# browser.close()
