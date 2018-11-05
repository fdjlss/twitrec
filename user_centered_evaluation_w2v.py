# coding=utf-8

# SOLO EN PYTHON 3.X
from gensim.models import KeyedVectors
from word2vec_evaluation import doc2vec, docs2vecs
from wmd_evaluation import flat_doc, flat_user, get_extremes, flatten_all_docs
from svd_evaluation import remove_consumed
from urllib.request import urlopen
from svd_evaluation import consumption
import json
import numpy as np
from scipy import spatial
import operator
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
def w2v_recs(data_path, solr, which_model, items, userId, model):
	test_c  = consumption(ratings_path=data_path+'test/test_N20.data', rel_thresh=0, with_ratings=True)
	train_c = consumption(ratings_path=data_path+'eval_train_N20.data', rel_thresh=0, with_ratings=False)
	consumpt = [ str(itemId) for itemId, rating, auth1, auth2, auth3 in items ]
	

	# Alt 1: en docs2vec & users2vec no están los libros nuevos y el usuario nuevo, entonces hago acá el embedding manualmente y luego los guardo 
	flat_docs = np.load('./w2v-tmp/flattened_docs.npy').item()
	extremes = get_extremes(flat_docs= flat_docs, n_below= 1, n_above= len(flat_docs) * 0.75)
	flat_user_books = {}
	for itemId, rating, auth1, auth2, auth3 in items:
		logging.info("Flattening item {}".format(itemId))
		url      = solr + '/query?q=goodreadsId:' + str(itemId)
		response = json.loads( urlopen(url).read().decode('utf8') )
		doc      = response['response']['docs']
		flat_user_books[itemId] = flat_doc(document= doc[0], model= model, extremes= extremes)
	flat_dude = flat_user(flat_docs= flat_user_books, consumption= consumpt)
	embd_user = doc2vec(list_document= flat_dude, model= model)

	embd_user_books = {}
	for itemId, flat_item in flat_user_books.items():
		embd_user_books[itemId] = doc2vec(list_document= flat_item, model= model)

	docs2vec  = np.load('./w2v-tmp/'+which_model+'/docs2vec_'+which_model+'.npy').item()
	for itemId in embd_user_books:
		docs2vec[itemId] = embd_user_books[itemId]
	np.save('./w2v-tmp/'+which_model+'/docs2vec_'+which_model+'.npy', docs2vec) #

	distances = dict((bookId, 0.0) for bookId in docs2vec)
	for bookId in docs2vec:
		distances[bookId] = spatial.distance.cosine(embd_user, docs2vec[bookId])


	# Alt 2: en docs2vec y flattened_docs ya están los libros nuevos, dado que corrí el pipeline para hacer el embedding de todos los libros del index (incluídos los nuevos)
	# (sería lo adecuado si es que descargo libros adicionales de GR a parte de los libros de los usuarios de prueba)
	# flat_docs = np.load('./w2v-tmp/flattened_docs_fea075b1.npy').item()
	# flat_user_books = dict( (itemId, flat_docs[itemId]) for itemId, rating, auth1, auth2, auth3 in items )
	# flat_dude = flat_user(flat_docs= flat_user_books, consumption= consumpt)
	# embd_user = doc2vec(list_document= flat_dude, model= model)
	# docs2vec  = np.load('./w2v-tmp/'+which_model+'/docs2vec_'+which_model+'.npy').item()
	# distances = dict((bookId, 0.0) for bookId in docs2vec)
	# for bookId in docs2vec:
	# 	distances[bookId] = spatial.distance.cosine(embd_user, docs2vec[bookId])

	# LA RECOMENDACIÓN
	sorted_sims = sorted(distances.items(), key=operator.itemgetter(1), reverse=False) #[(<grId>, MENOR dist), ..., (<grId>, MAYOR dist)]
	recs   = [ bookId for bookId, sim in sorted_sims ]
	recs   = remove_consumed(user_consumption=consumpt, rec_list=recs)

	# Por si necesito usar el usuario para subsiguientes recomendaciones con w2v 
	users2vec = np.load('./w2v-tmp/'+which_model+'/users2vec_books_'+which_model+'.npy').item()
	# Si es que es user de GR: ID de GR
	# Si no: mock ID ("A0", "A1", etc..)
	users2vec[userId] = embd_user
	np.save('./w2v-tmp/'+which_model+'/users2vec_books_'+which_model+'.npy', users2vec) 

	return recs

def recs_cleaner(solr, consumpt_hrefs, recs):
	# Saca todos los items cuyos hrefs ya los tenga el usuario
	for item in reversed(recs):
		url      = solr + '/select?q=goodreadsId:' + item + '&wt=json' 
		response = json.loads( urlopen(url).read().decode('utf8') )
		doc      = response['response']['docs'][0]
		rec_href = doc['href'][0]
		if rec_href in consumpt_hrefs: recs.remove(item)

	# Saca todos los ítems con hrefs iguales
	lista_dict = {}
	for item in recs:
		url      = solr + '/select?q=goodreadsId:' + item + '&wt=json' 
		response = json.loads( urlopen(url).read().decode('utf8') )
		doc      = response['response']['docs'][0]
		rec_href = doc['href'][0]		
		if rec_href not in lista_dict:
			lista_dict[rec_href] = []
			lista_dict[rec_href].append( item )
		else:
			lista_dict[rec_href].append( item )
		
	clean_recs = recs
	rep_hrefs = []
	for href in lista_dict: lista_dict[href] = lista_dict[href][:-1]
	for href in lista_dict: rep_hrefs += lista_dict[href]

	for rep_href in rep_hrefs: clean_recs.remove(rep_href)

	return clean_recs
	

def main():
	data_path_context = 'TwitterRatings/funkSVD/data_with_authors/'
	data_path = 'TwitterRatings/funkSVD/data/'
	solr = "http://localhost:8983/solr/grrecsys"

	denis = [('8267287', 0, '5650904', '83881', '0'), 
					 ('158014', 0, '1113469', '0', '0'), 
					 ('230514', 0, '134770', '0', '0'),
					 ('19508', 0, '909675', '0', '0'),
					 ('21031', 0, '909675', '0', '0'), 
					 ('123632', 0, '38285', '100950', '574570'),
					 ('17690', 0, '5223', '0', '0'),
					 ('485894', 0, '5223', '0', '0'),
					 ('56919', 0, '30195', '0', '0'),
					 ('191373', 0, '25824', '0', '0'),
					 ('4662', 0, '3190', '0', '0'),
					 ('16207545', 0, '128195', '0', '0'),
					 ('46170', 0, '1455', '0', '0'),
					 ('9646', 0, '3706', '0', '0'),
					 ('7144', 0, '3137322', '0', '0'),
					 ('4934', 0, '3137322', '0', '0'),
					 ('63034', 0, '72039', '0', '0'),
					 ('820183', 0, '500', '0', '0'),
					 ('426504', 0, '500', '0', '0'),
					 ('338798', 0, '5144', '0', '0'),
					 ('7588', 0, '5144', '0', '0'),
					 ('847635', 0, '201906', '0', '0'),
					 ('13031462', 0, '25591', '0', '0'),
					 ('214319', 0, '51808', '0', '0')] #[ (itemId, rating, authId1, authId2, authId3) ]
	reschilling=[
								(	'84369',4,'1069006','0','0'),
								(	'65641',3,'1069006','25523','0'),
								(	'84119',4,'1069006','0','0'),
								(	'121749',3,'1069006','0','0'),
								(	'65605',4,'1069006','0','0'),
								(	'140225',3,'1069006','0','0'),
								(	'7332',5,'656983','9533','4938'),
								(	'18512',5,'656983','0','0'),
								(	'15241',5,'656983','0','0'),
								(	'29579',0,'16667','0','0'),
								(	'23566799',5,'6567626','0','0'),
								(	'13615',5,'15577045','0','0'),
								(	'5907',5,'656983','0','0'),
								(	'100915',3,'1069006','0','0'),
								(	'34',5,'656983','0','0'),
								(	'17157681',5,'656983','0','0'),
								(	'3',4,'1077326','0','0')]
	andrescarvallo = [
										('297673', 5, '55215', '0', '0'),
										('13037816', 3, '5324300', '242659', '0'),
										('60371', 2, '34031', '0', '0'),
										('60364', 3, '34031', '3874874', '0'),
										('24529201', 2, '34031', '3197885', '0'),
										('7779571', 3, '34031', '222757', '0'),
										('22041881', 3, '766034', '0', '0'),
										('11869206', 5, '766034', '0', '0'),
										('6745991', 5, '766034', '0', '0'),
										('42424958', 4, '1063732', '0', '0'),
										('40595529', 3, '4725841', '0', '0'),
										('7056969', 4, '4142981', '0', '0'),
										('859888', 5, '2934', '0', '0'),
										('7992363', 4, '3465954', '0', '0'),
										('41725366', 4, '1662945', '0', '0'),
										('24796', 3, '2238', '0', '0'),
										('522776', 3, '289953', '0', '0'),
										('5987468', 5, '706255', '4967061', '0'),
										('23212883', 4, '395812', '0', '0'),
										('10876733', 5, '43139', '0', '0'),
										('41973399', 3, '9887', '0', '0'),
										('22085341', 4, '822613', '3098123', '0'),
										('18050082', 5, '38457', '0', '0'),
										('1089597', 3, '38457', '0', '0'),
										('8062894', 5, '38457', '0', '0')
											]
	mfsepulveda = [
									('41804', 3, '16667', '0', '0'),
									('7082', 3, '4764', '0', '0'),
									('1836303', 5, '957894', '1086744', '0'),
									('2998', 3, '2041', '0', '0'),
									('52036', 4, '1113469', '0', '0'),
									('17689', 4, '877884', '3070002', '5223'),
									('485894', 5, '5223', '0', '0'),
									('157993', 5, '1020792', '0', '0'),
									('23704928', 5, '13661', '0', '0'),
									('1470844', 1, '21778', '0', '0'),
									('916114', 4, '227771', '0', '0'),
									('53447', 4, '5775606', '0', '0'),
									('11531083', 3, '13450', '0', '0'),
									('54008', 3, '25824', '0', '0'),
									('370523', 5, '13450', '0', '0'),
									('7144', 5, '3137322', '0', '0'),
									('40961427', 4, '3706', '0', '0'),
									('457264', 4, '5548', '0', '0'),
									('36525023', 5, '97783', '0', '0'),
									('2099048', 0, '35668', '0', '0'),
									('7278752', 2, '3389', '0', '0'),
									('6320534', 1, '3389', '0', '0'),
									('10614', 2, '3389', '0', '0'),
									('292228', 5, '27804', '0', '0'),
									('42428455', 4, '27804', '0', '0'),
									('67932', 5, '27804', '0', '0'),
									('16141559', 3, '27804', '0', '0'),
									('67931', 4, '27804', '0', '0')
										]
	hfvaldivieso = [
									('18927777', 0, '9640322', '0', '0'),
									('29740489', 5, '9640322', '0', '0'),
									('34948542', 0, '9640322', '0', '0'),
									('34546496', 0, '9640322', '0', '0'),
									('36406402', 0, '9640322', '0', '0'),
									('34550147', 5, '9640322', '0', '0'),
									('24872479', 5, '9640322', '0', '0'),
									('34546480', 0, '9640322', '0', '0'),
									('23381203', 0, '9640322', '0', '0'),
									('18042462', 0, '9640322', '0', '0'),
									('18042459', 0, '9640322', '0', '0'),
									('34546558', 0, '9640322', '0', '0'),
									('27852745', 5, '9640322', '0', '0'),
									('34546505', 0, '9640322', '0', '0'),
									('23382497', 5, '9640322', '0', '0'),
									('23384429', 5, '9640322', '0', '0'),
									('24872400', 0, '9640322', '0', '0'),
									('18927785', 0, '9640322', '0', '0'),
									('23381215', 0, '9640322', '0', '0'),
									('23382498', 0, '9640322', '0', '0'),
									('17900389', 5, '9640322', '0', '0'),
									('24872366', 0, '9640322', '0', '0'),
									('13145328', 5, '9640322', '0', '0'),
									('25989238', 5, '9640322', '0', '0'),
									('17900370', 5, '9640322', '0', '0'),
									('19539271', 5, '9640322', '0', '0'),
									('17900388', 5, '9640322', '0', '0'),
									('17900362', 5, '9640322', '0', '0'),
									('12254015', 5, '9640322', '0', '0'),
									('18042460', 5, '9640322', '0', '0'),
									('12254020', 5, '9640322', '0', '0'),
									('17900364', 5, '9640322', '0', '0'),
									('23788065', 5, '9640322', '0', '0'),
									('20263218', 4, '7009690', '0', '0'),
									('22800468', 4, '7009690', '0', '0'),
									('22608495', 4, '7009690', '0', '0'),
									('22884125', 4, '7009690', '0', '0'),
									('25011719', 4, '7009690', '0', '0'),
									('23546982', 4, '7009690', '0', '0'),
									('17661519', 4, '7009690', '0', '0'),
									('13562860', 0, '1351002', '0', '0'),
									('36665733', 5, '17177260', '0', '0'),
									('30811396', 1, '6293730', '0', '0'),
									('36665721', 5, '17177260', '0', '0'),
									('36290178', 5, '17177260', '0', '0'),
									('15835797', 0, '6472544', '0', '0'),
									('18207591', 0, '7022844', '0', '0'),
									('23345520', 5, '14010620', '0', '0'),
									('23345514', 5, '14010620', '0', '0'),
									('23345527', 5, '14010620', '0', '0'),
									('23270963', 5, '14010620', '0', '0'),
									('10397256', 3, '5786', '0', '0'),
									('7419434', 2, '5786', '0', '0'),
									('7104807', 2, '43697', '0', '0'),
									('23264935', 5, '8591556', '0', '0'),
									('23264934', 5, '8591556', '0', '0'),
									('23264932', 5, '8591556', '0', '0'),
									('23264926', 5, '8591556', '0', '0'),
									('29069989', 3, '5042201', '3439408', '1077326'),
									('41899', 3, '57983', '1077326', '0'),
									('13153400', 4, '1510073', '0', '0'),
									('3398257', 5, '700231', '0', '0'),
									('3398249', 5, '700231', '0', '0'),
									('1620063', 5, '700231', '0', '0'),
									('3235059', 5, '700231', '0', '0'),
									('1408083', 4, '43784', '0', '0'),
									('186669', 5, '43784', '0', '0'),
									('77161', 5, '43784', '0', '0'),
									('77160', 5, '43784', '0', '0')
										]
	alan004 = [
							('1202', 4, '798', '1928', '0'),
							('129327', 5, '9494', '0', '0'),
							('5129', 5, '3487', '0', '0'),
							('5470', 5, '3706', '0', '0'),
							('19288043', 5, '2383', '0', '0'),
							('9969571', 5, '31712', '0', '0'),
							('29059', 3, '2546', '0', '0'),
							('7076703', 3, '2546', '0', '0'),
							('375802', 5, '589', '0', '0')
							]
	evelonce = [
							('9460487', 3, '3046613', '0', '0'),
							('15783514', 0, '1221698', '0', '0'),
							('186074', 0, '108424', '0', '0'),
							('3', 5, '1077326', '0', '0'),
							('31685789', 0, '23613', '0', '0'),
							('11387515', 3, '4859212', '0', '0'),
							('8909152', 2, '4208569', '0', '0'),
							('15745753', 5, '4208569', '0', '0'),
							('16068905', 5, '4208569', '0', '0'),
							('20820994', 5, '2982266', '0', '0'),
							('25855506', 0, '14144506', '0', '0')
								]
	carlarc28 = [('27188596', 5, '7074943', '0', '0'),
							('5907', 3, '656983', '0', '0'),
							('34506912', 0, '1557381', '0', '0'),
							('34499221', 0, '14684499', '0', '0'),
							('26032887', 0, '25422', '25422', '0'),
							('39331868', 4, '25422', '0', '0'),
							('8667848', 0, '3849415', '0', '0'),
							('36370046', 0, '4', '0', '0'),
							('24347197', 1, '874602', '0', '0'),
							('10724399', 3, '9887', '3097120', '0'),
							('18481271', 4, '4637539', '0', '0'),
							('18315788', 4, '3433047', '0', '0'),
							('41047644', 0, '18270790', '0', '0'),
							('22055262', 0, '7168230', '3099544', '0'),
							('25578831', 4, '25052', '0', '0'),
							('20613470', 0, '3433047', '0', '0'),
							('18006496', 0, '3433047', '0', '0'),
							('17670709', 4, '3433047', '0', '0'),
							('36341204', 0, '7579036', '7577278', '0'),
							('7171637', 0, '150038', '0', '0'),
							('10507293', 0, '2987125', '0', '0'),
							('17927395', 0, '3433047', '0', '0'),
							('22839894', 4, '3433047', '0', '0'),
							('4250', 5, '630', '9654', '0'),
							('27883214', 0, '14137787', '0', '0'),
							('36546635', 0, '239569', '0', '0'),
							('12408024', 3, '874602', '5331077', '0'),
							('6603726', 1, '874602', '0', '0'),
							('13519397', 3, '3433047', '0', '0'),
							('6456519', 3, '941441', '4870003', '0'),
							('11196993', 2, '190887', '0', '0'),
							('10560691', 3, '190887', '0', '0'),
							('18937911', 3, '348878', '7752438', '8440205'),
							('17729104', 3, '348878', '7752438', '5043181'),
							('15677818', 4, '348878', '7752438', '5043181'),
							('12043291', 4, '348878', '5043181', '7752438'),
							('9167899', 5, '348878', '7752438', '0'),
							('15920163', 3, '25052', '0', '0'),
							('12310706', 4, '25052', '0', '0'),
							('10969220', 4, '25052', '0', '0'),
							('20944743', 3, '25052', '5351593', '0'),
							('34992929', 0, '4637539', '0', '0'),
							('21920669', 5, '4637539', '0', '0'),
							('13104080', 4, '4637539', '0', '0'),
							('13455782', 3, '4637539', '0', '0'),
							('32505753', 0, '15945524', '0', '0'),
							('30226723', 4, '7074943', '0', '0'),
							('18138213', 0, '835348', '0', '0'),
							('32320661', 0, '16022936', '0', '0'),
							('7137327', 2, '835348', '0', '0'),
							('23174274', 4, '7074943', '0', '0'),
							('22328546', 5, '7074943', '0', '0'),
							('7461177', 2, '124319', '0', '0'),
							('29563587', 0, '25052', '0', '0'),
							('3777732', 3, '150038', '0', '0'),
							('1582996', 3, '150038', '0', '0'),
							('256683', 4, '150038', '0', '0'),
							('23267628', 4, '348878', '0', '0'),
							('24980633', 3, '348878', '0', '0'),
							('28175163', 3, '348878', '0', '0'),
							('20733600', 4, '348878', '0', '0'),
							('3425375', 4, '941441', '0', '0'),
							('2123654', 3, '941441', '0', '0'),
							('1881429', 2, '941441', '0', '0'),
							('1570814', 5, '941441', '3104327', '0'),
							('18376563', 0, '1330292', '0', '0'),
							('7693632', 0, '1330292', '2961772', '6434390'),
							('25306860', 0, '5768611', '0', '0'),
							('34108953', 0, '4086715', '0', '0')]
	flowerscata = [('74312', 4, '11270', '3890528', '4111263'),
								('10589899', 5, '1514864', '0', '0'),
								('27106851', 5, '14439352', '0', '0'),
								('27107205', 5, '5622776', '14439352', '0'),
								('12460502', 5, '6062', '298170', '5331'),
								('35383853', 5, '5624429', '0', '0'),
								('12593555', 5, '5805355', '0', '0'),
								('12593548', 5, '5805353', '5805354', '0'),
								('33955945', 5, '3720', '0', '0'),
								('11575725', 0, '42475', '0', '0'),
								('28444964', 0, '5624429', '0', '0'),
								('485812', 0, '10538', '0', '0'),
								('28490127', 4, '3389', '0', '0'),
								('28497932', 4, '10874652', '0', '0'),
								('26110915', 5, '151353', '0', '0'),
								('3869', 0, '1401', '0', '0'),
								('18721523', 4, '7364043', '0', '0'),
								('3222147', 5, '418034', '0', '0'),
								('17861507', 0, '7911220', '7068611', '14318227'),
								('381522', 0, '217230', '217229', '0'),
								('7755701', 0, '2959320', '3383826', '1268585'),
								('849065', 4, '1949', '200217', '0'),
								('25439283', 3, '637415', '13856856', '13856857'),
								('157993', 0, '1020792', '27112', '4479191'),
								('5507', 0, '3720', '0', '0'),
								('4808763', 5, '16667', '139505', '1663513'),
								('433567', 0, '3093075', '168631', '0'),
								('13355981', 5, '5424320', '0', '0'),
								('21480930', 5, '2383', '0', '0'),
								('20662423', 5, '1406392', '0', '0'),
								('2767052', 0, '153394', '0', '0'),
								('6186357', 5, '348878', '0', '0'),
								('16279856', 5, '348878', '0', '0'),
								('70771', 0, '16667', '0', '0'),
								('63137', 5, '25536', '0', '0'),
								('16155330', 4, '28181', '0', '0'),
								('73716', 5, '4694', '0', '0'),
								('256683', 3, '150038', '0', '0'),
								('10818853', 1, '4725841', '0', '0'),
								('43814', 5, '7577', '0', '0'),
								('43763', 5, '7577', '0', '0'),
								('16640', 5, '285217', '410875', '0'),
								('776877', 5, '1036736', '0', '0'),
								('324505', 5, '6895971', '1036736', '0'),
								('125333', 5, '6895971', '1036736', '0'),
								('3685', 5, '2530', '4327031', '0'),
								('366784', 5, '6895971', '1036736', '0'),
								('15881', 5, '1077326', '2927', '1946159'),
								('3', 5, '1077326', '2927', '3911856'),
								('16002308', 3, '5818588', '0', '0'),
								('23878', 5, '13450', '50060', '0'),
								('485894', 5, '5223', '141558', '0'),
								('102868', 5, '2448', '0', '0'),
								('13599706', 5, '5816342', '0', '0')]

	user = carlarc28

	hrefs = []
	for itemId, rating, author1, author, author3 in user:
		url      = solr + '/select?q=goodreadsId:' + itemId + '&wt=json' 
		response = json.loads( urlopen(url).read().decode('utf8') )
		doc      = response['response']['docs'][0]
		hrefs.append( doc['href'][0] )

	# # SOLO PYTHON 3.x y Alt. 2
	model = KeyedVectors.load_word2vec_format('/home/jschellman/gensim-data/word2vec-google-news-300/word2vec-google-news-300', binary=True)
	which_model = 'google'
	# # 1. Flatten todos los docs del index en Solr
	# dict_docs =	flatten_all_docs(solr= solr, model= model, filter_extremes= True)
	# # 2. Guarda flatten docs
	# np.save('./w2v-tmp/flattened_docs_fea075b1.npy', dict_docs)
	# # 3. Embedding de los flattened docs
	# dict_docs =	docs2vecs(model= model) # (por dentro carga "./w2v-tmp/flattened_docs_fea075b1.npy")
	# # 4. Guarda los embeddings
	# np.save('./w2v-tmp/'+which_model+'/docs2vec_'+which_model+'.npy', dict_docs)
	# # 5. Genera las recomendaciones
	lista_w2v = w2v_recs(data_path= data_path, solr= solr, which_model= which_model, items= user, userId="carlarc28", model= model)
	lista_w2v = recs_cleaner(solr= solr, consumpt_hrefs= hrefs, recs= lista_w2v)


	i = 0
	logging.info("---------------")
	logging.info("RECOMENDADOR C")
	logging.info("---------------")
	for item in lista_w2v:
		i += 1
		url      = solr + '/select?q=goodreadsId:'+item+'&wt=json'
		response = json.loads( urlopen(url).read().decode('utf8') )
		doc      = response['response']['docs'][0]
		logging.info(i)
		logging.info("Título: " + doc['title.titleOfficial'][0])
		logging.info("Autor: " + doc['author.authors.authorName'][0])
		logging.info("URL: " + doc['href'][0])
		logging.info("- RELEVANTE / ALGO RELEVANTE / IRRELEVANTE")
		logging.info("- NOVEDOSO / ALGO NOVEDOSO / NO NOVEDOSO")
		logging.info(" ")
		if i==10: break
	logging.info("Con repsecto al recomendador C:")
	logging.info("Respecto al recomendador C:")
	logging.info("- SATISFECHO / NO SATISFECHO")	

	# diversity_calculation(data_path= data_path, solr= solr, params_cb= params_solr, params_cf= params_imp, params_hy= params_hy)

if __name__ == '__main__':
	main()