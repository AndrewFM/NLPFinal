from time import time
from nltk import word_tokenize
from nltk.corpus import names
from nltk.corpus import wordnet as wn
from nltk.corpus import conll2000
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction import DictVectorizer
import pickle
import question_chunker
import numpy
import nltk
import re
import os

#Li and Roth's question hierarchy
question_hierarchy_coarse = ['ABBR', 'DESC', 'NUM', 'ENTY', 'LOC', 'HUM']
question_hierarchy_fine   = {'ABBR': ['exp', 'abb'],
					 	     'DESC': ['def', 'desc', 'manner', 'reason'],
					 	     'NUM':  ['code', 'count', 'date', 'dist', 'money', 'other', 'ord', 'period', 'perc', 'speed', 'temp', 'volsize', 'weight'],
					 	     'ENTY': ['animal', 'body', 'color', 'cremat', 'currency', 'dismed', 'event', 'food', 'instru', 'lang', 'letter', 'other', 'plant', 'product', 'religion', 'sport', 'substance', 'symbol', 'techmeth', 'termeq', 'veh', 'word'],
					 	     'LOC':  ['city', 'country', 'state', 'mount', 'other'],
					 	     'HUM':  ['ind', 'gr', 'title', 'desc']}

#Extract Class-Specific Relations from the question
print("Loading Class-Specific Relations dictionary...")
semCSR_dict = dict()
for sem_class in os.listdir("data/SemCSR"):
	infile = open("data/SemCSR/"+sem_class, 'r')
	for line in infile:
		if semCSR_dict.get(line) == None:
			semCSR_dict[line] = [sem_class]
		else:
			semCSR_dict[line].append(sem_class)
	infile.close()

def question_features_SemCSR(tok_question):
	global semCSR_dict
	features = dict()

	for word in tok_question:
		if semCSR_dict.get(word) != None:
			for semclass in semCSR_dict[word]:
				features["class_"+semclass] = 1

	return features

#Extract WordNet synonyms, hyponyms, and hypernyms of all words in the question
def question_features_WordNet(tok_question):
	return_features = []

	for word in tok_question:
		synsets = wn.synsets(word.lower())
		synonyms = [lemma for syn in synsets for lemma in syn.lemma_names()]
		hyponyms = [lemma for syn in synsets for hypo in syn.hyponyms() for lemma in hypo.lemma_names()]
		hypernyms = [lemma for syn in synsets for hyper in syn.hypernyms() for lemma in hyper.lemma_names()]
		return_features += synonyms + hyponyms + hypernyms
		#synonyms = [re.sub(r"\..+", "", syn.name()) for syn in synsets]
		#return_features += synonyms

	return list(set(return_features))

#Extract POS Tags and Chunks from the question
print("Training Part-of-speech chunker...")
t0 = time()
chunker = question_chunker.PosChunker(conll2000.chunked_sents('train.txt'))
print("done in %0.3fs" % (time() - t0))	
'''print("Evaluating Part-of-speech chunker...")
t0 = time()
print(chunker.evaluate(conll2000.chunked_sents('test.txt', chunk_types=['NP','VP'])))
print("done in %0.3fs" % (time() - t0))	'''

def question_features_POS(tok_question):
	features = dict()
	found_head_np = False
	found_head_vp = False
	cur_chunk = []
	cur_type = None	#NP, VP, PP

	tag_question = nltk.pos_tag(tok_question)
	chunk_question = [((word, tag), chunk) for (word,tag,chunk) in nltk.chunk.tree2conlltags(chunker.parse(tag_question))]
	chunk_question.append((('<END>', '.'), 'O')) #Need this in case the user forgets to end their question with a punctuation mark
	for i in range(len(chunk_question)):
		item = chunk_question[i]
		if item[0][1] != '.':
			features[item[0][0].lower()] = 1 #Word
			features['tag_'+item[0][0].lower()] = item[0][1]  #Tag

		#Chunks
		chunk_tag = item[1]
		if chunk_tag[0] != "I":
			if len(cur_chunk) > 0:
				if cur_type != None:
					features['chunk_'+'_'.join(cur_chunk)] = cur_type
				if cur_type == "NP" and found_head_np == False:
					head_np = ' '.join(cur_chunk)
					if head_np.lower() not in ['what', 'where', 'when', 'why', 'who', 'how']: #Don't count question words as head noun phrases
						features["np_head"] = head_np
						found_head_np = True
				elif cur_type == "VP" and found_head_vp == False:
					features["vp_head"] = ' '.join(cur_chunk)
					found_head_vp = True
			cur_chunk = []
			if chunk_tag[0] == "O":
				cur_type = None
			else:
				cur_type = chunk_tag[2:]
		cur_chunk.append(item[0][0].lower())

	return features

#Build and train hierarchial classifier for answer type extraction
def get_coarse_features(tok_question):
	coarse_features = dict()
	coarse_features.update(question_features_POS(tok_question))
	coarse_features.update(question_features_SemCSR(tok_question))
	return coarse_features

def file_answer_type_features(filename):
	t0 = time()
	print("Processing answer type features in file "+filename+"...")
	at_coarse_trdata = []
	at_fine_trdata = []
	at_coarse_trtargets = []
	at_fine_trtargets = []

	infile = open(filename, 'r')
	for line in infile:
		tok_line = line.split(' ')
		tok_class = tok_line[0].split(':')

		coarse_features = get_coarse_features(tok_line[1:])
		at_coarse_trdata.append(coarse_features)
		at_coarse_trtargets.append(tok_class[0])

		fine_features = dict()
		for fclass in question_hierarchy_fine[tok_class[0]]:
			fine_features[tok_class[0]+":"+fclass] = 1
		fine_features.update(coarse_features)
		at_fine_trdata.append(fine_features)
		at_fine_trtargets.append(tok_line[0])
	infile.close()
	print("done in %0.3fs" % (time() - t0))

	return {'coarse_data':at_coarse_trdata, 'coarse_targets':at_coarse_trtargets, 'fine_data':at_fine_trdata, 'fine_targets':at_fine_trtargets}


coarse_classifier = Pipeline([('vect', DictVectorizer()),('clf', SGDClassifier(loss='log', n_jobs=-1))])
fine_classifier = Pipeline([('vect', DictVectorizer()),('clf', SGDClassifier(n_jobs=-1))])

if os.path.isfile("dumps/coarse_atype_classifier.pkl") and os.path.isfile("dumps/fine_atype_classifier.pkl"):
	print("Found pickled answer type classifiers... loading them.")
	dump = open("dumps/coarse_atype_classifier.pkl", 'rb')
	coarse_classifier = pickle.load(dump)
	dump.close()

	dump = open("dumps/fine_atype_classifier.pkl", 'rb')
	fine_classifier = pickle.load(dump)
	dump.close()
else:
	features = file_answer_type_features("data/train_5500.label")

	#Train Answer Type prediction
	print("Training coarse answer type classifier...")
	t0 = time()
	coarse_classifier.fit_transform(features['coarse_data'], features['coarse_targets'])	
	dump = open('dumps/coarse_atype_classifier.pkl', 'wb')
	pickle.dump(coarse_classifier, dump)
	dump.close()
	print("done in %0.3fs" % (time() - t0))

	print("Training fine answer type classifier...")
	t0 = time()
	fine_classifier.fit_transform(features['fine_data'], features['fine_targets'])
	dump = open('dumps/fine_atype_classifier.pkl', 'wb')
	pickle.dump(fine_classifier, dump)
	dump.close()
	print("done in %0.3fs" % (time() - t0))

#Test Answer Type prediction
'''print("Evaluating answer type classifiers...")
features = file_answer_type_features("data/TREC_10.label")

predictions = coarse_classifier.predict(features['coarse_data'])	
print("Coarse Classifier accuracy is: "+str(numpy.mean(predictions == features['coarse_targets'])*100)+"%")
predictions = fine_classifier.predict(features['fine_data'])	
print("Fine Classifier accuracy is: "+str(numpy.mean(predictions == features['fine_targets'])*100)+"%")'''

#Returns an answer type tuple: (Coarse type, Fine type)
def get_answer_type(tok_question):
	coarse_features = get_coarse_features(tok_question)
	coarse_dist = coarse_classifier.predict_proba(coarse_features)
	coarse_dist = [(coarse_classifier.get_params()['clf'].classes_[i],coarse_dist[0][i]) for i in range(len(coarse_dist[0]))]
	coarse_dist = sorted(coarse_dist, key=lambda x:(-x[1],x[0]))
	#print(coarse_dist)
	
	#Grab coarse classes up to 95% total certainty, or 5 classes total, whichever comes first
	coarse_collection = []
	certainty_so_far = 0
	for cclass in coarse_dist: 
		coarse_collection.append(cclass[0])
		certainty_so_far += cclass[1]

		if certainty_so_far >= 0.95 or len(coarse_collection) >= 5:
			break
	#coarse_collection = [coarse_dist[0][0]]

	#Decompose selected coarse classes into their associated fine classes
	fine_features = dict()
	for cclass in coarse_collection:
		for fclass in question_hierarchy_fine[cclass]:
			fine_features[cclass+":"+fclass] = 1
	fine_features.update(coarse_features)

	#TODO: pass features to answer type classifier
	fine_predict = fine_classifier.predict(fine_features)[0]
	print("Predicted answer type is:", fine_predict)
	return (fine_predict.split(':')[0], fine_predict)

#Search relevant stack exchange domains for potential answers to the question.
def get_candidate_answers(question, domains):
	return [""]

#Get relevant sentence(s) and/or paragraph(s) from the returned answers.
def extract_passage(question, atype, answers):
	return ""

#Find matches between a file's contents and a passage
def intersect_with_file(filename, passage):
	intersects = []

	f = open(filename, 'rb')
	for line in f:
		if passage.find(str(line).lower()) != -1:
			intersects.append(str(line))
	f.close()

	return intersects

#Find any title-cased words in the passage
def get_proper_nouns(tok_passage):
	return [word for word in tok_passage if word.istitle() == True]

#Extract the final answer to be output, from the summarized, relevant passages, using answer-type pattern extraction.
def extract_answer(question, atype, passage):
	answer_fragments = []
	tok_passage = passage.split(' ')

	pat_realnum = r"[0-9]+[.]*[0-9]*"	# 10, 50.2, 0.01, etc
	pat_symbolic_date = r"(?:[0-9]+[\\/-])+[0-9]+"	# 10/21/1991, 2014-10-11, 04/01, etc
	pat_written_date = r"(?:[Jj]anuary|[Ff]ebruary|[Mm]arch|[Aa]pril|[Mm]ay|[Jj]une|[Jj]uly|[Aa]ugust|[Ss]eptember|[Oo]ctober|[Nn]ovember|[Dd]ecember)\s[0-9]+(?:[\s\,]*[0-9]+)?" #ie: March 21, 2015
	pat_year = r"[0-9]{4}"	# 4 digit numbers
	pat_ancient_year = r"(?:[0-9]|\,)+\s*(?:AD|BC)" # "123 AD", "25,000,000 BC", etc

	#Fine pass (Numeric)
	if atype[1] == 'NUM:date':
		answer_fragments = re.findall(pat_written_date+r"|"+pat_symbolic_date+r"|"+pat_year+r"|"+pat_ancient_year, passage)
	elif atype[1] == 'NUM:money':
		answer_fragments = re.findall(r"[\$£¥¢]"+pat_realnum, passage)
	elif atype[1] == 'NUM:temp':
		answer_fragments = re.findall(pat_realnum+r"\s*°[A-Z]*", passage)
	elif atype[1] == 'NUM:perc':
		answer_fragments = re.findall(pat_realnum+r"\s*\%", passage)
	elif atype[1] == 'NUM:weight' or atype[1] == 'NUM:volsize' or atype[1] == 'NUM:speed' or atype[1] == 'NUM:dist': #Number with unit
		answer_fragments = re.findall(pat_realnum+r"\s*\S+\s", passage)
	elif atype[0] == 'NUM':
		answer_fragments = re.findall(pat_realnum, passage)

	#Fine pass (Human)
	elif atype[1] == 'HUM:ind':
		name_list = set([name for name in names.words('male.txt')] + [name for name in names.words('female.txt')])
		current_name = ""
		for i in range(len(tok_passage)):
			if len(set([re.sub("\W", "", tok_passage[i]).title()]) & name_list) != 0: #If this word matches a name in the name list
				current_name = tok_passage[i]
				if i < len(tok_passage)-1 and tok_passage[i+1].istitle(): #Check for last name
					current_name += " "+tok_passage[i+1]
					i += 1
			if current_name != "":
				answer_fragments.append(current_name)
				current_name = ""

	#Fine pass (Location)
	elif atype[1] == 'LOC:state':
		answer_fragments = intersect_with_file("data/states.txt", passage.lower())
	elif atype[1] == 'LOC:country':
		answer_fragments = intersect_with_file("data/countries.txt", passage.lower())
	elif atype[1] == 'LOC:city':
		answer_fragments = intersect_with_file("data/cities.txt", passage.lower())
	elif atype[1] == 'LOC:mount':
		answer_fragments = intersect_with_file("data/mountains.txt", passage.lower())

	#Fine pass (Entity)
	elif atype[1] == 'ENTY:instru':
		answer_fragments = intersect_with_file("data/instruments.txt", passage.lower())
	elif atype[1] == 'ENTY:currency':
		answer_fragments = intersect_with_file("data/currencies.txt", passage.lower())
	elif atype[1] == 'ENTY:lang':
		answer_fragments = intersect_with_file("data/languages.txt", passage.lower())
	elif atype[1] == 'ENTY:religion':
		answer_fragments = intersect_with_file("data/religions.txt", passage.lower())

	#Coarse pass
	if len(answer_fragments) == 0:
		if atype[0] == 'NUM':
			answer_fragments = re.findall(pat_realnum, passage) 
		elif atype[0] == 'HUM' and (atype[1] == 'HUM:ind' or atype[1] == 'HUM:gr'):
			#I didn't find a name, let's become naive and just return any title-cased words
			answer_fragments = get_proper_nouns(tok_passage)
		elif atype[0] == 'LOC':
			#I didn't find the location, let's become naive and just return any title-cased words
			answer_fragments = get_proper_nouns(tok_passage)

	#Output
	if len(answer_fragments) == 0: #Unhandled answer type, or failed to extract answer. Just return the entire passage.
		return passage
	else:
		return ', '.join(set(answer_fragments))
