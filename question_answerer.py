from time import time
from nltk import word_tokenize
from nltk.corpus import names
from nltk.corpus import wordnet as wn
from nltk.corpus import conll2000
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction import DictVectorizer
from collections import OrderedDict
import question_settings as settings
import stackexchange
import pickle
import question_chunker
import numpy
import nltk
import re
import os
import sys

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
		clean_line = line.strip().lower()
		if semCSR_dict.get(clean_line) == None:
			semCSR_dict[clean_line] = [sem_class]
		else:
			semCSR_dict[clean_line].append(sem_class)
	infile.close()

def question_features_SemCSR(tok_question):
	global semCSR_dict
	features = dict()

	for word in tok_question:
		if semCSR_dict.get(word.lower()) != None:
			for semclass in semCSR_dict[word.lower()]:
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
if settings.SHOW_EVALUATION:
	print("Evaluating Part-of-speech chunker...")
	t0 = time()
	print(chunker.evaluate(conll2000.chunked_sents('test.txt')))
	print("done in %0.3fs" % (time() - t0))

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

#Check if a pickled version of the Answer Type classifier exists on disk
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
if settings.SHOW_EVALUATION:
	print("Evaluating answer type classifiers...")
	features = file_answer_type_features("data/TREC_10.label")

	predictions = coarse_classifier.predict(features['coarse_data'])	
	print("Coarse Classifier accuracy is: "+str(numpy.mean(predictions == features['coarse_targets'])*100)+"%")
	predictions = fine_classifier.predict(features['fine_data'])	
	print("Fine Classifier accuracy is: "+str(numpy.mean(predictions == features['fine_targets'])*100)+"%")

#Returns an answer type tuple: (Coarse type, Fine type)
def get_answer_type(tok_question):
	coarse_features = get_coarse_features(tok_question)
	if settings.SHOW_DETAILED_METRICS:
		print("Question's featureset:")
		print(coarse_features)
		print()
	coarse_dist = coarse_classifier.predict_proba(coarse_features)
	coarse_dist = [(coarse_classifier.get_params()['clf'].classes_[i],coarse_dist[0][i]) for i in range(len(coarse_dist[0]))]
	coarse_dist = sorted(coarse_dist, key=lambda x:(-x[1],x[0]))
	if settings.SHOW_DETAILED_METRICS:
		print("Coarse Class probabilities:")
		print(coarse_dist)
		print()
	
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
	answers = []
	#grab all similar questions from all relevant domains, and find accepted answers
	for s in domains:
		site = stackexchange.Site(s, impose_throttling=True, app_key=settings.user_api_key)
		q_throttle = 0
		test_qoutput = 0
		for q in site.similar(question, pagesize=25, sort="relevance", order="desc"):
			get_answer = site.build('questions/'+str(q.id)+"/answers", stackexchange.Answer, "answers", kw={"sort":"votes", "order":"desc", "include_body":"true", "filter":"withbody"})
			if len(get_answer) > 0:
				answers.append(get_answer[0].body) #cluttered string
				if test_qoutput == 0:
					print(q.title)
					test_qoutput = 1

			q_throttle += 1
			#Stop after finding 5 answers
			if len(answers) > 5 or q_throttle > 20:
				break

	return answers

#Get relevant sentence(s) and/or paragraph(s) from the returned answers.
#Returns 0 if there are no named entities
def extract_passage(question, atype, answers):
	#sort by ranking
	name_entries = []
	num_keywords = []
	keyword = atype.split(':')[1]
	for answer in answers:
		name_entries.append(extract_answer(question, atype, answer))
		num_keywords.append(len(intersect_with_file('/data/SemCSR/' + keyword, answer.lower(), False)))
	if len(names_entries) == 0:
		return 0
	name_entries.sort(key=lambda key: len(key))	
	num_keywords.sort()

	ranked = dict(zip(answers:name_entries + num_keywords + [reversed(i for i in range(len(answers)))]))
	ranked =  OrderedDict(sorted(ranked.items(), key=lambda kv: kv[1], reverse=True))
	return ranked

#Find matches between a file's contents and a passage
def intersect_with_file(filename, passage, case_sensitive):
	intersects = []

	f = open(filename, 'r', encoding='utf-8')
	for line in f:
		mod_line = str(line).strip()
		if case_sensitive == False:
			mod_line = mod_line.lower()
		if len(re.findall(r'[\s\W]'+mod_line+r'[\s\W]', passage)) > 0:
			intersects.append(mod_line)
	f.close()

	return intersects

def get_proper_nouns(tok_passage):
	#Criteria:
	#	-Must be title-cased
	#	-Must come after some other word (ie: ignore capitalized first words of sentences)
	#	-Ignore special cases of "I", "I'm", "I'd", and "I'll"
	proper_nouns = []
	prevword = ""
	ignore_prevwords = [".", "!", "?", "", "'", '"']
	for word in tok_passage:
		if word.istitle():
			if prevword not in ignore_prevwords and word != "I":
				proper_nouns.append(word)
		prevword = word

	return proper_nouns

#Extract the final answer to be output, from the summarized, relevant passages, using answer-type pattern extraction.
#fallback: Boolean parameter. If True, will return the original passage if no named entities are found.
#		   					  If false, will return an empty list if no named entities are found.
def extract_answer(question, atype, passage, fallback):
	answer_fragments = []
	tok_passage = word_tokenize(passage)

	pat_realnum = r"[0-9][0-9,]*\.?[0-9]*"	# 10, 999,999.25, 0.01, etc
	pat_largenum = r"[0-9]+\s(?:[Tt]housand|[a-zA-Z]+illion)" # ie: 150 Billion
	pat_vague_largenum = r"(?:[Hh]undreds|[Tt]ens)\sof\s(?:[Tt]housands|[a-zA-Z]+illions)" # ie: hundreds of billions
	pat_anynumber = pat_realnum + r'|' + pat_largenum + r'|' + pat_vague_largenum
	pat_symbolic_date = r"(?:[0-9]+[\\/-])+[0-9]+"	# 10/21/1991, 2014-10-11, 04/01, etc
	pat_written_date = r"(?:[Jj]anuary|[Ff]ebruary|[Mm]arch|[Aa]pril|[Mm]ay|[Jj]une|[Jj]uly|[Aa]ugust|[Ss]eptember|[Oo]ctober|[Nn]ovember|[Dd]ecember)\s[0-9]+(?:[\s\,]*[0-9]+)?" #ie: March 21, 2015
	pat_year = r"[0-9]{4}"	# 4 digit numbers
	pat_ancient_year = r"(?:[0-9]|\,)+\s*(?:AD|BC)" # "123 AD", "25,000,000 BC", etc
	pat_symbols_nopunct = r"(?!(?:[A-Za-z0-9]|\s|[\.\?!\,\:;\"\'\(\)]))." #Uncommon symbols

	#Fine pass (Numeric)
	if atype[1] == 'NUM:date':
		answer_fragments = re.findall(pat_written_date+r"|"+pat_symbolic_date+r"|"+pat_year+r"|"+pat_ancient_year, passage)
	elif atype[1] == 'NUM:money':
		answer_fragments = re.findall(r"[\$£¥¢]"+pat_anynumber, passage)
	elif atype[1] == 'NUM:temp':
		answer_fragments = re.findall(pat_realnum+r"\s*°[A-Z]*", passage)
	elif atype[1] == 'NUM:perc':
		answer_fragments = re.findall(pat_realnum+r"\s*\%", passage)
	elif atype[1] == 'NUM:weight' or atype[1] == 'NUM:volsize' or atype[1] == 'NUM:speed' or atype[1] == 'NUM:dist': #Number with unit
		answer_fragments = re.findall(pat_anynumber+r"\s*\S+\s", passage)
	elif atype[0] == 'NUM':
		answer_fragments = re.findall(pat_anynumber, passage)

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
		answer_fragments = intersect_with_file("data/states.txt", passage.lower(), False)
	elif atype[1] == 'LOC:country':
		answer_fragments = intersect_with_file("data/countries.txt", passage.lower(), False)
	elif atype[1] == 'LOC:city':
		answer_fragments = intersect_with_file("data/cities.txt", passage.lower(), False)
	elif atype[1] == 'LOC:mount':
		answer_fragments = intersect_with_file("data/mountains.txt", passage.lower(), False)

	#Fine pass (Entity)
	elif atype[1] == 'ENTY:instru':
		answer_fragments = intersect_with_file("data/instruments.txt", passage.lower(), False)
	elif atype[1] == 'ENTY:currency':
		answer_fragments = intersect_with_file("data/currencies.txt", passage.lower(), False)
	elif atype[1] == 'ENTY:lang':
		answer_fragments = intersect_with_file("data/languages.txt", passage.lower(), False)
	elif atype[1] == 'ENTY:religion':
		answer_fragments = intersect_with_file("data/religions.txt", passage.lower(), False)
	elif atype[1] == 'ENTY:animal':
		answer_fragments = intersect_with_file("data/animals.txt", passage.lower(), False)
	elif atype[1] == 'ENTY:body':
		answer_fragments = intersect_with_file("data/body.txt", passage.lower(), False)
	elif atype[1] == 'ENTY:color':
		answer_fragments = intersect_with_file("data/colors.txt", passage.lower(), False)
	elif atype[1] == 'ENTY:food':
		answer_fragments = intersect_with_file("data/foods.txt", passage.lower(), False)
	elif atype[1] == 'ENTY:sport':
		answer_fragments = intersect_with_file("data/sports.txt", passage.lower(), False)
	elif atype[1] == 'ENTY:plant':
		answer_fragments = intersect_with_file("data/plants.txt", passage.lower(), False)
	elif atype[1] == 'ENTY:symbol':
		answer_fragments = intersect_with_file("data/symbols.txt", passage, True)
		answer_fragments += [re.findall(r'[A-Z]+', item)[0] for item in re.findall(r'\W[A-Z]+\W', passage)]
		answer_fragments += re.findall(pat_symbols_nopunct, passage)

	#Fine pass (Other)
	elif atype[1] == 'ABBR:abb' or atype[1] == 'ENTY:letter':
		#We can at least try to find acronyms...
		answer_fragments = [re.findall(r'[A-Z]+', item)[0] for item in re.findall(r'\W[A-Z]+\W', passage)]

	#Coarse pass
	if len(answer_fragments) == 0:
		if atype[0] == 'NUM':
			answer_fragments = re.findall(pat_anynumber, passage) 
		elif atype[0] == 'HUM' and (atype[1] == 'HUM:ind' or atype[1] == 'HUM:gr'):
			#I didn't find a name, let's become naive and just return any potential proper nouns
			answer_fragments = get_proper_nouns(tok_passage)
		elif atype[0] == 'LOC':
			#I didn't find the location, let's become naive and just return any potential proper nouns
			answer_fragments = get_proper_nouns(tok_passage)

	#Output
	if len(answer_fragments) == 0: #Unhandled answer type, or failed to extract answer. Just return the entire passage.
		if fallback:
			return passage
		else:
			return []
	else:
		return ', '.join(set(answer_fragments))
