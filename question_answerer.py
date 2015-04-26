from enum import Enum
from time import time
from nltk import word_tokenize
from nltk.corpus import names
from nltk.corpus import wordnet as wn
from nltk.corpus import conll2000
import question_chunker
import nltk
import re
import os

#Li and Roth's question hierarchy
class Coarse(Enum):
	abbreviation = 1
	description = 2
	entity = 3
	human = 4
	location = 5
	numeric = 6

class Fine(Enum):
	# [Abbreviation]
	abbreviation = 1
	expression = 2

	# [Description]
	definition = 3
	description = 4
	manner = 5
	reason = 6

	# [Entity]
	animal = 7
	body = 8
	color = 9
	creative = 10
	currency = 11
	medicine = 12
	event = 13
	food = 14
	instrument = 15
	lang = 16
	letter = 17
	other_entity = 18
	plant = 19
	product = 20
	religion = 21
	sport = 22
	substance = 23
	symbol = 24
	technique = 25
	term = 26
	vehicle = 27
	word = 28

	# [Human]
	group = 29
	individual = 30
	title = 31
	human_description = 32

	# [Location]
	city = 33
	country = 34
	mountain = 35
	other_location = 36
	state = 37

	# [Numeric]
	code = 38
	count = 39
	date = 40
	distance = 41
	money = 42
	order = 43
	other_numeric = 44
	period = 45
	percent = 46
	speed = 47
	temp = 48
	size = 49
	weight = 50

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
	return_classes = []

	for word in tok_question:
		if semCSR_dict.get(word) != None:
			return_classes += semCSR_dict[word]

	return list(set(return_classes))

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
chunker = question_chunker.PosChunker(conll2000.chunked_sents('train.txt', chunk_types=['NP','VP']))
print("done in %0.3fs" % (time() - t0))	
'''print("Evaluating Part-of-speech chunker...")
t0 = time()
print(chunker.evaluate(conll2000.chunked_sents('test.txt', chunk_types=['NP','VP'])))
print("done in %0.3fs" % (time() - t0))	'''

def question_features_POS(tok_question):
	tag_question = nltk.pos_tag(tok_question)
	pos_tag_features = [tag[1] for tag in tag_question]

	chunk_question = [((word, tag), chunk) for (word,tag,chunk) in nltk.chunk.tree2conlltags(chunker.parse(tag_question))]
	print(chunk_question)
	pos_chunk_features = []

	return pos_tag_features + pos_chunk_features

#Returns an answer type tuple: (Coarse type, Fine type)
def get_answer_type(question):
	return (Coarse.description, Fine.definition)

#Search relevant stack exchange domains for potential answers to the question.
def get_candidate_answers(question, domains):
	return [""]

#Get relevant sentence(s) and/or paragraph(s) from the returned answers.
def extract_passage(question, atype, answers):
	return ""

#Find matches between a file's contents and a passage
def intersect_with_file(filename, passage):
	intersects = []

	f = open(filename, 'r')
	for line in f:
		if passage.find(line.lower()) != -1:
			intersects.append(line)
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
	if atype[1] == Fine.date:
		answer_fragments = re.findall(pat_written_date+r"|"+pat_symbolic_date+r"|"+pat_year+r"|"+pat_ancient_year, passage)
	elif atype[1] == Fine.money:
		answer_fragments = re.findall(r"[\$£¥¢]"+pat_realnum, passage)
	elif atype[1] == Fine.temp:
		answer_fragments = re.findall(pat_realnum+r"\s*°[A-Z]*", passage)
	elif atype[1] == Fine.percent:
		answer_fragments = re.findall(pat_realnum+r"\s*\%", passage)
	elif atype[1] == Fine.weight or atype[1] == Fine.size or atype[1] == Fine.speed or atype[1] == Fine.distance: #Number with unit
		answer_fragments = re.findall(pat_realnum+r"\s*\S+\s", passage)
	elif atype[0] == Coarse.Numeric:
		answer_fragments = re.findall(pat_realnum, passage)

	#Fine pass (Human)
	elif atype[1] == Fine.individual:
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
	elif atype[1] == Fine.state:
		answer_fragments = intersect_with_file("data/states.txt", passage.lower())
	elif atype[1] == Fine.country:
		answer_fragments = intersect_with_file("data/countries.txt", passage.lower())
	elif atype[1] == Fine.city:
		answer_fragments = intersect_with_file("data/cities.txt", passage.lower())
	elif atype[1] == Fine.mountain:
		answer_fragments = intersect_with_file("data/mountains.txt", passage.lower())

	#Fine pass (Entity)
	elif atype[1] == Fine.instrument:
		answer_fragments = intersect_with_file("data/instruments.txt", passage.lower())
	elif atype[1] == Fine.currency:
		answer_fragments = intersect_with_file("data/currencies.txt", passage.lower())
	elif atype[1] == Fine.lang:
		answer_fragments = intersect_with_file("data/languages.txt", passage.lower())
	elif atype[1] == Fine.religion:
		answer_fragments = intersect_with_file("data/religions.txt", passage.lower())

	#Coarse pass
	if len(answer_fragments) == 0:
		if atype[0] == Coarse.Numeric:
			answer_fragments = re.findall(pat_realnum, passage) 
		elif atype[0] == Coarse.human and (atype[1] == Fine.individual or atype[1] == Fine.group):
			#I didn't find a name, let's become naive and just return any title-cased words
			answer_fragments = get_proper_nouns(tok_passage)
		elif atype[0] == Coarse.location:
			#I didn't find the location, let's become naive and just return any title-cased words
			answer_fragments = get_proper_nouns(tok_passage)

	#Output
	if len(answer_fragments) == 0: #Unhandled answer type, or failed to extract answer. Just return the entire passage.
		return passage
	else:
		return ', '.join(set(answer_fragments))
