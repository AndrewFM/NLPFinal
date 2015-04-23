from enum import Enum

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
	description = 32

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

#Returns an answer type tuple: (Coarse type, Fine type)
def get_answer_type(question):
	return (Coarse.description, Fine.definition)

#Search relevant stack exchange domains for potential answers to the question.
def get_candidate_answers(question, domains):
	return [""]

#Get relevant sentence(s) and/or paragraph(s) from the returned answers.
def extract_passage(question, type, answers):
	return ""

#Extract the final answer to be output, from the summarized, relevant passages
def extract_answer(question, type, passage):
	return ""
