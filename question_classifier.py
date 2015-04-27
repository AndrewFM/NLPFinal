#Takes a question as input, and soft-classifies it into the most likely stack-exchange sites in our corpus.
import os
import sys
import numpy
import random
import operator
import question_answerer as QA
import question_stemmer
from nltk import word_tokenize
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from time import time

#Print with only 2 decimal places
def format_decimal(num):
	return "{0:.2f}".format(num)

def get_question_features(question):
	return question
	#tok_question = word_tokenize(str(question))
	#return str(question) + ' '.join(QA.question_features_WordNet(tok_question))

#Gather together the data
category_size = 36
folds = 10
data = []

t0 = time()
cat_so_far = 0
for site in os.listdir("Question Corpus"):
	print("Processing", site)
	infile = open("Question Corpus/"+site, 'rb')
	for line in infile:
		stemmed_line = ' '.join(question_stemmer.porter_stem_question(word_tokenize(str(line))))
		data.append((get_question_features(stemmed_line),site))
	infile.close()
	cat_so_far += 1
	if cat_so_far >= category_size:
		break
print("done in %0.3fs" % (time() - t0))	

random.seed(123)
random.shuffle(data)
best_fold = 0
classifier = None
for i in range(folds):
	test_size = int(len(data)/folds)
	train_size = len(data)-test_size

	train_data, train_targets = zip(*(data[:test_size*i]+data[test_size*(i+1):]))
	test_data, test_targets = zip(*data[test_size*i:test_size*(i+1)])

	#Train the classifier
	#	HashingVectorizer = Bag of Words (used HashingVectorizer instead of CountVectorizer for memory conservation)
	#	TfidfTransformer  = Proportional word counts
	#	SGDClassifier	  = SVM w/ Stochastic gradient descent
	print("Building classifier (Fold "+str(i+1)+")...")
	t0 = time()
	classifier = Pipeline([('vect', HashingVectorizer(input='content', strip_accents='unicode', ngram_range=(1,2), stop_words='english')),		
	                      ('tfidf', TfidfTransformer()),								
	                      ('clf', SGDClassifier(loss='log', n_jobs=-1))])	
	classifier.fit_transform(train_data, train_targets)		
	print("done in %0.3fs" % (time() - t0))					

	#Check accuracy of classifier
	print("Testing classifier accuracy (Fold "+str(i+1)+")...")
	t0 = time()
	test_predictions = classifier.predict(test_data)	
	#print(metrics.classification_report(test_targets, test_predictions))
	accuracy = numpy.mean(test_predictions == test_targets)
	if accuracy > best_fold:
		best_fold = accuracy
	print("Classifier accuracy is: "+str(accuracy*100)+"%")
	print("done in %0.3fs" % (time() - t0))	

print("Best accuracy:", best_fold)
print("\n")

#Get questions from user
while True:
	user_question = get_question_features(input("Enter a question: "))
	tok_question = word_tokenize(user_question)
	predict_dist = classifier.predict_proba([user_question])
	predict_dist = [(classifier.get_params()['clf'].classes_[i],predict_dist[0][i]) for i in range(len(predict_dist[0]))]
	predict_dist.sort(key=operator.itemgetter(1), reverse=True)
	print("Your question belongs to:", predict_dist[0][0].replace('.txt', ''))#, "(I'm", format_decimal(predict_dist[0][1]*100)+"% sure)")
	#print("(It could also be", predict_dist[1][0], "["+format_decimal(predict_dist[1][1]*100)+"%])")
	#print("(Or, perhaps,", predict_dist[2][0], "["+format_decimal(predict_dist[2][1]*100)+"%])\n")

	# From playing around with this a bit, it looks to me like...
	# 	Less than 4% certainty, the program is failing to classify the question
	# 	4-6% certainty, it's not entirely sure
	# 	6%+ certainty, it probably has the right answer

	#TODO: answers = QA.get_candidate_answers(...)
	#TODO: passage = QA.extract_passage(...)

	passage = "Not implemented yet"
	atype = QA.get_answer_type(tok_question)
	final_answer = QA.extract_answer(tok_question, atype, passage)
	print("The answer is:", final_answer)
	print("\n")