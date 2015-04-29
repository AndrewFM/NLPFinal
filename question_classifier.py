#Takes a question as input, and soft-classifies it into the most likely stack-exchange sites in our corpus.
import os
import numpy
import random
import operator
import question_answerer as QA
import question_stemmer
import question_settings as settings
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
data = []
classifier = None
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
	if cat_so_far >= settings.CATEGORY_LIMIT:
		break
print("done in %0.3fs" % (time() - t0))	

random.seed(123)
random.shuffle(data)
best_fold = 0
fold_range = 1
if settings.KFOLD_CVALIDATION:
	fold_range = settings.NUM_FOLDS
for i in range(fold_range):
	test_size = int(len(data)/settings.NUM_FOLDS)
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

	if settings.SHOW_EVALUATION:
		#Check accuracy of classifier
		print("Testing classifier accuracy (Fold "+str(i+1)+")...")
		t0 = time()
		test_predictions = classifier.predict(test_data)	
		if settings.SHOW_DETAILED_METRICS:
			print(metrics.classification_report(test_targets, test_predictions))
		accuracy = numpy.mean(test_predictions == test_targets)
		if accuracy > best_fold:
			best_fold = accuracy
		print("Classifier accuracy is: "+str(accuracy*100)+"%")
		print("done in %0.3fs" % (time() - t0))	

if settings.SHOW_EVALUATION:
	print("Best accuracy:", best_fold)

#Get questions from user
while True:
	print()
	user_question = get_question_features(input("Enter a question: "))
	tok_question = word_tokenize(user_question)
	predict_dist = classifier.predict_proba([user_question])
	predict_dist = [(classifier.get_params()['clf'].classes_[i],predict_dist[0][i]) for i in range(len(predict_dist[0]))]
	predict_dist.sort(key=operator.itemgetter(1), reverse=True)
	if settings.SHOW_PREDICTIONS:
		print("Your question belongs to:", predict_dist[0][0].replace('.txt', ''))#, "(I'm", format_decimal(predict_dist[0][1]*100)+"% sure)")
	#print("(It could also be", predict_dist[1][0], "["+format_decimal(predict_dist[1][1]*100)+"%])")
	#print("(Or, perhaps,", predict_dist[2][0], "["+format_decimal(predict_dist[2][1]*100)+"%])\n")

	#TODO: Change subdomain searching behavior based on categorical certainty
	# From playing around with this a bit, it looks to me like...
	# 	Less than 4% certainty, the program is failing to classify the question
	# 	4-6% certainty, it's not entirely sure
	# 	6%+ certainty, it probably has the right answer
	answers = QA.get_candidate_answers(user_question, [predict_dist[0][0].replace('.txt', '')])
	atype = QA.get_answer_type(tok_question)
	final_answer = QA.extract_passage(user_question, atype, answers)

	print("The answer is:", final_answer)