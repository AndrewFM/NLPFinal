#Takes a question as input, and soft-classifies it into the most likely stack-exchange sites in our corpus.
import os
import sys
import numpy
import random
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer

#Gather together the data
train_size = 2000
data = []

for site in os.listdir("Question Corpus"):
	infile = open("Question Corpus/"+site, 'rb')
	for line in infile:
		data.append((line,site))
	infile.close()

#random.seed(123)
random.shuffle(data)
train_data, train_targets = zip(*data[:train_size])
test_data, test_targets = zip(*data[train_size:])

#Train the classifier
#	HashingVectorizer = Bag of Words (used HashingVectorizer instead of CountVectorizer for memory conservation)
#	TfidfTransformer  = Proportional word counts
#	SGDClassifier	  = SVM w/ Stochastic gradient descent
print("Building classifier...")
classifier = Pipeline([('vect', HashingVectorizer(input='content', strip_accents='unicode', ngram_range=(1,2), stop_words='english')),		
                      ('tfidf', TfidfTransformer()),								
                      ('clf', SGDClassifier(n_jobs=-1))])	
classifier.fit_transform(train_data, train_targets)							

#Check accuracy of classifier
test_predictions = classifier.predict(test_data)	
print("Classifier accuracy is: "+str(numpy.mean(test_predictions == test_targets)*100)+"%")

#Get questions from user
while True:
    user_question = input("Enter a question: ")
    print("I think your question belongs to: "+classifier.predict([user_question])[0])









