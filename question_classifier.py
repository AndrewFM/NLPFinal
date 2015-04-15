#Takes a question as input, and soft-classifies it into the most likely stack-exchange sites in our corpus.
import os
import sys
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer

#Gather together the data
data = []
targets = []
for site in os.listdir("Question Corpus"):
	infile = open("Question Corpus/"+site, 'rb')
	for line in infile:
		data.append(line)
		targets.append(site)
	infile.close()

#Train the classifier
#	HashingVectorizer = Bag of Words (used HashingVectorizer instead of CountVectorizer for memory conservation)
#	TfidfTransformer  = Proportional word counts
#	SGDClassifier	  = SVM w/ Stochastic gradient descent
print("Building classifier...")
classifier = Pipeline([('vect', HashingVectorizer(input='content', strip_accents='unicode', ngram_range=(1,2), stop_words='english')),		
                      ('tfidf', TfidfTransformer()),								
                      ('clf', SGDClassifier(n_jobs=-1))])	
classifier.fit_transform(data, targets)								

#(Todo): Gather together test data (some subset of the training data, not used during training), and output classification accuracy

#Get questions
while True:
    user_question = input("Enter a question: ")
    print("I think your question belongs to: "+classifier.predict([user_question])[0])









