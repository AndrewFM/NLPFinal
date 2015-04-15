#Takes a question as input, and soft-classifies it into the most likely stack-exchange sites in our corpus.

import nltk
from nltk.corpus import stopwords
import os
import string
import pickle

question_limit = 500 #Number of training questions to use per file (since the full 2500 is wrecking the computer's memory)
classifier = None
table = str.maketrans("", "", string.punctuation) #Used to strip punctuation

#If we don't have a pre-trained classifier saved to disk, train one on the data.
if os.path.isfile("SerializedClassifier.dat") == False:
	#Construct Bag-of-words model for each website
	train_set = [] 

	for infile in os.listdir("Question Corpus"):
	    site = open("Question Corpus/"+infile, 'r', encoding='utf-8')
	    print("Compiling featureset for "+infile)
	    
	    words = set()
	    words_so_far = 0
	    for question in site:
	    	words = words.union(set([word.lower() for word in question.translate(table).split(' ')]))
	    	words_so_far += 1
	    	if words_so_far > question_limit:
	    		break
	    site.close()

	    train_set.append( (dict.fromkeys(words, True), infile) )

	#Train the classifier
	print("Building classifier...")
	classifier = nltk.NaiveBayesClassifier.train(train_set)
	p = pickle.Pickler(open("SerializedClassifier.dat", "wb"))
	p.fast = True
	p.dump(classifier)

#Otherwise, load the classifier from disk.
else:
	print("Loading the classifier from file...")
	classifier = pickle.load( open("SerializedClassifier.dat", "rb") )

#Get questions
while True:
    user_question = input("Enter a question: ")
    word_arr = user_question.lower().translate(table).split(" ")
    print("I think your question belongs to: "+classifier.classify(nltk.FreqDist(word_arr)))









