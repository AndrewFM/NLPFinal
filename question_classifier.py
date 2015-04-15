#Takes a question as input, and soft-classifies it into the most likely stack-exchange sites in our corpus.

import nltk
import os
import string

#Construct Bag-of-words model for each website
word_counts = []
ids = []
maketrans = ''.maketrans

for infile in os.listdir("Question Corpus"):
	site = open("Question Corpus/"+infile, 'r')
	print("Question Corpus/"+infile)
	words = []
	for question in site:
		for word in question.split(' '):
			words.append(word.lower().translate(maketrans("",""), string.punctuation))
	site.close()

	word_counts.append(nltk.FreqDist(words))
	ids.append(infile)

#Train the classifier
train_set = [(words_counts[i], ids[i]) for i in range(len(ids))]
classifier = nltk.NaiveBayesClassifier(train_set)

#Get questions
while True:
	user_question = input("Enter a question: ")
	print("I think your question belongs to: "+classifier.classify(user_question))









