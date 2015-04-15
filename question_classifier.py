#Takes a question as input, and soft-classifies it into the most likely stack-exchange sites in our corpus.

import nltk
import os
import string
from nltk.corpus import stopwords

punctuation = """!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""

def remove_punctuation(word):
    stripped_str = ""
    for c in word:
        if c in punctuation:
            continue
        stripped_str += c
    return stripped_str

#Construct Bag-of-words model for each website
word_counts = []
ids = []   

for infile in os.listdir("Question Corpus"):
    site = open("Question Corpus/"+infile, 'r')
    print("Question Corpus/"+infile)
    words = []
    questions_seen = 0
    for question in site:
        for word in question.split(' '):
               word = remove_punctuation(word.lower())
               if word not in stopwords.words('english'):
                   words.append(word)
        questions_seen += 1
        if questions_seen > 500:
            break
    site.close()

    word_counts.append(nltk.FreqDist(set(words)))
    ids.append(infile)

#Train the classifier
train_set = [(word_counts[i], ids[i]) for i in range(len(ids))]
print("Building classifier...")
classifier = nltk.NaiveBayesClassifier.train(train_set)

#Get questions
while True:
    user_question = raw_input("Enter a question: ")
    user_question = remove_punctuation(user_question.lower())
    word_arr = user_question.split(" ")
    print("I think your question belongs to: "+classifier.classify(nltk.FreqDist(word_arr)))









