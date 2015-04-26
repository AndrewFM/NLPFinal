from nltk.stem.porter import *
from nltk.stem.snowball import SnowballStemmer
from nltk import word_tokenize

def porter_stem_question(question):
    """Input a list of words from a question, output a stemmed list of words"""
    """Less accurate than Snowball Stemmer"""
    stemmer = PorterStemmer()
    stemmed_question = []

    [stemmed_question.append(stemmer.stem(w)) for w in question]

    return stemmed_question

def snowball_stem_question(question):
    """Input a list of words from a question, output a stemmed list of words"""
    """More accurate than Porter Stemmer and ignores stopwords"""
    stemmer = SnowballStemmer("english", ignore_stopwords=True)
    stemmed_question = []
    
    [stemmed_question.append(stemmer.stem(w)) for w in question]

    return stemmed_question
