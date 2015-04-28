import nltk
import pickle
import os
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction import DictVectorizer

#Collection of tags that occured in the question since the last occurance of some POS tag
def tags_since(tag_question, word_ind, pos):
	tags = set()
	for word, tag in reversed(tag_question[:word_ind]):
		if tag == pos:
			break
		else:
			tags.add(tag)
	return '+'.join(sorted(tags))

#tag_question: each index contains (word,tag)
def chunk_features(tag_question, word_ind):
	word, cur_tag = tag_question[word_ind]
	if word_ind == 0: 								#First word
		prev_word, prev_tag = "<START>", "<START>"
	else:
		prev_word, prev_tag = tag_question[word_ind-1]

	if word_ind == len(tag_question)-1:				#Last word
		next_word, next_tag = "<END>", "<END>"
	else:
		next_word, next_tag = tag_question[word_ind+1]

	##TODO (Try different features)
	return {"tag": cur_tag, "word": word, "prev_tag": prev_tag, "next_tag": next_tag
		  , "prev_comb": "%s+%s" % (prev_tag, cur_tag), "next_comb": "%s+%s" % (cur_tag, next_tag)
		  , "tags_since_dt": tags_since(tag_question, word_ind, "DT")}

#A chunk tagger that uses supervised machine learning to learn the chunking rules.
class ChunkTagger(nltk.TaggerI):

	#Train set should be a list of chunk-tagged sentences in ((word, tag), chunk) IOB format.
	def __init__(self, train_set):
		if os.path.isfile("dumps/chunker.pkl"):
			print("Found pickled chunker... loading it.")
			dump = open("dumps/chunker.pkl", 'rb')
			self.classifier = pickle.load(dump)
			dump.close()
		else:
			train_data = []
			train_targets = []

			for tag_question in train_set:
				stripped_chunk_tags = nltk.tag.untag(tag_question)
				for i in range(len(tag_question)):
					features = chunk_features(stripped_chunk_tags, i)
					train_data.append(features)
					train_targets.append(tag_question[i][1])
			
			self.classifier = Pipeline([('vect', DictVectorizer()),										
	                      				('clf', SGDClassifier(n_jobs=-1))])	
			self.classifier.fit_transform(train_data, train_targets)
			dump = open('dumps/chunker.pkl', 'wb')
			pickle.dump(self.classifier, dump)
			dump.close()		

	#Given a POS tagged sentence, additionally tag the sentence with POS chunks
	def tag(self, tag_question):
		return_tag = []
		for i in range(len(tag_question)):
			features = chunk_features(tag_question, i)
			chunk = self.classifier.predict(features)[0]
			return_tag.append((tag_question[i],chunk))
		return return_tag

#A POS chunker that uses a trained chunk tagger to tag sentences.
class PosChunker(nltk.ChunkParserI):

	#Train set should be a collection of chunk-tagged sentences in tree format.
	def __init__(self, train_set):
		train_iob = [[((word, tag), chunk) for (word,tag,chunk) in nltk.chunk.tree2conlltags(sent)] for sent in train_set]
		self.tagger = ChunkTagger(train_iob)

	def parse(self, tag_question):
		chunked_question = self.tagger.tag(tag_question)
		conlltags = [(word, tag, chunk) for ((word, tag), chunk) in chunked_question] # Must convert back to tree in order for the
		return nltk.chunk.conlltags2tree(conlltags)									  # built-in evaluate() function to work properly.