from sklearn.grid_search import GridSearchCV

#returns the parameters used for classification
#runs an exhaustive search in which all possible parameters are tested for learning
def grid_search_classifier(classifier):
    parameters = {'vect__ngram__range': [(1, 1), (1, 2)],
                  'tfidf__use_idf': (True, False),
                  'clf__alpha': (1e-2, 1e-3),
    }

    gs_classifier = GridSearchCV(classifier, parameters, n_jobs=-1)

    return gs_classifier, parameters

#returns a dict of the best parameters used in training
#parameter names are the keys from the parameters dict argument
#value is the best value for that parameter
def determine_best_scores(gs_classifier, parameters):
    best_parameters, score, _ = max(gs_classifier.grid_scores_, key=lambda x:x[1])
    return best_parameters, score
