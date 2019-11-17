import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from pprint import pprint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import ShuffleSplit
import matplotlib.pyplot as plt
# import seaborn as sns
import pandas as pd


# Dataframe
path_df = "otApp/Pickles/df.pickle"
with open(path_df, 'rb') as data:
  df = pickle.load(data)

# features_train
path_features_train = "otApp/Pickles/features_train.pickle"
with open(path_features_train, 'rb') as data:
  features_train = pickle.load(data)

# labels_train
path_labels_train = "otApp/Pickles/labels_train.pickle"
with open(path_labels_train, 'rb') as data:
  labels_train = pickle.load(data)

# features_test
path_features_test = "otApp/Pickles/features_test.pickle"
with open(path_features_test, 'rb') as data:
  features_test = pickle.load(data)

# labels_test
path_labels_test = "otApp/Pickles/labels_test.pickle"
with open(path_labels_test, 'rb') as data:
  labels_test = pickle.load(data)

sentiment_codes = {
    0: 'worry',
    1: 'sadness',
    2: 'hate',
    3: 'empty'
}
# Create the parameter grid based on the results of random search
# C = [.0001, .001, .01, .1]
# degree = [3, 4, 5]
# gamma = [1, 10, 100]
# probability = [True]

# param_grid = [
#     {'C': C, 'kernel': ['linear'], 'probability':probability},
#     {'C': C, 'kernel': ['poly'], 'degree':degree, 'probability':probability},
#     {'C': C, 'kernel': ['rbf'], 'gamma':gamma, 'probability':probability}
# ]
# cv_sets = ShuffleSplit(n_splits=3, test_size=.33, random_state=8)

# svc = svm.SVC(random_state=8)

# # Instantiate the grid search model
# grid_search = GridSearchCV(estimator=svc,
#                            param_grid=param_grid,
#                            scoring='accuracy',
#                            cv=cv_sets,
#                            verbose=1)
# grid_search.fit(features_train, labels_train)
# best_svc = grid_search.best_estimator_

# best_svc.fit(features_train, labels_train)
# svc_pred = best_svc.predict(features_test)

# # Training accuracy
# print("The training accuracy is: ")
# print(accuracy_score(labels_train, best_svc.predict(features_train)))

# # Test accuracy
# print("The test accuracy is: ")
# print(accuracy_score(labels_test, svc_pred))
# -------------------------------multinomial nb----------------------
print(features_train.shape)
print(features_test.shape)

mnbc = MultinomialNB()
print(mnbc)

mnbc.fit(features_train, labels_train)


save_classifier = open("naivebayes.pickle", "wb")
pickle.dump(mnbc, save_classifier)
save_classifier.close()
# ------------------------------------------------------------------


def check_emo(inp):
  classifier_f = open("naivebayes.pickle", "rb")
  classifier = pickle.load(classifier_f)
  classifier_f.close()
  path_tfidf = "otApp/Pickles/tfidf.pickle"
  with open(path_tfidf, 'rb') as data:
    tfidf = pickle.load(data)
  inp_test = tfidf.transform([inp]).toarray()
  inp_pred = classifier.predict(inp_test)
  return sentiment_codes[int(inp_pred)]


mnbc_pred = mnbc.predict(features_test)

# Training accuracy
print("The training accuracy is: ")
print(accuracy_score(labels_train, mnbc.predict(features_train)))

# Test accuracy
print("The test accuracy is: ")
print(accuracy_score(labels_test, mnbc_pred))

# Classification report
print("Classification report")
print(classification_report(labels_test, mnbc_pred))
