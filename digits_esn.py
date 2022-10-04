#!/usr/bin/env python
from sklearn.model_selection import train_test_split
from pyrcn.datasets import load_digits
from pyrcn.echo_state_network import ESNClassifier
import numpy as np

X, y = load_digits(return_X_y=True, as_sequence=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 
# X_train is training data and y_train is labels for that training data 
# X_test is testing data and y_test is labels for that test data

clf = ESNClassifier()
clf.fit(X=X_train, y=y_train) # Returns a trained classifier

y_pred_classes = clf.predict(X=X_test)  # output is the class for each input example
y_pred_proba = clf.predict_proba(X=X_test)  #  output are the class probabilities for each input example

# print(y_pred_classes - y_test)

diff = np.array(y_pred_classes - y_test)

res = len(diff) - np.count_nonzero(diff)

print("Accuracy: " + str((res/len(diff))*100) + "%")
# print(np.count_nonzero(diff))

