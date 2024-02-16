from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


import os
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl


def unpickle_data(filename):
    with open(filename, "rb") as file:
        data = pkl.load(file)
    return data

X_train = unpickle_data("X_train.pkl")
X_test = unpickle_data("X_test.pkl")
y_train = unpickle_data("y_train.pkl")
y_test = unpickle_data("y_test.pkl")

rfc = rfc(n_estimators=1, random_state=50)

rfc.fit(X_train, y_train)

y_pred = rfc.predict(X_test)

# Accuracy score
acc = accuracy_score(y_test, y_pred)
print("Accuracy: ", acc)

# Classification report
print(classification_report(y_test, y_pred))