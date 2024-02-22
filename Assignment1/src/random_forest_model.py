from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import confusion_matrix
from skimage.io import imread
from skimage.transform import resize
from tensorflow.keras.models import Model


import random
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import seaborn as sns

def unpickle_data(filename):
    with open(filename, "rb") as file:
        data = pkl.load(file)
    return data

X_train = unpickle_data("X_train.pkl")
X_test = unpickle_data("X_test.pkl")
y_train = unpickle_data("y_train.pkl")
y_test = unpickle_data("y_test.pkl")

rfc = rfc(n_estimators=10, random_state=50)

rfc.fit(X_train, y_train)

# Save Model
with open("rfc.pkl", "wb") as file:
    pkl.dump(rfc, file)

