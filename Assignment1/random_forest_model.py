from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import confusion_matrix


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



# Train Data Analysis
y_pred_train = rfc.predict(X_train)

# Accuracy score for train data
acc = accuracy_score(y_train, y_pred_train)
print("Accuracy for Train Data: ", acc)

# Classification report for train data
print("Classification report for Train Data: \n", classification_report(y_train, y_pred_train))

#--------------------------------------------------------------

# Test Data Analysis
y_pred_test = rfc.predict(X_test)

# Accuracy score for test data
acc = accuracy_score(y_test, y_pred_test)
print("Accuracy for Test Data: ", acc)

# Classification report for test data
print("Classification report for Test Data: \n", classification_report(y_test, y_pred_test))


# Plotting Confusion Matricies

# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    total_predictions = np.sum(cm)
    accuracy = np.trace(cm) / total_predictions * 100

    classes = np.unique(y_true)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted labels\nAccuracy: {:.2f}%'.format(accuracy))
    plt.ylabel('True labels')
    plt.title(title)
    plt.savefig(title + ".png")
    plt.show()
    
    
plot_confusion_matrix(y_train, y_pred_train, title="Confusion Matrix for Train Data")
plot_confusion_matrix(y_test, y_pred_test, title="Confusion Matrix for Test Data")