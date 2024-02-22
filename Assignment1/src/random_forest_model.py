from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import confusion_matrix
from skimage.io import imread
from skimage.transform import resize

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

rfc = rfc(n_estimators=1, random_state=50)

rfc.fit(X_train, y_train)

#--------------------------------------------------------------

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
   
    
    
plot_confusion_matrix(y_train, y_pred_train, title="Confusion Matrix for Train Data")
plot_confusion_matrix(y_test, y_pred_test, title="Confusion Matrix for Test Data")


#--------------------------------------------------------------

# Define the directory containing the prediction images
pred_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "Intel_Image_classification", "seg_pred", "seg_pred")

# Get a list of image file names in the directory
image_files = os.listdir(pred_dir)

# Randomly select 10 images
random_images = random.sample(image_files, 10)

# List to store predictions
predictions = []

# Loop through each randomly selected image file
plt.figure(figsize=(15, 8))
for i, image_file in enumerate(random_images, 1):
    # Load the image
    image_path = os.path.join(pred_dir, image_file)
    image = imread(image_path)
    image = resize(image, output_shape=(256,256), anti_aliasing=True)
    image = image.flatten()
    image = image.reshape(1, -1)
    
    
    # Predict class
    pred_class = rfc.predict(image)[0]
    
    # Map numeric label to class name
    if pred_class == 0:
        pred_class = "buildings"
    elif pred_class == 1:
        pred_class = "forest"
    elif pred_class == 2:
        pred_class = "glacier"
    elif pred_class == 3:
        pred_class = "mountain"
    elif pred_class == 4:
        pred_class = "sea"
    elif pred_class == 5:
        pred_class = "street"
    
    
    # Visualize the image along with its predicted class
    plt.subplot(2, 5, i)
    plt.imshow(image.reshape(256,256,3))
    plt.title(f"Predicted Class: {pred_class}")
    plt.axis('on')

    
    # Store the predictions
    predictions.append(pred_class)


plt.tight_layout()
plt.savefig("rfc_pred_" + ".png")


print(predictions)

