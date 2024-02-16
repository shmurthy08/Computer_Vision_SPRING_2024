from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score
from skimage.io import imread
from skimage.transform import resize

import os
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl

# Load the data - The dataset used is the Intel Image Classification dataset
# there is a seg_pred, seg_test, seg_train folder in the dataset

# Define the directories
train_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "Intel_Image_classification", "seg_train", "seg_train")
test_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "Intel_Image_classification", "seg_test", "seg_test")

#pickle the data
def pickle_data(data, filename):
    with open(filename, "wb") as file:
        pkl.dump(data, file)
        


def load_imgs(dire):
    images = []
    labels = []
    for label in os.listdir(dire):
        for img in os.listdir(os.path.join(dire, label)):
            img_path = os.path.join(dire, label, img)
            img = imread(img_path)
            img = resize(img, output_shape=(256,256), anti_aliasing=True)
            #flatten image
            img = img.flatten()
            images.append(img)
            labels.append(label)
    return np.array(images), np.array(labels)




# TTS
X_train, y_train = load_imgs(train_dir)

X_test, y_test = load_imgs(test_dir)

# Pickle the data
pickle_data(X_train, "X_train.pkl")
pickle_data(X_test, "X_test.pkl")
pickle_data(y_train, "y_train.pkl")
pickle_data(y_test, "y_test.pkl")

print("Data pickled, good luck creating the model!")


        




