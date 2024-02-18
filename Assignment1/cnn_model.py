import numpy as np
import tensorflow as tf 
import os
import matplotlib.pyplot as plt

from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model 
from tensorflow.keras.optimizers import Adam

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Parameters for images
img_size = (256,256)
batch_size = 64
num_classes = 6
epochs = 1

# Load VGG16
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(256,256,3))

# Freeze layers to prevent weight updates
for layer in base_model.layers:
    layer.trainable = False
    
# Add custom layers
x = Flatten()(base_model.output)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
x = BatchNormalization()(x)
output = Dense(num_classes, activation="softmax")(x)

# combine base and custom
model = Model(inputs=base_model.input, outputs=output)

# Early Stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Compile
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])


#--------------------------------------------

# Load and Preprocess Data

# Define the directories
train_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "Intel_Image_classification", "seg_train", "seg_train")
test_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "Intel_Image_classification", "seg_test", "seg_test")


train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical')
test_gen = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical')



# Fit the model
model.fit(
    train_gen,
    validation_data=test_gen,
    epochs=epochs,
    steps_per_epoch=len(train_gen),
    validation_steps=len(test_gen), 
    callbacks=[early_stopping]
)

# Save the model
model.save("model.h5")