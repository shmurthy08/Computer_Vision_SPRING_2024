import numpy as np
import tensorflow as tf 
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model 
from tensorflow.keras.optimizers import Adam

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split



# Preprocessing:

# Parameters for images
img_size = (256,256)
batch_size = 64
num_classes = 6
epochs = 10

# Define the directories
train_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "Intel_Image_classification", "seg_train", "seg_train")
test_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "Intel_Image_classification", "seg_test", "seg_test")

# List all subdirectories (classes) in the train directory
classes = os.listdir(train_dir)

# Create a list to hold the file paths of all images in the train directory
train_files = []
for cls in classes:
    cls_dir = os.path.join(train_dir, cls)
    cls_files = [os.path.join(cls_dir, f) for f in os.listdir(cls_dir)]
    train_files.extend(cls_files)

# Create a DataFrame with file paths and labels to ensure proper train test split
train_df = pd.DataFrame({'filepath': train_files, 'label': [os.path.basename(os.path.dirname(f)) for f in train_files]})

# Train-test split for the DataFrame
train_df, val_df = train_test_split(train_df, test_size=0.2, shuffle=True, random_state=42)   

# Define ImageDataGenerator for train and validation data
train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input, horizontal_flip=True, rotation_range=30 ,fill_mode='nearest')
val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input, horizontal_flip=True, rotation_range=30 ,fill_mode='nearest')
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

# Flow from DataFrame for train and validation data
train_gen = train_datagen.flow_from_dataframe(
    train_df,
    x_col='filepath',
    y_col='label',
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)
val_gen = val_datagen.flow_from_dataframe(
    val_df,
    x_col='filepath',
    y_col='label',
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)
test_gen = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False)





#--------------------------------------------
# Model Building: VGG16




# Load VGG16
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(256,256,3))

# Fine-tuning: Unfreeze some layers of VGG16
for layer in base_model.layers:
    layer.trainable = False

# Add custom layers
x = Flatten()(base_model.output)
x = Dense(1024, activation="relu")(x)
x = Dropout(0.5)(x)
x = BatchNormalization()(x)
x = Dense(512, activation="relu")(x)
x = BatchNormalization()(x)
x = Dense(256, activation="relu")(x)
x = BatchNormalization()(x)
output = Dense(num_classes, activation="softmax")(x)

# combine base and custom
model = Model(inputs=base_model.input, outputs=output)

# Early Stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

# Compile
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])


#--------------------------------------------

# Fit the model
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=epochs,
    steps_per_epoch=len(train_gen),
    validation_steps=len(val_gen), 
    callbacks=[early_stopping]
)

# Save the model
model.save("model.h5")

# Plot training and validation loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.savefig("Training and Validation Loss of CNN Model" + ".png")
plt.show()


# -----------------------------------------------

# Make evaluation on the test data
model.evaluate(test_gen)

# Confusion Matrix for test dir
# Run for each file in the test directory and create matrix
predictions = model.predict(test_gen)

y_pred = np.argmax(predictions, axis=1)
y_true = test_gen.classes


# Get the confusion matrix
cm = confusion_matrix(y_true, y_pred)
total_predictions = np.sum(cm)
accuracy = np.trace(cm) / total_predictions * 100

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.xlabel('Predicted labels\nAccuracy: {:.2f}%'.format(accuracy))
plt.ylabel('True labels')
plt.title("Confusion Matrix - CNN Model for Test Data")
plt.savefig("Confusion Matrix of Test Data for CNN Model" + ".png")
plt.show()


# Classification report for test data
print("Classification report for Test Data: \n", classification_report(y_true, y_pred))
print("Legend for the classes: ")
print("(0: 'buildings', 1: 'forest', 2: 'glacier', 3: 'mountain', 4: 'sea', 5: 'street')")