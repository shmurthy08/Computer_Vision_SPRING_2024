import os
import numpy as np
import tensorflow as tf 
import matplotlib.pyplot as plt
import random
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('model.h5')

# Define the directory containing the prediction images
pred_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "Intel_Image_classification", "seg_pred", "seg_pred")

# Get a list of image file names in the directory
image_files = os.listdir(pred_dir)

# Randomly select 10 images
random_images = random.sample(image_files, 10)

# List to store predictions
predictions = []

# Loop through each randomly selected image file
for image_file in random_images:
    # Load the image
    image_path = os.path.join(pred_dir, image_file)
    image = load_img(image_path, target_size=(256, 256))
    image = img_to_array(image)
    image = preprocess_input(image)
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    
    # Predict class probabilities
    pred_probs = model.predict(image)
    
    # Get the predicted class label
    predicted_class = np.argmax(pred_probs)
    if predicted_class == 0:
        predicted_class = "buildings"
    elif predicted_class == 1:
        predicted_class = "forest"
    elif predicted_class == 2:
        predicted_class = "glacier"
    elif predicted_class == 3:
        predicted_class = "mountain"
    elif predicted_class == 4:
        predicted_class = "sea"
    elif predicted_class == 5:
        predicted_class = "street"
    
    # Optionally, you can print or visualize the image along with its predicted class
    plt.imshow(load_img(image_path))
    plt.title(f"Predicted Class: {predicted_class}")
    plt.axis('off')
    plt.savefig(image_file+'_prediction.png')
    plt.show()
    
    # Store the predictions
    predictions.append(predicted_class)

# Print the predictions
print(predictions)