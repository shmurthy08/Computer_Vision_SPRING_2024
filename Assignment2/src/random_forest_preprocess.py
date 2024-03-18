import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from scipy.ndimage import zoom

# Function to preprocess images and masks
def preprocess_images_and_masks(dataset_dir, desired_size=(256, 256, 3)):
    images = []
    masks = []
    
    # Iterate through the dataset directory
    for file in sorted(os.listdir(dataset_dir)):
        if file.endswith(".png"):
            if file.endswith("_L.png"):
                # Load and preprocess mask
                mask_path = os.path.join(dataset_dir, file)
                mask = np.array(Image.open(mask_path))
                
                
                # Resize mask
                mask_resized = np.array(zoom(mask, (desired_size[0] / mask.shape[0], desired_size[1] / mask.shape[1], 1)))
                masks.append(mask_resized.flatten())
                
            else:
                # Load and preprocess image
                image_path = os.path.join(dataset_dir, file)
                image = np.array(Image.open(image_path))
                
                # Resize image
                image_resized = np.array(zoom(image, (desired_size[0] / image.shape[0], desired_size[1] / image.shape[1], 1)))
                image_resized = image_resized / 255.0  # Normalize image
                images.append(image_resized.flatten())  # Append resized image
    
    # Convert lists to numpy arrays
    images = np.array(images)
    masks = np.array(masks)
    # Shuffle images and masks together
    images_shuffled, masks_shuffled = shuffle(images, masks, random_state=42)
    return images_shuffled, masks_shuffled




# Define directory containing the dataset
dataset_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "CamSeq07")

# Preprocess images and masks
images, masks = preprocess_images_and_masks(dataset_dir)

# Check shapes and other information
print("Shape of images array:", images.shape)
print("Shape of masks array:", masks.shape)



# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, masks, test_size=0.3, random_state=42)

# Save preprocessed data
np.savez("preprocessed_data_for_RF.npz", X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

print("Preprocessing complete.")

# Print shapes and other information
print("Shape of X_train:", X_train.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_test:", y_test.shape)
print("Number of images:", len(images))
print("Number of masks:", len(masks))
