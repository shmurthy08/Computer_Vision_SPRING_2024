import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from scipy.ndimage import zoom

# Function to preprocess images and masks
def preprocess_images_and_masks(dataset_dir, label_colors, desired_size=(256, 256, 3)):
    images = []
    masks = []
    
    # Iterate through the dataset directory
    for file in sorted(os.listdir(dataset_dir)):
        if file.endswith(".png"):
            if file.endswith("_L.png"):
                # Load and preprocess mask
                mask_path = os.path.join(dataset_dir, file)
                mask = np.array(Image.open(mask_path))
                
                
                # Encode mask using label colors
                encoded_mask = encode_mask(mask, label_colors)
                
                # Resize mask
                mask_resized = np.array(zoom(mask, (desired_size[0] / mask.shape[0], desired_size[1] / mask.shape[1], 1)))
                masks.append(mask_resized.flatten())
                
            else:
                # Load and preprocess image
                image_path = os.path.join(dataset_dir, file)
                image = np.array(Image.open(image_path))
                
                # Resize image
                image_resized = np.array(zoom(image, (desired_size[0] / image.shape[0], desired_size[1] / image.shape[1], 1)))
                images.append(image_resized.flatten())  # Append resized image
    
    # Convert lists to numpy arrays
    images = np.array(images)
    masks = np.array(masks)
    # Shuffle images and masks together
    images_shuffled, masks_shuffled = shuffle(images, masks, random_state=42)
    return images_shuffled, masks_shuffled


# Function to encode mask using label colors
def encode_mask(mask, label_colors):
    encoded_mask = np.zeros_like(mask, dtype=np.uint8)
    for label, color in label_colors.items():
        encoded_mask[(mask == color).all(axis=-1)] = label
    return encoded_mask

# Define directory containing the dataset
dataset_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "CamSeq07")

# Define label colors dictionary
label_colors = {
    0: [64, 128, 64], 1: [192, 0, 128], 2: [0, 128, 192], 3: [0, 128, 64],
    4: [128, 0, 0], 5: [64, 0, 128], 6: [64, 0, 192], 7: [192, 128, 64],
    8: [192, 192, 128], 9: [64, 64, 128], 10: [128, 0, 192], 11: [192, 0, 64],
    12: [128, 128, 64], 13: [192, 0, 192], 14: [128, 64, 64], 15: [64, 192, 128],
    16: [64, 64, 0], 17: [128, 64, 128], 18: [128, 128, 192], 19: [0, 0, 192],
    20: [192, 128, 128], 21: [128, 128, 128], 22: [64, 128, 192], 23: [0, 0, 64],
    24: [0, 64, 64], 25: [192, 64, 128], 26: [128, 128, 0], 27: [192, 128, 192],
    28: [64, 0, 64], 29: [192, 192, 0], 30: [0, 0, 0], 31: [64, 192, 0]
}

# Preprocess images and masks
images, masks = preprocess_images_and_masks(dataset_dir, label_colors)

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
