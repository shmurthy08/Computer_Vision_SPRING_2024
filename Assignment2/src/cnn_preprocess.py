import os
import numpy as np
from PIL import Image
from scipy.ndimage import zoom
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

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
                mask_encoded = RGBtoLabel(mask, label_colors)
                # Resize mask using nearest-neighbor interpolation
                mask_resized = np.array(zoom(mask_encoded, (desired_size[0] / mask_encoded.shape[0], desired_size[1] / mask_encoded.shape[1], 1), order=3))
                masks.append(mask_resized)
                
            else:
                # Load and preprocess image
                image_path = os.path.join(dataset_dir, file)
                image = np.array(Image.open(image_path))
                
                # Resize image
                image_resized = np.array(zoom(image, (desired_size[0] / image.shape[0], desired_size[1] / image.shape[1], 1), order=0))
                image_resized = image_resized / 255.0  # Normalize image
                images.append(image_resized)  # Append resized image
    
    # Convert lists to numpy arrays
    images = np.array(images)
    masks = np.array(masks)
    
    # Shuffle images and masks together
    images_shuffled, masks_shuffled = shuffle(images, masks, random_state=42)
    return images_shuffled, masks_shuffled

def RGBtoLabel(rgb_image, colormap, num_classes=32):
    num_classes = len(colormap)
    shape = rgb_image.shape[:2]+(num_classes,)
    encoded_image = np.zeros( shape, dtype=np.float32 )
    for i, cls in enumerate(colormap):
        encoded_image[:,:,i] = np.all(rgb_image.reshape( (-1,3) ) == colormap[i], axis=1).reshape(shape[:2])
    return encoded_image

# Define directory containing the dataset
dataset_dir = "CamSeq07"

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

# Define the size of each set
total_samples = len(images)
train_size = int(0.65 * total_samples)
val_size = int(0.15 * total_samples)

# Split the data into train, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(images, masks, test_size=(1 - train_size / total_samples), random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=val_size / (total_samples - train_size), random_state=42)

# Check the sizes of the sets
print("Train set size:", len(X_train))
print("Validation set size:", len(X_val))
print("Test set size:", len(X_test))

# Save train, validation, and test sets
np.savez("dataset.npz", X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, X_test=X_test, y_test=y_test)
