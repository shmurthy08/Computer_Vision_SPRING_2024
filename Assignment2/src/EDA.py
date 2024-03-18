import os
import cv2

import matplotlib.pyplot as plt

dataset_dir = 'CamSeq07'  # Replace with the actual path to your dataset directory

image_sizes = []
mask_sizes = []

# Iterate over the images and masks
for filename in sorted(os.listdir(dataset_dir)):
    if filename.endswith('.png'):
        if filename.endswith('_L.png'):
            image_path = os.path.join(dataset_dir, filename)
            mask_path = os.path.join(dataset_dir, filename.replace('_L.png', '.png'))

            # Read the image and mask
            image = cv2.imread(image_path)
            mask = cv2.imread(mask_path)

            # Get the sizes of the image and mask
            image_size = image.shape[:2]  # Use [:2] to get width and height only
            mask_size = mask.shape[:2]  # Use [:2] to get width and height only

            # Append the sizes to the respective lists
            image_sizes.append(image_size)
            mask_sizes.append(mask_size)

            # Plot the image and mask
            plt.figure()
            plt.subplot(1, 2, 1)
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.title('Mask')
            plt.subplot(1, 2, 2)
            plt.imshow(cv2.cvtColor(mask, cv2.COLOR_BGR2RGB))
            plt.title('Image')

            # Save the plot as an image
            plt.savefig('example_image_and_mask.png')

# Print the sizes of the images and masks
print("Image Sizes:")
for size in image_sizes:
    print(size)

print("\nMask Sizes:")
for size in mask_sizes:
    print(size)