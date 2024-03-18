import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
import tensorflow as tf
import tf_explain.core.grad_cam as grad_cam

# Define categorical dice loss function
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
def categorical_dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

smooth = 1.

# Define Jaccard index (IoU) metric
def jaccard_index(y_true, y_pred):
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3]) - intersection
    return K.mean((intersection + smooth) / (union + smooth), axis=0)



load_data = np.load('dataset.npz')
X_test, y_test = load_data['X_test'], load_data['y_test']

y_out = np.argmax(y_test, axis=-1)
print(len(y_out), len(y_test))
# Define the custom objects dictionary with the custom loss function
custom_objects = {'categorical_dice_loss': categorical_dice_loss, 'dice_coef': dice_coef, 'jaccard_index': jaccard_index}

# Load the model with the custom loss function
vgg_model = load_model("vgg_unet_model.h5", custom_objects=custom_objects)

# ------------- Decoding Masks ---------------
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



# Decode masks
def decode_mask(encoded_mask, label_colors):
    # Initialize decoded_mask with zeros
    decoded_mask = np.zeros((encoded_mask.shape[0], encoded_mask.shape[1], 3), dtype=np.uint8)
    # Iterate over each color and set the decoded mask to the corresponding color
    for k in label_colors.keys():
        decoded_mask[encoded_mask == k] = label_colors[k]
    return decoded_mask


loss, dice_coef, jaccard_index, accuracy = vgg_model.evaluate(X_test, y_test)
print("Test Accuracy: {:.3f}".format(accuracy))
print("Test Jaccard Index: {:.3f}".format(jaccard_index))
print("Test Dice Coefficient: {:.3f}".format(dice_coef))
print("Test Loss: {:.3f}".format(loss))
predicted_masks = vgg_model.predict(X_test)

# Convert every decoded mask to RGB using the label colors
single_layer = np.argmax(predicted_masks, axis=-1)

# Initialize decoded_masks with the correct shape
decoded_masks = np.array([decode_mask(mask, label_colors) for mask in single_layer])

# Initialize decoded_masks with the correct shape for y_test
decoded_y_test = np.array([decode_mask(mask, label_colors) for mask in y_out])
plt.imshow(decoded_masks[0])
plt.savefig('cnn_decoded_mask.png')


# ------------- Visualize the Segmentation ---------------

# Visualization
def visualize_segmentation(image, original_mask, mask, label_colors):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title("Image")
    plt.axis("off")
    
    plt.subplot(1, 3, 2)
    plt.imshow(original_mask)
    plt.title("Original Mask")
    plt.axis("off")
    
    plt.subplot(1, 3, 3)
    plt.imshow(mask)
    plt.title("Segmentation Mask")
    plt.axis("off")
    plt.savefig('cnn_segmentation.png')
    plt.show()

# # Choose a random sample for visualization
sample_index = np.random.randint(len(X_test))
visualize_segmentation(X_test[sample_index], decoded_y_test[sample_index], decoded_masks[sample_index], label_colors)


# ------------- Saliency Map and GradCAM ---------------
def compute_saliency_map(model, input_image):
    input_tensor = tf.convert_to_tensor(input_image)
    input_tensor = tf.expand_dims(input_tensor, axis=0)  # Add batch dimension
    with tf.GradientTape() as tape:
        tape.watch(input_tensor)
        predictions = model(input_tensor)
        predicted_class = tf.argmax(predictions, axis=-1)  # Get predicted class for each pixel
        predicted_class = tf.expand_dims(predicted_class, axis=-1)  # Add channel dimension

    gradients = tape.gradient(predictions, input_tensor)
    saliency_map = tf.reduce_max(tf.abs(gradients), axis=-1)  # Take maximum gradient across channels

    return saliency_map


# Choose a random sample for saliency map generation
sample_index = np.random.randint(len(X_test))
input_image = X_test[sample_index]

# Compute saliency map
saliency_map = compute_saliency_map(vgg_model, input_image)

print("Input Image Shape:", input_image.shape)
print(saliency_map[0])

# Plot the input image and the saliency map
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(input_image)
plt.title("Input Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(saliency_map[0], cmap='viridis')
plt.title("Saliency Map")
plt.axis("off")
plt.savefig('cnn_saliency_map.png')
plt.show()

# GradCAM using tf-explain
# Create a GradCAM explainer
explainer = grad_cam.GradCAM()


for i in range(9):
    # Call to explain() method
    j = i + 1
    output = explainer.explain((X_test, y_test), vgg_model, None, f"conv2d_{j}")

    # Save output
    explainer.save(output, '.', f'cnn_gradcam_{j}.png')





