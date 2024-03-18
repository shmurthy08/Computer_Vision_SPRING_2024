import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
from sklearn.metrics import accuracy_score, classification_report

# Load preprocessed data
preprocessed_data = np.load("preprocessed_data_for_RF.npz")
X_test, y_test = preprocessed_data['X_test'], preprocessed_data['y_test']
X_train, y_train = preprocessed_data['X_train'], preprocessed_data['y_train']

# Load the trained Random Forest model
with open("rf_model.pkl", "rb") as f:
    rf_model = pkl.load(f)


# Predict the train set
y_pred_train = rf_model.predict(X_train)

# Predict the test set
y_pred = rf_model.predict(X_test)


# ------------- Print Sample Images ---------------

# Select a random index
idx = np.random.randint(len(X_test))

# Predict on a single image
y_output = rf_model.predict(X_test[idx].reshape(1, -1))

# Reshape the output to match the original mask shape
y_output = y_output.reshape(256, 256, 3)

# Plot the image, actual mask, and predicted mask
plt.figure(figsize=(12, 6))

# Original image
plt.subplot(1, 3, 1)
plt.imshow(X_test[idx].reshape(256, 256, 3))
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(y_test[idx].reshape(256, 256, 3))
plt.title('Original Mask')
plt.axis('off')
# Predicted mask
plt.subplot(1, 3, 3)
plt.imshow(y_output, cmap='jet', vmin=0, vmax=31)  
plt.title('Predicted Mask')
plt.axis('off')
plt.tight_layout()
plt.savefig("predicted_mask.png")
plt.show()


### Analysis for Test and Train

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

# Decode the encoded masks
def decode_mask(encoded_mask, label_colors):
    decoded_mask = np.array(encoded_mask.shape[:2])
    # Initialize decoded mask with the same shape as the encoded mask   
    decoded_mask = np.zeros(encoded_mask.shape[:2], dtype=np.uint8)

    # Iterate over each pixel in the encoded mask
    for i in range(encoded_mask.shape[0]):
        for j in range(encoded_mask.shape[1]):
            pixel_color = tuple(encoded_mask[i, j])  # Get the color code of the current pixel
            # Find the corresponding label in the label colors dictionary
            for label, color in label_colors.items():
                if pixel_color == tuple(color):
                    decoded_mask[i, j] = label  
                    break  

    return decoded_mask

# Flatten the ground truth and predicted labels
y_test_decoded = np.array([decode_mask(mask.reshape(256, 256, 3), label_colors) for mask in y_test])
y_pred_decoded = np.array([decode_mask(mask.reshape(256, 256, 3), label_colors) for mask in y_pred])

y_train_decoded = np.array([decode_mask(mask.reshape(256, 256, 3), label_colors) for mask in y_train])
y_pred_train_decoded = np.array([decode_mask(mask.reshape(256, 256, 3), label_colors) for mask in y_pred_train])


# Flatten the decoded masks
y_test_flattened = y_test_decoded.flatten()
y_pred_flattened = y_pred_decoded.flatten()

y_train_flattened = y_train_decoded.flatten()
y_pred_train_flattened = y_pred_train_decoded.flatten()

# Find unique classes in flattened test and prediction arrays
unique_classes_test = np.unique(y_test_flattened)
unique_classes_pred = np.unique(y_pred_flattened)

# Print unique classes
print("Unique classes in flattened test array:", unique_classes_test)
print("Unique classes in flattened prediction array:", unique_classes_pred)


print("Test Classification Report: ", classification_report(y_test_flattened, y_pred_flattened), flush=True)
print("Train Classification Report: ", classification_report(y_train_flattened, y_pred_train_flattened), flush=True)
# Accuracy score
accuracy = accuracy_score(y_test_flattened, y_pred_flattened)
print("Test Accuracy: ", accuracy, flush=True)

# IoU score
intersection = np.logical_and(y_test_decoded, y_pred_decoded)
union = np.logical_or(y_test_decoded, y_pred_decoded)
iou_score = np.sum(intersection) / np.sum(union)
print("Test IoU Score: ", iou_score, flush=True)


# Accuracy score
accuracy = accuracy_score(y_train_flattened, y_pred_train_flattened)
print("Train Accuracy: ", accuracy)

# IoU score
intersection = np.logical_and(y_train_decoded, y_pred_train_decoded)
union = np.logical_or(y_train_decoded, y_pred_train_decoded)
iou_score = np.sum(intersection) / np.sum(union)
print("Train IoU Score: ", iou_score)

