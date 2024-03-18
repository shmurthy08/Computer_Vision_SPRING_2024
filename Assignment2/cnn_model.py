import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Conv2D, UpSampling2D, Concatenate, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ReduceLROnPlateau
import matplotlib.pyplot as plt


# Load Preprocessed data
load_data = np.load('dataset.npz')
X_train, y_train, X_val, y_val, X_test, y_test = load_data['X_train'], load_data['y_train'], load_data['X_val'], load_data['y_val'], load_data['X_test'], load_data['y_test']

# Define dice coefficient metric
def dice_coef(y_true, y_pred):
    """""
        Dice coefficient metric
        :param y_true: ground truth mask
        :param y_pred: predicted mask
        :return: dice coefficient
    """""
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

# Define Jaccard index (IoU) metric
def jaccard_index(y_true, y_pred):
    """""
        Jaccard index (IoU) metric
        :param y_true: ground truth mask
        :param y_pred: predicted mask
        :return: jaccard index
    """""
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3]) - intersection
    return K.mean((intersection + smooth) / (union + smooth), axis=0)


# Define dice coefficient loss function
def categorical_dice_loss(y_true, y_pred):
    """""
        Categorical dice loss function
        :param y_true: ground truth mask
        :param y_pred: predicted mask
        :return: dice loss
    """""
    return 1 - dice_coef(y_true, y_pred)

smooth = 1.

# Define U-Net architecture with transfer learning using VGG16
def unet_vgg(input_shape):
    """""
        U-Net architecture with transfer learning using VGG16
        :param input_shape: input shape of the model
        :return: U-Net model with VGG16 encoder
    """""
    
    
    # Load pre-trained VGG16 model as encoder
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)

    # Freeze weights of the encoder
    for layer in base_model.layers:
        layer.trainable = False

    # Get encoder output
    encoder_output = base_model.output

    # Decoder part
    x = Conv2D(512, 3, activation='relu', padding='same')(encoder_output)
    x = BatchNormalization()(x)
    x = UpSampling2D((4, 4))(x)
    x = Concatenate()([x, base_model.get_layer('block4_conv3').output])
    x = Conv2D(512, 3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(512, 3, activation='relu', padding='same')(x)



    x = UpSampling2D((2, 2))(x)
    x = Concatenate()([x, base_model.get_layer('block3_conv3').output])
    x = Conv2D(256, 3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(256, 3, activation='relu', padding='same')(x)

    x = UpSampling2D((2, 2))(x)
    x = Concatenate()([x, base_model.get_layer('block2_conv2').output])
    x = Conv2D(128, 3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, 3, activation='relu', padding='same')(x)

    x = UpSampling2D((2, 2))(x)
    x = Concatenate()([x, base_model.get_layer('block1_conv2').output])
    x = Conv2D(64, 3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)    
    x = Conv2D(64, 3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)

    # Output layer
    output = Conv2D(32, 1, activation='softmax', padding='same')(x)

    # Create and compile model
    model = Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer='adam', loss=categorical_dice_loss, metrics=[dice_coef, jaccard_index, 'accuracy'])



    return model


# Define input shape
input_shape = (256, 256, 3)

# Instantiate the VGG16 model
vgg_model = unet_vgg(input_shape)

# Define callbacks for VGG16 model
vgg_checkpoint_path = "vgg_model_checkpoint.h5"
vgg_checkpoint = ModelCheckpoint(vgg_checkpoint_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
vgg_lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=7, min_lr=1e-6, verbose=1)

# Train the VGG16 model
history = vgg_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=75, batch_size=32, callbacks=[vgg_checkpoint, vgg_lr_scheduler])

# Plot the training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Loss and Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Loss/Accuracy')
plt.legend()
plt.savefig("vgg_model_loss_accuracy.png")

# Load best weights for VGG16 model
vgg_model.load_weights(vgg_checkpoint_path)

# Evaluate the VGG16 model
vgg_model.evaluate(X_test, y_test)

# Save the entire VGG16 model
vgg_model.save("history_vgg_unet_model.h5")
