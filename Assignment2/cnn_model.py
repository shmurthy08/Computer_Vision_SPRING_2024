import os
import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Conv2D, UpSampling2D, Concatenate, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import LearningRateScheduler

# Load Preprocessed data
load_data = np.load('cnn_train_test_val_data.npz')
X_train, y_train, X_val, y_val, X_test, y_test = load_data['X_train'], load_data['y_train'], load_data['X_val'], load_data['y_val'], load_data['X_test'], load_data['y_test']


# Define the integer-encoded label masks
y_train_encoded = np.argmax(y_train, axis=-1)
y_val_encoded = np.argmax(y_val, axis=-1)
y_test_encoded = np.argmax(y_test, axis=-1)


# Define U-Net architecture with transfer learning using VGG16
def unet_vgg(input_shape):
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

    # Output layer
    output = Conv2D(3, 1, activation='softmax')(x)

    # Create and compile model
    model = Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

# Define input shape
input_shape = (256, 256, 3)

# Instantiate the VGG16 model
vgg_model = unet_vgg(input_shape)

# Define callbacks for VGG16 model
vgg_checkpoint_path = "vgg_model_checkpoint.h5"
vgg_checkpoint = ModelCheckpoint(vgg_checkpoint_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
vgg_lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-7, verbose=1)

# Train the VGG16 model
vgg_model.fit(X_train, y_train_encoded, validation_data=(X_val, y_val_encoded), epochs=50, batch_size=32, callbacks=[vgg_checkpoint, vgg_lr_scheduler])

# Load best weights for VGG16 model
vgg_model.load_weights(vgg_checkpoint_path)

# Evaluate the VGG16 model
vgg_model.evaluate(X_test, y_test_encoded)

# Save the entire VGG16 model
vgg_model.save("vgg_unet_model.h5")
