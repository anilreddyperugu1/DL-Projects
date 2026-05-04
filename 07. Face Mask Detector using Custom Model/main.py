# Importing dependencies
import tensorflow as tf
import numpy as np
from keras.applications import MobileNetV2
from keras.layers import Dense, Flatten
from keras.models import Model
from keras.layers import GlobalAveragePooling2D

#Defining Batch side and Input size
Batch_size = 6
IMG_SIZE = (224, 224)

#Reading and pre processing input for training
training_set = tf.keras.utils.image_dataset_from_directory(
    directory="/Users/anilreddyperugu/Git/DL-Projects/07. Face Mask Detector using Custom Model/datasets/train", 
    batch_size=Batch_size,
    image_size=(224, 224),
    label_mode='binary')

#reading and pre processing the input for validation
val_set = tf.keras.utils.image_dataset_from_directory(
    directory="/Users/anilreddyperugu/Git/DL-Projects/07. Face Mask Detector using Custom Model/datasets/val",
    batch_size=Batch_size, 
    image_size=(224, 224),
    label_mode='binary')

# Normalizing the data
train_dataset = training_set.map(lambda x, y: (x/255.0, y))
val_dataset = val_set.map(lambda x, y: (x/255.0, y))


# Performance optimization (optional)
train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
val_dataset = val_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

#Intializing the base model
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

#Freezing the layers
for layer in base_model.layers:
    layer.trainable=False

#Applying Pooling and adding Dense layer
x = base_model.output
x = GlobalAveragePooling2D() (x)
x = Dense(128, activation='relu') (x)

output = Dense(1, activation='sigmoid') (x)

#Defining input and output to the final model
model = Model(inputs=base_model.input, outputs=output)

#Compiling the data
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# model.summary()

#Training the model
history = model.fit(train_dataset,
                    validation_data=val_dataset,
                    epochs=10, verbose=1
                    )

#Evaluating the model performance
loss, accuracy = model.evaluate(val_dataset)
print(f"Validation Loss: {loss}")
print(f"Validation Accuracy: {accuracy}")

#Saving the model for validation use
model.save("mask_detector_model.keras")