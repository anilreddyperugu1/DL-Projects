#Importing dependencies
import numpy as np
import matplotlib.pyplot as plt
import pathlib
from keras.models import Model
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.applications import MobileNetV2
print("All Libraries imported!")

#Reading data from the path
data_dir = pathlib.Path("dataset")
image_count = len(list(data_dir.glob('*/*.jpg')))
print("Total Images: ", image_count)

#Learning all classes
classes = np.array([i.name for i in data_dir.glob('*')])

#Defining Batch Size
batch_size=20

#Normalizing and splitting the data
image_generator = ImageDataGenerator(rescale=1/255, validation_split=0.2)

#Data to be trained
train_set = image_generator.flow_from_directory(directory='dataset', batch_size=batch_size,
                                                classes=list(classes), target_size=(224,224), 
                                                subset='training')

#Data for test
test_dataset = image_generator.flow_from_directory(directory='dataset', batch_size=batch_size,
                                                   classes=list(classes), target_size=(224,224), 
                                                   subset='validation')
#Intializing the model
model = MobileNetV2(input_shape=(224, 224, 3))
# model.summary() #To check the architecture

#Remove the last layer
model.layers.pop()

#Freeze all the layers except last 4
for layer in model.layers[:-4]:
    layer.trainable=False

#adding the dense layer
output = Dense(10, activation='softmax')

#Printing output and combining Dense layer to MobileNet Model
output = output(model.layers[-1].output)
model = Model(inputs=model.inputs, outputs=output)
# model.summary()

#Definfing loss and optimizer and accuracy
model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['acc'])

#Checkpoint to save the best model
checkpoint = ModelCheckpoint('model_mobilenet.keras',
                             save_best_only=True,
                             verbose=1)

# count number of steps per epoch
trainingStepsPerEpoch = np.ceil(train_set.samples / batch_size)
validationStepsPerEpoch = np.ceil(test_dataset.samples / batch_size)
# print(validationStepsPerEpoch)

#Training the model
history = model.fit(train_set, steps_per_epoch=int(trainingStepsPerEpoch),
                              validation_data=test_dataset, validation_steps=int(validationStepsPerEpoch),
                              epochs=15,  verbose=1, callbacks=[checkpoint])

#Saving the weights of the model
model.load_weights('model_mobilenet.keras')
validationStepsPerEpoch = np.ceil(test_dataset.samples / batch_size)
evaluation = model.evaluate(test_dataset)

acc = history.history['acc']
val_acc = history.history['val_acc']
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


loss = history.history['loss']
val_loss = history.history['val_loss']
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()