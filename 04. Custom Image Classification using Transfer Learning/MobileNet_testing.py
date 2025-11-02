#Importing dependencies
import numpy as np
from keras.preprocessing import image
from keras.models import load_model
from keras.applications.mobilenet_v2 import preprocess_input

#Loading the model
model = load_model('model_mobilenet.keras')
print("Model loaded!")

#Represent Labels
Labels = ["burger", "cookies", "donuts","french_fries","fried_chicken","fried_egg","hotdogs","pizza","steak","toast"]

#Preprocess and input the test image
img_path = 'test/pizza_2.jpg'
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = preprocess_input(img_array)
prediction = model.predict(img_array)

#Prediction
predicted_class = np.argmax(prediction[0])
confidence = np.max(prediction[0])

print(f'Predicted class: {Labels[predicted_class]} with confidence of {confidence*100:.2f}')