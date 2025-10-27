from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import model_from_json
import numpy as np
from sklearn.metrics import accuracy_score

json_file = open('model.json', 'r')
loaded_json_model = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_json_model)
loaded_model.load_weights('model.weights.h5')


print("Model loaded")
label = ['Benign', 'Malign', 'Normal']
img_path = 'TestDataset/B_2.jpg'
test_image = image.load_img(img_path, target_size=(128, 128))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)

result = loaded_model.predict(test_image)
result = label[result.argmax()]
print(result)