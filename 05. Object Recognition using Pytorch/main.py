import torch
from torchvision.models import inception_v3
import cv2
import numpy as np

# Defining image path
imgPath = 'peacock.jpeg'

# Load the model
model = inception_v3(pretrained=True)

# Device handling
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # use GPU if available else CPU
model = model.to(device)

# Preprocessing the image
def preprocess_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # converting the image from BGR to RGB
    img = cv2.resize(img, (299,299)) # Resizing as per the model requirement
    img = img.astype('float32') / 255.0 # Normalization

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    img = (img - mean) / std # Statistical operation as per model need

    img = img.transpose(2,0,1) # re ordering the image dimensions
    img = np.expand_dims(img, axis=0) # adding the batch size dimension at axis 0
    return img

# Reading the image and sending it to preprocessing
print("[INFO] Loading Image..")
image = cv2.imread(imgPath) # reading the image
image = preprocess_image(image) # preprocess the image (above function)
image = torch.from_numpy(image).float().to(device) # convert img from numpy to torch

# Reading the labels from the txt file
with open('ilsvrc2012_wordnet_lemmas.txt', 'r') as f:
    labels = [line.strip() for line in f.readlines()] # Read the file line wise

# Predicting the output
print('[INFO] Classifying the image..')
model.eval() #Preparing for evaluation

with torch.no_grad(): # Ignoring gradients since we are not dealing with NN layers
    output = model(image)

class_index = torch.argmax(output, dim=1).item() # Max argument index from the output
predicted_label = labels[class_index] # Find the corresponding label with the index
confidence = torch.softmax(output, dim=1)[0][class_index].item() # Calculating confidence

print(f'Predicted Label: {predicted_label}')
print(f'Confidence: {confidence * 100:.2f}%')

# Finding top 5 probabilities [OPTIONAL]
top5_probs, top5_idx = torch.topk(torch.softmax(output, dim=1), 5)

print('                                   ')
print('***  TOP 5 probabilities   ***')
for i in range(5):
    print(labels[top5_idx[0][i]],':', top5_probs[0][i].item())

