import torch
from torchvision.models import inception_v3
import cv2
import numpy as np

imgPath = 'peacock.jpeg'

model = inception_v3(pretrained=True)

#Device handling
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

#Preprocessing the image
def preprocess_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (299,299))
    img = img.astype('float32') / 255.0

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    img = (img - mean) / std

    img = img.transpose(2,0,1)
    img = np.expand_dims(img, axis=0)
    return img

#Reading the image and sending it to preprocessing
print("[INFO] Loading Image..")
image = cv2.imread(imgPath)
image = preprocess_image(image)
image = torch.from_numpy(image).float().to(device)

#Reading the labels from the txt file
with open('ilsvrc2012_wordnet_lemmas.txt', 'r') as f:
    labels = [line.strip() for line in f.readlines()]

#Predicting the output
print('[INFO] Classifying the image..')
model.eval()

with torch.no_grad():
    output = model(image)

class_index = torch.argmax(output, dim=1).item()
predicted_label = labels[class_index]
confidence = torch.softmax(output, dim=1)[0][class_index].item()

print(f'Predicted Label: {predicted_label}')
print(f'Confidence: {confidence * 100:.2f}%')

top5_probs, top5_idx = torch.topk(torch.softmax(output, dim=1), 5)

print('                                   ')
print('***  TOP 5 probabilities   ***')
for i in range(5):
    print(labels[top5_idx[0][i]],':', top5_probs[0][i].item())

