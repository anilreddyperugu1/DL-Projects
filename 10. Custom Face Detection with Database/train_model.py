# Importing dependencies
import pathlib
import numpy as np
import os
import pandas as pd
import cv2

# Finding the path
data_dir = pathlib.Path('/Users/anilreddyperugu/Git/DL-Projects/10. Custom Face Detection with Database/dataset')

dataset = '/Users/anilreddyperugu/Git/DL-Projects/10. Custom Face Detection with Database/dataset' # Locating the dataset

image_count = len(list(data_dir.glob('*/*.jpg'))) # Finding image count

# classes = np.array([i.name for i in data_dir.glob('*')]) # Learning classes

ID = 0
faces = []
labels = []
label_map = {}
for folder in os.listdir(dataset): # List dir name under dataset dir
    folder_path = os.path.join(dataset, folder) # Creating full folder path

    if not os.path.isdir(folder_path): # Ignore if any files exist
        continue

    label_map[ID] = folder # Assigning numeric IDs to the directories
   

    for image in os.listdir(folder_path): # List dir name under folder path
        image_path = os.path.join(folder_path, image) # Loop through every image
        img = cv2.imread(image_path) # Read every img
        if img is None: #if no img found
            continue # skip
        grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # COnvert to gray
        faces.append(grayImg) # add face to faces list
        labels.append(ID) # add label to labels list

    ID += 1 # Increment ID
        
# print(label_map)
# print('Total faces', len(faces))
# print('Total Lables', len(labels))


recognizer = cv2.face.LBPHFaceRecognizer_create() #creating LBPH face recognizer model
recognizer.train(faces, np.array(labels)) # Training the recognizer using face images and labels
recognizer.save('trainer.yml') # Saving the model