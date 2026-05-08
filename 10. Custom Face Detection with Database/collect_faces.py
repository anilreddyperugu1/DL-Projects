# Importing dependencies
import os
import time
import cv2

# Reading the dataset path
dataset = '/Users/anilreddyperugu/Git/DL-Projects/10. Custom Face Detection with Database/dataset'
name = '' # Directory name

# Joining dataset path with person name
path = os.path.join(dataset, name) 
if not os.path.isdir(path): #If not exist create one
    os.mkdir(path)

# Initiating the camera
cam = cv2.VideoCapture(0)

# Loading haarcascade path
path1 = cv2.data.haarcascades
face_cascade = cv2.CascadeClassifier(path1 + 'haarcascade_frontalface_default.xml') # Loading the Haar Cascade classifier

(width, height) = (130, 100)
count = 0
while count < 31: # Take 30 pictures
    print(count)
    ret, frame = cam.read() # capturing the frame
    if not ret:
        print('cannot grab the camera')
        break

    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #Converting BGR to Gray
    faces = face_cascade.detectMultiScale(gray_img, 1.3, 4) # Detecting faces in the frame
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2) #Drawing bounding boxes
 
        faceOnly = gray_img[y:y+h, x:x+w] 
        resizeimg = cv2.resize(faceOnly, (width, height)) # Resizing face image to fixed dimensions
        cv2.imwrite("%s/%s.jpg" %(path,count), resizeimg) # Saving the face image
        time.sleep(0.5)
        count += 1
    cv2.imshow('Face Detection', frame) # Display the camera feed

    if cv2.waitKey(1) & 0xFF == ord('q'): # Exit on 'q'
        break

cam.release() # releasing camera
cv2.destroyAllWindows() # destroy other windows(if any)