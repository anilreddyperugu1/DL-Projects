#Importing dependencies
import cv2
import os
import pathlib

# Loading haarcascade path
path1 = cv2.data.haarcascades
face_cascade = cv2.CascadeClassifier(path1 + 'haarcascade_frontalface_default.xml') # Loading the Haar Cascade classifier

# Opening the video camera
cam = cv2.VideoCapture(0)
if not cam.isOpened():
    print("Camera not opened")
    exit()

# Loading the model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('/Users/anilreddyperugu/Git/DL-Projects/10. Custom Face Detection with Database/trainer.yml') # Read data from the trained file

(width, height) = (130, 100)

data_dir = pathlib.Path('/Users/anilreddyperugu/Git/DL-Projects/10. Custom Face Detection with Database/dataset') 

dataset = '/Users/anilreddyperugu/Git/DL-Projects/10. Custom Face Detection with Database/dataset'

ID = 0
label_map = {}
for folder in sorted(os.listdir(dataset)): # List dir name under dataset dir
    folder_path = os.path.join(dataset, folder) # Creating full folder path

    if not os.path.isdir(folder_path): # Ignore if any files exists
        continue

    label_map[ID] = folder # Map numeric ID to folder/person name
    ID += 1 # Increment value

while True: 
    ret, frame = cam.read() # read the frame
    if not ret: # If frame not captured
        print("cannot grab the camera")
        break
    
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Convert to gray scale
    faces = face_cascade.detectMultiScale(gray_img, 1.3, 4) # Detect faces in the frame
   
    for (x,y,w,h) in faces: 
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2) # Drawing bounding boxes around the faces

        faceOnly = gray_img[y:y+h, x:x+w] # Crop only the detected face region
        resizeImg = cv2.resize(faceOnly, (width, height)) # Resize face image to match training size
        label, confidence = recognizer.predict(resizeImg) # Predict the face
        name = label_map.get(label, "Unknown") # Convert predicted label into person name

        cv2.putText(frame, name, (x,y-10), cv2.FONT_HERSHEY_COMPLEX, 3.5, (0,0,255), 2) # Show name on the frame
        # print(name)

    cv2.imshow("FaceDetector", frame) # Display the webcam feed
    
    key = cv2.waitKey(1) #quit on ESC
    if key == 27:
        break


cam.release() # Release the camera
cv2.destroyAllWindows() # Destroy other windows(if any)
