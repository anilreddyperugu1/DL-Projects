#Importing dependencies
import cv2
import numpy as np
from ultralytics import YOLO
from scipy.spatial import distance

#Intialising the model
model = YOLO("yolov8n.pt")

#Reading the source file
cap = cv2.VideoCapture("/Users/anilreddyperugu/Git/DL-Projects/06. Social DIstance Monitoring/videos/vid1.mp4")

#Main Loop
while True:
    ret, frame=cap.read() #reading the frame
    # print("RET:", ret)

    if not ret: #if not true
        break
    
    # cv2.imshow("Frame", frame) 
    key = cv2.waitKey(30) 
    if key == 27: #If key=esc
        break
    results=model(frame) 

    boxes = []
    centroids = []
    for r in results: #for each frame
        for box in r.boxes:
            cls = int(box.cls[0]) #choose only people

            if cls == 0: #If Human
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                boxes.append((x1,y1,x2,y2))

                cx = int((x1+x2)/2) #X Centroid
                cy = int((y1+y2)/2) #Y Centroid
                centroids.append((cx, cy))
                
    violations = set()
    for i in range(len(centroids)): #for each i
        for j in range(i + 1, len(centroids)): #find range to all other points(Boxes)
            d = distance.euclidean(centroids[i], centroids[j]) #Euclidean distance

            if d < 50:  # threshold
                violations.add(i)       
                violations.add(j)

    for i, (x1, y1, x2, y2) in enumerate(boxes): 
        color = (0, 0, 255) if i in violations else (0, 255, 0) #Coloring the boxes
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2) #Bounding Box

    #Violations count per frame
    cv2.putText(frame, f"Violations: {len(violations)}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 2)

    #Show the frame
    if frame is not None:
        cv2.imshow("Social Distance Monitoring", frame)

# print("Total boxes:", len(results[0].boxes))

cap.release()
cv2.destroyAllWindows()