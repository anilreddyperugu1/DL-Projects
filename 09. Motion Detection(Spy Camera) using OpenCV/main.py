# Importing dependencies
import cv2
import imutils

#Starting video camera
cam = cv2.VideoCapture(0)

firstFrame = None
area = 500

#The main loop
while True:
    ret, frame = cam.read() #Reading the frame
    text = 'Normal'
    img = imutils.resize(frame, width=900) #resizing the frame
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #converting img from color to gray
    gaussian_blur = cv2.GaussianBlur(gray_img, (21, 21), 0) #providing gaussian blur 
    if firstFrame is None: #storing the first frame initailly
        firstFrame = gaussian_blur 
        continue
    imgDiff = cv2.absdiff(firstFrame, gaussian_blur) # extracting the difference between 2 frames
    thres_img = cv2.threshold(imgDiff, 25, 255, cv2.THRESH_BINARY)[1] # defining the threshold
    thres_img = cv2.dilate(thres_img, None, iterations=2) # dilation = connects nearby white pixels between moving objects
    contours, _ = cv2.findContours(thres_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #locating moving object
    for contour in contours:
        if cv2.contourArea(contour) < area: #ignoring the small movements which are less than area size
            continue
        (x, y, w, h) = cv2.boundingRect(contour) #defining bouding box
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2) #drawing bounding box
        text = "Moving Object Detected"
    cv2.putText(img, text, (10, 20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 2) #text on the frame
    print(text)
    firstFrame = gaussian_blur #update first frame

    if cv2.waitKey(1) & 0xFF == ord('q'): # quitting
        break

    cv2.imshow("Video Frame", img) #showing the frame

cam.release() #releasing the frame
cv2.destroyAllWindows() # destroy all windows(if any)