# Importing dependencies
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

# Loading the trained mask model
model = load_model("mask_detector_model.keras", compile=False)
print("Mask model loaded")

# Loading DNN face detector
net = cv2.dnn.readNetFromCaffe(
    "/Users/anilreddyperugu/Git/DL-Projects/07. Face Mask Detector using Custom Model/deploy.prototxt",
    "/Users/anilreddyperugu/Git/DL-Projects/07. Face Mask Detector using Custom Model/res10_300x300_ssd_iter_140000.caffemodel"
)
# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera")
    exit()

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    (h, w) = frame.shape[:2]    

    # Prepare blob for face detection
    blob = cv2.dnn.blobFromImage(
        frame,
        1.0,
        (300, 300),
        (104.0, 177.0, 123.0)
    )

    net.setInput(blob)
    detections = net.forward()

    # Loop over detected faces
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            (x1, y1, x2, y2) = box.astype("int")

            # Ensure box is within frame
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            # Crop face
            face = frame[y1:y2, x1:x2]

            if face.size == 0:
                continue

            # Preprocess for model
            face = cv2.resize(face, (224, 224))
            face = face / 255.0
            face = np.reshape(face, (1, 224, 224, 3))

            # Predict
            pred = model.predict(face, verbose=0)[0][0]

            # Label
            if pred < 0.4:
                label = "Mask"
                color = (0, 255, 0)
            else:
                label = "No Mask"
                color = (0, 0, 255)

            # Drawing Bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2
            )

    # Show output
    cv2.imshow("Mask Detection", frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()