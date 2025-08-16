# predict_from_webcam.py

import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

# === Load trained model ===
model = load_model('face_cnn_model.h5')

# === Load class labels from dataset folders ===
class_names = sorted(os.listdir("dataset"))  # ['Alice', 'Bob', 'Charlie']

# === Face detection using Haar Cascade ===
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# === Webcam setup ===
cap = cv2.VideoCapture(0)
IMG_SIZE = (160, 160)

print("📷 Starting webcam. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale for detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Preprocess face for prediction
        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face, IMG_SIZE)
        face = face.astype('float32') / 255.0
        face = np.expand_dims(face, axis=0)  # (1, 160, 160, 3)

        # Predict
        predictions = model.predict(face)
        index = np.argmax(predictions)
        confidence = predictions[0][index]
        name = class_names[index]

        label = f"{name} ({confidence:.2f})"
        cv2.putText(frame, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

