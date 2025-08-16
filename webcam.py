# webcam_predict.py

import cv2
import numpy as np
import face_recognition
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

# === Load Trained Model ===
model = load_model("face_cnn_model.h5")
class_names = sorted(os.listdir("dataset"))  # Ensure class order is the same as in training

# === Open Webcam ===
cap = cv2.VideoCapture(0)  # Use the first webcam (0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # === Detect faces in the frame ===
    rgb_frame = frame[:, :, ::-1]  # Convert BGR to RGB
    face_locations = face_recognition.face_locations(rgb_frame)
    
    # === Process each detected face ===
    for top, right, bottom, left in face_locations:
        # Crop the face
        face_image = frame[top:bottom, left:right]
        
        # Resize the face image to 160x160 for the CNN model
        face_resized = cv2.resize(face_image, (160, 160))
        
        # Preprocess: Convert to RGB and normalize
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        face_array = image.img_to_array(face_rgb) / 255.0
        face_array = np.expand_dims(face_array, axis=0)
        
        # === Make prediction ===
        predictions = model.predict(face_array)
        predicted_index = np.argmax(predictions)
        confidence = predictions[0][predicted_index]
        
        # Display the result
        name = class_names[predicted_index]
        label = f"{name}: {confidence*100:.2f}%"
        
        # Draw rectangle and label around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    # === Show the frame with predicted label ===
    cv2.imshow('Face Recognition - Press Q to Exit', frame)
    
    # Exit the loop when the user presses 'Q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# === Clean up ===
cap.release()
cv2.destroyAllWindows()

