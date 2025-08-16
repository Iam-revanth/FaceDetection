# train_model.py

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import os

# === PARAMETERS ===
IMG_SIZE = (160, 160)
BATCH_SIZE = 1                # Use 1 for small datasets
EPOCHS = 10
DATASET_PATH = "dataset"

# === PREPROCESSING ===
# Simple rescaling of pixel values (0–255 → 0–1)
datagen = ImageDataGenerator(rescale=1./255)

# === LOAD TRAINING DATA ONLY ===
train_gen = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

# === DEFINE CNN MODEL ===
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(train_gen.num_classes, activation='softmax')  # Output layer with # of classes
])

# === COMPILE THE MODEL ===
model.compile(
    optimizer=Adam(),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# === TRAIN THE MODEL ===
model.fit(train_gen, epochs=EPOCHS)

# === SAVE THE MODEL ===
model.save("face_cnn_model.h5")
print("✅ Model training complete and saved as 'face_cnn_model.h5'")

