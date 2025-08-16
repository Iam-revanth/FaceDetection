# Face Detection using CNN 🧠📸

This project implements a **Convolutional Neural Network (CNN)** for face detection. The model is trained on a custom dataset of images and can predict whether a given image contains a specific person’s face. It also supports real-time face detection using a webcam.

---

## 🚀 Features
- Face detection using **CNN** (Convolutional Neural Network).
- Custom dataset with multiple classes (different persons).
- Training and prediction scripts provided.
- Real-time detection using **OpenCV + Webcam**.
- Saved model (`face_cnn_model.h5`) for quick reuse.

---

## 📂 Project Structure
Facedetection/
│── dataset/ # Training dataset (organized by person)
│ ├── person1/
│ ├── person2/
│ └── ...
│── train_model.py # Script to train the CNN model
│── predict.py # Script to test predictions on images
│── webcam.py # Real-time face detection via webcam
│── face_cnn_model.h5 # Trained CNN model

---

## 🛠️ Tech Stack
- **Python 3.x**
- **TensorFlow / Keras** – for building and training the CNN
- **OpenCV** – for real-time webcam detection
- **NumPy, Pandas** – for data handling
- **Matplotlib** – for visualization

---

## ⚙️ Installation & Setup (Windows)

### 1️⃣ Clone the repository
git clone https://github.com/your-username/Facedetection.git
cd Facedetection

### 2️⃣ Create and activate a virtual environment (Windows)
# Create virtual environment
python -m venv venv
# Activate it
venv\Scripts\activate

### 3️⃣ Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

### 4️⃣ Train the model (optional if you want to retrain)
python train_model.py

### 5️⃣ Run prediction on sample images
python predict.py

### 6️⃣ Real-time face detection with webcam
python webcam.py

### 📊 Model Training
Model: CNN with Conv2D, MaxPooling, Flatten, Dense layers.
Loss Function: categorical_crossentropy
Optimizer: adam
Metrics: accuracy

### 🎯 Results
Achieved high accuracy on the training dataset.
Successfully detects faces from live webcam feed.
Can be extended to larger datasets for real-world applications.

### 🚀 Future Improvements
Add more classes with larger dataset.
Improve accuracy with transfer learning (e.g., VGG16, ResNet).
Deploy model with Flask / FastAPI as a backend service.
Create a frontend interface for easy usage.
