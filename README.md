Self-Driving Car – Behavioral Cloning

This project implements a deep learning model that learns to steer a car by observing human driving behavior. The model processes images from a front-facing camera and predicts steering angles, effectively cloning human driving behavior.

✨ Key Features

Data Augmentation: Robust augmentation techniques for better generalization

CNN Architecture: Custom network inspired by NVIDIA’s self-driving car model

Real-time Prediction: Socket.IO server for live steering angle predictions

Data Balancing: Prevents model bias towards straight driving

Preprocessing Pipeline: Optimized image preprocessing for model performance

🎥 Demo & Presentation

📹 Video Demonstration
 – Watch the model in action

📑 Project Presentation
 – View the PowerPoint slides

🏗️ Model Architecture

The network is based on NVIDIA’s architecture with modifications:

Input: (66, 200, 3) preprocessed image
↓
Conv2D (24 filters, 5x5, strides=2x2) + ELU
↓
Conv2D (36 filters, 5x5, strides=2x2) + ELU
↓
Conv2D (48 filters, 5x5, strides=2x2) + ELU
↓
Conv2D (64 filters, 3x3) + ELU
↓
Conv2D (64 filters, 3x3) + ELU
↓
Flatten
↓
Dense (100 neurons) + ELU
↓
Dense (50 neurons) + ELU
↓
Dense (10 neurons) + ELU
↓
Output: 1 neuron (steering angle)

⚙️ Installation & Usage
✅ Prerequisites

Python 3.7+

TensorFlow 2.x

OpenCV

Flask, SocketIO, Eventlet

🔽 Installation

Clone the repository and install dependencies:

git clone https://github.com/your-username/self-driving-car.git
cd self-driving-car
pip install tensorflow opencv-python socketio eventlet flask pandas scikit-learn imgaug

🚗 Training the Model

Place training data inside the data/ folder containing:

IMG/ directory (images)

driving_log.csv

Run training:

python train.py


The trained model will be saved as model.h5

🔌 Running the Prediction Server

Start the server for real-time steering:

python test.py


Runs on port 4567

Waits for connections from the driving simulator

🔧 Technical Details
🖼️ Data Preprocessing

Cropping to focus on road area

RGB → YUV color space conversion

Gaussian blur for noise reduction

Resizing to 200×66 pixels

Normalization (pixel values scaled 0–1)

🧪 Data Augmentation

Random panning & shifting

Zoom variations

Brightness adjustments

Horizontal flipping (with steering angle correction)

Random blurring

Rotation transformations

📚 Training Approach

Balanced dataset to avoid steering bias

80/20 train-validation split

Batch generator for efficiency

Loss: Mean Squared Error (MSE)

Optimizer: Adam (lr = 1e-6)

📊 Results

Achieved low validation loss → strong generalization

Smooth steering predictions in varied conditions

Stable performance due to diverse augmentations
