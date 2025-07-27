# ASL recognition systemL

A real-time American Sign Language (ASL)  recognition system developed using Convolutional Neural Networks (CNNs) and OpenCV. The system is trained on a labeled dataset of ASL hand gestures and deployed for real-time inference through a webcam.

---

## Project Aim

The objective of this project is to develop a deep learning-based system capable of recognizing ASL alphabets from static hand gestures. By training a Convolutional Neural Network on a dataset of labeled ASL gestures, the system can classify signs from webcam input in real-time. This project contributes to the development of accessible communication tools for individuals with hearing or speech impairments.

---

## Features

- Real-time ASL alphabet recognition using webcam input
- Convolutional Neural Network (CNN) trained on image data
- Modular scripts for model training and live prediction
- Scalable foundation for full sign language sentence recognition

---

## Model Architecture

The model is implemented using TensorFlow and Keras, with the following architecture:

- Input Layer: 64x64 RGB image
- Conv2D Layer: 32 filters, 3x3 kernel, ReLU activation
- MaxPooling2D: 2x2 pool size
- Conv2D Layer: 64 filters, 3x3 kernel, ReLU activation
- MaxPooling2D: 2x2 pool size
- Flatten Layer
- Dense Layer: 128 units, ReLU activation
- Output Layer: Softmax activation with 30 output classes

The model is compiled using the Adam optimizer and categorical crossentropy loss function. EarlyStopping is used to prevent overfitting by monitoring validation loss.

---

Create a virtual environment (optional):
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
Install dependencies:
pip install -r requirements.txt

Dataset (Kaggle) :https://www.kaggle.com/datasets/grassknoted/asl-alphabet

Training the Model
Then run the training script: python model.py
This will preprocess the data, train the CNN model, and save the trained model as asl_cnn_model.keras.

Real-time Prediction
Ensure you have the trained model saved as asl_cnn_model.keras (or converted to asl_model.h5 if needed).

To start real-time prediction:  python predict.py
A window will open showing the webcam feed. The system will draw a bounding box to identify the region of interest (ROI) and predict the ASL alphabet being shown within that region. Press q to quit the prediction window.

Acknowledgments
TensorFlow and Keras for deep learning libraries

OpenCV for real-time computer vision

References :  1) https://www.sciencedirect.com/science/article/pii/S2666990021000471
              2) https://www.sciencedirect.com/science/article/pii/S2773186325000143
