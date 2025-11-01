# American Sign Language (ASL) Recognition System

A **real-time American Sign Language (ASL) recognition system** developed using **Convolutional Neural Networks (CNNs)**, **OpenCV**, and **Python**.  
The model is trained on 30 distinct ASL hand gestures and deployed for live classification through a webcam interface with bounding box tracking.

---

##  Project Aim

The goal of this project is to develop a deep learning-based model capable of recognizing **ASL alphabets and basic gestures** from static hand signs.  
This system aims to enhance **communication accessibility** for individuals with hearing or speech impairments.

---

##  Features

- Real-time ASL alphabet recognition using webcam input  
- CNN trained on **30 distinct gestures (A–Z + del, none, nothing, space)**  
- OpenCV-based interface with **live bounding box tracking**  
- Modular scripts for **training (`model.py`)** and **live prediction (`predict.py`)**  
- Documented in a **Jupyter Notebook (`train_asl.ipynb`)** for clarity and reproducibility  

---

##  Model Architecture

| Layer Type | Details |
|-------------|----------|
| Input | 64×64 RGB image |
| Conv2D | 32 filters, 3×3 kernel, ReLU activation |
| MaxPooling2D | 2×2 pool size |
| Conv2D | 64 filters, 3×3 kernel, ReLU activation |
| MaxPooling2D | 2×2 pool size |
| Flatten | — |
| Dense | 128 units, ReLU activation |
| Output | 30 units, Softmax activation |

**Optimizer:** Adam  
**Loss Function:** Categorical Crossentropy  
**Callbacks:** EarlyStopping, ModelCheckpoint  

 **Achieved ~94% validation accuracy** on the ASL Alphabet dataset.

---

##  Installation

Create a virtual environment (optional):

```bash
python -m venv venv
source venv/bin/activate  # For Windows: venv\Scripts\activate
````

Install dependencies:

```bash
pip install -r requirements.txt
```

---

##  Dataset

**Dataset:** [Kaggle - ASL Alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)

Organize your dataset as:

```
ASL_Train/
  A/
  B/
  ...
  Z/
  del/
  none/
  nothing/
  space/
```

---

##  Training

Run the model training script:

```bash
python model.py
```

This will:

* Preprocess the dataset
* Train the CNN
* Save the trained model as `asl_model.h5`
* Store training history as `training_history.npz` for visualization

---

##  Real-time Prediction

Ensure `asl_model.h5` is present, then run:

```bash
python predict.py
```

A webcam window will open with a green bounding box showing the **Region of Interest (ROI)**.
The model will display predicted ASL alphabets in real time.
Press **q** to quit.

---

## 📈 Results

| Metric              | Value                        |
| ------------------- | ---------------------------- |
| Validation Accuracy | ~94%                         |
| Input Size          | 64×64                        |
| Total Classes       | 30                           |
| Frameworks          | TensorFlow, Keras, OpenCV    |
| Interface           | Webcam (real-time detection) |

---

##  Future Work

Planned extensions:

* Incorporate **skeleton-based gesture detection** (MediaPipe or PoseNet) to capture **dynamic ASL motion**.
* Expand to **sentence-level ASL translation** using temporal models (LSTMs or Transformers).

---

##  Acknowledgments

* **TensorFlow** and **Keras** — Deep Learning Framework
* **OpenCV** — Real-time Computer Vision
* **Kaggle** — ASL Alphabet Dataset

### References:

1. [https://www.sciencedirect.com/science/article/pii/S2666990021000471](https://www.sciencedirect.com/science/article/pii/S2666990021000471)
2. [https://www.sciencedirect.com/science/article/pii/S2773186325000143](https://www.sciencedirect.com/science/article/pii/S2773186325000143)

---

## 🧾 Project Documentation

Notebook: [`train_asl.ipynb`](train_asl.ipynb)

* Visualizes training accuracy and loss curves
* Summarizes final validation performance

---

**Author:** Fadhil
**Tech Stack:** Python, TensorFlow, Keras, OpenCV
**Accuracy:** ~94%
**License:** MIT

```
```

