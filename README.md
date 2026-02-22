AI SYSTEM FOR EARLY DETECTION OF PARKINSON FROM SPEECH AND HANDWRITING IN TAMIL

MLP + CNN Classification

📌 Project Overview

This project focuses on detecting Parkinson’s Disease using voice recordings.

Parkinson’s disease affects motor control, including muscles involved in speech production. By analyzing acoustic characteristics of voice recordings, we build deep learning models to classify:

Healthy (0)

Parkinson (1)

Two approaches were implemented:

🔹 MLP (Feature-based model)

🔹 CNN (Spectrogram-based model)

🎯 Problem Statement

To develop a deep learning system that can classify speech samples as:

Healthy

Parkinson

using acoustic features extracted from voice recordings.

📂 Dataset Description

The dataset consists of .wav speech recordings organized as:

Tamil_Parkinson_Dataset/
│
├── train/
│   ├── healthy/
│   └── parkinson/
│
├── test/
│   ├── healthy/
│   └── parkinson/

Dataset Characteristics

Type: Audio (.wav)

Sampling Rate: 16 kHz

Input to CNN: 128×128 Mel Spectrogram

Task: Binary Classification

🔬 Why Voice Data?

Parkinson’s disease affects:

Vocal stability

Pitch variation

Tremor in speech

Amplitude irregularities

Harmonic-to-noise ratio

Speech analysis provides a non-invasive diagnostic support tool.

🧾 Data Preprocessing Pipeline
1️⃣ Audio Loading

Loaded using librosa

Resampled to 16kHz

2️⃣ Mel Spectrogram Conversion

Each audio file is converted to:

128 Mel frequency bands

Fixed width (128 frames)

3️⃣ Normalization

Spectrogram values scaled to [0, 1]

4️⃣ Train / Validation / Test Split

Stratified splitting

Prevents class imbalance issues

5️⃣ Imbalance Handling

class_weight used during training

6️⃣ Data Augmentation (CNN)

Random translation

Random zoom

Random contrast

🏗 Model Architectures

🔹 MLP Model

Used flattened spectrogram features.

Architecture:

Dense (128) + ReLU

Dropout (0.5)

Dense (64) + ReLU

Dropout (0.5)

Output: Sigmoid

Loss: Binary Crossentropy
Optimizer: Adam
Learning Rate: 0.001

🔹 CNN Model (Main Model)

Uses 2D convolution on spectrogram images.

Architecture:

Conv2D (16) + BatchNorm + MaxPooling

Conv2D (32) + BatchNorm + MaxPooling

Conv2D (64) + BatchNorm + MaxPooling

Dense (64) + Dropout

Output: Sigmoid

Loss: Binary Crossentropy
Optimizer: Adam
Learning Rate: 0.0005
EarlyStopping used

📊 Evaluation Metrics

Accuracy

Precision

Recall

F1-score

Confusion Matrix

Training & Validation curves

📈 Comparative Analysis (MLP vs CNN)
Model	Strength	Limitation
MLP	Simple, faster training	Ignores spatial structure
CNN	Captures time-frequency patterns	Higher computation
Key Insight:

CNN performs better because Mel Spectrograms contain spatial time-frequency dependencies that convolutional layers can capture effectively.

🛠 Overfitting Control

Dropout (0.5)

Batch Normalization

EarlyStopping

Data Augmentation

Class Weights

📦 Technologies Used

Python

TensorFlow / Keras

Librosa

NumPy

Matplotlib

Seaborn

Scikit-learn

🚀 How to Run

Clone repository

git clone https://github.com/yourusername/parkinson-detection.git

Install dependencies

pip install -r requirements.txt

Run notebook

jupyter notebook
🔁 Reproducibility

Random seeds are fixed for:

NumPy

TensorFlow

Python

Ensuring reproducible results.

📌 Key Learning Outcomes

Audio signal processing using Mel Spectrogram

CNN for biomedical signal classification

Handling imbalanced datasets

Hyperparameter tuning

Comparative deep learning analysis

Model evaluation & interpretation

📚 Future Improvements

Add CNN + LSTM hybrid model

Use larger multi-speaker dataset

Apply K-Fold cross validation

Deploy as web application

Integrate real-time audio recording

👩‍💻 Author

Yehaasary KM
