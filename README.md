# Speech-Emotion-Recognition using CNN (PyTorch)
This project aims to classify human emotions from speech audio files using a Convolutional Neural Network (CNN) implemented in PyTorch. The model is trained on the RAVDESS dataset, which contains recordings of actors expressing various emotions.
## Project Structure
Speech-Emotion-Recognition/
- Predicting_emotion.ipynb - Loads the trained model and predicts emotion from a new .wav file
- train_emotion_model.ipynb - Full training pipeline (data loading, feature extraction, model training)
- requirements.txt - List of Python packages required to run the project
- emotion_cnn.pth - Saved PyTorch model weights
- README.md - Project overview and instructions (this file)
---

## Objective

The goal of this project is to build a **Speech Emotion Recognition (SER)** system using deep learning techniques, specifically a **Convolutional Neural Network (CNN)** implemented in PyTorch. 
Given a `.wav` file of a human voice, the system should accurately classify it into one of **eight emotions**:
- Angry
- Calm
- Disgust
- Fearful
- Happy
- Neutral
- Sad
- Surprised

This project aims to demonstrate:
- How raw audio can be converted into MFCC features suitable for deep learning,
- How CNNs can capture spatial patterns in spectrogram-like data,
- And how PyTorch enables easy training, evaluation, and deployment of emotion recognition models.

---

## Features Used

- **MFCC (Mel-Frequency Cepstral Coefficients):**
  - Extracted using `librosa`
  - 40 MFCCs per audio file
  - Zero-padded or trimmed to maintain a fixed shape (`(40, 174)`)

---

## Model Architecture

Defined in `train_emotion_model.ipynb` as `EmotionCNN`:
- **3 Convolutional Layers**:
  - Each followed by BatchNorm + ReLU + MaxPooling
  - Channels: 1 → 16 → 32 → 64
- **Flatten + Fully Connected Layers**:
  - FC1: Linear(64×5×21 → 128) + Dropout
  - FC2: Linear(128 → 8) → Final emotion class prediction
- **Loss Function**: CrossEntropyLoss
- **Optimizer**: Adam (lr=0.0005)
- **Accuracy Achieved**: ~98% training accuracy, ~96.9% test accuracy

---

## Evaluation

- Confusion matrix and classification report (precision, recall, f1-score) generated after training.
- Labels decoded using `LabelEncoder`.

---


## Predicting on New Audio

You can predict the emotion of any `.wav` audio file using the notebook:

-  **`Predicting_emotion.ipynb`**
  - Loads the trained model (`emotion_cnn.pth`)
  - Extracts MFCC features from a new `.wav` file using `librosa`
  - Predicts emotion using the CNN model
  - Decodes the predicted label using the original `LabelEncoder`


---

## How to Run
1. Train the Model:
Open train_emotion_model.ipynb
Run all cells
2. Predict Emotion from Audio
Open Predicting_emotion.ipynb
Upload a .wav file (or use one from RAVDESS)
Run the notebook to see predicted emotion

---

## Final Results

- **Test Accuracy Achieved**: **96.88%** on the test set (576 samples).
- **Per-Class Performance**:
  - High accuracy for most emotions.
  - Emotions like `calm`, `happy`, and `surprised` achieved near-perfect scores.
  - Slightly lower recall for emotions like `fearful` and `sad`, suggesting these are harder to distinguish.

> These results reflect the model’s generalization ability and performance on real unseen audio samples.

---

## Dataset
RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)

8 emotions, 24 actors, 60 recordings per actor

Source: https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio

---


## Dataset Attribution
This project uses the RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song) dataset for educational and non-commercial purposes.

Non-Academic Use Attribution:

The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)" by Livingstone & Russo is licensed under CC BY-NA-SC 4.0.



