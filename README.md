# Speech-Emotion-Recognition using CNN (PyTorch)
This project aims to classify human emotions from speech audio files using a Convolutional Neural Network (CNN) implemented in PyTorch. The model is trained on the RAVDESS dataset, which contains recordings of actors expressing various emotions.
## Project Structure
Speech-Emotion-Recognition/
- Predicting_emotion.ipynb - Loads the trained model and predicts emotion from a new .wav file
- train_emotion_model.ipynb - Full training pipeline (data loading, feature extraction, model training)
- emotion_cnn.pth - Saved PyTorch model weights
- README.md - Project overview and instructions (this file)
---

## Objective

Build a deep learning model that recognizes **8 emotions** from speech audio:
- Angry
- Calm
- Disgust
- Fearful
- Happy
- Neutral
- Sad
- Surprised

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

## Predicting on New Audio

You can predict the emotion of any `.wav` file using:
- `Predicting_emotion.ipynb`
- It uses the saved `emotion_cnn.pth` weights and a new audio file.

## How to Run
1. Train the Model:
Open train_emotion_model.ipynb
Run all cells
2. Predict Emotion from Audio
Open Predicting_emotion.ipynb
Upload a .wav file (or use one from RAVDESS)
Run the notebook to see predicted emotion

## Dataset
RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)

8 emotions, 24 actors, 60 recordings per actor

Source: https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio

## Dataset Attribution
This project uses the RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song) dataset for educational and non-commercial purposes.

Non-Academic Use Attribution:

The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)" by Livingstone & Russo is licensed under CC BY-NA-SC 4.0.
