import os
import numpy as np
from sklearn.svm import SVC
import joblib

from utils.feature_extraction import extract_features

X = []
y = []

dataset_path = "datasets"

print("Loading dataset...")

for file in os.listdir(dataset_path):

    if file.endswith(".wav"):

        file_path = os.path.join(dataset_path, file)

        features = extract_features(file_path)

        X.append(features)

        # Extract speaker name
        speaker = file.split("_")[1]

        y.append(speaker)

print("Training model...")

model = SVC(kernel="linear")

model.fit(X, y)

joblib.dump(model, "speaker_model.pkl")

print("Model saved as speaker_model.pkl")