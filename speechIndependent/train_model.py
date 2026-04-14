import os
import joblib
from sklearn.svm import SVC
from speechdependent.utils.feature_extraction import extract_features

X = []
y = []

dataset_path = "datasets"

print("Loading dataset...")

for file in os.listdir(dataset_path):

    if file.endswith(".wav"):

        # FULL PATH
        file_path = os.path.join(dataset_path, file)

        # Use file_path here
        features = extract_features(file_path)

        X.append(features)

        # Extract speaker name
        # Example: 1_george_10.wav

        speaker = file.split("_")[1]

        

        y.append(speaker)

print("Training model...")

model = SVC(kernel="rbf")

model.fit(X, y)

joblib.dump(model, "speaker_model.pkl")

print("Training complete")