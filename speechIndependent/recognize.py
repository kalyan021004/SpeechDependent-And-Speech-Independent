import joblib
from utils.feature_extraction import extract_features

model = joblib.load("speaker_model.pkl")

def recognize(file):

    features = extract_features(file)

    prediction = model.predict([features])

    return prediction[0]