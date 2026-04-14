import librosa
import numpy as np

def extract_features(file):

    audio, sr = librosa.load(file, sr=16000)

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=13
    )

    features = np.mean(mfcc.T, axis=0)

    return features