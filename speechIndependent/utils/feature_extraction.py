import librosa
import numpy as np

def extract_features(file):

    audio, sr = librosa.load(file, sr=16000)

    # Normalize
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio))

    # Extract MFCC
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=13
    )

    # FIX: handle short signals
    if mfcc.shape[1] < 9:

        # pad frames
        pad_width = 9 - mfcc.shape[1]

        mfcc = np.pad(
            mfcc,
            pad_width=((0, 0), (0, pad_width)),
            mode='edge'
        )

    delta = librosa.feature.delta(mfcc)

    delta2 = librosa.feature.delta(mfcc, order=2)

    features = np.hstack([
        np.mean(mfcc, axis=1),
        np.mean(delta, axis=1),
        np.mean(delta2, axis=1)
    ])

    return features