# utils/preprocess_audio.py

import librosa
import numpy as np

def preprocess_audio(file):

    audio, sr = librosa.load(file, sr=16000)

    # Normalize
    audio = audio / np.max(np.abs(audio))

    # Remove silence
    audio, _ = librosa.effects.trim(
        audio,
        top_db=20
    )

    return audio, sr