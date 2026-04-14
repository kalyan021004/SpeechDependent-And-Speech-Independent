# utils/record_audio.py

import sounddevice as sd
from scipy.io.wavfile import write

def record_audio(filename):

    fs = 16000
    seconds = 3

    print("Recording...")

    audio = sd.rec(
        int(seconds * fs),
        samplerate=fs,
        channels=1
    )

    sd.wait()

    write(filename, fs, audio)

    print("Saved:", filename)