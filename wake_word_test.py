"""
Wake word detection test script.
Listens through your mic and prints a message when it hears "Hey Jarvis."

This is step 1 of the voice assistant project: get the wake word
working on its own before wiring it into the full voice loop.
"""

import pyaudio
import numpy as np
from openwakeword.model import Model

# openWakeWord ships with a few pretrained models, including "hey_jarvis."
# You can add more model names to this list later if you want other phrases.
model = Model(wakeword_models=["hey_jarvis"])

# Audio settings openWakeWord expects: 16kHz, mono, 16-bit audio,
# read in 80ms chunks (1280 samples at 16kHz).
CHUNK = 1280
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

# How confident the model needs to be before we count it as a real trigger.
# Raise this if it's triggering on background noise. Lower it if it's not
# picking up your voice at all.
CONFIDENCE_THRESHOLD = 0.5

audio = pyaudio.PyAudio()
stream = audio.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    frames_per_buffer=CHUNK,
)

print("Listening for wake word... say 'Hey Jarvis'")
print("Press Ctrl+C to stop.\n")

try:
    while True:
        # Read one chunk of raw audio from the mic
        raw_audio = stream.read(CHUNK, exception_on_overflow=False)
        audio_chunk = np.frombuffer(raw_audio, dtype=np.int16)

        # Feed it to the model and get a confidence score per wake word
        prediction = model.predict(audio_chunk)

        for wake_word, score in prediction.items():
            if score > CONFIDENCE_THRESHOLD:
                print(f"Heard you! (wake word: {wake_word}, confidence: {score:.2f})")

except KeyboardInterrupt:
    print("\nStopping...")

finally:
    stream.stop_stream()
    stream.close()
    audio.terminate()
