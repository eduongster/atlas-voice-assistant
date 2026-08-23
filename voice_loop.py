"""
Atlas voice assistant - basic voice loop.

Flow: wake word triggers recording -> whisper.cpp transcribes it ->
Claude answers -> macOS 'say' speaks the answer out loud.

This builds directly on wake_word_test.py. Once this loop works
reliably, later steps add calendar access, timers, memory, and
multi-LLM routing on top of it.
"""

import os
import subprocess
import wave

import numpy as np
import pyaudio
from dotenv import load_dotenv
from openwakeword.model import Model
from pywhispercpp.model import Model as WhisperModel
import anthropic

# Load your API key from the .env file (never hardcode it here)
load_dotenv()
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# --- Wake word setup ---
wake_model = Model(wakeword_models=["hey_jarvis"], inference_framework="onnx")
WAKE_CONFIDENCE_THRESHOLD = 0.5

# --- Audio settings ---
CHUNK = 1280
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 5  # how long to listen after the wake word triggers

# --- Whisper setup ---
# First run downloads the model automatically. "base" is a good
# balance of speed and accuracy for a personal project like this.
whisper_model = WhisperModel("base")

# --- Claude's personality ---
SYSTEM_PROMPT = (
    "You are Atlas, a helpful voice assistant. Keep answers short and "
    "conversational, since they get read out loud. No markdown, no lists, "
    "just plain spoken sentences."
)

audio = pyaudio.PyAudio()


def record_audio(filename="temp_recording.wav"):
    """Records a fixed number of seconds of audio and saves it as a wav file."""
    stream = audio.open(
        format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK
    )
    print("Listening for your question...")

    frames = []
    for _ in range(int(RATE / CHUNK * RECORD_SECONDS)):
        frames.append(stream.read(CHUNK, exception_on_overflow=False))

    stream.stop_stream()
    stream.close()

    with wave.open(filename, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b"".join(frames))

    return filename


def transcribe(filename):
    """Runs the recorded audio through local whisper.cpp and returns the text."""
    segments = whisper_model.transcribe(filename)
    text = " ".join(segment.text for segment in segments).strip()
    return text


def ask_claude(question):
    """Sends the question to Claude and returns the reply text."""
    response = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=300,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": question}],
    )
    # Claude can return multiple content blocks (e.g. a thinking block
    # before the actual answer), so find the text block instead of
    # assuming it's always first.
    for block in response.content:
        if block.type == "text":
            return block.text
    return "Sorry, I didn't get a text response back."


def speak(text):
    """Uses macOS's built-in 'say' command to read text out loud."""
    subprocess.run(["say", text])


def main():
    print("Atlas is listening for the wake word... say 'Hey Jarvis'")
    print("Press Ctrl+C to stop.\n")

    wake_stream = audio.open(
        format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK
    )

    try:
        while True:
            raw_audio = wake_stream.read(CHUNK, exception_on_overflow=False)
            audio_chunk = np.frombuffer(raw_audio, dtype=np.int16)
            prediction = wake_model.predict(audio_chunk)

            for wake_word, score in prediction.items():
                if score > WAKE_CONFIDENCE_THRESHOLD:
                    print(f"\nWake word heard! (confidence: {score:.2f})")

                    # Pause the wake word stream while we record the question,
                    # so the two don't fight over the mic
                    wake_stream.stop_stream()

                    filename = record_audio()
                    print("Transcribing...")
                    question = transcribe(filename)
                    print(f"You said: {question}")

                    # Whisper sometimes labels silence or background noise as
                    # things like "[Music]" or "[BLANK_AUDIO]" instead of
                    # returning empty text. Catch those so we don't waste an
                    # API call asking Claude to respond to nothing.
                    noise_artifacts = ["[music]", "[blank_audio]", "[silence]", "(silence)"]
                    is_real_question = question and question.lower().strip() not in noise_artifacts

                    if is_real_question:
                        print("Asking Claude...")
                        answer = ask_claude(question)
                        print(f"Atlas: {answer}")
                        speak(answer)
                    else:
                        print("Didn't catch anything, try again.")

                    os.remove(filename)
                    print("\nListening for wake word again...")
                    wake_stream.start_stream()

    except KeyboardInterrupt:
        print("\nStopping...")

    finally:
        wake_stream.stop_stream()
        wake_stream.close()
        audio.terminate()


if __name__ == "__main__":
    main()
