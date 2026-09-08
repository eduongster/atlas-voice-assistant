"""
Atlas voice assistant - voice loop with memory and auto-stop recording.

Flow: wake word triggers recording -> records until you stop talking
(instead of a fixed timer) -> whisper.cpp transcribes it -> Claude
answers, with the full conversation so far as context -> macOS 'say'
speaks the answer out loud -> goes back to listening for the wake word.

This builds on the earlier basic voice_loop.py. Two upgrades:
1. Conversation memory: every turn gets added to a running history list,
   and the whole list gets sent to Claude each time, so it can refer
   back to earlier things you said in the same session.
2. Voice activity detection (VAD): instead of always recording for a
   fixed number of seconds, this listens for a stretch of silence and
   stops recording right then, so you're not cut off or left waiting.
"""

import os
import subprocess
import wave

import numpy as np
import pyaudio
import webrtcvad
from dotenv import load_dotenv
from openwakeword.model import Model
from pywhispercpp.model import Model as WhisperModel
import anthropic

# Load your API key from the .env file (never hardcode it here)
load_dotenv()
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# --- Wake word setup ---
wake_model = Model(wakeword_models=["hey_jarvis"], inference_framework="onnx")
WAKE_CONFIDENCE_THRESHOLD = 0.7

# --- Audio settings ---
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
WAKE_CHUNK = 1280  # 80ms, what openWakeWord expects

# --- Voice activity detection settings ---
VAD_FRAME_MS = 20  # webrtcvad only accepts 10, 20, or 30ms frames
VAD_FRAME_SIZE = int(RATE * VAD_FRAME_MS / 1000)  # samples per frame
SILENCE_DURATION = 0.8  # seconds of quiet before we consider you done talking
SILENCE_FRAMES_NEEDED = int(SILENCE_DURATION * 1000 / VAD_FRAME_MS)
INITIAL_TIMEOUT = 4  # seconds to wait for you to start talking at all
MAX_RECORD_SECONDS = 20  # hard safety cap, in case VAD never detects silence
VAD_AGGRESSIVENESS = 2  # 0 (lenient) to 3 (strict) about what counts as speech

# --- Whisper setup ---
whisper_model = WhisperModel("base")

# --- Claude's personality ---
SYSTEM_PROMPT = (
    "You are Atlas, a helpful voice assistant. Keep answers short and "
    "conversational, since they get read out loud. No markdown, no lists, "
    "just plain spoken sentences."
)

# How many turns to keep in memory. Each question + answer is 2 entries,
# so 20 keeps roughly the last 10 exchanges. Trimmed to control cost and
# keep old, less relevant context from crowding out the recent stuff.
MAX_HISTORY_LENGTH = 20
conversation_history = []

audio = pyaudio.PyAudio()
vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)


def record_until_silence(filename="temp_recording.wav"):
    """
    Records audio starting right after the wake word, and stops as soon
    as it detects you've gone quiet for SILENCE_DURATION seconds.
    Returns (filename, whether_any_speech_was_detected).
    """
    stream = audio.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=VAD_FRAME_SIZE,
    )
    print("Listening for your question...")

    frames = []
    speech_detected = False
    silence_streak = 0
    frames_read = 0

    max_frames = int(MAX_RECORD_SECONDS * 1000 / VAD_FRAME_MS)
    initial_timeout_frames = int(INITIAL_TIMEOUT * 1000 / VAD_FRAME_MS)

    while frames_read < max_frames:
        chunk = stream.read(VAD_FRAME_SIZE, exception_on_overflow=False)
        frames.append(chunk)
        frames_read += 1

        # webrtcvad only accepts exact 10/20/30ms frames. If PortAudio ever
        # hands back a short buffer (device glitch, stream closing mid-read),
        # is_speech() would raise and kill the loop. Keep the audio, but skip
        # VAD analysis for that partial frame.
        if len(chunk) != VAD_FRAME_SIZE * 2:  # 2 bytes per int16 sample, mono
            continue

        is_speech = vad.is_speech(chunk, RATE)

        if is_speech:
            speech_detected = True
            silence_streak = 0
        elif speech_detected:
            # You were talking, now you've gone quiet. Count how long.
            silence_streak += 1
            if silence_streak >= SILENCE_FRAMES_NEEDED:
                break
        else:
            # You haven't said anything yet. Give up if it's been too long.
            if frames_read >= initial_timeout_frames:
                break

    stream.stop_stream()
    stream.close()

    with wave.open(filename, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b"".join(frames))

    return filename, speech_detected


def transcribe(filename):
    """Runs the recorded audio through local whisper.cpp and returns the text."""
    segments = whisper_model.transcribe(filename)
    text = " ".join(segment.text for segment in segments).strip()
    return text


def ask_claude(question):
    """
    Sends the question to Claude along with the conversation so far,
    so it can refer back to earlier things you said. Updates the
    running history with both your question and Claude's answer.

    History is only updated after a successful API call. If the call
    raises (rate limit, server error, network blip), conversation_history
    is left untouched, so the next call won't send two 'user' turns in a
    row (which the API rejects, breaking the rest of the session).
    """
    user_turn = {"role": "user", "content": question}

    response = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=300,
        system=SYSTEM_PROMPT,
        messages=conversation_history + [user_turn],
    )

    answer = "Sorry, I didn't get a text response back."
    for block in response.content:
        if block.type == "text":
            answer = block.text
            break

    conversation_history.append(user_turn)
    conversation_history.append({"role": "assistant", "content": answer})

    # Trim old history so the list doesn't grow forever
    if len(conversation_history) > MAX_HISTORY_LENGTH:
        del conversation_history[: len(conversation_history) - MAX_HISTORY_LENGTH]

    return answer


def speak(text):
    """Uses macOS's built-in 'say' command to read text out loud."""
    subprocess.run(["say", text])


def main():
    print("Atlas is listening for the wake word... say 'Hey Jarvis'")
    print("Press Ctrl+C to stop.\n")

    wake_stream = audio.open(
        format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=WAKE_CHUNK
    )

    try:
        while True:
            raw_audio = wake_stream.read(WAKE_CHUNK, exception_on_overflow=False)
            audio_chunk = np.frombuffer(raw_audio, dtype=np.int16)
            prediction = wake_model.predict(audio_chunk)

            for wake_word, score in prediction.items():
                if score > WAKE_CONFIDENCE_THRESHOLD:
                    print(f"\nWake word heard! (confidence: {score:.2f})")

                    wake_stream.stop_stream()

                    filename, speech_detected = record_until_silence()

                    if not speech_detected:
                        print("Didn't hear anything, try again.")
                        os.remove(filename)
                        print("\nListening for wake word again...")
                        wake_stream.start_stream()
                        continue

                    print("Transcribing...")
                    question = transcribe(filename)
                    print(f"You said: {question}")

                    noise_artifacts = [
                        "[music]", "[blank_audio]", "[silence]", "(silence)", ""
                    ]
                    is_real_question = question.lower().strip() not in noise_artifacts

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
