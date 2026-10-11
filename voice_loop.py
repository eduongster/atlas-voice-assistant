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

It also reports each step to the live dashboard (dashboard/), which starts
automatically at http://127.0.0.1:8765. The dashboard is optional: if it
can't start, Atlas runs without it. See `python3 voice_loop.py --help`.
"""

import argparse
import os
import re
import subprocess
import time
import wave
from pathlib import Path

import numpy as np
import pyaudio
import webrtcvad
from dotenv import load_dotenv
from openwakeword.model import Model
from pywhispercpp.model import Model as WhisperModel
import anthropic

from dashboard.events import emit, get_bus, latency_status
from dashboard.server import DEFAULT_PORT, start_dashboard

# Load your API key from the .env file (never hardcode it here)
load_dotenv()
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# --- Wake word setup ---
WAKE_WORD = "hey_jarvis"
wake_model = Model(wakeword_models=[WAKE_WORD], inference_framework="onnx")
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
STT_LABEL = "Whisper base"
whisper_model = WhisperModel("base")

# --- Claude model ---
MODEL = "claude-sonnet-5"

# --- Dashboard ---
LEVEL_EVERY_N_FRAMES = 5  # mic level to the dashboard every 100ms while recording
LEVEL_FULL_SCALE = 4000   # int16 RMS that counts as "loud speech" (level 1.0)
mic_label = "Default microphone"  # replaced with the real device name in main()

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

        if frames_read % LEVEL_EVERY_N_FRAMES == 0:
            emit("audio.level", rms=_rms(chunk))

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
        model=MODEL,
        max_tokens=1024,
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


def _rms(chunk):
    """Loudness of an int16 audio chunk as a 0..1 level for the dashboard meter."""
    samples = np.frombuffer(chunk, dtype=np.int16).astype(np.float32)
    if samples.size == 0:
        return 0.0
    rms = float(np.sqrt(np.mean(samples * samples)))
    return round(min(1.0, rms / LEVEL_FULL_SCALE), 3)


# Short, readable names for API status codes (shown in the terminal and dashboard).
STATUS_WORDS = {
    400: "bad request", 401: "API key rejected", 403: "permission denied",
    404: "model not found", 408: "request timed out", 413: "request too large",
    429: "rate limited", 500: "server error", 502: "bad gateway",
    503: "service unavailable", 529: "overloaded",
}

# Anything that looks like a credential is replaced before an error message
# is printed, sent to the dashboard, or written to a --record file.
_SECRET_PATTERNS = [
    re.compile(r"sk-ant-[\w-]+"),
    re.compile(r"(?i)\b(x-api-key|api[_-]?key|authorization|proxy-authorization|bearer|"
               r"token|secret|password|cookie)\b(['\"]?\s*[:=]\s*|\s+)(['\"]?)"
               r"(?:(?:bearer|basic|token)\s+)?[^\s'\",}]+"),   # "Bearer <token>" as one value
    re.compile(r"[A-Za-z0-9_\-]{32,}"),   # long opaque tokens
]


def _redact(text):
    key = os.getenv("ANTHROPIC_API_KEY")
    if key:
        text = text.replace(key, "[redacted]")
    text = _SECRET_PATTERNS[0].sub("[redacted]", text)
    text = _SECRET_PATTERNS[1].sub(r"\1\2\3[redacted]", text)
    return _SECRET_PATTERNS[2].sub("[redacted]", text)


def describe_error(e):
    """
    A short, human description of an exception, safe to show anywhere.
    API errors are described by status code only (never their request,
    response, or headers); other errors keep their message with anything
    secret-looking redacted, cut to 120 characters.
    """
    status = getattr(e, "status_code", None)
    if isinstance(status, int) and not isinstance(status, bool):
        words = STATUS_WORDS.get(status, "server error" if status >= 500 else "request failed")
        return f"{words} (error {status})"
    name = type(e).__name__
    if "Timeout" in name:
        return "timed out waiting for a response"
    if "Connection" in name:
        return "couldn't connect (check your internet connection)"
    text = _redact(str(e)).strip()
    return f"{name}: {text[:120]}" if text else name


STAGE_WORDS = {
    "record": "recording", "transcribe": "transcribing",
    "llm": "asking Claude", "speak": "speaking",
}


def handle_turn(wake_stream, wake_word, score, turn):
    """
    One exchange, from wake word to spoken answer, reported to the dashboard
    at each step. If any step fails, the error is printed and shown on the
    dashboard and Atlas goes back to listening instead of crashing. Ctrl+C
    still propagates to main(), which exits cleanly.
    """
    print(f"\nWake word heard! (confidence: {score:.2f})")
    emit("wake.detected", turn=turn, word=wake_word, confidence=round(float(score), 3))
    emit("state", state="wake", turn=turn)

    wake_stream.stop_stream()

    filename = None
    stage = "record"
    interrupted = False
    try:
        emit("state", state="recording", turn=turn)
        started = time.perf_counter()
        filename, speech_detected = record_until_silence()
        emit("recording.finished", turn=turn, speech_detected=speech_detected,
             duration_s=round(time.perf_counter() - started, 2))

        if not speech_detected:
            print("Didn't hear anything, try again.")
            emit("turn.discarded", turn=turn, reason="no_speech")
            return

        stage = "transcribe"
        print("Transcribing...")
        emit("state", state="transcribing", turn=turn)
        started = time.perf_counter()
        question = transcribe(filename)
        stt_ms = round((time.perf_counter() - started) * 1000)
        print(f"You said: {question}")
        emit("health", component="stt", status=latency_status("stt", stt_ms), label=STT_LABEL,
             detail=f"last took {stt_ms / 1000:.1f} s", latency_ms=stt_ms)

        noise_artifacts = [
            "[music]", "[blank_audio]", "[silence]", "(silence)", ""
        ]
        is_real_question = question.lower().strip() not in noise_artifacts

        if not is_real_question:
            print("Didn't catch anything, try again.")
            emit("turn.discarded", turn=turn, reason="noise", heard=question)
            return

        emit("turn.user", turn=turn, text=question, stt_ms=stt_ms)

        stage = "llm"
        print("Asking Claude...")
        emit("state", state="thinking", turn=turn)
        started = time.perf_counter()
        answer = ask_claude(question)
        llm_ms = round((time.perf_counter() - started) * 1000)
        print(f"Atlas: {answer}")
        emit("turn.assistant", turn=turn, text=answer, llm_ms=llm_ms, model=MODEL)
        emit("health", component="llm", status=latency_status("llm", llm_ms), label=MODEL,
             detail=f"last answer {llm_ms / 1000:.1f} s", latency_ms=llm_ms)

        stage = "speak"
        emit("state", state="speaking", turn=turn)
        speak(answer)

    except KeyboardInterrupt:
        interrupted = True
        raise

    except Exception as e:  # noqa: BLE001 - report it and keep listening
        message = describe_error(e)
        print(f"Something went wrong while {STAGE_WORDS[stage]}: {message}")
        emit("error", turn=turn, stage=stage, message=message, recoverable=True)
        failing = {"llm": ("llm", MODEL), "transcribe": ("stt", STT_LABEL), "record": ("mic", mic_label)}
        if stage in failing:
            component, label = failing[stage]
            emit("health", component=component, status="failing", label=label, error=message)
        emit("state", state="error", turn=turn)

    finally:
        if filename and os.path.exists(filename):
            os.remove(filename)
        if not interrupted:
            print("\nListening for wake word again...")
            emit("state", state="idle", turn=turn)
            wake_stream.start_stream()


def microphone_name():
    try:
        return audio.get_default_input_device_info()["name"]
    except Exception:  # noqa: BLE001 - only a label for the dashboard
        return "Default microphone"


def main():
    global mic_label
    print("Atlas is listening for the wake word... say 'Hey Jarvis'")
    print("Press Ctrl+C to stop.\n")

    emit("session.started", mode="live", wake_word=WAKE_WORD, model=MODEL, stt=STT_LABEL)
    emit("health", component="wake", status="ok", label=WAKE_WORD)
    emit("health", component="stt", status="idle", label=STT_LABEL, detail="no audio yet")
    emit("health", component="llm", status="idle", label=MODEL, detail="No calls yet")

    wake_stream = audio.open(
        format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=WAKE_CHUNK
    )
    mic_label = microphone_name()
    emit("health", component="mic", status="ok", label=mic_label)
    emit("state", state="idle")

    turn = 0
    end_reason = "stopped"
    try:
        while True:
            raw_audio = wake_stream.read(WAKE_CHUNK, exception_on_overflow=False)
            emit("audio.level", rms=_rms(raw_audio))
            audio_chunk = np.frombuffer(raw_audio, dtype=np.int16)
            prediction = wake_model.predict(audio_chunk)

            for wake_word, score in prediction.items():
                if score > WAKE_CONFIDENCE_THRESHOLD:
                    turn += 1
                    handle_turn(wake_stream, wake_word, score, turn)

    except KeyboardInterrupt:
        print("\nStopping...")
        end_reason = "keyboard_interrupt"

    except Exception as e:
        end_reason = f"crashed: {describe_error(e)}"
        raise

    finally:
        emit("session.ended", reason=end_reason)
        wake_stream.stop_stream()
        wake_stream.close()
        audio.terminate()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Atlas voice assistant.")
    parser.add_argument("--no-dashboard", action="store_true",
                        help="don't start the live dashboard")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help=f"dashboard port (default {DEFAULT_PORT})")
    parser.add_argument("--open", action="store_true",
                        help="open the dashboard in your browser")
    parser.add_argument("--record", action="store_true",
                        help="save this session's events, including transcripts, to "
                             "sessions/<id>.jsonl (gitignored) for "
                             "`python3 -m dashboard --demo FILE`")
    return parser.parse_args(argv)


def run(argv=None):
    """Command-line entry point: start the dashboard (optional, never fatal),
    optionally record the session, then run the voice loop."""
    args = parse_args(argv)

    dashboard = None
    if not args.no_dashboard:
        try:
            dashboard = start_dashboard(port=args.port, open_browser=args.open)
        except Exception as e:  # noqa: BLE001 - the dashboard must never stop Atlas
            print(f"Dashboard not started ({describe_error(e)}). Atlas will run without it.")

    if args.record:
        sessions = Path(__file__).resolve().parent / "sessions"
        try:
            sessions.mkdir(exist_ok=True)
            path = sessions / f"{get_bus().session}.jsonl"
            get_bus().record_to(path)
            print(f"Recording this session to {path}")
        except OSError as e:
            print(f"Not recording ({describe_error(e)}).")

    try:
        main()
    finally:
        if args.record:
            get_bus().record_to(None)
        if dashboard:
            dashboard.stop()


if __name__ == "__main__":
    run()
