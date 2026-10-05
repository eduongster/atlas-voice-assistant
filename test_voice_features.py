"""
Offline test harness for the two features in voice_loop.py:

  1. Conversation memory  (ask_claude keeps a running history list)
  2. Voice activity detection (record_until_silence stops on ~0.8s of silence)

No microphone, no network, no Anthropic API, no whisper/wakeword model loads.
Hardware/network deps are stubbed in sys.modules before the target module is
imported; webrtcvad is the REAL library so the VAD state machine is exercised
for real, fed by a fake audio stream built from temp_recording.wav + silence.

Run:  python3 test_voice_features.py
Exit code is non-zero if any check fails.
"""

import importlib.util
import sys
import types
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import webrtcvad

HERE = Path(__file__).resolve().parent
TARGET = HERE / "voice_loop.py"
# 16kHz mono speech clip (5.3s, no pause longer than ~0.3s), generated on macOS with:
#   say -v Samantha -o tests/fixtures/speech_sample.wav --data-format=LEI16@16000 \
#     "Hey Atlas, what's on my calendar for tomorrow morning, and do I have anything before ten?"
# Other macOS versions may synthesize a slightly different length; anything ~4.2-5.5s passes.
REAL_WAV = HERE / "tests/fixtures/speech_sample.wav"
SCRATCH = Path("/private/tmp/claude-501/-Users-eduong-atlas-voice-assistant"
               "/bf44a04e-4190-4e06-94c4-c61b18da9c15/scratchpad")
SCRATCH.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# tiny test runner
# --------------------------------------------------------------------------
_results = []


def check(name, ok, detail=""):
    _results.append((name, bool(ok), detail))
    mark = "PASS" if ok else "FAIL"
    line = f"  [{mark}] {name}"
    if detail:
        line += f"\n         {detail}"
    print(line)


def section(title):
    print(f"\n=== {title} ===")


# --------------------------------------------------------------------------
# stub the heavy / hardware / network imports BEFORE importing the target
# --------------------------------------------------------------------------
class _FakeStream:
    """Stand-in for a pyaudio input stream: serves a fixed PCM buffer in
    frame-sized reads, padding the tail with silence once it runs out."""

    def __init__(self, pcm: bytes):
        self._pcm = pcm
        self._pos = 0
        self.stopped = False
        self.closed = False

    def read(self, num_frames, exception_on_overflow=True):
        nbytes = num_frames * 2  # int16 mono
        chunk = self._pcm[self._pos:self._pos + nbytes]
        self._pos += nbytes
        if len(chunk) < nbytes:                       # ran out -> pad with silence
            chunk = chunk + b"\x00" * (nbytes - len(chunk))
        return chunk

    def stop_stream(self):
        self.stopped = True

    def close(self):
        self.closed = True

    def start_stream(self):
        self.stopped = False


class _ShortReadStream:
    """Like _FakeStream, but returns one deliberately short (300-byte) read at
    frame index `short_at`, simulating a device glitch / stream closing."""

    def __init__(self, pcm: bytes, short_at: int):
        self._pcm = pcm
        self._pos = 0
        self._n = 0
        self._short_at = short_at
        self.stopped = False
        self.closed = False

    def read(self, num_frames, exception_on_overflow=True):
        self._n += 1
        if self._n == self._short_at:
            return b"\x11" * 300                      # not 640 -> guard must skip it
        nbytes = num_frames * 2
        chunk = self._pcm[self._pos:self._pos + nbytes]
        self._pos += nbytes
        if len(chunk) < nbytes:
            chunk = chunk + b"\x00" * (nbytes - len(chunk))
        return chunk

    def stop_stream(self):
        self.stopped = True

    def close(self):
        self.closed = True


class _FakePyAudio:
    def __init__(self, *a, **k):
        # tests overwrite .open per-scenario via mod.audio.open = ...
        self._pcm = b""

    def open(self, *a, **k):
        return _FakeStream(self._pcm)

    def get_sample_size(self, _fmt):
        return 2

    def terminate(self):
        pass


def _install_stubs():
    pyaudio_mod = types.ModuleType("pyaudio")
    pyaudio_mod.paInt16 = 8
    pyaudio_mod.PyAudio = _FakePyAudio
    sys.modules["pyaudio"] = pyaudio_mod

    oww = types.ModuleType("openwakeword")
    oww_model = types.ModuleType("openwakeword.model")

    class _WakeModel:
        def __init__(self, *a, **k):
            pass

        def predict(self, *a, **k):
            return {}

    oww_model.Model = _WakeModel
    oww.model = oww_model
    sys.modules["openwakeword"] = oww
    sys.modules["openwakeword.model"] = oww_model

    pw = types.ModuleType("pywhispercpp")
    pw_model = types.ModuleType("pywhispercpp.model")

    class _WhisperModel:
        def __init__(self, *a, **k):
            pass

        def transcribe(self, *a, **k):
            return []

    pw_model.Model = _WhisperModel
    pw.model = pw_model
    sys.modules["pywhispercpp"] = pw
    sys.modules["pywhispercpp.model"] = pw_model

    anthropic_mod = types.ModuleType("anthropic")

    class _Anthropic:
        def __init__(self, *a, **k):
            self.messages = SimpleNamespace(create=lambda **kw: None)

    anthropic_mod.Anthropic = _Anthropic
    sys.modules["anthropic"] = anthropic_mod

    dotenv_mod = types.ModuleType("dotenv")
    dotenv_mod.load_dotenv = lambda *a, **k: False
    sys.modules["dotenv"] = dotenv_mod


def _load_target():
    spec = importlib.util.spec_from_file_location("voice_loop_under_test", str(TARGET))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["voice_loop_under_test"] = mod
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------
# audio fixtures
# --------------------------------------------------------------------------
def _real_speech_pcm() -> bytes:
    with wave.open(str(REAL_WAV), "rb") as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1 and w.getsampwidth() == 2
        return w.readframes(w.getnframes())


def _silence_pcm(seconds: float) -> bytes:
    return b"\x00\x00" * int(16000 * seconds)


def _loud_noise_pcm(seconds: float) -> bytes:
    rng = np.random.default_rng(0)
    return rng.integers(-9000, 9000, int(16000 * seconds)).astype("<i2").tobytes()


def _wav_duration(path) -> float:
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / w.getframerate()


# --------------------------------------------------------------------------
# VAD tests
# --------------------------------------------------------------------------
def _reset_vad(mod):
    """The target module keeps ONE module-level webrtcvad.Vad for the whole
    session; webrtcvad adapts its internal model to whatever audio it has
    already seen, so recording N's exact cut point depends on recordings
    1..N-1. Reset it between synthetic scenarios so each checks one property
    in isolation. (See review note: the real app does not reset it.)"""
    mod.vad = webrtcvad.Vad(mod.VAD_AGGRESSIVENESS)


def test_vad(mod):
    section("Voice activity detection (record_until_silence)")

    check(
        "frame math: 20ms frames, 40 silent frames = 0.8s, 200 frames = 4s initial timeout",
        mod.VAD_FRAME_MS in (10, 20, 30)
        and mod.VAD_FRAME_SIZE == 320
        and mod.SILENCE_FRAMES_NEEDED == 40
        and int(mod.INITIAL_TIMEOUT * 1000 / mod.VAD_FRAME_MS) == 200,
        f"VAD_FRAME_SIZE={mod.VAD_FRAME_SIZE} SILENCE_FRAMES_NEEDED={mod.SILENCE_FRAMES_NEEDED}",
    )

    max_frames = int(mod.MAX_RECORD_SECONDS * 1000 / mod.VAD_FRAME_MS)

    # --- A: speech then silence -> stops shortly after you stop talking ---
    _reset_vad(mod)
    mod.audio._pcm = _real_speech_pcm() + _silence_pcm(1.5)   # 4.96s speech + 1.5s quiet
    out = SCRATCH / "vad_a.wav"
    fname, speech = mod.record_until_silence(str(out))
    dur = _wav_duration(fname)
    check(
        "A: detects speech and auto-stops on trailing silence (not the safety cap)",
        speech is True and 4.8 < dur < 6.4 and dur < mod.MAX_RECORD_SECONDS - 1,
        f"speech_detected={speech}, recorded {dur:.2f}s (expected ~5.0-5.9s: 4.96s speech + ~0.8s)",
    )

    # --- B: pure silence -> initial-timeout bail, no speech ---
    _reset_vad(mod)
    mod.audio._pcm = _silence_pcm(6.0)
    out = SCRATCH / "vad_b.wav"
    fname, speech = mod.record_until_silence(str(out))
    dur = _wav_duration(fname)
    check(
        "B: no speech within INITIAL_TIMEOUT -> returns speech_detected=False at ~4s",
        speech is False and 3.8 < dur < 4.3,
        f"speech_detected={speech}, recorded {dur:.2f}s (expected ~4.00s)",
    )

    # --- C: continuous speech -> MAX_RECORD_SECONDS safety cap ---
    _reset_vad(mod)
    mod.audio._pcm = _loud_noise_pcm(mod.MAX_RECORD_SECONDS + 5)
    out = SCRATCH / "vad_c.wav"
    fname, speech = mod.record_until_silence(str(out))
    dur = _wav_duration(fname)
    check(
        "C: never goes silent -> hard-stops at MAX_RECORD_SECONDS",
        speech is True and abs(dur - mod.MAX_RECORD_SECONDS) < 0.3,
        f"speech_detected={speech}, recorded {dur:.2f}s (expected ~{mod.MAX_RECORD_SECONDS}.00s cap)",
    )

    # --- D: internal pauses shorter than 0.8s do NOT end the recording ---
    # real wav has internal silence runs up to ~0.52s; feeding it alone (then
    # padding silence) must not stop until the 0.8s trailing-silence threshold.
    _reset_vad(mod)
    mod.audio._pcm = _real_speech_pcm() + _silence_pcm(1.5)
    out = SCRATCH / "vad_d.wav"
    fname, speech = mod.record_until_silence(str(out))
    dur = _wav_duration(fname)
    check(
        "D: mid-sentence pauses (<0.8s) don't cut the recording early",
        speech is True and dur > 4.9,
        f"recorded {dur:.2f}s; real speech is 4.96s so anything >4.9s means no premature stop",
    )

    # --- E: a short/partial read from the device is skipped, not fatal ---
    _reset_vad(mod)
    pcm = _real_speech_pcm() + _silence_pcm(1.5)
    mod.audio.open = lambda *a, **k: _ShortReadStream(pcm, short_at=50)
    out = SCRATCH / "vad_e.wav"
    try:
        fname, speech = mod.record_until_silence(str(out))
        raised = None
    except Exception as e:                       # noqa: BLE001 - we want any exception
        raised, speech = e, None
    check(
        "E: a short/partial audio frame is skipped by the VAD guard, not fatal",
        raised is None and speech is True,
        f"raised={raised!r}, speech_detected={speech} "
        "(without the guard, webrtcvad.is_speech() raises Error on a non-20ms buffer)",
    )
    mod.audio.open = _FakePyAudio.open.__get__(mod.audio)   # restore default


# --------------------------------------------------------------------------
# conversation-memory tests
# --------------------------------------------------------------------------
class _FakeMessages:
    def __init__(self):
        self.calls = []
        self.raise_once = None

    def create(self, *, model, max_tokens, system, messages):
        # snapshot exactly what would be sent to the real API
        self.calls.append({"model": model, "system": system,
                           "messages": [dict(m) for m in messages]})
        if self.raise_once is not None:
            err, self.raise_once = self.raise_once, None
            raise err
        text = f"reply#{len(self.calls)}"
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)])


class _FakeClient:
    def __init__(self):
        self.messages = _FakeMessages()


def _fresh_client(mod):
    c = _FakeClient()
    mod.client = c
    mod.conversation_history.clear()
    return c


def test_memory(mod):
    section("Conversation memory (ask_claude)")

    # --- 1: one turn appends user+assistant to history ---
    c = _fresh_client(mod)
    ans = mod.ask_claude("My name is Sam.")
    check(
        "1: first turn returns the reply and records both sides",
        ans == "reply#1"
        and mod.conversation_history == [
            {"role": "user", "content": "My name is Sam."},
            {"role": "assistant", "content": "reply#1"},
        ],
        f"history={mod.conversation_history}",
    )

    # --- 2: second turn actually SENDS the prior turn to the API ---
    mod.ask_claude("What's my name?")
    sent = c.messages.calls[1]["messages"]
    check(
        "2: second call includes the earlier exchange as context",
        [m["role"] for m in sent] == ["user", "assistant", "user"]
        and sent[0]["content"] == "My name is Sam."
        and sent[1]["content"] == "reply#1"
        and c.messages.calls[1]["system"] == mod.SYSTEM_PROMPT,
        f"messages sent on call 2 = {[ (m['role'], m['content']) for m in sent ]}",
    )

    # --- 3: history is trimmed to MAX_HISTORY_LENGTH, alternation preserved ---
    c = _fresh_client(mod)
    for i in range(15):                       # 15 turns -> 30 entries -> trim to 20
        mod.ask_claude(f"q{i}")
    hist = mod.conversation_history
    roles = [m["role"] for m in hist]
    alternating = all(
        roles[j] == ("user" if j % 2 == 0 else "assistant") for j in range(len(roles))
    )
    check(
        "3: long conversation trims to MAX_HISTORY_LENGTH without breaking user/assistant order",
        len(hist) == mod.MAX_HISTORY_LENGTH
        and roles[0] == "user"
        and alternating
        and hist[0]["content"] == "q5"          # oldest retained = turn 6
        and hist[-2]["content"] == "q14"        # newest user turn
        and hist[-1]["content"] == "reply#15",
        f"len={len(hist)} first={hist[0]['content']!r} last_user={hist[-2]['content']!r}",
    )

    # --- 4: trailing text block is picked even after a leading non-text block ---
    c = _fresh_client(mod)
    c.messages.create = lambda **kw: SimpleNamespace(content=[
        SimpleNamespace(type="thinking", thinking="hmm"),
        SimpleNamespace(type="text", text="the answer"),
    ])
    ans = mod.ask_claude("hi")
    check(
        "4: extracts the text block when a non-text block comes first",
        ans == "the answer" and mod.conversation_history[-1] == {
            "role": "assistant", "content": "the answer"},
        f"got {ans!r}",
    )

    # --- 5: regression - a failed API call must NOT corrupt history ---
    c = _fresh_client(mod)
    c.messages.raise_once = RuntimeError("simulated API 500")
    raised = False
    try:
        mod.ask_claude("boom")
    except RuntimeError:
        raised = True
    untouched = mod.conversation_history == []          # nothing committed on failure
    # a later call succeeds and sends a clean single user turn (no leftover)
    mod.ask_claude("are you there?")
    next_sent = c.messages.calls[-1]["messages"]
    clean = [m["role"] for m in next_sent] == ["user"] and \
        next_sent[0]["content"] == "are you there?"
    check(
        "5: a failed API call leaves history uncorrupted (no dangling user turn)",
        raised and untouched and clean,
        f"error propagated={raised}, history after error={mod.conversation_history}, "
        f"roles on next call={[m['role'] for m in next_sent]}",
    )

    # --- 6: history recovers cleanly and keeps working after the failure ---
    ans = mod.ask_claude("second question")
    check(
        "6: conversation continues normally after a recovered failure",
        ans == f"reply#{len(c.messages.calls)}"
        and [m["role"] for m in mod.conversation_history] == [
            "user", "assistant", "user", "assistant"],
        f"history roles={[m['role'] for m in mod.conversation_history]}",
    )


# --------------------------------------------------------------------------
def main():
    if not TARGET.exists():
        print(f"target not found: {TARGET}")
        return 2
    if not REAL_WAV.exists():
        print(f"fixture wav not found: {REAL_WAV}")
        return 2

    _install_stubs()
    mod = _load_target()
    print(f"loaded {TARGET.name}")

    test_vad(mod)
    test_memory(mod)

    section("summary")
    passed = sum(1 for _, ok, _ in _results if ok)
    failed = [n for n, ok, _ in _results if not ok]
    print(f"  {passed}/{len(_results)} checks passed")
    for n in failed:
        print(f"  FAILED: {n}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
