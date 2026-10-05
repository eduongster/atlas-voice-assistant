"""
Demo mode: drive the dashboard with scripted events. No mic, no models, no API.

    python3 -m dashboard --demo                 scripted scenario (below)
    python3 -m dashboard --demo --loop          ...repeat forever (for recording video)
    python3 -m dashboard --demo --speed 2       ...twice as fast
    python3 -m dashboard --demo sessions/x.jsonl  replay a session saved with
                                                  `python3 voice_loop.py --record`

Events go through the same EventBus as live Atlas, in strict mode, so the
script doubles as a check that the schema covers everything the UI shows.
"""

import json
import random
import threading
import time
from datetime import datetime, timedelta

from . import events
from .events import latency_status

WAKE_WORD = "hey_jarvis"
MODEL = "claude-sonnet-5"
STT = "Whisper base"
MIC = "MacBook Pro Microphone"


def _iso(dt):
    return dt.astimezone().isoformat(timespec="seconds")


def _next_weekday_at(weekday, hour):
    """Next given weekday (Mon=0) at hour:00, at least one day from now."""
    now = datetime.now()
    days = (weekday - now.weekday()) % 7 or 7
    return (now + timedelta(days=days)).replace(hour=hour, minute=0, second=0, microsecond=0)


def _today_at(hour):
    now = datetime.now()
    at = now.replace(hour=hour, minute=0, second=0, microsecond=0)
    return at if at > now else at + timedelta(days=1)


class _Script:
    """Builds a time-ordered list of (t_seconds, type, data) steps.
    `data` may be a callable(speed) for values that depend on emit time."""

    def __init__(self, seed=7):
        self.t = 0.0
        self.turn = 0
        self.steps = []
        self._actions = 0
        self._rng = random.Random(seed)

    def at(self, delay, etype, data=None, **fields):
        self.t += delay
        self.steps.append((round(self.t, 3), etype, data if data is not None else fields))

    def later(self, offset, etype, data=None, **fields):
        """Schedule an event `offset` seconds from now without advancing time."""
        self.steps.append((round(self.t + offset, 3), etype, data if data is not None else fields))

    def levels(self, seconds, kind):
        lo, hi = {"idle": (0.02, 0.06), "quiet": (0.03, 0.09), "speech": (0.15, 0.75)}[kind]
        for i in range(int(seconds * 10)):
            if kind == "speech":
                # syllable-ish envelope with small gaps between words
                env = abs((i % 7) - 3) / 3
                rms = lo + (hi - lo) * (1 - env) * self._rng.uniform(0.55, 1.0)
            else:
                rms = self._rng.uniform(lo, hi)
            self.at(0.1, "audio.level", rms=round(rms, 3))

    def action_id(self):
        self._actions += 1
        return f"act_{self._actions}"

    def health(self, component, status, label, **extra):
        self.at(0, "health", component=component, status=status, label=label, **extra)

    def turn_start(self, rec_s, speech=True):
        self.turn += 1
        t = self.turn
        self.at(0, "wake.detected", turn=t, word=WAKE_WORD,
                confidence=round(self._rng.uniform(0.82, 0.97), 2))
        self.at(0, "state", state="wake", turn=t)
        self.at(0.7, "state", state="recording", turn=t)
        self.levels(rec_s, "speech" if speech else "quiet")
        self.at(0, "recording.finished", turn=t, duration_s=rec_s, speech_detected=speech)
        return t

    def heard(self, t, question, stt_ms):
        self.at(0, "state", state="transcribing", turn=t)
        self.at(stt_ms / 1000, "turn.user", turn=t, text=question, stt_ms=stt_ms)
        self.health("stt", latency_status("stt", stt_ms), STT,
                    detail=f"last took {stt_ms / 1000:.1f} s", latency_ms=stt_ms)
        self.at(0, "state", state="thinking", turn=t)

    def answer(self, t, text, llm_ms, already_waited=0.0):
        self.at(max(llm_ms / 1000 - already_waited, 0), "turn.assistant",
                turn=t, text=text, llm_ms=llm_ms, model=MODEL)
        self.health("llm", latency_status("llm", llm_ms), MODEL,
                    detail=f"last answer {llm_ms / 1000:.1f} s", latency_ms=llm_ms)
        self.at(0, "state", state="speaking", turn=t)
        self.at(max(2.0, len(text) / 15), "state", state="idle", turn=t)

    def qa(self, question, answer, stt_ms=400, llm_ms=1200, rec_s=2.2):
        t = self.turn_start(rec_s)
        self.heard(t, question, stt_ms)
        self.answer(t, answer, llm_ms)
        return t


def scenario():
    """The scripted demo: about 95 seconds at speed 1."""
    s = _Script()
    s.at(0, "session.started", mode="demo", wake_word=WAKE_WORD, model=MODEL, stt=STT)
    s.health("wake", "ok", WAKE_WORD)
    s.health("stt", "idle", STT, detail="no audio yet")
    s.health("llm", "idle", MODEL, detail="No calls yet")
    s.health("mic", "ok", MIC)
    s.at(0, "state", state="idle")
    s.levels(4, "idle")                                   # empty state on screen

    # 1. plain question
    s.qa("What's a good side dish for salmon?",
         "Roasted asparagus with lemon is quick and goes well with salmon. "
         "Rice or a simple salad work too.", stt_ms=380, llm_ms=1300)
    s.levels(2, "idle")

    # 2. timer (short, so it rings on camera)
    t = s.turn_start(2.0)
    s.heard(t, "Set a tea timer for thirty seconds.", 340)
    tea = s.action_id()
    s.at(0.4, "action", action_id=tea, turn=t, kind="timer.start", status="pending", title="Tea",
         payload={"label": "Tea", "duration_s": 30, "ends_at": _iso(datetime.now())})
    s.at(0.3, "action", lambda speed: {
        "action_id": tea, "turn": t, "kind": "timer.start", "status": "done", "title": "Tea",
        "payload": {"label": "Tea", "duration_s": 30,
                    "ends_at": _iso(datetime.now() + timedelta(seconds=30 / speed))}})
    s.later(30, "action", lambda speed: {
        "action_id": tea, "turn": t, "kind": "timer.start", "status": "triggered", "title": "Tea",
        "payload": {"label": "Tea", "duration_s": 30, "ends_at": _iso(datetime.now())}})
    s.answer(t, "Tea timer set for thirty seconds.", 950, already_waited=0.7)
    s.levels(2, "idle")

    # 3. calendar event
    t = s.turn_start(2.4)
    s.heard(t, "Put the dentist on Thursday at three.", 410)
    dentist = s.action_id()
    start = _next_weekday_at(3, 15)
    cal = {"title": "Dentist", "start": _iso(start), "end": _iso(start + timedelta(hours=1)),
           "location": "Elm Street Dental", "calendar": "Personal"}
    s.at(0.5, "action", action_id=dentist, turn=t, kind="calendar.create", status="pending",
         title="Dentist", payload=cal)
    s.at(0.9, "action", action_id=dentist, turn=t, kind="calendar.create", status="done",
         title="Dentist", payload=cal)
    s.answer(t, "Done. The dentist is on your calendar for Thursday at 3 PM.", 1700,
             already_waited=1.4)
    s.levels(2, "idle")

    # 4. wake word, then nothing said
    t = s.turn_start(4.0, speech=False)
    s.at(0, "turn.discarded", turn=t, reason="no_speech")
    s.at(0, "state", state="idle", turn=t)
    s.levels(2, "idle")

    # 5. reminder
    t = s.turn_start(1.8)
    s.heard(t, "Remind me to call Mom at six.", 360)
    rem = s.action_id()
    s.at(0.5, "action", action_id=rem, turn=t, kind="reminder.create", status="done",
         title="Call Mom", payload={"text": "Call Mom", "due_at": _iso(_today_at(18))})
    s.answer(t, "I'll remind you to call Mom at 6 PM.", 1100, already_waited=0.5)
    s.levels(3, "idle")

    # 6. Claude fails
    t = s.turn_start(2.1)
    s.heard(t, "How long does fresh pasta take to cook?", 390)
    s.at(2.6, "error", turn=t, stage="llm", message="overloaded (error 529)", recoverable=True)
    s.health("llm", "failing", MODEL, error="overloaded (529)")
    s.at(0, "state", state="error", turn=t)
    s.at(4.0, "state", state="idle", turn=t)
    s.levels(3, "idle")

    # 7. retry works, but slowly (amber)
    s.qa("How long does fresh pasta take to cook?",
         "Fresh pasta cooks fast, usually two to four minutes in salted boiling water. "
         "Taste a piece at two minutes.", stt_ms=420, llm_ms=4600, rec_s=2.0)
    s.levels(6, "idle")

    return sorted(s.steps, key=lambda step: step[0])


def load_recording(path):
    """Turn a `--record` JSONL file into replayable steps (same timing).
    Timer/reminder times are shifted so countdowns are live during replay."""
    rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    if not rows:
        return []
    t0 = datetime.fromisoformat(rows[0]["ts"])
    steps = []
    for row in rows:
        when = datetime.fromisoformat(row["ts"])
        data = dict(row["data"])
        if row["type"] == "session.started":
            data["mode"] = "replay"
        payload = data.get("payload") or {}
        if any(k in payload for k in ("ends_at", "due_at")):
            data = _shifted(data, when)
        steps.append(((when - t0).total_seconds(), row["type"], data))
    return steps


def _shifted(data, recorded_at):
    def build(_speed):
        shift = datetime.now().astimezone() - recorded_at
        payload = dict(data["payload"])
        for key in ("ends_at", "due_at"):
            if key in payload:
                payload[key] = _iso(datetime.fromisoformat(payload[key]) + shift)
        return {**data, "payload": payload}
    return build


def run(bus, steps, speed=1.0, stop=None):
    """Emit steps in real time (divided by speed). Returns False if stopped."""
    stop = stop or threading.Event()
    start = time.monotonic()
    for t, etype, data in steps:
        delay = start + t / speed - time.monotonic()
        if delay > 0 and stop.wait(delay):
            return False
        bus.emit(etype, **(data(speed) if callable(data) else data))
    return True


def play(bus, make_steps, speed=1.0, loop=False, stop=None):
    """Run make_steps() once (or forever with loop). Each pass starts a new
    session and rebuilds the steps, so dates and countdowns stay current."""
    stop = stop or threading.Event()
    while True:
        bus.reset()
        if not run(bus, make_steps(), speed, stop) or not loop:
            return
        if stop.wait(4):
            return


def default_bus():
    """Strict bus installed as the process-wide one, for demo runs."""
    bus = events.EventBus(strict=True)
    events.set_bus(bus)
    return bus
