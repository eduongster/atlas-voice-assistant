"""
Atlas event layer.

voice_loop.py (and demo mode) call emit("type", **data). Every event is wrapped
in the same envelope, kept in an in-memory history for the session, and fanned
out to subscribers (one queue per browser tab, see server.py):

    {"v": 1, "id": 42, "ts": "2026-10-05T14:41:07.123-07:00",
     "session": "s_20261005_144100_ab12", "type": "state", "data": {...}}

Event types (data fields; "?" marks optional):

    session.started   mode ("live"|"demo"|"replay"), wake_word?, model?, stt?
    session.ended     reason?
    state             state (see STATES), turn?, detail?
    wake.detected     turn, word, confidence
    recording.finished turn, duration_s, speech_detected
    turn.user         turn, text, stt_ms?
    turn.assistant    turn, text, llm_ms?, model?
    turn.discarded    turn, reason ("no_speech"|"noise"), heard?
    error             stage (see STAGES), message, turn?, recoverable?
    health            component ("wake"|"stt"|"llm"|"mic"),
                      status ("ok"|"slow"|"failing"|"idle"), label, detail?,
                      latency_ms?, error?
    audio.level       rms (0..1)                        -- transient, not stored
    action            action_id, kind, status (see ACTION_STATUSES), turn?,
                      title?, payload?, error?

Actions are how calendar/timer/reminder features (and later, Claude tool calls)
plug in: the tool name is `kind`, the call emits status "pending", the result
emits "done"/"failed", and timers/reminders later emit "triggered". The
frontend upserts cards by action_id and falls back to a generic card for kinds
it doesn't know. Known kinds have required payload fields (ACTION_PAYLOADS).

Claude's health comes from the last real API call, never a ping: it starts
"idle" ("No calls yet") and becomes ok/slow/failing after each call.

emit() never raises in live mode and never blocks: an invalid event is logged
once and still delivered, and a subscriber that falls too far behind is
dropped (its browser reconnects and gets a fresh snapshot). Strict mode (tests,
demo) raises EventError on an invalid event instead.
"""

import itertools
import json
import logging
import queue
import secrets
import threading
from collections import deque, namedtuple
from datetime import datetime

log = logging.getLogger("atlas.events")

SCHEMA_VERSION = 1

STATES = ("idle", "wake", "recording", "transcribing", "thinking", "speaking", "error")
STAGES = ("wake", "record", "transcribe", "llm", "speak", "action")
SESSION_MODES = ("live", "demo", "replay")
DISCARD_REASONS = ("no_speech", "noise")
HEALTH_COMPONENTS = ("wake", "stt", "llm", "mic")
HEALTH_STATUSES = ("ok", "slow", "failing", "idle")
ACTION_STATUSES = ("pending", "done", "failed", "cancelled", "triggered")

# Events that are delivered live but not kept for snapshot/replay.
TRANSIENT = frozenset({"audio.level"})

# A step slower than this shows amber ("slow") in the status strip.
SLOW_MS = {"stt": 2000, "llm": 4000}

NUM = (int, float)
Field = namedtuple("Field", "types required choices", defaults=(True, None))


def _opt(types, choices=None):
    return Field(types, False, choices)


SCHEMA = {
    "session.started": {
        "mode": Field(str, True, SESSION_MODES),
        "wake_word": _opt(str), "model": _opt(str), "stt": _opt(str),
    },
    "session.ended": {"reason": _opt(str)},
    "state": {
        "state": Field(str, True, STATES),
        "turn": _opt(int), "detail": _opt(str),
    },
    "wake.detected": {"turn": Field(int), "word": Field(str), "confidence": Field(NUM)},
    "recording.finished": {
        "turn": Field(int), "duration_s": Field(NUM), "speech_detected": Field(bool),
    },
    "turn.user": {"turn": Field(int), "text": Field(str), "stt_ms": _opt(NUM)},
    "turn.assistant": {
        "turn": Field(int), "text": Field(str), "llm_ms": _opt(NUM), "model": _opt(str),
    },
    "turn.discarded": {
        "turn": Field(int), "reason": Field(str, True, DISCARD_REASONS), "heard": _opt(str),
    },
    "error": {
        "stage": Field(str, True, STAGES), "message": Field(str),
        "turn": _opt(int), "recoverable": _opt(bool),
    },
    "health": {
        "component": Field(str, True, HEALTH_COMPONENTS),
        "status": Field(str, True, HEALTH_STATUSES),
        "label": Field(str), "detail": _opt(str),
        "latency_ms": _opt(NUM), "error": _opt(str),
    },
    "audio.level": {"rms": Field(NUM)},
    "action": {
        "action_id": Field(str), "kind": Field(str),
        "status": Field(str, True, ACTION_STATUSES),
        "turn": _opt(int), "title": _opt(str), "payload": _opt(dict), "error": _opt(str),
    },
}

# Required payload fields for the action kinds the frontend has cards for.
# Unknown kinds are allowed (generic card), so new features need no schema change.
ACTION_PAYLOADS = {
    "calendar.create": {"title": str, "start": str},
    "timer.start": {"label": str, "duration_s": NUM, "ends_at": str},
    "reminder.create": {"text": str, "due_at": str},
    "calendar.list": {"range": str, "events": list},
}


class EventError(ValueError):
    """Raised by emit() in strict mode when an event doesn't match SCHEMA."""


def _type_ok(value, types):
    # bool is a subclass of int; a True where a number belongs is a bug.
    if isinstance(value, bool) and types is not bool:
        return False
    return isinstance(value, types)


def validate(event):
    """Return a list of problems with `event` (empty list means valid)."""
    if not isinstance(event, dict):
        return ["event is not a dict"]
    problems = []
    if event.get("v") != SCHEMA_VERSION:
        problems.append(f"v must be {SCHEMA_VERSION}")
    if not isinstance(event.get("id"), int) or isinstance(event.get("id"), bool):
        problems.append("id must be an int")
    for key in ("ts", "session"):
        if not isinstance(event.get(key), str):
            problems.append(f"{key} must be a string")

    etype, data = event.get("type"), event.get("data")
    spec = SCHEMA.get(etype)
    if spec is None:
        return problems + [f"unknown event type {etype!r}"]
    if not isinstance(data, dict):
        return problems + ["data must be a dict"]

    for name, field in spec.items():
        if name not in data:
            if field.required:
                problems.append(f"{etype}: missing {name}")
            continue
        value = data[name]
        if not _type_ok(value, field.types):
            problems.append(f"{etype}: {name} has wrong type {type(value).__name__}")
        elif field.choices and value not in field.choices:
            problems.append(f"{etype}: {name}={value!r} not in {field.choices}")
    for name in data:
        if name not in spec:
            problems.append(f"{etype}: unexpected field {name}")

    if etype == "audio.level" and _type_ok(data.get("rms"), NUM) and not 0 <= data["rms"] <= 1:
        problems.append("audio.level: rms must be between 0 and 1")
    if etype == "action" and data.get("kind") in ACTION_PAYLOADS:
        payload = data.get("payload")
        if not isinstance(payload, dict):
            problems.append(f"action {data['kind']}: payload required")
        else:
            for name, types in ACTION_PAYLOADS[data["kind"]].items():
                if not _type_ok(payload.get(name), types):
                    problems.append(f"action {data['kind']}: payload.{name} missing or wrong type")
    return problems


def latency_status(component, latency_ms):
    """'slow' if a step took longer than SLOW_MS for its component, else 'ok'."""
    return "slow" if latency_ms > SLOW_MS.get(component, float("inf")) else "ok"


def now_iso():
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def new_session_id():
    return datetime.now().strftime("s_%Y%m%d_%H%M%S_") + secrets.token_hex(2)


class EventBus:
    """Thread-safe event history + fan-out. One per process in practice."""

    def __init__(self, strict=False, max_history=5000, queue_size=2000):
        self.strict = strict
        self._max_history = max_history
        self._queue_size = queue_size
        self._lock = threading.Lock()
        self._subscribers = set()
        self._record_file = None
        self._warned = set()
        self.reset()

    # -- session --------------------------------------------------------
    def reset(self, session=None):
        """Start a new session: fresh id counter and empty history.
        Subscribers stay connected; the next event carries the new session id,
        which tells the browser to clear its view."""
        with self._lock:
            self.session = session or new_session_id()
            self._ids = itertools.count(1)
            self._history = deque(maxlen=self._max_history)
        return self.session

    def record_to(self, path):
        """Append every stored (non-transient) event to a JSONL file."""
        with self._lock:
            if self._record_file:
                self._record_file.close()
            self._record_file = open(path, "a", encoding="utf-8") if path else None

    # -- emit -----------------------------------------------------------
    def emit(self, etype, **data):
        try:
            with self._lock:
                event = {"v": SCHEMA_VERSION, "id": next(self._ids), "ts": now_iso(),
                         "session": self.session, "type": etype, "data": data}
            problems = validate(event)
            if problems:
                if self.strict:
                    raise EventError("; ".join(problems))
                key = (etype, problems[0])
                if key not in self._warned:
                    self._warned.add(key)
                    log.warning("invalid dashboard event %s: %s", etype, "; ".join(problems))

            with self._lock:
                if etype not in TRANSIENT:
                    self._history.append(event)
                    if self._record_file:
                        self._record_file.write(json.dumps(event) + "\n")
                        self._record_file.flush()
                subscribers = list(self._subscribers)

            for q in subscribers:
                try:
                    q.put_nowait(event)
                except Exception:  # noqa: BLE001 - queue.Full or a broken subscriber
                    # A tab that stopped reading (or worse). Drop it; server.py
                    # sees q.overflowed, closes the stream, the browser reconnects.
                    q.overflowed = True
                    self.unsubscribe(q)
            return event
        except EventError:
            raise
        except Exception:  # noqa: BLE001 - the dashboard must never break Atlas
            log.exception("dashboard emit failed for %s", etype)
            return None

    # -- subscribe ------------------------------------------------------
    def subscribe(self):
        """Return (snapshot_events, queue). Taken under one lock, so every
        event lands in exactly one of the two: none lost, none duplicated."""
        q = queue.Queue(maxsize=self._queue_size)
        q.overflowed = False
        with self._lock:
            snapshot = list(self._history)
            self._subscribers.add(q)
        return snapshot, q

    def unsubscribe(self, q):
        with self._lock:
            self._subscribers.discard(q)

    def snapshot(self):
        with self._lock:
            return list(self._history)

    @property
    def subscriber_count(self):
        with self._lock:
            return len(self._subscribers)


_bus = EventBus()


def emit(etype, **data):
    """Emit an event on the process-wide bus. Safe to call from anywhere."""
    return _bus.emit(etype, **data)


def get_bus():
    return _bus


def set_bus(bus):
    """Swap the process-wide bus (tests, demo). Returns the previous one."""
    global _bus
    previous, _bus = _bus, bus
    return previous
