"""
Offline tests for the dashboard event layer (dashboard/).

No microphone, no network beyond 127.0.0.1, no Anthropic API, no model loads.
Same tiny runner style as test_voice_features.py.

Run:  python3 test_events.py
Exit code is non-zero if any check fails.
"""

import http.client
import json
import logging
import re
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from dashboard import events  # noqa: E402
from dashboard.events import EventBus, EventError, validate  # noqa: E402
from dashboard.server import DashboardServer  # noqa: E402

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


# One valid example per event type (and per known action kind). Mirrors the
# schema table in the plan and the docstring in dashboard/events.py.
EXAMPLES = [
    ("session.started", {"mode": "live", "wake_word": "hey_jarvis",
                         "model": "claude-sonnet-5", "stt": "whisper.cpp base"}),
    ("session.ended", {"reason": "keyboard_interrupt"}),
    ("state", {"state": "recording", "turn": 3}),
    ("wake.detected", {"turn": 3, "word": "hey_jarvis", "confidence": 0.91}),
    ("recording.finished", {"turn": 3, "duration_s": 2.4, "speech_detected": True}),
    ("turn.user", {"turn": 3, "text": "Set a timer for ten minutes", "stt_ms": 410}),
    ("turn.assistant", {"turn": 3, "text": "Timer set for ten minutes.",
                        "llm_ms": 1240, "model": "claude-sonnet-5"}),
    ("turn.discarded", {"turn": 4, "reason": "noise", "heard": "[BLANK_AUDIO]"}),
    ("error", {"turn": 5, "stage": "llm", "message": "APIStatusError 529 overloaded",
               "recoverable": True}),
    ("health", {"component": "llm", "status": "ok", "label": "claude-sonnet-5",
                "detail": "1.2 s", "latency_ms": 1180}),
    ("health", {"component": "llm", "status": "idle", "label": "claude-sonnet-5",
                "detail": "No calls yet"}),
    ("audio.level", {"rms": 0.42}),
    ("action", {"action_id": "act_1", "turn": 3, "kind": "calendar.create", "status": "done",
                "title": "Dentist", "payload": {"title": "Dentist",
                                                "start": "2026-10-08T15:00:00-07:00",
                                                "end": "2026-10-08T16:00:00-07:00"}}),
    ("action", {"action_id": "act_2", "turn": 3, "kind": "timer.start", "status": "pending",
                "title": "Pasta", "payload": {"label": "Pasta", "duration_s": 600,
                                              "ends_at": "2026-10-05T14:51:07-07:00"}}),
    ("action", {"action_id": "act_3", "kind": "reminder.create", "status": "triggered",
                "payload": {"text": "Call mom", "due_at": "2026-10-05T18:00:00-07:00"}}),
    ("action", {"action_id": "act_4", "kind": "calendar.list", "status": "done",
                "payload": {"range": "tomorrow", "events": []}}),
    ("action", {"action_id": "act_5", "kind": "music.play", "status": "failed",
                "title": "Play jazz", "error": "not supported yet"}),
]


def _envelope(etype, data):
    return {"v": 1, "id": 1, "ts": "2026-10-05T14:41:07.123-07:00",
            "session": "s_test", "type": etype, "data": data}


# --------------------------------------------------------------------------
# 1. envelope
# --------------------------------------------------------------------------
def test_envelope():
    section("1. Envelope")
    bus = EventBus(strict=True)
    a = bus.emit("state", state="idle")
    b = bus.emit("state", state="wake", turn=1)
    check("1a: envelope has v, id, ts, session, type, data",
          set(a) == {"v", "id", "ts", "session", "type", "data"} and a["v"] == 1,
          f"keys={sorted(a)}")
    check("1b: ids increase monotonically within a session",
          (a["id"], b["id"]) == (1, 2), f"ids={a['id']},{b['id']}")
    iso = re.compile(r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}[+-]\d\d:\d\d$")
    check("1c: ts is ISO 8601 with milliseconds and UTC offset",
          iso.match(a["ts"]) is not None, a["ts"])
    old = bus.session
    bus.reset()
    c = bus.emit("state", state="idle")
    check("1d: reset() starts a new session id with ids from 1 and empty history",
          c["session"] != old and c["id"] == 1 and bus.snapshot() == [c],
          f"{old} -> {c['session']}, id={c['id']}")


# --------------------------------------------------------------------------
# 2. validation
# --------------------------------------------------------------------------
def test_validation():
    section("2. Schema validation")
    bad_examples = [p for t, d in EXAMPLES if (p := validate(_envelope(t, d)))]
    check(f"2a: all {len(EXAMPLES)} documented example payloads validate",
          not bad_examples, f"problems={bad_examples}")

    covered = {t for t, _ in EXAMPLES}
    check("2b: examples cover every event type in SCHEMA",
          covered == set(events.SCHEMA), f"missing={set(events.SCHEMA) - covered}")

    rejects = {
        "unknown event type": ("state.changed", {"state": "idle"}),
        "unknown state": ("state", {"state": "napping"}),
        "missing action_id": ("action", {"kind": "timer.start", "status": "done",
                                         "payload": {"label": "x", "duration_s": 1,
                                                     "ends_at": "t"}}),
        "unknown action status": ("action", {"action_id": "a", "kind": "x", "status": "maybe"}),
        "timer payload missing ends_at": ("action", {"action_id": "a", "kind": "timer.start",
                                                     "status": "done",
                                                     "payload": {"label": "x", "duration_s": 1}}),
        "non-numeric rms": ("audio.level", {"rms": "loud"}),
        "rms out of range": ("audio.level", {"rms": 1.7}),
        "bool where number expected": ("wake.detected", {"turn": 1, "word": "w",
                                                         "confidence": True}),
        "unknown health status": ("health", {"component": "llm", "status": "green",
                                             "label": "x"}),
        "typo'd field name": ("turn.user", {"turn": 1, "txt": "hi"}),
    }
    accepted = [name for name, (t, d) in rejects.items() if not validate(_envelope(t, d))]
    check(f"2c: {len(rejects)} malformed events are all rejected",
          not accepted, f"wrongly accepted={accepted}")

    check("2d: latency_status marks slow STT (>2 s) and slow Claude (>4 s)",
          events.latency_status("stt", 2500) == "slow"
          and events.latency_status("stt", 400) == "ok"
          and events.latency_status("llm", 4100) == "slow"
          and events.latency_status("llm", 3900) == "ok")


# --------------------------------------------------------------------------
# 3. emit never raises / never blocks (live mode)
# --------------------------------------------------------------------------
def test_emit_safety():
    section("3. emit() is safe in live mode")
    logging.getLogger("atlas.events").setLevel(logging.CRITICAL)   # keep output clean

    bus = EventBus(strict=False)
    try:
        ev = bus.emit("state", state="napping")
        raised = None
    except Exception as e:  # noqa: BLE001
        ev, raised = None, e
    check("3a: invalid event in live mode logs instead of raising (and is still delivered)",
          raised is None and ev is not None and ev["data"]["state"] == "napping",
          f"raised={raised!r}")

    strict = EventBus(strict=True)
    try:
        strict.emit("state", state="napping")
        strict_raised = False
    except EventError:
        strict_raised = True
    check("3b: strict mode (tests, demo) raises EventError on an invalid event", strict_raised)

    class _Exploding:
        overflowed = False

        def put_nowait(self, _):
            raise RuntimeError("subscriber bug")

    bus = EventBus()
    bad = _Exploding()
    bus._subscribers.add(bad)
    _, healthy = bus.subscribe()
    try:
        result = bus.emit("state", state="idle")
        raised = None
    except Exception as e:  # noqa: BLE001
        result, raised = None, e
    check("3c: a broken subscriber can't make emit() raise or starve the others",
          raised is None and result is not None and healthy.qsize() == 1
          and bad not in bus._subscribers,
          f"raised={raised!r}, returned={result is not None}, healthy got {healthy.qsize()}")

    bus = EventBus(queue_size=3)
    _, q = bus.subscribe()
    for i in range(10):
        bus.emit("audio.level", rms=0.1)
    check("3d: a subscriber that stops reading is dropped, not waited on",
          q.overflowed and bus.subscriber_count == 0,
          f"overflowed={q.overflowed}, subscribers={bus.subscriber_count}")


# --------------------------------------------------------------------------
# 4. bus: fan-out, transient events, bounded history, atomic subscribe
# --------------------------------------------------------------------------
def test_bus():
    section("4. Event bus")
    bus = EventBus(strict=True)
    bus.emit("state", state="idle")
    snap1, q1 = bus.subscribe()
    snap2, q2 = bus.subscribe()
    live = bus.emit("state", state="wake", turn=1)
    check("4a: every subscriber gets the snapshot so far plus each live event",
          len(snap1) == len(snap2) == 1
          and q1.get_nowait() == live and q2.get_nowait() == live)

    bus.emit("audio.level", rms=0.3)
    got = q1.get_nowait()
    check("4b: audio.level reaches live subscribers but is not stored for replay",
          got["type"] == "audio.level"
          and all(e["type"] != "audio.level" for e in bus.snapshot()))

    small = EventBus(max_history=5)
    for i in range(12):
        small.emit("state", state="idle")
    snap = small.snapshot()
    check("4c: history is bounded (oldest events dropped first)",
          len(snap) == 5 and snap[0]["id"] == 8 and snap[-1]["id"] == 12,
          f"len={len(snap)} ids={[e['id'] for e in snap]}")

    # Emit from a thread while subscribing: every event must appear exactly
    # once across (snapshot + queue), never twice and never missing.
    bus = EventBus(max_history=10**6)
    stop = threading.Event()

    def spam():
        for _ in range(200_000):
            if stop.is_set():
                break
            bus.emit("state", state="idle")

    t = threading.Thread(target=spam)
    t.start()
    snap, q = bus.subscribe()
    stop.set()
    t.join()
    queued = []
    while not q.empty():
        queued.append(q.get_nowait())
    ids = [e["id"] for e in snap] + [e["id"] for e in queued]
    all_ids = [e["id"] for e in bus.snapshot()]
    check("4d: subscribing while another thread emits loses and duplicates nothing",
          ids == all_ids, f"{len(snap)} in snapshot + {len(queued)} queued vs {len(all_ids)} total")

    path = HERE / "tests" / "_record_test.jsonl"
    path.unlink(missing_ok=True)
    bus = EventBus()
    bus.record_to(path)
    bus.emit("state", state="idle")
    bus.emit("audio.level", rms=0.2)
    bus.emit("state", state="wake", turn=1)
    bus.record_to(None)
    lines = path.read_text().splitlines()
    path.unlink()
    check("4e: record_to() writes stored events as JSONL (transient ones skipped)",
          len(lines) == 2 and '"wake"' in lines[1], f"{len(lines)} lines")


# --------------------------------------------------------------------------
# 6. server
# --------------------------------------------------------------------------
def _get(port, path, host="127.0.0.1"):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    conn.putrequest("GET", path, skip_host=True)
    conn.putheader("Host", f"{host}:{port}")
    conn.endheaders()
    resp = conn.getresponse()
    body = resp.read()
    conn.close()
    return resp.status, resp.getheader("Content-Type", ""), body


def _read_sse_message(resp):
    """Read one SSE message (skipping heartbeat comments); returns (name, id, data)."""
    name, event_id, data = "message", None, None
    while True:
        line = resp.fp.readline().decode().rstrip("\n")
        if line == "":
            if data is not None:
                return name, event_id, json.loads(data)
            continue
        if line.startswith(":"):
            continue
        field, _, value = line.partition(": ")
        if field == "event":
            name = value
        elif field == "id":
            event_id = int(value)
        elif field == "data":
            data = value


def test_server():
    section("6. Dashboard server")
    bus = EventBus(strict=True)
    server = DashboardServer(bus=bus, port=0).start()
    port = server.port
    try:
        bus.emit("session.started", mode="demo")
        bus.emit("state", state="idle")

        status, ctype, body = _get(port, "/")
        check("6a: / serves the dashboard HTML",
              status == 200 and ctype.startswith("text/html") and b"<title>" in body,
              f"status={status} type={ctype}")

        attempts = ["/static/../events.py", "/static/%2e%2e/server.py",
                    "/static/..%2f..%2fvoice_loop.py", "/static//etc/passwd"]
        leaks = [(p, s) for p in attempts if (s := _get(port, p)[0]) != 404]
        check("6b: path traversal out of dashboard/static is refused (404)",
              not leaks, f"leaked={leaks}")

        status, _, _ = _get(port, "/", host="evil.example")
        check("6c: requests with a non-localhost Host header are refused (DNS rebinding)",
              status == 403, f"status={status}")

        status, _, body = _get(port, "/api/snapshot")
        snap = json.loads(body)
        check("6d: /api/snapshot returns the session's events",
              status == 200 and snap["session"] == bus.session and len(snap["events"]) == 2)

        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request("GET", "/events")
        resp = conn.getresponse()
        name, _, first = _read_sse_message(resp)
        check("6e: /events opens with a snapshot of everything so far",
              resp.status == 200
              and resp.getheader("Content-Type", "").startswith("text/event-stream")
              and name == "snapshot"
              and [e["type"] for e in first["events"]] == ["session.started", "state"]
              and isinstance(first["now"], float),
              f"name={name} types={[e['type'] for e in first['events']]}")

        live = bus.emit("state", state="wake", turn=1)
        name, event_id, got = _read_sse_message(resp)
        check("6f: live events stream after the snapshot, with SSE ids",
              name == "message" and event_id == live["id"] and got == live,
              f"name={name} id={event_id}")

        t0 = time.monotonic()
        server.stop()
        stopped_in = time.monotonic() - t0
        conn.close()
        check("6g: server stops promptly even with a stream open",
              stopped_in < 3, f"stopped in {stopped_in:.2f}s")
    finally:
        server.stop()   # idempotent


# --------------------------------------------------------------------------
def main():
    test_envelope()
    test_validation()
    test_emit_safety()
    test_bus()
    test_server()

    section("summary")
    passed = sum(1 for _, ok, _ in _results if ok)
    failed = [n for n, ok, _ in _results if not ok]
    print(f"  {passed}/{len(_results)} checks passed")
    for n in failed:
        print(f"  FAILED: {n}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
