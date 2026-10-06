"""
Offline tests for the dashboard wiring in voice_loop.py.

Loads voice_loop.py with the same hardware/network stubs as
test_voice_features.py, drives main() with a fake wake model (fires on cue,
then raises KeyboardInterrupt like Ctrl+C), and checks the events it emits.
No microphone, no `say`, no network beyond 127.0.0.1, no Anthropic API.

Run:  python3 test_voice_dashboard.py
Exit code is non-zero if any check fails.
"""

import contextlib
import io
import os
import sys
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import test_voice_features as tvf  # noqa: E402  (stubs + loader; runs nothing on import)
from dashboard import events  # noqa: E402
from dashboard.events import EventBus, validate  # noqa: E402
from dashboard.server import DashboardServer, start_dashboard  # noqa: E402

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
# harness
# --------------------------------------------------------------------------
WAKE = {"hey_jarvis": 0.91}
TEMP_WAV = tvf.SCRATCH / "dash_turn.wav"


class RecordingBus(EventBus):
    """Live-mode bus (never raises, like the real one) that keeps every event
    and any schema problems, so a bad event can't hide inside handle_turn's
    error handling."""

    def __init__(self):
        super().__init__(strict=False)
        self.all, self.problems = [], []

    def emit(self, etype, **data):
        ev = super().emit(etype, **data)
        self.all.append(ev)
        self.problems += validate(ev)
        return ev


class FakeWakeModel:
    """Returns the scripted predictions, then raises KeyboardInterrupt (Ctrl+C)."""

    def __init__(self, predictions):
        self.predictions = list(predictions)

    def predict(self, _chunk):
        if not self.predictions:
            raise KeyboardInterrupt
        return self.predictions.pop(0)


def fake_record(speech=True):
    def record(filename="temp_recording.wav"):
        TEMP_WAV.write_bytes(b"RIFF fake")
        return str(TEMP_WAV), speech
    return record


def run_main(mod, predictions, *, speech=True, heard="What's the capital of Australia?",
             client=None, speak=None):
    """Run voice_loop.main() once. Returns (bus, raised, wake_stream, stdout)."""
    bus = RecordingBus()
    events.set_bus(bus)
    stream = tvf._FakeStream(b"")
    mod.audio.open = lambda *a, **k: stream
    mod.wake_model = FakeWakeModel(predictions)
    mod.record_until_silence = fake_record(speech)
    mod.transcribe = lambda _f: heard
    mod.speak = speak or (lambda _t: None)          # never run the real `say`
    mod.client = client or tvf._FakeClient()
    mod.conversation_history.clear()
    out = io.StringIO()
    try:
        with contextlib.redirect_stdout(out):
            mod.main()
        raised = None
    except BaseException as e:  # noqa: BLE001 - KeyboardInterrupt escaping is a failure
        raised = e
    return bus, raised, stream, out.getvalue()


def timeline(bus):
    """Events without mic levels, as (type, state-or-detail) pairs."""
    out = []
    for e in bus.all:
        if e["type"] == "audio.level":
            continue
        d = e["data"]
        tag = d.get("state") or d.get("component") or d.get("reason") or d.get("stage") or ""
        out.append((e["type"], tag))
    return out


STARTUP = [("session.started", ""), ("health", "wake"), ("health", "stt"),
           ("health", "llm"), ("health", "mic"), ("state", "idle")]


# --------------------------------------------------------------------------
# tests
# --------------------------------------------------------------------------
def test_turns(mod):
    section("Turns emit the right events")

    bus, raised, stream, _ = run_main(mod, [WAKE])
    expected = STARTUP + [
        ("wake.detected", ""), ("state", "wake"), ("state", "recording"),
        ("recording.finished", ""), ("state", "transcribing"), ("health", "stt"),
        ("turn.user", ""), ("state", "thinking"), ("turn.assistant", ""), ("health", "llm"),
        ("state", "speaking"), ("state", "idle"), ("session.ended", "keyboard_interrupt"),
    ]
    got = timeline(bus)
    check("D1: a normal turn walks the pipeline and every event matches the schema",
          raised is None and got == expected and not bus.problems
          and any(e["type"] == "audio.level" for e in bus.all),
          f"raised={raised!r} problems={bus.problems}\n         got={got}")
    check("D2: the turn's temp recording is deleted and history has the exchange",
          not TEMP_WAV.exists() and [m["role"] for m in mod.conversation_history] == ["user", "assistant"],
          f"temp exists={TEMP_WAV.exists()} history={mod.conversation_history}")

    bus, raised, _, _ = run_main(mod, [WAKE], speech=False)
    got = timeline(bus)
    check("D3: wake word with no speech -> turn.discarded(no_speech), back to idle",
          raised is None and ("turn.discarded", "no_speech") in got
          and ("state", "transcribing") not in got and got[-2] == ("state", "idle")
          and not bus.problems, f"got={got[len(STARTUP):]}")

    client = tvf._FakeClient()
    bus, raised, _, _ = run_main(mod, [WAKE], heard="[BLANK_AUDIO]", client=client)
    got = timeline(bus)
    check("D4: background noise -> turn.discarded(noise), Claude is not called",
          raised is None and ("turn.discarded", "noise") in got
          and ("turn.user", "") not in got and client.messages.calls == []
          and not bus.problems, f"got={got[len(STARTUP):]} calls={len(client.messages.calls)}")


def test_errors(mod):
    section("Errors are reported and Atlas keeps listening")

    class Overloaded(Exception):
        status_code = 529

    client = tvf._FakeClient()
    client.messages.raise_once = Overloaded("Error code: 529 - overloaded")
    bus, raised, _, out = run_main(mod, [WAKE, {}, WAKE], client=client)
    got = timeline(bus)
    errors = [e["data"] for e in bus.all if e["type"] == "error"]
    llm_health = [e["data"]["status"] for e in bus.all
                  if e["type"] == "health" and e["data"]["component"] == "llm"]
    first_turn = got[got.index(("state", "thinking")):]
    check("D5: a Claude failure emits error(llm) + failing health, then error -> idle",
          raised is None and errors == [{"turn": 1, "stage": "llm",
                                         "message": "overloaded (error 529)", "recoverable": True}]
          and first_turn[:4] == [("state", "thinking"), ("error", "llm"),
                                 ("health", "llm"), ("state", "error")]
          and first_turn[4] == ("state", "idle")
          and "Something went wrong while asking Claude: overloaded (error 529)" in out,
          f"raised={raised!r} errors={errors}\n         after thinking={first_turn[:5]}")
    check("D6: the next wake still works, health recovers, history stays clean",
          [e["data"]["turn"] for e in bus.all if e["type"] == "turn.assistant"] == [2]
          and llm_health == ["idle", "failing", "ok"]
          and [m["role"] for m in mod.conversation_history] == ["user", "assistant"]
          and not bus.problems,
          f"llm health={llm_health} history roles={[m['role'] for m in mod.conversation_history]}")


def test_ctrl_c(mod):
    section("Ctrl+C exits cleanly")

    bus, raised, stream, out = run_main(mod, [])
    check("D7: Ctrl+C while waiting for the wake word -> clean exit, session.ended, stream closed",
          raised is None and timeline(bus)[-1] == ("session.ended", "keyboard_interrupt")
          and stream.closed and "Stopping..." in out,
          f"raised={raised!r} closed={stream.closed} last={timeline(bus)[-1]}")

    def interrupted_speech(_text):
        raise KeyboardInterrupt

    bus, raised, stream, out = run_main(mod, [WAKE, WAKE], speak=interrupted_speech)
    got = timeline(bus)
    check("D8: Ctrl+C while speaking (inside handle_turn) -> clean exit, session.ended, cleanup",
          raised is None and got[-2:] == [("state", "speaking"), ("session.ended", "keyboard_interrupt")]
          and stream.closed and not TEMP_WAV.exists()
          and "Listening for wake word again" not in out
          and sum(1 for t in got if t == ("wake.detected", "")) == 1,
          f"raised={raised!r} closed={stream.closed} temp={TEMP_WAV.exists()} tail={got[-3:]}")


def test_describe_error(mod):
    section("describe_error never leaks secrets")
    os.environ["ANTHROPIC_API_KEY"] = "plainkey-0042"   # restored below
    secrets = ["sk-ant-api03-Zx9VqLmN8pQr", "abc.def.ghi", "hunter2", "swordfish",
               "s3ss10n", "plainkey-0042", "f" * 40, "SECRETVALUE"]

    class APIStatusError(Exception):
        def __init__(self):
            super().__init__("Error code: 401 x-api-key: sk-ant-api03-Zx9VqLmN8pQr")
            self.status_code = 401
            self.request = SimpleNamespace(headers={"x-api-key": "sk-ant-api03-Zx9VqLmN8pQr"})
            self.response = SimpleNamespace(headers={"authorization": "Bearer abc.def.ghi"})

    class ReadTimeout(Exception):
        pass

    cases = [
        APIStatusError(),
        Exception("headers={'x-api-key': 'sk-ant-api03-Zx9VqLmN8pQr', "
                  "'authorization': 'Bearer abc.def.ghi'}"),
        Exception("Authorization: Bearer abc.def.ghi"),
        Exception("token=hunter2 password: swordfish cookie=s3ss10n"),
        Exception("bad key plainkey-0042 rejected"),
        Exception("upstream said " + "f" * 40),
        Exception('{"api_key": "SECRETVALUE"}'),
        ReadTimeout("GET https://api.anthropic.com x-api-key=SECRETVALUE"),
    ]
    outputs = [mod.describe_error(e) for e in cases]
    leaks = [(o, s) for o in outputs for s in secrets if s in o]
    check("D9: no API keys, headers, tokens or passwords in any output",
          not leaks and all(len(o) <= 140 for o in outputs),
          f"leaks={leaks}\n         outputs={outputs}")
    check("D10: messages stay useful (status words, timeouts, plain errors kept)",
          outputs[0] == "API key rejected (error 401)"
          and outputs[-1] == "timed out waiting for a response"
          and mod.describe_error(OSError("Input device unavailable")) == "OSError: Input device unavailable",
          f"outputs[0]={outputs[0]!r} outputs[-1]={outputs[-1]!r}")

    client = tvf._FakeClient()
    client.messages.raise_once = RuntimeError("401 from proxy, sent x-api-key: sk-ant-api03-Zx9VqLmN8pQr")
    bus, raised, _, out = run_main(mod, [WAKE], client=client)
    shown = [e["data"]["message"] for e in bus.all if e["type"] == "error"] + \
            [e["data"].get("error", "") for e in bus.all if e["type"] == "health"]
    check("D11: end to end, the dashboard error and terminal output are redacted too",
          raised is None and shown and not any("sk-ant" in s for s in shown)
          and "sk-ant" not in out,
          f"shown={shown}")
    del os.environ["ANTHROPIC_API_KEY"]


def test_entry_point(mod):
    section("Dashboard problems never stop Atlas")
    calls = []
    real_main, real_start = mod.main, mod.start_dashboard
    mod.main = lambda: calls.append("main")
    try:
        taken = DashboardServer(port=0).start()
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            result = start_dashboard(port=taken.port)
            mod.run(["--port", str(taken.port)])
        taken.stop()
        check("D12: port already in use -> warning, no dashboard, Atlas still runs",
              result is None and calls == ["main"] and "Dashboard not started" in out.getvalue(),
              f"start_dashboard={result!r} main calls={calls}")

        def explode(**_):
            raise RuntimeError("unexpected dashboard bug")

        mod.start_dashboard = explode
        calls.clear()
        with contextlib.redirect_stdout(io.StringIO()):
            mod.run([])
        check("D13: any other dashboard failure is caught and Atlas still runs",
              calls == ["main"], f"main calls={calls}")

        mod.start_dashboard = lambda **_: None
        bus = EventBus()
        events.set_bus(bus)
        sessions_existed = (HERE / "sessions").exists()
        before = set((HERE / "sessions").glob("*.jsonl")) if sessions_existed else set()
        with contextlib.redirect_stdout(io.StringIO()):
            mod.run(["--record", "--no-dashboard"])
        new = set((HERE / "sessions").glob("*.jsonl")) - before
        for f in new:
            f.unlink()
        if not sessions_existed and not any((HERE / "sessions").iterdir()):
            (HERE / "sessions").rmdir()
        ignore = (HERE / ".gitignore").read_text().split()
        check("D14: --record writes to sessions/, and sessions/ is gitignored",
              len(new) == 1 and next(iter(new)).name == f"{bus.session}.jsonl"
              and "sessions/" in ignore,
              f"new files={[f.name for f in new]} gitignored={'sessions/' in ignore}")
    finally:
        mod.main, mod.start_dashboard = real_main, real_start


# --------------------------------------------------------------------------
def main():
    tvf._install_stubs()
    mod = tvf._load_target()
    print("loaded voice_loop.py with stubs")
    real_bus = events.get_bus()
    try:
        test_turns(mod)
        test_errors(mod)
        test_ctrl_c(mod)
        test_describe_error(mod)
        test_entry_point(mod)
    finally:
        events.set_bus(real_bus)
        TEMP_WAV.unlink(missing_ok=True)

    section("summary")
    passed = sum(1 for _, ok, _ in _results if ok)
    failed = [n for n, ok, _ in _results if not ok]
    print(f"  {passed}/{len(_results)} checks passed")
    for n in failed:
        print(f"  FAILED: {n}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
