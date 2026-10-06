# Atlas Voice Assistant

A voice-activated personal assistant that listens, talks back, and actually helps with day-to-day stuff. Say a wake word, ask it something, get an answer out loud. No typing required.

Built as a hands-on project to learn how voice pipelines, LLM orchestration, and real-world API integrations fit together.

## What it does (or will do)

- [x] **Wake word activation** — listens for a wake phrase using [openWakeWord](https://github.com/dscripka/openWakeWord), no button press needed
- [ ] **Voice in, voice out** — speech-to-text via Whisper, text-to-speech for the response, so the whole interaction stays hands-free
- [ ] **Google Calendar access** — ask what's on your schedule, add events, get a heads up on what's coming
- [ ] **Timers and reminders** — set them by voice, get notified when they're up
- [ ] **General Q&A** — quick answers to everyday questions, so you don't have to break focus to search something up
- [ ] **A real personality** — not a flat robotic assistant, has a defined tone and way of responding
- [ ] **Persistent memory** — remembers things across sessions, not just within one conversation

## Why this project

Most voice assistant tutorials wire together a single API call and call it done. This one is different: it routes requests intelligently, remembers context over time, and connects to real services I actually use. The goal was to build something with a genuine architecture behind it, not just a demo.

## Tech stack

- **Wake word detection:** openWakeWord
- **Speech-to-text:** OpenAI Whisper
- **Text-to-speech:** TBD
- **LLM orchestration:** custom routing logic (in progress)
- **Memory:** SQLite
- **Calendar integration:** Google Calendar API

## Getting started

### Requirements
- Python 3.10+
- A working microphone
- macOS, Linux, or Windows (tested on macOS)

### Setup

```bash
# Clone the repo
git clone https://github.com/eduongster/atlas-voice-assistant.git
cd atlas-voice-assistant

# Install system dependency for audio
brew install portaudio   # macOS
# sudo apt install portaudio19-dev   # Linux

# Install Python packages
pip3 install -r requirements.txt
```

Install from `requirements.txt` rather than adding packages one at a time — it has pinned versions and a note on why `setuptools` is held below 80 (needed for the `webrtcvad` build on Python 3.12+).

### Run the wake word test

```bash
python3 wake_word_test.py
```

Say your wake phrase and watch your terminal. First run downloads the model files, so give it a few seconds.

### Run Atlas

Put your Anthropic API key in a `.env` file next to `voice_loop.py` (`ANTHROPIC_API_KEY=...`), then:

```bash
python3 voice_loop.py --open
```

Say "Hey Jarvis", ask a question, and Atlas answers out loud. The live dashboard opens at http://127.0.0.1:8765. Press Ctrl+C to stop.

| Flag | What it does |
|---|---|
| `--open` | Open the dashboard in your browser |
| `--port N` | Serve the dashboard on another port (default 8765) |
| `--no-dashboard` | Run without the dashboard |
| `--record` | Also save this session's events to `sessions/<id>.jsonl` for replay (see below) |

## Live dashboard

The dashboard shows what Atlas is doing as it happens:

- **Status:** a sphere that changes with each state (listening for the wake word, heard you, recording, transcribing, thinking, speaking, something went wrong), with a plain-language line underneath.
- **Conversation:** every turn with timestamps and how long transcription and Claude took. It survives a page reload for the whole session.
- **Actions:** cards for calendar events, timers and reminders, shown under the turn that created them. Running timers also show under the sphere.
- **Model status:** wake word, speech to text, Claude and microphone, each with a dot (green ok, amber slow, red failing, grey idle) and a live mic level. Claude's status comes from its last real call, never a ping.

It's plain HTML/CSS/JS served by Python's standard library: no build step and no extra dependencies. It only listens on `127.0.0.1`. If it can't start (for example the port is taken), Atlas prints a warning and runs without it.

### Demo mode

Run the dashboard with scripted events. You don't need a microphone, models or API key, so it's useful for UI work or recording a video:

```bash
python3 -m dashboard --demo --open          # ~90 s scripted session
python3 -m dashboard --demo --loop          # repeat forever
python3 -m dashboard --demo --speed 2       # twice as fast
```

The script covers every state, a timer that rings, a calendar event, a reminder, a wake with no speech, a Claude failure and a slow answer.

### Recording and replaying a real session

```bash
python3 voice_loop.py --record                          # saves sessions/<id>.jsonl
python3 -m dashboard --demo sessions/<id>.jsonl         # replays it with the same timing
```

Recordings contain your transcripts. `sessions/` is in `.gitignore`, so they're never committed.

### How it works

```
voice_loop.py --emit()--> dashboard/events.py --queue per tab--> dashboard/server.py --SSE--> browser
                          (history + fan-out)                    (snapshot, then live)     (static/app.js)
```

- `emit(type, **data)` never raises and never blocks the voice loop.
- Each browser tab first gets a snapshot of the session so far, then live events. A reload rebuilds the same view.
- The full event schema (every type, its fields and example payloads) is documented at the top of [`dashboard/events.py`](dashboard/events.py).

### Adding a feature that does something (calendar, timers, ...)

Actions use one generic event, so new features plug in without changing the dashboard's plumbing:

```python
emit("action", action_id="act_7", turn=turn, kind="timer.start", status="pending",
     title="Pasta", payload={"label": "Pasta", "duration_s": 600, "ends_at": "2026-10-05T18:10:00-07:00"})
# ...when it succeeds:
emit("action", action_id="act_7", turn=turn, kind="timer.start", status="done", ...)
# ...later, when the timer goes off:
emit("action", action_id="act_7", turn=turn, kind="timer.start", status="triggered", ...)
```

Statuses are `pending`, `done`, `failed`, `cancelled` and `triggered`. Cards for `calendar.create`, `timer.start`, `reminder.create` and `calendar.list` live in [`dashboard/static/cards.js`](dashboard/static/cards.js). Any other `kind` gets a generic card until you add a renderer there.

## Tests

All offline: no microphone, network, models or API calls.

```bash
python3 test_voice_features.py    # conversation memory and voice activity detection
python3 test_events.py            # event schema, event bus, demo script, dashboard server
python3 test_voice_dashboard.py   # voice_loop.py's dashboard events, error handling, Ctrl+C, secret redaction
```

The VAD tests use `tests/fixtures/speech_sample.wav`, a clip generated with macOS `say`. The exact command is in a comment in `test_voice_features.py`.

## Project status

Early stage. Wake word detection is working. Everything else below it in the feature list is actively in progress. Check the commit history for the latest.

## License

MIT — see [LICENSE](LICENSE) for details.
