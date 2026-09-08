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

## Project status

Early stage. Wake word detection is working. Everything else below it in the feature list is actively in progress. Check the commit history for the latest.

## License

MIT — see [LICENSE](LICENSE) for details.
