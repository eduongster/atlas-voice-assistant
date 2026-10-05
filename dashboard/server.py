"""
Local dashboard server (stdlib only).

    GET /               dashboard page (static/index.html)
    GET /static/<path>  frontend files, confined to dashboard/static/
    GET /events         Server-Sent Events: one `snapshot` message with the
                        session so far, then every live event as it happens
    GET /api/snapshot   the same snapshot as JSON (debugging, tests)

Binds to 127.0.0.1 only (transcripts are private) and rejects requests whose
Host header isn't localhost, so other websites can't read the stream through
DNS rebinding. Runs in daemon threads, so it never keeps Atlas alive on exit.
"""

import json
import mimetypes
import queue
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

from . import events

STATIC_DIR = (Path(__file__).resolve().parent / "static").resolve()
DEFAULT_PORT = 8765
HEARTBEAT_S = 15

CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".json": "application/json",
    ".svg": "image/svg+xml",
    ".woff2": "font/woff2",
    ".txt": "text/plain; charset=utf-8",
}


def snapshot_payload(bus, snapshot):
    return {"session": bus.session, "now": time.time() * 1000, "events": snapshot}


class _Handler(BaseHTTPRequestHandler):
    server_version = "AtlasDashboard/1"
    protocol_version = "HTTP/1.1"

    # quiet: voice_loop.py's terminal output is for the user, not access logs
    def log_message(self, *args):
        pass

    def do_GET(self):
        if not self._host_allowed():
            return self._send_text(403, "Forbidden host")
        path = unquote(urlsplit(self.path).path)
        if path == "/":
            return self._send_file(STATIC_DIR / "index.html")
        if path == "/events":
            return self._stream_events()
        if path == "/api/snapshot":
            bus = self.server.bus
            body = json.dumps(snapshot_payload(bus, bus.snapshot())).encode()
            return self._send(200, "application/json", body)
        if path.startswith("/static/"):
            target = (STATIC_DIR / path[len("/static/"):]).resolve()
            if target.is_relative_to(STATIC_DIR) and target.is_file():
                return self._send_file(target)
        return self._send_text(404, "Not found")

    # -- helpers --------------------------------------------------------
    def _host_allowed(self):
        host = (self.headers.get("Host") or "").rsplit(":", 1)[0].strip("[]")
        return host in ("127.0.0.1", "localhost", "::1")

    def _send(self, status, content_type, body):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_text(self, status, text):
        self._send(status, "text/plain; charset=utf-8", text.encode())

    def _send_file(self, path):
        ctype = CONTENT_TYPES.get(path.suffix) or mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self._send(200, ctype, path.read_bytes())

    def _stream_events(self):
        bus = self.server.bus
        snapshot, q = bus.subscribe()
        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            self._write_sse(snapshot_payload(bus, snapshot), name="snapshot")

            last_write = time.monotonic()
            while not self.server.stopping.is_set() and not q.overflowed:
                try:
                    event = q.get(timeout=1.0)
                except queue.Empty:
                    if time.monotonic() - last_write >= HEARTBEAT_S:
                        self.wfile.write(b": ping\n\n")
                        self.wfile.flush()
                        last_write = time.monotonic()
                    continue
                self._write_sse(event, event_id=event["id"])
                last_write = time.monotonic()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass  # tab closed
        finally:
            bus.unsubscribe(q)
            self.close_connection = True

    def _write_sse(self, payload, name=None, event_id=None):
        chunk = ""
        if name:
            chunk += f"event: {name}\n"
        if event_id is not None:
            chunk += f"id: {event_id}\n"
        chunk += f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"
        self.wfile.write(chunk.encode())
        self.wfile.flush()


class DashboardServer:
    """The HTTP server plus the thread running it."""

    def __init__(self, bus=None, host="127.0.0.1", port=DEFAULT_PORT):
        self._httpd = ThreadingHTTPServer((host, port), _Handler)
        self._httpd.daemon_threads = True
        self._httpd.bus = bus or events.get_bus()
        self._httpd.stopping = threading.Event()
        self._thread = None

    @property
    def port(self):
        return self._httpd.server_address[1]

    @property
    def url(self):
        return f"http://127.0.0.1:{self.port}"

    def start(self):
        self._thread = threading.Thread(target=self._httpd.serve_forever,
                                        name="atlas-dashboard", daemon=True)
        self._thread.start()
        return self

    def stop(self):
        if self._httpd.stopping.is_set():
            return
        self._httpd.stopping.set()
        self._httpd.shutdown()
        self._httpd.server_close()


def start_dashboard(bus=None, port=DEFAULT_PORT, open_browser=False):
    """Start the dashboard in a background thread and return the server, or
    None (with a printed warning) if it can't start. Atlas runs either way."""
    try:
        server = DashboardServer(bus=bus, port=port).start()
    except OSError as e:
        print(f"Dashboard not started ({e}). Atlas will run without it; "
              f"try --port to pick another port.")
        return None
    print(f"Dashboard: {server.url}")
    if open_browser:
        webbrowser.open(server.url)
    return server
