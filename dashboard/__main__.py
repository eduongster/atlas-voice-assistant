"""python3 -m dashboard --demo [FILE] [--loop] [--speed N] [--port N] [--open]"""

import argparse
import sys
import threading

from . import demo
from .server import DEFAULT_PORT, start_dashboard


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python3 -m dashboard",
        description="Run the Atlas dashboard on its own, driven by demo events "
                    "(no microphone, models or API key needed).")
    parser.add_argument("--demo", nargs="?", const="script", metavar="FILE",
                        help="play the scripted demo, or replay a sessions/*.jsonl "
                             "file saved with `python3 voice_loop.py --record`")
    parser.add_argument("--loop", action="store_true", help="repeat the demo forever")
    parser.add_argument("--speed", type=float, default=1.0, help="playback speed (default 1)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--open", action="store_true", help="open the dashboard in a browser")
    args = parser.parse_args(argv)

    if not args.demo:
        parser.print_help()
        print("\nTo run Atlas itself (with the dashboard): python3 voice_loop.py")
        return 2

    make_steps = demo.scenario if args.demo == "script" else (lambda: demo.load_recording(args.demo))
    bus = demo.default_bus()
    server = start_dashboard(bus, port=args.port, open_browser=args.open)
    if server is None:
        return 1

    stop = threading.Event()
    try:
        print("Playing demo. Press Ctrl+C to stop.")
        demo.play(bus, make_steps, speed=args.speed, loop=args.loop, stop=stop)
        print("Demo finished. The page stays up; press Ctrl+C to quit.")
        while not stop.wait(1):
            pass
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        stop.set()
        server.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
