"""Single entry point — starts the gesture lab server and opens the web UI."""

import http.server
import os
import socketserver
import subprocess
import sys
import threading
import time
import webbrowser

ROOT = os.path.dirname(os.path.abspath(__file__))
PORT = 8000
URL = f"http://localhost:{PORT}/demo_ui.html"


def run_http_server():
    os.chdir(ROOT)
    handler = http.server.SimpleHTTPRequestHandler
    with socketserver.TCPServer(("", PORT), handler) as httpd:
        print(f"Serving UI at {URL}")
        httpd.serve_forever()


def main():
    os.chdir(ROOT)

    server_proc = subprocess.Popen(
        [sys.executable, "demo_server.py"],
        cwd=ROOT,
    )

    http_thread = threading.Thread(target=run_http_server, daemon=True)
    http_thread.start()

    time.sleep(1.5)
    print(f"\nOpening browser → {URL}")
    print("Everything runs in the browser UI. Press Ctrl+C to stop.\n")
    webbrowser.open(URL)

    try:
        server_proc.wait()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server_proc.terminate()
        server_proc.wait(timeout=5)


if __name__ == "__main__":
    main()
