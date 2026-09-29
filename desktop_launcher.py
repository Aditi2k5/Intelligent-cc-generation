from __future__ import annotations
import os, threading, time, webbrowser
import uvicorn

def open_browser():
    time.sleep(2)
    webbrowser.open("http://127.0.0.1:8000")

if __name__ == "__main__":
    os.environ.setdefault("PLANETREAD_FRONTEND_URL", "http://127.0.0.1:8000")
    threading.Thread(target=open_browser, daemon=True).start()
    uvicorn.run("backend.app:app", host="127.0.0.1", port=8000, log_level="info")
