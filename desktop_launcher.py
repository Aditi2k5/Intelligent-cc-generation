"""Entry point for the packaged PlanetRead app (PyInstaller).

Double-clicking PlanetRead.exe runs this: it loads the .env next to the exe,
starts the FastAPI backend (which also serves the built frontend), waits until
the server actually answers, and then opens the UI in the default browser.

`PlanetRead.exe --self-test` imports the whole pipeline and checks bundled
assets without starting the server; `--self-test --models` also loads every
model (PANNs, Music2Emo, Silero, Florence-2, ...). CI runs both so a broken
build fails there instead of on the recipient's machine.
"""
from __future__ import annotations

import json
import multiprocessing
import os
import socket
import subprocess
import sys
import threading
import time
import traceback
import urllib.request
import webbrowser
from pathlib import Path

HOST = "127.0.0.1"
PREFERRED_PORT = 8000

FROZEN = getattr(sys, "frozen", False)
# Folder the user sees (where PlanetRead.exe lives) vs. folder holding bundled code/data.
APP_DIR = Path(sys.executable).resolve().parent if FROZEN else Path(__file__).resolve().parent
BUNDLE_DIR = Path(getattr(sys, "_MEIPASS", APP_DIR))


def configure_environment() -> None:
    from dotenv import load_dotenv

    load_dotenv(APP_DIR / ".env")
    data_dir = Path(os.environ.get("PLANETREAD_DATA_DIR", "PlanetRead_data"))
    if not data_dir.is_absolute():
        data_dir = APP_DIR / data_dir
    try:
        data_dir.mkdir(parents=True, exist_ok=True)
        probe = data_dir / ".write-test"
        probe.write_text("ok")
        probe.unlink()
    except OSError:
        # e.g. the exe was put under Program Files; fall back to the user profile.
        data_dir = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "PlanetRead"
        data_dir.mkdir(parents=True, exist_ok=True)
    os.environ["PLANETREAD_DATA_DIR"] = str(data_dir)
    if not os.environ.get("OPENAI_API_KEY"):
        print(f"WARNING: OPENAI_API_KEY is not set. Copy .env.example to .env in\n  {APP_DIR}\n"
              "and add your keys, or video processing will fail.\n", flush=True)
    os.environ.setdefault("MPLCONFIGDIR", str(data_dir / ".matplotlib"))


def is_planetread(url: str) -> bool:
    try:
        with urllib.request.urlopen(f"{url}/api/health", timeout=2) as resp:
            return json.load(resp).get("status") == "ok"
    except Exception:
        return False


def port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((HOST, port))
            return True
        except OSError:
            return False


def pick_port() -> int:
    if port_free(PREFERRED_PORT):
        return PREFERRED_PORT
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, 0))
        return s.getsockname()[1]


def open_browser(url: str) -> None:
    try:
        if webbrowser.open(url):
            return
    except Exception:
        pass
    if sys.platform == "win32":
        try:
            os.startfile(url)  # type: ignore[attr-defined]
            return
        except OSError:
            subprocess.run(["cmd", "/c", "start", "", url], check=False)
    elif sys.platform == "darwin":
        subprocess.run(["open", url], check=False)
    else:
        subprocess.run(["xdg-open", url], check=False)


def open_when_ready(url: str, server) -> None:
    deadline = time.monotonic() + 300
    while time.monotonic() < deadline and not server.should_exit:
        if server.started and is_planetread(url):
            print(f"\nPlanetRead is running at {url}")
            print("Keep this window open while you use the app; close it to quit.\n", flush=True)
            open_browser(url)
            return
        time.sleep(0.5)
    print(f"\nThe server did not come up in time. Try opening {url} manually.", flush=True)


def model_checks(check) -> None:
    """Load every model the pipeline uses and run the audio ones on a test
    tone. Downloads the weights on first use (several GB)."""
    import numpy as np
    import newone

    sr = newone.SAMPLE_RATE
    t = np.arange(6 * sr) / sr
    tone = (0.3 * np.sin(2 * np.pi * 440 * t) * (1 + np.sin(2 * np.pi * 2 * t)) / 2).astype(np.float32)

    def panns():
        model = newone.get_panns()
        clipwise, _ = model.inference(tone[None, :])
        return f"top label: {model.labels[int(np.argmax(clipwise[0]))]}"

    def music2emo():
        if newone._get_music2emo_model() is None:
            raise RuntimeError("Music2Emo failed to load (see warning above)")
        newone.set_audio_context(tone, sr)
        valence, arousal, moods = newone.get_music_valence_arousal(0.0, 6.0)
        if valence is None:
            raise RuntimeError("Music2Emo prediction failed (see warning above)")
        return f"valence={valence:.2f} arousal={arousal:.2f} moods={moods}"

    check("PANNs model + inference", panns)
    check("Music2Emo model + inference", music2emo)
    check("Silero VAD", lambda: type(newone.get_silero_vad()[0]).__name__)
    check("Sentence Transformer", lambda: type(newone.get_sentence_model()).__name__)
    check("Florence-2", lambda: type(newone.get_florence()[1]).__name__)


def self_test() -> int:
    """Import everything the pipeline needs and verify bundled assets."""
    checks = []

    def check(name, fn):
        try:
            detail = fn()
            checks.append((name, True, detail or ""))
        except Exception as exc:
            checks.append((name, False, f"{type(exc).__name__}: {exc}"))
            traceback.print_exc()

    check("frontend bundle", lambda: str((BUNDLE_DIR / "frontend" / "dist" / "index.html").resolve(strict=True)))
    check("backend.app", lambda: __import__("backend.app").__name__)
    check("pipeline (newone)", lambda: __import__("newone").__name__)
    check("panns labels", lambda: f"{len(__import__('panns_inference').config.labels)} labels")
    check("ffmpeg", lambda: __import__("imageio_ffmpeg").get_ffmpeg_exe())
    check("tiktoken", lambda: __import__("tiktoken").get_encoding("cl100k_base").name)

    if "--models" in sys.argv:
        model_checks(check)

    failed = [c for c in checks if not c[1]]
    for name, ok, detail in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    return 1 if failed else 0


def main() -> int:
    configure_environment()
    if "--self-test" in sys.argv:
        return self_test()

    existing = f"http://{HOST}:{PREFERRED_PORT}"
    if is_planetread(existing):
        print(f"PlanetRead is already running at {existing}; opening it.")
        open_browser(existing)
        return 0

    import uvicorn

    port = pick_port()
    url = f"http://{HOST}:{port}"
    os.environ["PLANETREAD_FRONTEND_URL"] = url
    print(f"Starting PlanetRead (data folder: {os.environ['PLANETREAD_DATA_DIR']}) ...", flush=True)

    from backend.app import app

    server = uvicorn.Server(uvicorn.Config(app, host=HOST, port=port, log_level="info"))
    threading.Thread(target=open_when_ready, args=(url, server), daemon=True).start()
    server.run()
    return 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    try:
        code = main()
    except Exception:
        traceback.print_exc()
        code = 1
    if code and FROZEN and "--self-test" not in sys.argv:
        # Keep the console open so the user can read the error.
        input("\nPlanetRead stopped with an error. Press Enter to close...")
    sys.exit(code)
