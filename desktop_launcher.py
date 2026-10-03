"""Entry point for the packaged PlanetRead app (PyInstaller).

Double-clicking PlanetRead.exe runs this: it loads the .env next to the exe,
starts the FastAPI backend (which also serves the built frontend), waits until
the server actually answers, and then opens the UI in the default browser.

`PlanetRead.exe --self-test` imports the whole pipeline and checks bundled
assets without starting the server; `--self-test --models` also loads every
model (PANNs, Music2Emo, Silero, Florence-2, ...). CI runs both so a broken
build fails there instead of on the recipient's machine.

The bundled PyTorch is the CPU build; gpu_torch adds the CUDA build on
machines with an NVIDIA GPU. `--install-gpu-torch` sets that up immediately.
"""
from __future__ import annotations

import hashlib
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

import gpu_torch

# multiprocessing children re-run this module; give them the same torch as the parent.
if os.environ.get(gpu_torch.ENV_DIR):
    sys.path.insert(0, os.environ[gpu_torch.ENV_DIR])

HOST = "127.0.0.1"
PREFERRED_PORT = 8000

FROZEN = getattr(sys, "frozen", False)
# Folder the user sees (where PlanetRead.exe lives) vs. folder holding bundled code/data.
APP_DIR = Path(sys.executable).resolve().parent if FROZEN else Path(__file__).resolve().parent
BUNDLE_DIR = Path(getattr(sys, "_MEIPASS", APP_DIR))


def configure_environment() -> None:
    from dotenv import load_dotenv

    # Optional: settings normally come from the in-app Settings dialog (app_settings).
    load_dotenv(APP_DIR / ".env")
    if FROZEN:
        # Every model ships in the bundle (tools/pack_models.py); never reach the hub.
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
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
    import app_settings
    if not app_settings.openai_api_key():
        print("No OpenAI API key yet: the app will ask for it in the browser.\n", flush=True)
    os.environ.setdefault("MPLCONFIGDIR", str(data_dir / ".matplotlib"))


def planetread_health(url: str) -> dict | None:
    try:
        with urllib.request.urlopen(f"{url}/api/health", timeout=2) as resp:
            health = json.load(resp)
        return health if health.get("status") == "ok" else None
    except Exception:
        return None


def is_planetread(url: str) -> bool:
    return planetread_health(url) is not None


def frontend_build() -> str:
    """Same fingerprint backend.app reports as frontend_build in /api/health."""
    try:
        return hashlib.sha256((BUNDLE_DIR / "frontend" / "dist" / "index.html").read_bytes()).hexdigest()[:12]
    except OSError:
        return ""


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

    def torch_info():
        import torch
        where = "CUDA add-on" if os.environ.get(gpu_torch.ENV_DIR) else "bundled"
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no GPU, using CPU"
        return f"{torch.__version__} ({where}), {gpu}"

    check("torch", torch_info)
    def bundled_models():
        import model_store
        names = ["florence-2-large", "mert-v1-95m", "all-MiniLM-L6-v2", "panns", "silero-vad"]
        missing = [n for n in names if not model_store.bundled(n)]
        if FROZEN and missing:
            raise FileNotFoundError(f"not bundled: {missing}")
        return f"{model_store.MODELS_DIR} ({len(names) - len(missing)}/{len(names)})"

    check("bundled models", bundled_models)
    check("Hindi caption font", lambda: str((BUNDLE_DIR / "TiroDevanagariHindi-Italic.ttf").resolve(strict=True)))
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
    if "--probe-torch" in sys.argv:
        return gpu_torch.probe_main(sys.argv[sys.argv.index("--probe-torch") + 1])
    configure_environment()
    if "--install-gpu-torch" in sys.argv:
        return gpu_torch.install_now()
    gpu_torch.activate(setup="--self-test" not in sys.argv)
    if "--self-test" in sys.argv:
        return self_test()

    existing = f"http://{HOST}:{PREFERRED_PORT}"
    running = planetread_health(existing)
    if running and running.get("frontend_build") == frontend_build():
        print(f"PlanetRead is already running at {existing}; opening it.")
        open_browser(existing)
        return 0
    if running:
        # A different (usually older) PlanetRead still holds the port. Opening it would
        # show that version's page, so start this one on another port instead.
        print(f"Another version of PlanetRead is still running at {existing}.")
        print("Close its window (or end PlanetRead.exe in Task Manager) when you can.\n", flush=True)

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
    interactive = not {"--self-test", "--probe-torch", "--install-gpu-torch"}.intersection(sys.argv)
    if code and FROZEN and interactive:
        # Keep the console open so the user can read the error.
        input("\nPlanetRead stopped with an error. Press Enter to close...")
    sys.exit(code)
