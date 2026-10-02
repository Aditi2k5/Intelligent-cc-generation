"""Per-user settings entered in the app's Settings dialog.

Stored in the user's profile (%APPDATA%\\PlanetRead\\settings.json on Windows,
~/.config/planetread/settings.json elsewhere), so the key is never built into
the app or shared with the zip, and survives replacing the app folder.
A key saved here wins over an OPENAI_API_KEY environment variable, which
remains a fallback for running from source with a .env.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

if sys.platform == "win32":
    SETTINGS_DIR = Path(os.environ.get("APPDATA", Path.home())) / "PlanetRead"
else:
    SETTINGS_DIR = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "planetread"
SETTINGS_FILE = SETTINGS_DIR / "settings.json"

DEFAULTS = {"openai_api_key": "", "gpu": False}


def load() -> dict:
    try:
        return {**DEFAULTS, **json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))}
    except (OSError, ValueError):
        return dict(DEFAULTS)


def save(**changes) -> dict:
    settings = {**load(), **changes}
    SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = SETTINGS_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(settings, indent=2), encoding="utf-8")
    if sys.platform != "win32":
        os.chmod(tmp, 0o600)
    os.replace(tmp, SETTINGS_FILE)
    return settings


def openai_api_key() -> str:
    return load()["openai_api_key"] or os.environ.get("OPENAI_API_KEY", "")
