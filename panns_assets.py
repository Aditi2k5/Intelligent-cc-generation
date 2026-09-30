"""Provision the files panns_inference expects under ~/panns_data.

panns_inference fetches its label CSV (at import time) and its checkpoint
(on first AudioTagging()) by shelling out to `wget`, which does not exist on
Windows. The import then dies with "No such file or directory:
.../panns_data/class_labels_indices.csv". This module puts both files in
place first: the CSV is shipped with the app, the checkpoint is downloaded
with urllib.
"""
from __future__ import annotations

import os
import shutil
import sys
import urllib.request
from pathlib import Path

PANNS_DIR = Path.home() / "panns_data"
LABELS_CSV = PANNS_DIR / "class_labels_indices.csv"
CHECKPOINT = PANNS_DIR / "Cnn14_mAP=0.431.pth"
# panns_inference re-downloads any checkpoint smaller than this.
_CHECKPOINT_MIN_BYTES = 3e8
_CHECKPOINT_URL = "https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1"
_LABELS_URL = "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv"


def _bundled_labels() -> Path:
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    return base / "assets" / "panns" / "class_labels_indices.csv"


def _download(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"Downloading {dest.name} ...", flush=True)
    with urllib.request.urlopen(url, timeout=60) as resp, open(tmp, "wb") as out:
        total = int(resp.headers.get("Content-Length") or 0)
        done, last_pct = 0, -10
        while chunk := resp.read(1024 * 1024):
            out.write(chunk)
            done += len(chunk)
            if total:
                pct = done * 100 // total
                if pct >= last_pct + 10:
                    print(f"  {dest.name}: {pct}%", flush=True)
                    last_pct = pct
    os.replace(tmp, dest)


def ensure_labels() -> Path:
    if not LABELS_CSV.is_file() or LABELS_CSV.stat().st_size == 0:
        PANNS_DIR.mkdir(parents=True, exist_ok=True)
        bundled = _bundled_labels()
        if bundled.is_file():
            shutil.copyfile(bundled, LABELS_CSV)
        else:
            _download(_LABELS_URL, LABELS_CSV)
    return LABELS_CSV


def ensure_checkpoint() -> Path:
    if not CHECKPOINT.is_file() or CHECKPOINT.stat().st_size < _CHECKPOINT_MIN_BYTES:
        PANNS_DIR.mkdir(parents=True, exist_ok=True)
        _download(_CHECKPOINT_URL, CHECKPOINT)
    return CHECKPOINT
