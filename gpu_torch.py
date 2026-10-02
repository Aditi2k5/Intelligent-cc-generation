"""Optional NVIDIA GPU support for the packaged Windows app.

The package bundles the CPU build of PyTorch: the CUDA build adds ~2.5 GB of
NVIDIA libraries and would push the download past 2 GB. When an NVIDIA GPU is
present and the user turns on "Use NVIDIA GPU" in Settings (off by default,
as it is a 2.5 GB download), the CUDA builds of torch/torchvision/torchaudio
(same versions) are downloaded in the background into %LOCALAPPDATA%\\PlanetRead
and, from the next launch on, put ahead of the bundled CPU build on sys.path.
PyInstaller's importer is a sys.path entry, so the first match wins.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import urllib.request
import zipfile
from pathlib import Path

import app_settings

# Must match the torch pins in requirements.txt (the bundled CPU build).
CUDA_TAG = "cu121"
WHEELS = {"torch": "2.4.1", "torchvision": "0.19.1", "torchaudio": "2.4.1"}
INDEX_URL = f"https://download.pytorch.org/whl/{CUDA_TAG}"
ENV_DIR = "PLANETREAD_TORCH_DIR"  # set once active; inherited by child processes

_PY = f"cp{sys.version_info.major}{sys.version_info.minor}"
ROOT = (Path(os.environ.get("LOCALAPPDATA", Path.home())) / "PlanetRead"
        / f"torch-{WHEELS['torch']}-{CUDA_TAG}-{_PY}")
SITE = ROOT / "site"
STATUS = ROOT / "status.json"


# What the Settings dialog shows: off | downloading | restart | active | failed
STATE = {"state": "off", "message": ""}
_setup_thread: threading.Thread | None = None
_gpu_present: bool | None = None


def _frozen_windows() -> bool:
    return getattr(sys, "frozen", False) and sys.platform == "win32"


def enabled() -> bool:
    return _frozen_windows() and bool(app_settings.load()["gpu"])


def _set(state: str, message: str = "") -> None:
    STATE.update(state=state, message=message)


def use(site: Path) -> None:
    """Put the CUDA torch ahead of the bundled one. Must run before torch is imported."""
    sys.path.insert(0, str(site))
    os.environ[ENV_DIR] = str(site)


def nvidia_gpu_present() -> bool:
    global _gpu_present
    if _gpu_present is None:
        _gpu_present = _frozen_windows() and _run_nvidia_smi()
    return _gpu_present


def _run_nvidia_smi() -> bool:
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=15,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except (OSError, subprocess.SubprocessError):
        return False
    return out.returncode == 0 and "GPU" in out.stdout


def _download(url: str, dest: Path, log) -> None:
    """Download with resume, so an interrupted 2.4 GB download continues next launch."""
    part = dest.with_name(dest.name + ".part")
    have = part.stat().st_size if part.exists() else 0
    req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
    with urllib.request.urlopen(req, timeout=60) as resp:
        if have and resp.status != 206:
            have = 0  # server ignored the range; start over
        total = have + int(resp.headers.get("Content-Length", 0))
        next_report = 0
        with open(part, "ab" if have else "wb") as f:
            while chunk := resp.read(1 << 20):
                f.write(chunk)
                have += len(chunk)
                if total and have * 100 // total >= next_report:
                    pct = have * 100 // total
                    _set("downloading", f"Downloading {dest.name.split('-')[0]}: {pct}% of {total / 1e9:.2f} GB")
                    log(f"[GPU] {STATE['message']}")
                    next_report = pct + 10
    part.replace(dest)


def install(log=print) -> None:
    """Download and unpack the CUDA wheels into SITE (atomically)."""
    downloads = ROOT / "downloads"
    staging = ROOT / "site.partial"
    downloads.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(staging, ignore_errors=True)
    for name, version in WHEELS.items():
        wheel = downloads / f"{name}-{version}+{CUDA_TAG}-{_PY}-{_PY}-win_amd64.whl"
        if not wheel.exists():
            _download(f"{INDEX_URL}/{wheel.name.replace('+', '%2B')}", wheel, log)
        _set("downloading", f"Unpacking {name} ...")
        log(f"[GPU] {STATE['message']}")
        try:
            with zipfile.ZipFile(wheel) as zf:
                # C++ headers and .lib files are only for building extensions; skipping
                # them saves space and avoids paths over Windows' 260-character limit.
                members = [m for m in zf.namelist()
                           if not m.startswith(("torch/include/", "torch/share/")) and not m.endswith(".lib")]
                zf.extractall(staging, members)  # verifies each file's CRC
        except (zipfile.BadZipFile, OSError):
            wheel.unlink(missing_ok=True)  # corrupt download: fetch again next time
            raise
    shutil.rmtree(SITE, ignore_errors=True)
    staging.replace(SITE)
    shutil.rmtree(downloads, ignore_errors=True)


def probe() -> dict:
    """Import the CUDA torch in a separate process, so a broken install can't
    take down the app. Returns {"cuda": bool, ...}; raises if the import fails."""
    out = subprocess.run([sys.executable, "--probe-torch", str(SITE)], capture_output=True,
                         text=True, timeout=600, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    lines = out.stdout.strip().splitlines()
    if out.returncode != 0 or not lines:
        raise RuntimeError(f"CUDA torch failed to import:\n{out.stderr[-2000:]}")
    return json.loads(lines[-1])


def probe_main(site: str) -> int:
    """Body of `PlanetRead.exe --probe-torch <site>`."""
    use(Path(site))
    import torch

    info = {"version": torch.__version__, "cuda": torch.cuda.is_available()}
    if info["cuda"]:
        info["device"] = torch.cuda.get_device_name(0)
    print(json.dumps(info))
    return 0


def _installed() -> bool:
    return STATUS.exists() and (SITE / "torch" / "__init__.py").exists()


def _install_and_probe(log, require_cuda: bool) -> bool:
    try:
        if not (SITE / "torch" / "__init__.py").exists():
            install(log)
        _set("downloading", "Checking the GPU ...")
        info = probe()
    except Exception as exc:
        _set("failed", f"Could not set up GPU support ({exc}). PlanetRead keeps using the CPU; "
                       "turn the setting off and on again to retry.")
        log(f"[GPU] {STATE['message']}")
        shutil.rmtree(SITE, ignore_errors=True)
        return False
    if require_cuda and not info["cuda"]:
        _set("failed", "The NVIDIA GPU can't be used: its driver is probably too old (CUDA 12.1 needs "
                       "driver 527 or newer). Update the driver from nvidia.com and restart PlanetRead.")
        log(f"[GPU] {STATE['message']}")
        return False
    STATUS.write_text(json.dumps(info))
    _set("restart", "GPU support is ready. Restart PlanetRead to use the GPU.")
    log(f"[GPU] {STATE['message']}")
    return True


def start_setup(log=print) -> None:
    """Download and check the CUDA build in the background (no-op if running)."""
    global _setup_thread
    if _setup_thread is not None and _setup_thread.is_alive():
        return
    _set("downloading", "Starting download (~2.5 GB) ...")
    log("[GPU] Downloading GPU support (~2.5 GB, one time) in the background; "
        "PlanetRead keeps working on the CPU meanwhile.")
    _setup_thread = threading.Thread(target=_install_and_probe, args=(log, True),
                                     daemon=True, name="gpu-torch-setup")
    _setup_thread.start()


def activate(log=print, setup: bool = True) -> None:
    """Called at startup, before torch is imported. Uses the CUDA build if the
    setting is on and it is installed; if on but not installed (e.g. the
    download was interrupted), resumes the setup in the background."""
    if not enabled() or not nvidia_gpu_present() and not _installed():
        return
    if _installed():
        use(SITE)
        _set("active", f"Using {json.loads(STATUS.read_text()).get('device', 'the GPU')}.")
    elif setup:
        start_setup(log)


def set_enabled(on: bool, log=print) -> dict:
    """Settings dialog toggle."""
    app_settings.save(gpu=bool(on))
    if not on:
        _set("off", "Turned off. Restart PlanetRead to switch back to the CPU."
             if os.environ.get(ENV_DIR) else "")
    elif os.environ.get(ENV_DIR):
        _set("active", STATE["message"] or "Using the GPU.")
    elif _installed():
        _set("restart", "GPU support is ready. Restart PlanetRead to use the GPU.")
    else:
        start_setup(log)
    return status()


def status() -> dict:
    return {"supported": nvidia_gpu_present(), "enabled": enabled(), **STATE}


def install_now(log=print) -> int:
    """Body of `PlanetRead.exe --install-gpu-torch`: turn the setting on and set
    up synchronously, even without a GPU (CI uses this to test the swap)."""
    app_settings.save(gpu=True)
    return 0 if _install_and_probe(log, require_cuda=False) else 1
