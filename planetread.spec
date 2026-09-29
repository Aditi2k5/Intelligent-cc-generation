from pathlib import Path
from PyInstaller.utils.hooks import collect_all
root = Path(SPECPATH)
datas = [(str(root / "backend"), "backend"), (str(root / "frontend" / "dist"), "frontend/dist")]
for name in ["TiroDevanagariHindi-Italic.ttf", "face_landmarker.task", "pose_landmarker_heavy.task"]:
    if (root / name).exists(): datas.append((str(root / name), "."))
hiddenimports = ["backend", "backend.app", "uvicorn.logging", "uvicorn.loops.auto", "uvicorn.protocols.http.auto", "uvicorn.protocols.websockets.auto"]
for package in ["torch", "torchaudio", "transformers", "panns_inference", "sentence_transformers", "cv2", "PIL", "timm", "einops"]:
    try:
        d, b, h = collect_all(package)
        datas += d; hiddenimports += h
    except Exception:
        pass
a = Analysis([str(root / "desktop_launcher.py")], pathex=[str(root)], binaries=[], datas=datas, hiddenimports=hiddenimports, hookspath=[], hooksconfig={}, runtime_hooks=[], excludes=[], noarchive=False)
pyz = PYZ(a.pure)
EXE(pyz, a.scripts, a.binaries, a.datas, [], name="PlanetRead", debug=False, bootloader_ignore_signals=False, strip=False, upx=False, console=True)
