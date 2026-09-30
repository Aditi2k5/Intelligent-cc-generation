# PyInstaller spec for the Windows build. Produces dist/PlanetRead/PlanetRead.exe
# (a folder build: PyTorch is several GB, and a one-file exe would have to
# unpack all of it to a temp folder on every launch).
from pathlib import Path
from PyInstaller.utils.hooks import collect_all, collect_submodules

root = Path(SPECPATH)
datas = [
    (str(root / "frontend" / "dist"), "frontend/dist"),
    (str(root / "assets" / "panns"), "assets/panns"),
]
# Music2Emo (git submodule). Its code is compiled in via pathex below; the
# checkpoints and data it opens by relative path are shipped as files.
m2e = root / "Music2Emotion"
if not (m2e / "music2emo.py").exists():
    raise SystemExit("planetread.spec: Music2Emotion/ is missing - run: git submodule update --init")
_m2e_skip_dirs = {".git", "__pycache__", "output", "temp_out"}
for path in m2e.rglob("*"):
    rel = path.relative_to(m2e)
    if path.is_file() and not _m2e_skip_dirs.intersection(rel.parts) and path.suffix not in {".png", ".ipynb", ".pyc"}:
        datas.append((str(path), str(Path("Music2Emotion") / rel.parent)))
for name in ["TiroDevanagariHindi-Italic.ttf", "face_landmarker.task", "pose_landmarker_heavy.task"]:
    if (root / name).exists():
        datas.append((str(root / name), "."))

hiddenimports = [
    "backend.app",
    "newone",
    "panns_assets",
    "music2emo",
    "tiktoken_ext",
    "tiktoken_ext.openai_public",
] + collect_submodules("uvicorn")

binaries = []
# Packages with data files, native libs or dynamic imports that PyInstaller's
# static analysis misses. A missing package is a build error, not something to
# discover on the recipient's machine.
for package in [
    "torch", "torchaudio", "torchvision", "transformers", "sentence_transformers",
    "panns_inference", "torchlibrosa", "librosa", "soundfile", "noisereduce",
    "cv2", "PIL", "timm", "einops", "imageio_ffmpeg",
    "langchain_openai", "langchain_core", "langsmith", "openai", "tiktoken",
    # Music2Emo and the MERT model's remote code (nnAudio)
    "gradio", "gradio_client", "hydra", "omegaconf", "antlr4", "mir_eval", "music21",
    "pretty_midi", "mido", "nnAudio", "pytorch_lightning", "lightning_fabric",
    "lightning_utilities", "torchmetrics", "sklearn", "pandas", "scipy", "numba", "llvmlite",
]:
    d, b, h = collect_all(package)
    if not (d or b or h):
        raise SystemExit(f"planetread.spec: package '{package}' is not installed")
    datas += d; binaries += b; hiddenimports += h

a = Analysis(
    [str(root / "desktop_launcher.py")],
    pathex=[str(root), str(m2e)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    # gradio reads its own source files at import time.
    module_collection_mode={"gradio": "py", "gradio_client": "py"},
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="PlanetRead",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="PlanetRead")
