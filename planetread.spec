# PyInstaller spec for the Windows build. Produces dist/PlanetRead/PlanetRead.exe
# (a folder build: PyTorch is several GB, and a one-file exe would have to
# unpack all of it to a temp folder on every launch).
from pathlib import Path
from PyInstaller.utils.hooks import collect_all, collect_submodules

root = Path(SPECPATH)

# The CUDA build of torch bundles ~2.5 GB of NVIDIA libraries and pushes the
# package past 2 GB; requirements.txt installs the CPU build on Windows.
import torch
if torch.version.cuda:
    raise SystemExit(f"planetread.spec: torch {torch.__version__} is a CUDA build - install requirements.txt (CPU torch)")
# All models, compacted by tools/pack_models.py, so the app needs no downloads.
if not (root / "models" / "florence-2-large").is_dir():
    raise SystemExit("planetread.spec: models/ is missing - run: python tools/pack_models.py")
datas = [
    (str(root / "frontend" / "dist"), "frontend/dist"),
    (str(root / "assets" / "panns"), "assets/panns"),
    (str(root / "models"), "models"),
]
# Music2Emo (git submodule). Its code is compiled in via pathex below; the
# checkpoints and data it opens by relative path are shipped as files. Only
# what Music2emo() inference reads is shipped: the other checkpoints, the
# training metadata and the sample audio add ~100 MB.
m2e = root / "Music2Emotion"
if not (m2e / "music2emo.py").exists():
    raise SystemExit("planetread.spec: Music2Emotion/ is missing - run: git submodule update --init")
_m2e_skip_dirs = {".git", "__pycache__", "output", "temp_out", "dataset", "meta", "input"}
_m2e_skip_files = {"btc_model.pt"}
for path in m2e.rglob("*"):
    rel = path.relative_to(m2e)
    if (not path.is_file() or _m2e_skip_dirs.intersection(rel.parts) or path.name in _m2e_skip_files
            or path.suffix in {".png", ".ipynb", ".pyc", ".mp3"}):
        continue
    if rel.parts[0] == "saved_models" and path.name != "J_all.ckpt":
        continue
    datas.append((str(path), str(Path("Music2Emotion") / rel.parent)))
# newone.py burns Hindi captions into the video with this font, looked up next to
# its own file; without it Pillow's default font draws boxes instead of Hindi.
datas.append((str(root / "frontend" / "public" / "TiroDevanagariHindi-Italic.ttf"), "."))
for name in ["face_landmarker.task", "pose_landmarker_heavy.task"]:
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
    "hydra", "omegaconf", "antlr4", "mir_eval", "music21",
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
    excludes=["gradio", "gradio_client", "IPython", "jupyter", "notebook", "pytest"],
    noarchive=False,
    # PyInstaller's transformers hook also ships every .py file as source (~44 MB)
    # for TorchScript, which only its object-detection loss and fx tracing use;
    # none of our models do. Florence's and MERT's own code ships in models/.
    module_collection_mode={"transformers": "pyz"},
)
# Static/import .lib files and C++ headers are only used to compile extensions;
# torch alone ships ~825 MB of them (dnnl.lib is 647 MB). music21's corpus is
# 64 MB of sample scores; Music2Emo only parses its own MIDI output. Filtered
# here, after Analysis, so files added by PyInstaller hooks are covered too.
def _runtime_only(toc):
    def keep(dest):
        parts = Path(dest).parts
        return not dest.lower().endswith(".lib") and "include" not in parts[:-1] and parts[:2] != ("music21", "corpus")
    return [entry for entry in toc if keep(entry[0])]
a.datas, a.binaries = _runtime_only(a.datas), _runtime_only(a.binaries)

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
