# PlanetRead standalone distribution

## Music2Emotion

`Music2Emotion/` is a git submodule (pinned to the upstream commit the pipeline
was tested with). Clone with `git clone --recursive`, or run
`git submodule update --init` in an existing checkout. The Windows build bundles
its code, checkpoints and data; the build fails if the folder is missing.
Install only the top-level `requirements.txt` — it already covers Music2Emo, and
`Music2Emotion/requirements.txt` pins conflicting torch/transformers versions.

## Windows executable

The native Windows build is created on Windows because PyInstaller cannot
cross-compile a Windows `.exe` from macOS. Install Python 3.10+ and Node.js on
the Windows build machine, open PowerShell in this folder, and run:

```powershell
.\build_windows_exe.ps1
```

The output is the folder `dist\PlanetRead\` containing `PlanetRead.exe` plus an
`_internal` folder, under 2 GB in total with every model included (CI fails the
build if it grows past that). Zip and send the whole folder; the exe does not
work on its own. The recipient:

1. Unzips it anywhere they can write to (e.g. Desktop or Documents).
2. Double-clicks `PlanetRead.exe`. A console window shows progress and the UI
   opens in the browser once the server is ready. Closing the console quits.
3. Pastes their OpenAI API key into the Settings dialog, which opens by itself
   the first time. The key is checked with OpenAI and saved in their Windows
   profile (`%APPDATA%\PlanetRead\settings.json`), not in the app folder, so it
   is never part of the zip and survives replacing the app with a new version.

No `.env`, Hugging Face token, Git, Node.js, Python or ffmpeg is needed, and
nothing is downloaded at startup: all models ship inside the app and the app
runs with Hugging Face offline mode on. Only caption writing calls the OpenAI
API. Uploads and results are stored in `PlanetRead_data\` next to the exe.

### What keeps it under 2 GB

- **Models** (`python tools/pack_models.py` writes `models/`, ~1.0 GB): Florence-2-large,
  MERT (used by Music2Emo) and PANNs are stored with 8-bit weights (one scale per 64
  weights, see `model_pack.py`) and expanded back to float when loaded; MiniLM is
  stored in float16; Silero ships only its 2 MB JIT model. Revisions are pinned.
  Checked against the original models: Florence-2 gave identical captions on 8
  of 10 episode frames (the other two differ in a detail), PANNs the same top-5
  sounds, Music2Emo the same moods with valence/arousal within 1%, Silero and
  MiniLM identical results.
- **CPU-only PyTorch**: the CUDA build adds ~2.5 GB of NVIDIA libraries; the build
  stops if a CUDA torch is installed.
- **No build-only files**: `.lib` files and C++ headers (~825 MB in torch alone) and
  music21's 64 MB sample corpus are left out.
- **Music2Emo**: only the files inference reads (the `J_all.ckpt` checkpoint, the chord
  model and `inference/data`). gradio, which it imports but does not use, is replaced
  by an empty stub.

### NVIDIA GPU (optional)

On a PC with an NVIDIA GPU, Settings shows **Use NVIDIA GPU** (off by default). Turning
it on downloads the CUDA build of the same PyTorch once in the background (2.5 GB,
3.7 GB on disk, into `%LOCALAPPDATA%\PlanetRead`), checks that it sees the GPU, and
uses it from the next start. This is outside the 2 GB app. `PlanetRead.exe --self-test`
shows which torch and device it uses; CI also runs the self-test with the CUDA build.

The build ends with `PlanetRead.exe --self-test --models`, which imports the whole
pipeline, checks the bundled assets and loads every model (PANNs, Music2Emo,
Silero, sentence-transformers, Florence-2), running the audio models on a test tone, so a missing package fails the build instead of
the recipient's run. CI first deletes the download caches, so this proves the app
runs from the bundled models alone. Recipients can run the same command to diagnose problems.

## Build it from this Mac using GitHub Actions

If you do not have a Windows computer, push this repository to GitHub and run
the `Build PlanetRead Windows executable` workflow from the Actions tab using
**Run workflow**. GitHub builds it on a real Windows runner and publishes
`PlanetRead-Windows.zip` as a downloadable artifact. Unzip it and send
the `PlanetRead` folder as described above. This avoids Wine, which is
not a reliable PyInstaller toolchain for Apple Silicon.

## macOS

Double-click `setup_planetread.command` once. Add `OPENAI_API_KEY` and `HF_TOKEN`
to the generated `.env`, then double-click `run_planetread.command`.

## Windows

Double-click `setup_planetread.bat`, add the two keys to `.env`, then double-click
`run_planetread.bat`.

The app opens at `http://127.0.0.1:8000`. The backend serves the bundled frontend
and runs the ML pipeline. The first setup downloads Python dependencies and
model weights, so it requires internet access and several GB of disk space.

## Sharing the folder

Create a zip that includes `backend/`, `frontend/dist/`, `frontend/public/`,
`panns_pipeline.py`, `requirements.txt`, `backend/requirements.txt`, the task
files and the two setup/run launchers. Do not include `.env`, `.venv`,
`backend_data`, or `node_modules`; each recipient creates their own environment
and supplies their own API keys.
