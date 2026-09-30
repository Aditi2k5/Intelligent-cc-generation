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
`_internal` folder. Zip and send the whole folder; the exe does not work on its own.
The recipient:

1. Unzips it anywhere they can write to (e.g. Desktop or Documents).
2. Renames `.env.example` to `.env` and fills in `OPENAI_API_KEY` and `HF_TOKEN`.
3. Double-clicks `PlanetRead.exe`. A console window shows progress and the UI
   opens in the browser once the server is ready. Closing the console quits.

Uploads and results are stored in `PlanetRead_data\` next to the exe. Model
weights (Florence-2, PANNs, Silero, sentence-transformers) download on the first
run, so that run needs internet and several GB of disk. They are cached under the
user's profile after that. They do not need Git, Node.js, Python or ffmpeg.

The build ends with `PlanetRead.exe --self-test --models`, which imports the whole
pipeline, checks the bundled assets and loads every model (PANNs, Music2Emo,
Silero, sentence-transformers, Florence-2), running the audio models on a test tone, so a missing package fails the build instead of
the recipient's run. Recipients can run the same command to diagnose problems.

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
