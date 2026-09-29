# PlanetRead standalone distribution

## Windows executable

The native Windows build is created on Windows because PyInstaller cannot
cross-compile a Windows `.exe` from macOS. Install Python 3.10+ and Node.js on
the Windows build machine, open PowerShell in this folder, and run:

```powershell
.\build_windows_exe.ps1
```

The output is a single file: `dist\\PlanetRead.exe`. Send that one file. Put a
`.env` file beside it containing the recipient's `OPENAI_API_KEY` and
`HF_TOKEN`. PyInstaller extracts the embedded app internally when it launches;
the recipient does not need the project folder, Git, Node.js, or Python.

The executable will be large because it embeds PyTorch, Florence, PANNs and
OpenCV. Model weights are downloaded on first use and API keys remain external.

## Build it from this Mac using GitHub Actions

If you do not have a Windows computer, push this repository to GitHub and run
the `Build PlanetRead Windows executable` workflow from the Actions tab using
**Run workflow**. GitHub builds it on a real Windows runner and publishes
`PlanetRead-Windows.zip` as a downloadable artifact. This avoids Wine, which is
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
