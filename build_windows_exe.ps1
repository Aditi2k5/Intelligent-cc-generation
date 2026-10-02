$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
git submodule update --init
python -m venv .venv-exe
& .\.venv-exe\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt -r backend\requirements.txt pyinstaller
Push-Location frontend
npm install
npm run build
Pop-Location
if (-not (Test-Path models\florence-2-large)) { python tools\pack_models.py }
Remove-Item -Recurse -Force build, dist\PlanetRead -ErrorAction SilentlyContinue
python -m PyInstaller --clean --noconfirm planetread.spec
& dist\PlanetRead\PlanetRead.exe --self-test --models
if ($LASTEXITCODE -ne 0) { throw "Self-test failed" }
Write-Host "Built dist\PlanetRead\PlanetRead.exe"
