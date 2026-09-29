$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
python -m venv .venv-exe
& .\.venv-exe\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt -r backend\requirements.txt pyinstaller
if (-not (Test-Path frontend\dist\index.html)) {
  Push-Location frontend
  npm install
  npm run build
  Pop-Location
}
Remove-Item -Recurse -Force build, dist\PlanetRead -ErrorAction SilentlyContinue
python -m PyInstaller --clean --noconfirm planetread.spec
Write-Host "Built dist\PlanetRead\PlanetRead.exe"
