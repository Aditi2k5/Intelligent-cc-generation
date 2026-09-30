$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
git submodule update --init
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
$env:OPENAI_API_KEY = if ($env:OPENAI_API_KEY) { $env:OPENAI_API_KEY } else { "self-test-placeholder" }
& dist\PlanetRead\PlanetRead.exe --self-test --models
if ($LASTEXITCODE -ne 0) { throw "Self-test failed" }
Copy-Item backend\.env.example dist\PlanetRead\.env.example
Write-Host "Built dist\PlanetRead\PlanetRead.exe"
