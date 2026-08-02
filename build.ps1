$ErrorActionPreference = "Stop"
$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $ProjectDir ".venv\Scripts\python.exe"
$PyInstaller = Join-Path $ProjectDir ".venv\Scripts\pyinstaller.exe"

if (-not (Test-Path -LiteralPath $Python)) {
    $BootstrapPython = Get-Command python -ErrorAction Stop
    & $BootstrapPython.Source -m venv (Join-Path $ProjectDir ".venv")
}

& $Python -m pip install --upgrade pip
& $Python -m pip install -r (Join-Path $ProjectDir "requirements.txt")
& $PyInstaller `
    --noconfirm `
    --clean `
    --onefile `
    --windowed `
    --name "VLM Analytics" `
    --hidden-import "matplotlib.backends.backend_qtagg" `
    (Join-Path $ProjectDir "main.py")
