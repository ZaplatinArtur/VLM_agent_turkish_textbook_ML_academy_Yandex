[CmdletBinding()]
param(
    [string]$Python,
    [switch]$CopyToDesktop
)

$ErrorActionPreference = "Stop"
$projectDir = Split-Path -Parent $MyInvocation.MyCommand.Path

if (-not $Python) {
    $localPython = Join-Path $projectDir ".venv\Scripts\python.exe"
    if (Test-Path -LiteralPath $localPython -PathType Leaf) {
        $Python = $localPython
    } else {
        $Python = (Get-Command python -ErrorAction Stop).Source
    }
}

$holdoutSummary = Join-Path $projectDir "vlm_trace_viewer\holdout80_verified_summary.json"
$distDir = Join-Path $projectDir "dist-desktop"
$workDir = Join-Path $projectDir "build-desktop"
$entryPoint = Join-Path $projectDir "trace_viewer.py"

& $Python -m PyInstaller `
    --noconfirm `
    --clean `
    --onefile `
    --windowed `
    --name "VLM Analytics 9B" `
    --distpath $distDir `
    --workpath $workDir `
    --specpath $workDir `
    --add-data "${holdoutSummary};vlm_trace_viewer" `
    $entryPoint

if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller exited with code $LASTEXITCODE."
}

$builtExe = Join-Path $distDir "VLM Analytics 9B.exe"
if (-not (Test-Path -LiteralPath $builtExe -PathType Leaf)) {
    throw "Expected executable was not produced: $builtExe"
}

if ($CopyToDesktop) {
    $desktopDir = [Environment]::GetFolderPath("Desktop")
    $desktopExe = Join-Path $desktopDir "VLM Analytics 9B.exe"
    Copy-Item -LiteralPath $builtExe -Destination $desktopExe -Force
    Write-Output $desktopExe
} else {
    Write-Output $builtExe
}
