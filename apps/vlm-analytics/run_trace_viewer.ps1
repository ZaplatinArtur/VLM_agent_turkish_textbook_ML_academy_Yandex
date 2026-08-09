[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [string]$ArtifactRoot,

    [switch]$ValidateOnly,

    [string]$Task = "val_0178",

    [switch]$Holdout,

    [switch]$Metrics
)

$ErrorActionPreference = "Stop"
$appDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$frozenMarker = "reports\maxim_official_exact_source_v2_20260805\V7_POST_SCORE_RESULT.json"
$artifactProjectName = "VLM_agent_turkish_textbook_basic_rag"

function Test-ArtifactRoot {
    param([string]$Path)

    if (-not $Path) {
        return $false
    }
    return Test-Path -LiteralPath (Join-Path $Path $frozenMarker) -PathType Leaf
}

function Resolve-ArtifactRoot {
    if ($ArtifactRoot) {
        $resolved = [System.IO.Path]::GetFullPath($ArtifactRoot)
        if (-not (Test-ArtifactRoot $resolved)) {
            throw "Frozen V7 marker was not found under --artifact-root: $resolved"
        }
        return $resolved
    }

    if ($env:VLM_TRACE_ARTIFACT_ROOT) {
        $resolved = [System.IO.Path]::GetFullPath($env:VLM_TRACE_ARTIFACT_ROOT)
        if (-not (Test-ArtifactRoot $resolved)) {
            throw "VLM_TRACE_ARTIFACT_ROOT has no frozen V7 marker: $resolved"
        }
        return $resolved
    }

    $cursor = [System.IO.DirectoryInfo]$appDir
    $seen = @{}
    for ($level = 0; $level -lt 8 -and $null -ne $cursor; $level++) {
        foreach ($candidate in @($cursor.FullName, (Join-Path $cursor.FullName $artifactProjectName))) {
            $full = [System.IO.Path]::GetFullPath($candidate)
            if (-not $seen.ContainsKey($full)) {
                $seen[$full] = $true
                if (Test-ArtifactRoot $full) {
                    return $full
                }
            }
        }
        $cursor = $cursor.Parent
    }

    throw @"
The project containing frozen V7 artifacts was not found.
Run: .\run_trace_viewer.ps1 -ArtifactRoot C:\path\to\VLM_agent_turkish_textbook_basic_rag
"@
}

function Resolve-Python {
    $candidates = @()
    if ($env:VIRTUAL_ENV) {
        $candidates += (Join-Path $env:VIRTUAL_ENV "Scripts\python.exe")
    }
    $candidates += (Join-Path $appDir ".venv\Scripts\python.exe")
    $candidates += (Join-Path (Split-Path -Parent (Split-Path -Parent $appDir)) ".venv\Scripts\python.exe")

    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate -PathType Leaf) {
            return [System.IO.Path]::GetFullPath($candidate)
        }
    }

    $command = Get-Command python -ErrorAction SilentlyContinue
    if ($command) {
        return $command.Source
    }
    throw "Python was not found. Install Python 3.11+ and create .venv as described in DEMO_GUIDE_RU.md."
}

$resolvedArtifacts = Resolve-ArtifactRoot
$pythonExe = Resolve-Python
$viewer = Join-Path $appDir "trace_viewer.py"
$baseArgs = @($viewer, "--artifact-root", $resolvedArtifacts)

Write-Host "VLM Trace preflight" -ForegroundColor Cyan
Write-Host "  Python:    $pythonExe"
Write-Host "  Artifacts: $resolvedArtifacts"
$savedErrorPreference = $ErrorActionPreference
$ErrorActionPreference = "Continue"
& $pythonExe @baseArgs --validate-only
$preflightExit = $LASTEXITCODE
$ErrorActionPreference = $savedErrorPreference
if ($preflightExit -ne 0) {
    throw "Trace Viewer preflight exited with code $preflightExit."
}

if ($ValidateOnly) {
    exit 0
}

$ErrorActionPreference = "SilentlyContinue"
& $pythonExe -c "import PySide6" 2>$null
$dependencyExit = $LASTEXITCODE
$ErrorActionPreference = $savedErrorPreference
if ($dependencyExit -ne 0) {
    throw @"
PySide6 is not installed in the selected Python.
Run from ${appDir}:
  python -m venv .venv
  .\.venv\Scripts\python.exe -m pip install -r requirements.txt
"@
}

$launchArgs = @($viewer, "--artifact-root", $resolvedArtifacts, "--task", $Task)
if ($Holdout) {
    $launchArgs += @("--screenshot-tab", "1")
} elseif ($Metrics) {
    $launchArgs += @("--screenshot-tab", "2")
}

$ErrorActionPreference = "Continue"
& $pythonExe @launchArgs
$viewerExit = $LASTEXITCODE
$ErrorActionPreference = $savedErrorPreference
exit $viewerExit
