param(
    [Parameter(Mandatory = $true)]
    [string]$RankingsZip,
    [string]$Tasks = "outputs/validation_merged_20260723/validation_image_tasks.jsonl",
    [string]$Manifest = "outputs/validation_merged_20260723/validation_manifest.jsonl",
    [string]$DataRoot = "outputs/validation_merged_20260723",
    [string]$OutputRoot = "results/openrouter_reranker_accuracy_experiment",
    [string]$RunId = "",
    [string]$AnalyticsDb = "apps/vlm-analytics/vlm_analytics.db",
    [string]$DatasetVersion = "validation_images_198",
    [int]$TopK = 5,
    [int]$Limit = 10,
    [int]$Workers = 1,
    [switch]$IncludeQwen
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repoRoot ".venv/Scripts/python.exe"
$managedEnvironment = @(
    "MLA_LLM_PROVIDER",
    "MLA_OPENROUTER_BASE_URL",
    "MLA_OPENROUTER_MODEL_NAME",
    "MLA_DATA_ROOT",
    "MLA_TEXT_ONLY",
    "MLA_INCLUDE_QUESTION_TEXT_WITH_IMAGES",
    "MLA_ENABLE_THINKING",
    "MLA_PROMPT_VERSION",
    "MLA_CONCURRENCY",
    "MLA_RETRIEVAL_TOP_K",
    "MLA_RETRIEVAL_CONTEXT_ORDER"
)
$previousEnvironment = @{}
foreach ($name in $managedEnvironment) {
    $previousEnvironment[$name] = [System.Environment]::GetEnvironmentVariable(
        $name,
        "Process"
    )
}

function Resolve-RepoPath {
    param([string]$Path)

    if ([System.IO.Path]::IsPathRooted($Path)) {
        return [System.IO.Path]::GetFullPath($Path)
    }
    return [System.IO.Path]::GetFullPath((Join-Path $repoRoot $Path))
}

$RankingsZip = Resolve-RepoPath $RankingsZip
$Tasks = Resolve-RepoPath $Tasks
$Manifest = Resolve-RepoPath $Manifest
$DataRoot = Resolve-RepoPath $DataRoot
$OutputRoot = Resolve-RepoPath $OutputRoot
$AnalyticsDb = Resolve-RepoPath $AnalyticsDb

if (-not (Test-Path -LiteralPath $python)) {
    throw "Python environment not found: $python"
}
if ($Limit -lt 0) {
    throw "Limit must be zero (full run) or a positive integer."
}
if ($Workers -lt 1) {
    throw "Workers must be at least 1."
}
if ($TopK -lt 1 -or $TopK -gt 20) {
    throw "TopK must be between 1 and 20."
}
if (-not $env:OPENROUTER_API_KEY) {
    throw "Set OPENROUTER_API_KEY in the current PowerShell session."
}

Push-Location $repoRoot
try {
    foreach ($requiredPath in @($RankingsZip, $Tasks, $Manifest, $DataRoot)) {
        if (-not (Test-Path -LiteralPath $requiredPath)) {
            throw "Required experiment input not found: $requiredPath"
        }
    }
    if (-not $RunId) {
        $RunId = Get-Date -Format "yyyyMMdd_HHmmss"
    }
    $outputDir = Join-Path $OutputRoot $RunId
    New-Item -ItemType Directory -Force -Path $outputDir | Out-Null

    $arms = @(
        [pscustomobject]@{ Key = "dense"; Label = "Frozen Dense top-$TopK" },
        [pscustomobject]@{ Key = "gte_multilingual"; Label = "Frozen GTE top-$TopK" },
        [pscustomobject]@{ Key = "bge_v2_m3"; Label = "Frozen BGE-v2-m3 top-$TopK" }
    )
    if ($IncludeQwen) {
        $arms += [pscustomobject]@{
            Key = "qwen3_reranker_06b"
            Label = "Frozen Qwen3-Reranker-0.6B top-$TopK"
        }
    }

    $runConfig = [ordered]@{
        run_id = $RunId
        rankings_zip = $RankingsZip
        tasks = $Tasks
        manifest = $Manifest
        data_root = $DataRoot
        dataset_version = $DatasetVersion
        arms = @($arms | ForEach-Object { $_.Key })
        top_k = $TopK
        model = "qwen/qwen3.5-9b"
        prompt_version = "v2_cot"
        thinking = $false
        limit = $Limit
        workers = $Workers
        protocol = "frozen_reranker_ablation_v1"
    }
    $runConfigJson = $runConfig | ConvertTo-Json -Depth 5
    $runConfigPath = Join-Path $outputDir "experiment_config.json"
    if (Test-Path -LiteralPath $runConfigPath) {
        $existingConfig = Get-Content -LiteralPath $runConfigPath -Raw
        if ($existingConfig.Trim() -ne $runConfigJson.Trim()) {
            throw "RunId '$RunId' already exists with a different experiment config."
        }
    }
    else {
        [System.IO.File]::WriteAllText($runConfigPath, $runConfigJson)
    }

    $env:MLA_LLM_PROVIDER = "openrouter"
    $env:MLA_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
    $env:MLA_OPENROUTER_MODEL_NAME = "qwen/qwen3.5-9b"
    $env:MLA_DATA_ROOT = $DataRoot
    $env:MLA_TEXT_ONLY = "false"
    $env:MLA_INCLUDE_QUESTION_TEXT_WITH_IMAGES = "false"
    $env:MLA_ENABLE_THINKING = "false"
    $env:MLA_PROMPT_VERSION = "v2_cot"
    $env:MLA_CONCURRENCY = [string]$Workers
    $env:MLA_RETRIEVAL_TOP_K = [string]$TopK
    $env:MLA_RETRIEVAL_CONTEXT_ORDER = "score"

    Write-Host "`n=== Validate archive and freeze top-$TopK contexts ==="
    $prepareArgs = @(
        "-m", "mla_baseline.reranker_accuracy_experiment",
        "prepare",
        "--tasks", $Tasks,
        "--rankings-zip", $RankingsZip,
        "--output-dir", $outputDir,
        "--top-k", [string]$TopK,
        "--max-text-chars", "6000"
    )
    if ($IncludeQwen) {
        $prepareArgs += "--include-qwen"
    }
    & $python @prepareArgs | Out-Host
    if ($LASTEXITCODE -ne 0) { throw "Reranker archive preparation failed." }

    $completedRuns = @()
    foreach ($arm in $arms) {
        $contexts = Join-Path $outputDir "contexts_$($arm.Key).jsonl"
        $raw = Join-Path $outputDir "$($arm.Key)_raw.jsonl"
        $judgeInput = Join-Path $outputDir "$($arm.Key)_judge_input.jsonl"
        $judge = Join-Path $outputDir "$($arm.Key)_judge.jsonl"
        $judgeAudit = Join-Path $outputDir "$($arm.Key)_judge_completion.json"

        Write-Host "`n=== Agent: $($arm.Label) ==="
        $runnerArgs = @(
            "-m", "mla_baseline.reranker_accuracy_experiment",
            "run",
            "--tasks", $Tasks,
            "--contexts", $contexts,
            "--arm", $arm.Key,
            "--output", $raw,
            "--retry-errors"
        )
        if ($Limit -gt 0) {
            $runnerArgs += @("--limit", [string]$Limit)
        }
        & $python @runnerArgs | Out-Host
        if ($LASTEXITCODE -ne 0) { throw "Agent run failed: $($arm.Key)" }

        Write-Host "`n=== Judge: $($arm.Label) ==="
        & $python -m vlm_judge.cli prepare-image-judge-input `
            --manifest $Manifest `
            --results $raw `
            --data-root $DataRoot `
            --output $judgeInput | Out-Host
        if ($LASTEXITCODE -ne 0) { throw "Judge input failed: $($arm.Key)" }

        & $python -m vlm_judge.cli run-judge `
            --input $judgeInput `
            --output $judge `
            --base-url "https://openrouter.ai/api/v1" `
            --model "qwen/qwen3.5-9b" `
            --api-key-env OPENROUTER_API_KEY `
            --provider openrouter `
            --image-mode data_url `
            --disable-thinking `
            --workers $Workers | Out-Host
        if ($LASTEXITCODE -ne 0) { throw "Judge failed: $($arm.Key)" }

        & $python -m vlm_judge.cli verify-judge-output `
            --expected $judgeInput `
            --judge $judge `
            --output $judgeAudit | Out-Host
        if ($LASTEXITCODE -ne 0) {
            throw "Judge output is incomplete or invalid: $($arm.Key). See $judgeAudit"
        }

        $analyticsRunKey = "reranker_accuracy_$($RunId)_$($arm.Key)"
        & $python apps/vlm-analytics/main.py `
            --db $AnalyticsDb `
            --import-run-key $analyticsRunKey `
            --display-name $arm.Label `
            --raw $raw `
            --judge $judge `
            --manifest $Manifest `
            --dataset-version $DatasetVersion | Out-Host
        if ($LASTEXITCODE -ne 0) { throw "Analytics import failed: $($arm.Key)" }

        $completedRuns += [pscustomobject]@{
            Key = $arm.Key
            Judge = $judge
        }
    }

    Write-Host "`n=== Accuracy and paired changes versus Dense ==="
    $summaryPath = Join-Path $outputDir "experiment_summary.json"
    $summaryArgs = @(
        "-m", "mla_baseline.reranker_accuracy_experiment",
        "summarize",
        "--preparation-manifest", (Join-Path $outputDir "preparation_manifest.json"),
        "--output", $summaryPath
    )
    foreach ($run in $completedRuns) {
        $summaryArgs += @("--judge", "$($run.Key)=$($run.Judge)")
    }
    & $python @summaryArgs | Out-Host
    if ($LASTEXITCODE -ne 0) { throw "Accuracy summary failed." }

    Write-Host "`nSummary: $summaryPath"
    Write-Host "Analytics database: $AnalyticsDb"
    Write-Host "Run artifacts: $outputDir"
}
finally {
    Pop-Location
    foreach ($name in $managedEnvironment) {
        [System.Environment]::SetEnvironmentVariable(
            $name,
            $previousEnvironment[$name],
            "Process"
        )
    }
}
