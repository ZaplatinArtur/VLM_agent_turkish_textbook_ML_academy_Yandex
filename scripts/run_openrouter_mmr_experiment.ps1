param(
    [string]$Tasks = "outputs/validation_merged_20260723/validation_image_tasks.jsonl",
    [string]$Manifest = "outputs/validation_merged_20260723/validation_manifest.jsonl",
    [string]$DataRoot = "outputs/validation_merged_20260723",
    [string]$OutputRoot = "results/openrouter_mmr_experiment",
    [string]$RunId = "",
    [string]$AnalyticsDb = "apps/vlm-analytics/vlm_analytics.db",
    [string]$DatasetVersion = "validation_images_198",
    [int]$Limit = 10,
    [int]$Workers = 1,
    [switch]$SkipIndex
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repoRoot ".venv/Scripts/python.exe"
$culture = [System.Globalization.CultureInfo]::InvariantCulture
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
    "MLA_RETRIEVAL_FETCH_K",
    "MLA_RETRIEVAL_MMR_ENABLED",
    "MLA_RETRIEVAL_MMR_LAMBDA",
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

function Get-JudgeMetrics {
    param([string]$Path)

    $rows = @(
        Get-Content -LiteralPath $Path |
            Where-Object { $_.Trim() } |
            ForEach-Object { $_ | ConvertFrom-Json }
    )
    $valid = @(
        $rows | Where-Object {
            $null -ne $_.verdict -and
            $null -ne $_.verdict.strict_correct
        }
    )
    $correct = @($valid | Where-Object { $_.verdict.strict_correct -eq $true }).Count
    $total = $valid.Count
    return [pscustomobject]@{
        Correct = $correct
        Total = $total
        Accuracy = if ($total) {
            [math]::Round(100.0 * $correct / $total, 2)
        }
        else { 0.0 }
    }
}

function Get-PairedMetrics {
    param(
        [string]$Baseline,
        [string]$Candidate
    )

    $baselineByTask = @{}
    Get-Content -LiteralPath $Baseline |
        Where-Object { $_.Trim() } |
        ForEach-Object {
            $row = $_ | ConvertFrom-Json
            if ($null -ne $row.verdict -and $null -ne $row.verdict.strict_correct) {
                $baselineByTask[[string]$row.task_id] = [bool]$row.verdict.strict_correct
            }
        }
    $paired = 0
    $fixed = 0
    $regressed = 0
    Get-Content -LiteralPath $Candidate |
        Where-Object { $_.Trim() } |
        ForEach-Object {
            $row = $_ | ConvertFrom-Json
            $taskId = [string]$row.task_id
            if (
                $null -ne $row.verdict -and
                $null -ne $row.verdict.strict_correct -and
                $baselineByTask.ContainsKey($taskId)
            ) {
                $paired += 1
                $before = [bool]$baselineByTask[$taskId]
                $after = [bool]$row.verdict.strict_correct
                if (-not $before -and $after) { $fixed += 1 }
                if ($before -and -not $after) { $regressed += 1 }
            }
        }
    return [ordered]@{
        paired = $paired
        fixed = $fixed
        regressed = $regressed
        net_fixes = $fixed - $regressed
    }
}

function Invoke-ExperimentArm {
    param([pscustomobject]$Arm)

    $lambdaText = if ($null -eq $Arm.Lambda) {
        "0.5"
    }
    else {
        ([double]$Arm.Lambda).ToString("0.0", $culture)
    }
    $env:MLA_RETRIEVAL_MMR_ENABLED = ([bool]$Arm.MmrEnabled).ToString().ToLowerInvariant()
    $env:MLA_RETRIEVAL_MMR_LAMBDA = $lambdaText
    $env:MLA_RETRIEVAL_CONTEXT_ORDER = [string]$Arm.ContextOrder

    $raw = Join-Path $outputDir "$($Arm.Key)_raw.jsonl"
    $judgeInput = Join-Path $outputDir "$($Arm.Key)_judge_input.jsonl"
    $judge = Join-Path $outputDir "$($Arm.Key)_judge.jsonl"
    $judgeAudit = Join-Path $outputDir "$($Arm.Key)_judge_completion.json"

    Write-Host "`n=== $($Arm.Label) ==="
    $runnerArgs = @(
        "-m", "mla_baseline.runner",
        "--tasks", $Tasks,
        "--condition", "agent_rag",
        "--out", $raw,
        "--retry-errors"
    )
    if ($Limit -gt 0) {
        $runnerArgs += @("--limit", [string]$Limit)
    }
    & $python @runnerArgs | Out-Host
    if ($LASTEXITCODE -ne 0) { throw "Agent run failed: $($Arm.Key)" }

    & $python -m vlm_judge.cli prepare-image-judge-input `
        --manifest $Manifest `
        --results $raw `
        --data-root $DataRoot `
        --output $judgeInput | Out-Host
    if ($LASTEXITCODE -ne 0) { throw "Judge input failed: $($Arm.Key)" }

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
    if ($LASTEXITCODE -ne 0) { throw "Judge failed: $($Arm.Key)" }

    & $python -m vlm_judge.cli verify-judge-output `
        --expected $judgeInput `
        --judge $judge `
        --output $judgeAudit | Out-Host
    if ($LASTEXITCODE -ne 0) {
        throw "Judge output is incomplete or invalid: $($Arm.Key). See $judgeAudit"
    }

    & $python apps/vlm-analytics/main.py `
        --db $AnalyticsDb `
        --import-run-key $Arm.Key `
        --display-name $Arm.Label `
        --raw $raw `
        --judge $judge `
        --manifest $Manifest `
        --dataset-version $effectiveDatasetVersion | Out-Host
    if ($LASTEXITCODE -ne 0) { throw "Analytics import failed: $($Arm.Key)" }

    $metrics = Get-JudgeMetrics $judge
    return [pscustomobject]@{
        Key = $Arm.Key
        Label = $Arm.Label
        MmrEnabled = [bool]$Arm.MmrEnabled
        Lambda = $Arm.Lambda
        ContextOrder = [string]$Arm.ContextOrder
        Raw = $raw
        Judge = $judge
        Correct = $metrics.Correct
        Total = $metrics.Total
        Accuracy = $metrics.Accuracy
    }
}

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
if (-not $env:OPENROUTER_API_KEY) {
    throw "Set OPENROUTER_API_KEY in the current PowerShell session."
}

Push-Location $repoRoot
try {
    foreach ($requiredPath in @($Tasks, $Manifest, $DataRoot)) {
        if (-not (Test-Path -LiteralPath $requiredPath)) {
            throw "Required experiment input not found: $requiredPath"
        }
    }
    if (-not $RunId) {
        $RunId = Get-Date -Format "yyyyMMdd_HHmmss"
    }
    $outputDir = Join-Path $OutputRoot $RunId
    New-Item -ItemType Directory -Force -Path $outputDir | Out-Null
    $effectiveDatasetVersion = if ($Limit -gt 0) {
        "$DatasetVersion.limit$Limit"
    }
    else { $DatasetVersion }

    $initialArms = @(
        [pscustomobject]@{
            Key = "mmr_r0_dense"
            Label = "R0 Dense top-5"
            MmrEnabled = $false
            Lambda = $null
            ContextOrder = "score"
        },
        [pscustomobject]@{
            Key = "mmr_r1_l03"
            Label = "R1 MMR lambda=0.3"
            MmrEnabled = $true
            Lambda = 0.3
            ContextOrder = "score"
        },
        [pscustomobject]@{
            Key = "mmr_r2_l05"
            Label = "R2 MMR lambda=0.5"
            MmrEnabled = $true
            Lambda = 0.5
            ContextOrder = "score"
        },
        [pscustomobject]@{
            Key = "mmr_r3_l07"
            Label = "R3 MMR lambda=0.7"
            MmrEnabled = $true
            Lambda = 0.7
            ContextOrder = "score"
        }
    )
    $runConfig = [ordered]@{
        run_id = $RunId
        tasks = $Tasks
        manifest = $Manifest
        data_root = $DataRoot
        dataset_version = $effectiveDatasetVersion
        model = "qwen/qwen3.5-9b"
        prompt_version = "v2_cot"
        thinking = $false
        retrieval_top_k = 5
        retrieval_fetch_k = 20
        arms = $initialArms
        limit = $Limit
        workers = $Workers
    }
    $runConfigJson = $runConfig | ConvertTo-Json -Depth 6
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
    $env:MLA_RETRIEVAL_TOP_K = "5"
    $env:MLA_RETRIEVAL_FETCH_K = "20"

    if (-not $SkipIndex) {
        & $python -m retrieve.build_index `
            --sample-query "dikdörtgen alan formülü" `
            --k 3 | Out-Host
        if ($LASTEXITCODE -ne 0) { throw "Retrieval index check failed." }
    }

    $runs = @()
    foreach ($arm in $initialArms) {
        $runs += Invoke-ExperimentArm $arm
    }

    $bestMmr = @(
        $runs |
            Where-Object { $_.MmrEnabled } |
            Sort-Object `
                @{ Expression = "Accuracy"; Descending = $true }, `
                @{ Expression = { [math]::Abs([double]$_.Lambda - 0.5) }; Ascending = $true }
    )[0]
    $lambdaText = ([double]$bestMmr.Lambda).ToString("0.0", $culture)
    $lambdaTag = $lambdaText.Replace(".", "")
    $edgeArm = [pscustomobject]@{
        Key = "mmr_order_edge_l$lambdaTag"
        Label = "MMR lambda=$lambdaText edge-order"
        MmrEnabled = $true
        Lambda = [double]$bestMmr.Lambda
        ContextOrder = "edge"
    }
    $edgeRun = Invoke-ExperimentArm $edgeArm
    $runs += $edgeRun

    $r0 = @($runs | Where-Object { $_.Key -eq "mmr_r0_dense" })[0]
    $vsR0 = @()
    foreach ($run in $runs | Where-Object { $_.Key -ne $r0.Key }) {
        $vsR0 += [ordered]@{
            candidate = $run.Key
            comparison = Get-PairedMetrics $r0.Judge $run.Judge
        }
    }
    $summary = [ordered]@{
        run_id = $RunId
        dataset_version = $effectiveDatasetVersion
        best_mmr = [ordered]@{
            key = $bestMmr.Key
            lambda = $bestMmr.Lambda
            accuracy = $bestMmr.Accuracy
        }
        arms = @(
            $runs | ForEach-Object {
                [ordered]@{
                    key = $_.Key
                    label = $_.Label
                    mmr_enabled = $_.MmrEnabled
                    lambda = $_.Lambda
                    context_order = $_.ContextOrder
                    correct = $_.Correct
                    total = $_.Total
                    accuracy = $_.Accuracy
                }
            }
        )
        comparisons_vs_r0 = $vsR0
        order_comparison = [ordered]@{
            baseline = $bestMmr.Key
            candidate = $edgeRun.Key
            comparison = Get-PairedMetrics $bestMmr.Judge $edgeRun.Judge
        }
    }
    $summaryPath = Join-Path $outputDir "experiment_summary.json"
    [System.IO.File]::WriteAllText(
        $summaryPath,
        ($summary | ConvertTo-Json -Depth 8)
    )

    Write-Host "`n=== Accuracy ==="
    $runs |
        Select-Object Label, Correct, Total, Accuracy |
        Format-Table -AutoSize |
        Out-Host
    Write-Host "Best MMR lambda: $lambdaText"
    Write-Host "Summary: $summaryPath"
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
