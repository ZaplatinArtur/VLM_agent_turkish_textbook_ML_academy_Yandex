param(
    [string]$Tasks = "outputs/validation_merged_20260723/validation_image_tasks.jsonl",
    [string]$Manifest = "outputs/validation_merged_20260723/validation_manifest.jsonl",
    [string]$DataRoot = "outputs/validation_merged_20260723",
    [string]$OutputRoot = "results/openrouter_routed_experiment",
    [string]$RunId = "",
    [string]$AnalyticsDb = "apps/vlm-analytics/vlm_analytics.db",
    [string]$DatasetVersion = "validation_images_198",
    [string]$NoRetrievalSubjects = "Math",
    [int]$Limit = 10,
    [int]$Workers = 1,
    [switch]$SkipIndex
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repoRoot ".venv/Scripts/python.exe"

function Resolve-RepoPath {
    param([string]$Path)

    if ([System.IO.Path]::IsPathRooted($Path)) {
        return [System.IO.Path]::GetFullPath($Path)
    }
    return [System.IO.Path]::GetFullPath((Join-Path $repoRoot $Path))
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

    $runConfig = [ordered]@{
        run_id = $RunId
        tasks = $Tasks
        manifest = $Manifest
        data_root = $DataRoot
        dataset_version = $DatasetVersion
        model = "qwen/qwen3.5-9b"
        prompt_version = "v2_cot"
        thinking = $false
        no_retrieval_subjects = $NoRetrievalSubjects
        limit = $Limit
        workers = $Workers
    }
    $runConfigJson = $runConfig | ConvertTo-Json -Depth 4
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
    $env:MLA_RAG_NO_RETRIEVAL_SUBJECTS = $NoRetrievalSubjects

    if (-not $SkipIndex) {
        & $python -m retrieve.build_index `
            --sample-query "dikdörtgen alan formülü" `
            --k 3
        if ($LASTEXITCODE -ne 0) { throw "Retrieval index check failed." }
    }

    $experiments = @(
        @{ Key = "b0_no_tools"; Name = "E0 Без тулов" },
        @{ Key = "agent_rag"; Name = "E3 Image-first checked RAG" },
        @{ Key = "agent_rag_routed"; Name = "E4 Routed image-first RAG" }
    )

    foreach ($experiment in $experiments) {
        $raw = Join-Path $outputDir "$($experiment.Key)_raw.jsonl"
        $judgeInput = Join-Path $outputDir "$($experiment.Key)_judge_input.jsonl"
        $judge = Join-Path $outputDir "$($experiment.Key)_judge.jsonl"
        $judgeAudit = Join-Path $outputDir "$($experiment.Key)_judge_completion.json"

        if ($experiment.Key -eq "agent_rag_routed") {
            & $python -m mla_baseline.compose_routed `
                --tasks $Tasks `
                --no-tools (Join-Path $outputDir "b0_no_tools_raw.jsonl") `
                --rag (Join-Path $outputDir "agent_rag_raw.jsonl") `
                --output $raw `
                --no-retrieval-subjects $NoRetrievalSubjects
            if ($LASTEXITCODE -ne 0) { throw "Routed composition failed." }
        }
        else {
            $runnerArgs = @(
                "-m", "mla_baseline.runner",
                "--tasks", $Tasks,
                "--condition", $experiment.Key,
                "--out", $raw,
                "--retry-errors"
            )
            if ($Limit -gt 0) {
                $runnerArgs += @("--limit", [string]$Limit)
            }
            & $python @runnerArgs
            if ($LASTEXITCODE -ne 0) { throw "Agent run failed: $($experiment.Key)" }
        }

        & $python -m vlm_judge.cli prepare-image-judge-input `
            --manifest $Manifest `
            --results $raw `
            --data-root $DataRoot `
            --output $judgeInput
        if ($LASTEXITCODE -ne 0) { throw "Judge input failed: $($experiment.Key)" }

        & $python -m vlm_judge.cli run-judge `
            --input $judgeInput `
            --output $judge `
            --base-url "https://openrouter.ai/api/v1" `
            --model "qwen/qwen3.5-9b" `
            --api-key-env OPENROUTER_API_KEY `
            --provider openrouter `
            --image-mode data_url `
            --disable-thinking `
            --workers $Workers
        if ($LASTEXITCODE -ne 0) { throw "Judge failed: $($experiment.Key)" }

        & $python -m vlm_judge.cli verify-judge-output `
            --expected $judgeInput `
            --judge $judge `
            --output $judgeAudit
        if ($LASTEXITCODE -ne 0) {
            throw "Judge output is incomplete or invalid: $($experiment.Key). See $judgeAudit"
        }

        & $python apps/vlm-analytics/main.py `
            --db $AnalyticsDb `
            --import-run-key $experiment.Key `
            --display-name $experiment.Name `
            --raw $raw `
            --judge $judge `
            --manifest $Manifest `
            --dataset-version $DatasetVersion
        if ($LASTEXITCODE -ne 0) { throw "Analytics import failed: $($experiment.Key)" }
    }

    Write-Host "`nPaired metrics against E0:"
    & $python apps/vlm-analytics/main.py --db $AnalyticsDb --paired-summary
    if ($LASTEXITCODE -ne 0) { throw "Paired analytics failed." }
    Write-Host "`nAnalytics database: $AnalyticsDb"
    Write-Host "Run artifacts: $outputDir"
}
finally {
    Pop-Location
}
