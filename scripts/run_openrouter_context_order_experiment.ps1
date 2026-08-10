param(
    [string]$Tasks = "outputs/validation_merged_20260723/validation_image_tasks.jsonl",
    [string]$Manifest = "outputs/validation_merged_20260723/validation_manifest.jsonl",
    [string]$DataRoot = "outputs/validation_merged_20260723",
    [string]$OutputRoot = "results/openrouter_context_order_experiment",
    [string]$RunId = "",
    [string]$SeedResults = "",
    [string]$AnalyticsDb = "apps/vlm-analytics/vlm_analytics.db",
    [string]$DatasetVersion = "validation_images_198",
    [int]$Limit = 10,
    [int]$Workers = 1,
    [switch]$SkipIndex
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
    "MLA_RETRIEVAL_FETCH_K",
    "MLA_RETRIEVAL_MMR_ENABLED",
    "MLA_RETRIEVAL_MMR_LAMBDA",
    "MLA_RETRIEVAL_CONTEXT_ORDER",
    "MLA_RETRIEVAL_MAX_CALLS"
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

function Read-JsonlByTask {
    param([string]$Path)

    $rows = @{}
    Get-Content -LiteralPath $Path |
        Where-Object { $_.Trim() } |
        ForEach-Object {
            $row = $_ | ConvertFrom-Json
            $taskId = [string]$row.task_id
            if (-not $taskId) {
                throw "JSONL row has no task_id: $Path"
            }
            if ($rows.ContainsKey($taskId)) {
                throw "Duplicate task_id '$taskId' in $Path"
            }
            $rows[$taskId] = $row
        }
    return $rows
}

function Get-EffectiveRetrievalCalls {
    param([object]$Row)

    return @(
        @($Row.tool_calls) | Where-Object {
            [string]$_.tool -eq "search_textbooks" -and -not $_.error
        }
    )
}

function Normalize-Query {
    param([object]$Call)

    if ($null -eq $Call -or $null -eq $Call.args) { return "" }
    $query = [string]$Call.args.query
    return (($query.ToLowerInvariant() -replace "[^\p{L}\p{N}_]+", " ").Trim())
}

function Get-RetrievalEquivalence {
    param(
        [string]$ScorePath,
        [string]$EdgePath
    )

    $scoreByTask = Read-JsonlByTask $ScorePath
    $edgeByTask = Read-JsonlByTask $EdgePath
    $taskIds = @(
        @($scoreByTask.Keys) + @($edgeByTask.Keys) |
            Sort-Object -Unique
    )
    $mismatches = @()
    $matchedContexts = 0
    $noContext = 0
    $orderChanged = 0

    foreach ($taskId in $taskIds) {
        if (-not $scoreByTask.ContainsKey($taskId)) {
            $mismatches += [ordered]@{
                task_id = $taskId
                reason = "missing_in_score_arm"
            }
            continue
        }
        if (-not $edgeByTask.ContainsKey($taskId)) {
            $mismatches += [ordered]@{
                task_id = $taskId
                reason = "missing_in_edge_arm"
            }
            continue
        }

        $scoreCalls = @(Get-EffectiveRetrievalCalls $scoreByTask[$taskId])
        $edgeCalls = @(Get-EffectiveRetrievalCalls $edgeByTask[$taskId])
        if ($scoreCalls.Count -ne $edgeCalls.Count) {
            $mismatches += [ordered]@{
                task_id = $taskId
                reason = "retrieval_call_count"
                score_calls = $scoreCalls.Count
                edge_calls = $edgeCalls.Count
            }
            continue
        }
        if ($scoreCalls.Count -eq 0) {
            $noContext += 1
            continue
        }
        if ($scoreCalls.Count -ne 1) {
            $mismatches += [ordered]@{
                task_id = $taskId
                reason = "expected_one_retrieval_call"
                calls = $scoreCalls.Count
            }
            continue
        }

        $scoreCall = $scoreCalls[0]
        $edgeCall = $edgeCalls[0]
        $scoreQuery = Normalize-Query $scoreCall
        $edgeQuery = Normalize-Query $edgeCall
        $scoreIds = @($scoreCall.returned_chunk_ids | ForEach-Object { [string]$_ })
        $edgeIds = @($edgeCall.returned_chunk_ids | ForEach-Object { [string]$_ })
        $scoreSet = @($scoreIds | Sort-Object)
        $edgeSet = @($edgeIds | Sort-Object)
        $sameSet = (
            $scoreSet.Count -eq $edgeSet.Count -and
            @(Compare-Object $scoreSet $edgeSet).Count -eq 0
        )

        if ($scoreQuery -ne $edgeQuery -or -not $sameSet) {
            $mismatches += [ordered]@{
                task_id = $taskId
                reason = if ($scoreQuery -ne $edgeQuery) {
                    "retrieval_query"
                }
                else { "returned_chunk_set" }
                score_query = $scoreQuery
                edge_query = $edgeQuery
                score_chunk_ids = $scoreIds
                edge_chunk_ids = $edgeIds
            }
            continue
        }

        if ($scoreIds.Count -eq 0) {
            $noContext += 1
            continue
        }
        $matchedContexts += 1
        if (($scoreIds | ConvertTo-Json -Compress) -ne ($edgeIds | ConvertTo-Json -Compress)) {
            $orderChanged += 1
        }
    }

    return [ordered]@{
        valid = ($mismatches.Count -eq 0 -and $orderChanged -gt 0)
        tasks = $taskIds.Count
        matched_context_tasks = $matchedContexts
        no_context_tasks = $noContext
        order_changed_tasks = $orderChanged
        mismatches_count = $mismatches.Count
        mismatches = $mismatches
    }
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

function Invoke-Preparation {
    param(
        [string]$SeedRaw,
        [string]$FrozenContexts,
        [string]$AffectedTasks,
        [bool]$GenerateSeed
    )

    $env:MLA_RETRIEVAL_CONTEXT_ORDER = "score"
    Write-Host "`n=== Prepare one frozen retrieval context per task ==="
    if ($GenerateSeed) {
        $runnerArgs = @(
            "-m", "mla_baseline.runner",
            "--tasks", $Tasks,
            "--condition", "agent_rag",
            "--out", $SeedRaw,
            "--retry-errors"
        )
        if ($Limit -gt 0) {
            $runnerArgs += @("--limit", [string]$Limit)
        }
        & $python @runnerArgs | Out-Host
        if ($LASTEXITCODE -ne 0) { throw "Retrieval preparation run failed." }
    }

    $seedCount = @(
        Get-Content -LiteralPath $SeedRaw | Where-Object { $_.Trim() }
    ).Count
    $expectedSeedCount = if ($Limit -gt 0) {
        $Limit
    }
    else {
        @(Get-Content -LiteralPath $Tasks | Where-Object { $_.Trim() }).Count
    }
    if ($seedCount -ne $expectedSeedCount) {
        throw "SeedResults has $seedCount rows; expected $expectedSeedCount."
    }

    & $python -m mla_baseline.context_order_experiment prepare `
        --tasks $Tasks `
        --seed-results $SeedRaw `
        --output $FrozenContexts `
        --affected-tasks $AffectedTasks | Out-Host
    if ($LASTEXITCODE -ne 0) { throw "Freezing retrieval contexts failed." }
}

function Invoke-FrozenArm {
    param(
        [pscustomobject]$Arm,
        [string]$FrozenContexts,
        [string]$AffectedTasks
    )

    $raw = Join-Path $outputDir "$($Arm.Key)_raw.jsonl"
    Write-Host "`n=== Generate from frozen context: $($Arm.Label) ==="
    & $python -m mla_baseline.context_order_experiment run `
        --tasks $AffectedTasks `
        --contexts $FrozenContexts `
        --order $Arm.ContextOrder `
        --output $raw `
        --retry-errors | Out-Host
    if ($LASTEXITCODE -ne 0) { throw "Frozen agent run failed: $($Arm.Key)" }

    return [pscustomobject]@{
        Key = $Arm.Key
        Label = $Arm.Label
        ContextOrder = $Arm.ContextOrder
        Raw = $raw
    }
}

function Invoke-JudgeAndImport {
    param([pscustomobject]$Run)

    $judgeInput = Join-Path $outputDir "$($Run.Key)_judge_input.jsonl"
    $judge = Join-Path $outputDir "$($Run.Key)_judge.jsonl"
    $judgeAudit = Join-Path $outputDir "$($Run.Key)_judge_completion.json"

    Write-Host "`n=== Judge: $($Run.Label) ==="
    & $python -m vlm_judge.cli prepare-image-judge-input `
        --manifest $Manifest `
        --results $Run.Raw `
        --data-root $DataRoot `
        --output $judgeInput | Out-Host
    if ($LASTEXITCODE -ne 0) { throw "Judge input failed: $($Run.Key)" }

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
    if ($LASTEXITCODE -ne 0) { throw "Judge failed: $($Run.Key)" }

    & $python -m vlm_judge.cli verify-judge-output `
        --expected $judgeInput `
        --judge $judge `
        --output $judgeAudit | Out-Host
    if ($LASTEXITCODE -ne 0) {
        throw "Judge output is incomplete or invalid: $($Run.Key). See $judgeAudit"
    }

    $analyticsRunKey = "context_order_$($RunId)_$($Run.Key)"
    & $python apps/vlm-analytics/main.py `
        --db $AnalyticsDb `
        --import-run-key $analyticsRunKey `
        --display-name $Run.Label `
        --raw $Run.Raw `
        --judge $judge `
        --manifest $Manifest `
        --dataset-version $effectiveDatasetVersion | Out-Host
    if ($LASTEXITCODE -ne 0) { throw "Analytics import failed: $($Run.Key)" }

    $metrics = Get-JudgeMetrics $judge
    return [pscustomobject]@{
        Key = $Run.Key
        Label = $Run.Label
        ContextOrder = $Run.ContextOrder
        Raw = $Run.Raw
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
if ($SeedResults) {
    $SeedResults = Resolve-RepoPath $SeedResults
}

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
    if ($SeedResults -and -not (Test-Path -LiteralPath $SeedResults)) {
        throw "Seed results not found: $SeedResults"
    }
    if (-not $RunId) {
        $RunId = Get-Date -Format "yyyyMMdd_HHmmss"
    }
    if ($RunId -notmatch "^[A-Za-z0-9._-]+$") {
        throw "RunId may contain only letters, digits, dot, underscore, and dash."
    }
    $outputDir = Join-Path $OutputRoot $RunId
    New-Item -ItemType Directory -Force -Path $outputDir | Out-Null
    $effectiveDatasetVersion = if ($Limit -gt 0) {
        "$DatasetVersion.context_order_affected.limit$Limit"
    }
    else { "$DatasetVersion.context_order_affected" }

    $arms = @(
        [pscustomobject]@{
            Key = "score_order"
            Label = "Dense top-5: score order"
            ContextOrder = "score"
        },
        [pscustomobject]@{
            Key = "edge_order"
            Label = "Dense top-5: strongest chunks at edges"
            ContextOrder = "edge"
        }
    )
    $runConfig = [ordered]@{
        run_id = $RunId
        tasks = $Tasks
        manifest = $Manifest
        data_root = $DataRoot
        seed_results = $SeedResults
        dataset_version = $effectiveDatasetVersion
        model = "qwen/qwen3.5-9b"
        prompt_version = "v2_cot"
        thinking = $false
        retrieval_top_k = 5
        retrieval_fetch_k = 20
        retrieval_mmr_enabled = $false
        retrieval_max_calls = 1
        context_source = "single_frozen_preparation"
        evaluation_subset = "tasks_where_context_order_changes"
        experiment_version = "context_order_v2"
        arms = $arms
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
    $env:MLA_RETRIEVAL_MMR_ENABLED = "false"
    $env:MLA_RETRIEVAL_MMR_LAMBDA = "0.5"
    $env:MLA_RETRIEVAL_MAX_CALLS = "1"

    if (-not $SkipIndex) {
        & $python -m retrieve.build_index `
            --sample-query "dikdörtgen alan formülü" `
            --k 3 | Out-Host
        if ($LASTEXITCODE -ne 0) { throw "Retrieval index check failed." }
    }

    $seedRaw = if ($SeedResults) {
        $SeedResults
    }
    else { Join-Path $outputDir "preparation_raw.jsonl" }
    $frozenContexts = Join-Path $outputDir "frozen_contexts.jsonl"
    $affectedTasks = Join-Path $outputDir "affected_tasks.jsonl"
    Invoke-Preparation `
        -SeedRaw $seedRaw `
        -FrozenContexts $frozenContexts `
        -AffectedTasks $affectedTasks `
        -GenerateSeed (-not [bool]$SeedResults)

    $affectedCount = @(
        Get-Content -LiteralPath $affectedTasks | Where-Object { $_.Trim() }
    ).Count
    Write-Host "Order-sensitive tasks selected: $affectedCount"

    $generatedRuns = @()
    foreach ($arm in $arms) {
        $generatedRuns += Invoke-FrozenArm `
            -Arm $arm `
            -FrozenContexts $frozenContexts `
            -AffectedTasks $affectedTasks
    }

    $scoreGenerated = @($generatedRuns | Where-Object { $_.Key -eq "score_order" })[0]
    $edgeGenerated = @($generatedRuns | Where-Object { $_.Key -eq "edge_order" })[0]
    $equivalence = Get-RetrievalEquivalence `
        -ScorePath $scoreGenerated.Raw `
        -EdgePath $edgeGenerated.Raw
    $equivalencePath = Join-Path $outputDir "retrieval_equivalence.json"
    [System.IO.File]::WriteAllText(
        $equivalencePath,
        ($equivalence | ConvertTo-Json -Depth 8)
    )
    if (-not $equivalence.valid) {
        throw (
            "Order-only audit failed; judge was not started. " +
            "See $equivalencePath and rerun with a new RunId after resolving mismatches."
        )
    }

    $judgedRuns = @()
    foreach ($run in $generatedRuns) {
        $judgedRuns += Invoke-JudgeAndImport $run
    }
    $scoreRun = @($judgedRuns | Where-Object { $_.Key -eq "score_order" })[0]
    $edgeRun = @($judgedRuns | Where-Object { $_.Key -eq "edge_order" })[0]
    $paired = Get-PairedMetrics $scoreRun.Judge $edgeRun.Judge

    $summary = [ordered]@{
        run_id = $RunId
        dataset_version = $effectiveDatasetVersion
        preparation = [ordered]@{
            seed_raw = $seedRaw
            frozen_contexts = $frozenContexts
            affected_tasks = $affectedTasks
            affected_count = $affectedCount
        }
        retrieval_equivalence = $equivalence
        arms = @(
            $judgedRuns | ForEach-Object {
                [ordered]@{
                    key = $_.Key
                    label = $_.Label
                    context_order = $_.ContextOrder
                    correct = $_.Correct
                    total = $_.Total
                    accuracy_percent = $_.Accuracy
                }
            }
        )
        order_comparison = [ordered]@{
            baseline = $scoreRun.Key
            candidate = $edgeRun.Key
            paired = $paired
            accuracy_delta_points = [math]::Round(
                $edgeRun.Accuracy - $scoreRun.Accuracy,
                2
            )
        }
    }
    $summaryPath = Join-Path $outputDir "experiment_summary.json"
    [System.IO.File]::WriteAllText(
        $summaryPath,
        ($summary | ConvertTo-Json -Depth 10)
    )

    Write-Host "`n=== Accuracy ==="
    $judgedRuns |
        Select-Object Label, Correct, Total, Accuracy |
        Format-Table -AutoSize |
        Out-Host
    Write-Host (
        "Paired: fixed=$($paired.fixed), regressed=$($paired.regressed), " +
        "net=$($paired.net_fixes)"
    )
    Write-Host "Retrieval audit: $equivalencePath"
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
