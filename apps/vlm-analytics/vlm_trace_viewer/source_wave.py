from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from .adapter import ArtifactError, discover_artifact_root
from .model import TaskTrace, TraceDataset, split_solution_steps
from .selector_wave import MODEL, SelectorWaveSummary


WAVE = Path("experiments/maxim_9b_source_expansion_wave_v1_1/final_wave")
FREEZE = WAVE / "FINAL_WAVE_FREEZE.json"
AMENDMENT = WAVE / "INDEPENDENT_AUDIT_AMENDMENT.json"
COMPLETION = WAVE / "execution/WAVE_COMPLETION.json"
OFFICIAL_METRICS = WAVE / "execution/outputs/official/official16/metrics.json"

PINNED_SHA256 = {
    FREEZE: "efcf854f011357e35f48bd86d934521ccaf252343988e04023403cced5c34a5c",
    AMENDMENT: "4d0720122a0a55d8c560f895ad3ab8b4bc1b24b9779e0132d24313cc9f6d6749",
    COMPLETION: "318be80043ffac433a9482d0fc2bde8acf99d1fe1c3b8ed44dfffcee8a36506e",
    OFFICIAL_METRICS: "969cece754bcf3eadd2fded4b519d9974e69a64c27e906e54e5ab9ecae470d8e",
}

OFFICIAL_FIX_TASK_IDS = (
    "val_0048",
    "val_0050",
    "val_0051",
    "val_0054",
    "val_0055",
    "val_0056",
    "val_0057",
    "val_0058",
    "val_0182",
)
OFFICIAL_TARGET_TASK_IDS = (
    "val_0054",
    "val_0055",
    "val_0056",
    "val_0057",
    "val_0058",
    "val_0181",
    "val_0182",
    "val_0183",
    "val_0184",
    "val_0185",
    "val_0048",
    "val_0049",
    "val_0050",
    "val_0051",
    "val_0052",
    "val_0053",
)
RESEARCH_ARM_IDS = (
    "bs11_8_research",
    "research_bs24",
    "fenomen12_research",
    "research_fenomen28",
    "research_all36",
)


@dataclass(frozen=True)
class SourceWaveTask:
    task_id: str
    subject: str
    final_answer: str
    correct: bool
    score_source: str
    score_method: str
    transition: str
    answer_changed_vs_240: bool
    correctness_fix_vs_240: bool
    solver_row: dict[str, Any]


@dataclass(frozen=True)
class ResearchWaveResult:
    arm_id: str
    correct: int
    rows: int
    accuracy: float
    classification: str
    eligible_for_official_headline: bool
    license_status: str
    metrics_sha256: str


@dataclass(frozen=True)
class SourceWaveSummary:
    model: str
    correct: int
    rows: int
    math_correct: int
    math_rows: int
    english_correct: int
    english_rows: int
    deterministic_correct: int
    deterministic_rows: int
    image_correct: int
    image_rows: int
    previous_correct: int
    fixes: int
    regressions: int
    fix_task_ids: tuple[str, ...]
    answer_changed_task_ids: tuple[str, ...]
    target_task_ids: tuple[str, ...]
    tasks: tuple[SourceWaveTask, ...]
    by_subject: dict[str, dict[str, Any]]
    freeze_sha256: str
    amendment_sha256: str
    completion_sha256: str
    official_metrics_sha256: str
    official_solver_sha256: str
    official_image_judge_sha256: str
    research_all36: ResearchWaveResult
    verified_files: tuple[Path, ...]

    @property
    def accuracy(self) -> float:
        return self.correct / self.rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise ArtifactError(f"cannot hash source-wave artifact {path}: {exc}") from exc
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ArtifactError(f"required source-wave artifact is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"cannot read source-wave JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ArtifactError(f"expected object in source-wave artifact: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ArtifactError(f"required source-wave artifact is missing: {path}")
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ArtifactError(f"expected object at {path}:{line_number}")
                rows.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"cannot read source-wave JSONL {path}: {exc}") from exc
    return rows


def _index(rows: Iterable[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        task_id = str(row.get("task_id") or "")
        if not task_id or task_id in result:
            raise ArtifactError(f"missing or duplicate task_id in {label}: {task_id!r}")
        result[task_id] = row
    return result


def _expect(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ArtifactError(
            f"source-wave contract mismatch for {label}: {actual!r} != {expected!r}"
        )


def _safe_relative(base: Path, value: Any, label: str) -> Path:
    text = str(value or "")
    pure = PurePosixPath(text.replace("\\", "/"))
    if (
        pure.is_absolute()
        or not pure.parts
        or ".." in pure.parts
        or pure.parts[0].endswith(":")
    ):
        raise ArtifactError(f"unsafe {label} path in source-wave completion: {text!r}")
    resolved_base = base.resolve()
    resolved = (resolved_base / Path(*pure.parts)).resolve()
    try:
        resolved.relative_to(resolved_base)
    except ValueError as exc:
        raise ArtifactError(f"{label} path escapes frozen wave: {text!r}") from exc
    return resolved


def _metric_tuple(metrics: dict[str, Any]) -> tuple[int, ...]:
    overall = metrics.get("overall") or {}
    subjects = metrics.get("by_subject") or {}
    sources = metrics.get("by_source") or {}
    math = subjects.get("Math") or {}
    english = subjects.get("English") or {}
    deterministic = sources.get("deterministic") or {}
    image = sources.get("image_judge") or {}
    return (
        int(overall.get("new_correct") or 0),
        int(overall.get("n") or 0),
        int(math.get("new_correct") or 0),
        int(math.get("n") or 0),
        int(english.get("new_correct") or 0),
        int(english.get("n") or 0),
        int(deterministic.get("new_correct") or 0),
        int(deterministic.get("n") or 0),
        int(image.get("new_correct") or 0),
        int(image.get("n") or 0),
    )


class SourceExpansionWaveAdapter:
    """Load the audited official source wave and keep research arms non-headline."""

    def __init__(self, artifact_root: Path | str | None = None):
        self.root = discover_artifact_root(artifact_root)

    def _path(self, relative: Path) -> Path:
        root = self.root.resolve()
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ArtifactError(f"source-wave path escapes artifact root: {relative}") from exc
        return path

    def _verify_static_pin(self, relative: Path) -> Path:
        path = self._path(relative)
        actual = _sha256(path)
        expected = PINNED_SHA256[relative]
        if actual != expected:
            raise ArtifactError(
                f"source-wave frozen hash mismatch for {relative}: {actual} != {expected}"
            )
        return path

    def _completion_artifact(
        self,
        completion: dict[str, Any],
        arm_id: str,
        kind: str,
    ) -> tuple[Path, str]:
        outputs = completion.get("output_artifacts") or {}
        arm = outputs.get(arm_id) or {}
        descriptor = arm.get(kind) or {}
        wave_root = self._path(WAVE)
        path = _safe_relative(wave_root, descriptor.get("path"), f"{arm_id}.{kind}")
        expected = str(descriptor.get("sha256") or "")
        if len(expected) != 64 or _sha256(path) != expected:
            raise ArtifactError(f"source-wave completion hash mismatch for {arm_id}.{kind}")
        return path, expected

    def load(self) -> SourceWaveSummary:
        freeze_path = self._verify_static_pin(FREEZE)
        amendment_path = self._verify_static_pin(AMENDMENT)
        completion_path = self._verify_static_pin(COMPLETION)
        official_metrics_path = self._verify_static_pin(OFFICIAL_METRICS)
        freeze = _read_json(freeze_path)
        amendment = _read_json(amendment_path)
        completion = _read_json(completion_path)

        _expect(
            freeze.get("schema_version"),
            "maxim-9b-source-expansion-wave-v1.1-final-freeze",
            "freeze schema",
        )
        _expect(freeze.get("official_headline_arm"), "official16", "freeze headline")
        _expect(
            tuple(freeze.get("research_evaluation_only_arm_ids") or ()),
            RESEARCH_ARM_IDS,
            "freeze research-only arms",
        )
        official_arm = (freeze.get("arms") or {}).get("official16") or {}
        _expect(
            official_arm.get("classification"),
            "official_headline_candidate",
            "official arm classification",
        )
        _expect(official_arm.get("eligible_for_official_headline"), True, "official eligibility")
        _expect(
            tuple(official_arm.get("target_task_ids") or ()),
            OFFICIAL_TARGET_TASK_IDS,
            "official target set/order",
        )
        research_arm = (freeze.get("arms") or {}).get("research_all36") or {}
        _expect(
            research_arm.get("classification"),
            "research_evaluation_only",
            "research_all36 classification",
        )
        _expect(
            research_arm.get("eligible_for_official_headline"),
            False,
            "research_all36 headline exclusion",
        )

        _expect(
            amendment.get("schema_version"),
            "maxim-9b-source-expansion-wave-v1.1-final-independent-audit-amendment",
            "audit schema",
        )
        _expect(amendment.get("wave_freeze_sha256"), PINNED_SHA256[FREEZE], "audit freeze pin")
        _expect(amendment.get("audit_status"), "pass", "audit status")
        _expect(amendment.get("evaluation_authorized"), True, "evaluation authorization")
        _expect(
            amendment.get("research_outputs_must_remain_separate"),
            True,
            "research separation authorization",
        )
        _expect(amendment.get("official_headline_arm"), "official16", "audit headline")

        _expect(
            completion.get("schema_version"),
            "maxim-9b-source-expansion-wave-v1.1-final-completion",
            "completion schema",
        )
        _expect(
            completion.get("status"),
            "all_ten_completed_outputs_hash_frozen",
            "completion status",
        )
        _expect(completion.get("wave_freeze_sha256"), PINNED_SHA256[FREEZE], "completion freeze")
        _expect(
            completion.get("independent_audit_amendment_sha256"),
            PINNED_SHA256[AMENDMENT],
            "completion amendment",
        )
        for key in (
            "all_returncodes_zero",
            "all_ten_completed_before_manifest",
            "all_ten_started_via_one_shared_barrier",
            "individual_output_content_never_parsed_or_printed_by_launcher",
            "research_outputs_separate",
            "same_wave_retuning_forbidden",
        ):
            _expect(completion.get(key), True, f"completion {key}")
        _expect(completion.get("official_headline_arm"), "official16", "completion headline")
        _expect(
            tuple(completion.get("research_evaluation_only_arm_ids") or ()),
            RESEARCH_ARM_IDS,
            "completion research-only arms",
        )

        completed_official_path, completed_official_hash = self._completion_artifact(
            completion, "official16", "json"
        )
        if completed_official_path != official_metrics_path:
            raise ArtifactError("completion does not point at the pinned official16 metric")
        _expect(completed_official_hash, PINNED_SHA256[OFFICIAL_METRICS], "official metric pin")
        base_metrics_path, _ = self._completion_artifact(completion, "base240", "json")
        research_metrics_path, research_metrics_hash = self._completion_artifact(
            completion, "research_all36", "json"
        )
        official_metrics = _read_json(official_metrics_path)
        base_metrics = _read_json(base_metrics_path)
        research_metrics = _read_json(research_metrics_path)

        _expect(official_metrics.get("schema_version"), "maxim-full274-score-v1", "score schema")
        _expect(official_metrics.get("label"), "official16", "score label")
        _expect(tuple(official_metrics.get("models") or ()), (MODEL,), "score model closure")
        _expect(
            _metric_tuple(official_metrics),
            (249, 274, 117, 139, 9, 9, 158, 177, 91, 97),
            "official metrics",
        )
        _expect(
            _metric_tuple(base_metrics),
            (240, 274, 109, 139, 8, 9, 158, 177, 82, 97),
            "selector baseline metrics",
        )
        overall = official_metrics.get("overall") or {}
        errors = (official_metrics.get("operational") or {}).get("errors") or {}
        guardrails = official_metrics.get("guardrails") or {}
        _expect(overall.get("new_accuracy"), 0.908759, "published rounded accuracy")
        _expect(overall.get("solver_errors"), 0, "official solver errors")
        _expect(overall.get("missing_answers"), 0, "official missing answers")
        _expect(errors.get("generation_failure_union_count"), 0, "generation failures")
        for key, expected in (
            ("baseline_rows_verified", 274),
            ("benchmark_rows_verified", 274),
            ("solver_rows_verified", 274),
            ("duplicate_task_ids", 0),
            ("forbidden_gold_fields_in_solver", 0),
            ("explicit_nonfalse_generation_gold_access", 0),
            ("task_id_sets_match", True),
            ("frozen_sha_pins_checked", True),
        ):
            _expect(guardrails.get(key), expected, f"official guardrail {key}")

        official_outcomes = _index(
            official_metrics.get("task_outcomes") or (), "official16 outcomes"
        )
        base_outcomes = _index(base_metrics.get("task_outcomes") or (), "base240 outcomes")
        _expect(len(official_outcomes), 274, "official outcome rows")
        _expect(set(official_outcomes), set(base_outcomes), "official/base task set")
        fixes = tuple(
            task_id
            for task_id in official_outcomes
            if bool(official_outcomes[task_id].get("new_correct"))
            and not bool(base_outcomes[task_id].get("new_correct"))
        )
        regressions = tuple(
            task_id
            for task_id in official_outcomes
            if not bool(official_outcomes[task_id].get("new_correct"))
            and bool(base_outcomes[task_id].get("new_correct"))
        )
        _expect(fixes, OFFICIAL_FIX_TASK_IDS, "fixes versus audited 240")
        _expect(regressions, (), "regressions versus audited 240")

        solver_descriptor = official_arm.get("solver") or {}
        solver_path = _safe_relative(self._path(WAVE), solver_descriptor.get("path"), "official solver")
        solver_sha = str(solver_descriptor.get("sha256") or "")
        if _sha256(solver_path) != solver_sha:
            raise ArtifactError("official16 solver hash differs from frozen arm")
        _expect(
            (amendment.get("authorized_solver_sha256") or {}).get("official16"),
            solver_sha,
            "audit-authorized official solver",
        )
        image_sha = str((official_arm.get("image_judge") or {}).get("sha256") or "")
        _expect(
            (amendment.get("authorized_image_judge_sha256") or {}).get("official16"),
            image_sha,
            "audit-authorized official image judge",
        )
        image_path = _safe_relative(
            self._path(WAVE),
            (official_arm.get("image_judge") or {}).get("path"),
            "official image judge",
        )
        if _sha256(image_path) != image_sha:
            raise ArtifactError("official16 image judge hash differs from frozen arm")
        image_rows = _index(_read_jsonl(image_path), "official16 image judge")
        _expect(len(image_rows), 97, "official image judge rows")
        provenance = official_metrics.get("provenance") or {}
        _expect(
            (provenance.get("solver_results") or {}).get("sha256"),
            solver_sha,
            "score-to-solver provenance",
        )
        _expect(
            (provenance.get("image_judge") or {}).get("sha256"),
            image_sha,
            "score-to-image-judge provenance",
        )
        solver = _index(_read_jsonl(solver_path), "official16 solver")
        base_solver_path = _safe_relative(
            self._path(WAVE),
            ((freeze.get("arms") or {}).get("base240") or {}).get("solver", {}).get("path"),
            "base240 solver",
        )
        base_solver_sha = str(
            ((freeze.get("arms") or {}).get("base240") or {}).get("solver", {}).get("sha256") or ""
        )
        if _sha256(base_solver_path) != base_solver_sha:
            raise ArtifactError("base240 solver hash differs from frozen arm")
        base_solver = _index(_read_jsonl(base_solver_path), "base240 solver")
        _expect(set(solver), set(official_outcomes), "solver/outcome task set")
        for task_id, row in solver.items():
            _expect(row.get("model"), MODEL, f"{task_id} model")
            if not str(row.get("final_answer") or "").strip():
                raise ArtifactError(f"{task_id}: official solver answer is empty")
        answer_changed = tuple(
            task_id
            for task_id in solver
            if str(solver[task_id].get("final_answer") or "")
            != str(base_solver[task_id].get("final_answer") or "")
        )
        if not set(answer_changed).issubset(set(OFFICIAL_TARGET_TASK_IDS)):
            raise ArtifactError("official solver changes a non-target answer")

        tasks = tuple(
            SourceWaveTask(
                task_id=task_id,
                subject=str(outcome.get("subject") or "Unknown"),
                final_answer=str(solver[task_id].get("final_answer") or ""),
                correct=bool(outcome.get("new_correct")),
                score_source=str(outcome.get("score_source") or ""),
                score_method=str(outcome.get("score_method") or ""),
                transition=str(outcome.get("transition") or ""),
                answer_changed_vs_240=task_id in answer_changed,
                correctness_fix_vs_240=task_id in fixes,
                solver_row=solver[task_id],
            )
            for task_id, outcome in official_outcomes.items()
        )
        by_subject = {
            str(subject): {
                "n": int(values.get("n") or 0),
                "new_correct": int(values.get("new_correct") or 0),
                "new_accuracy": float(values.get("new_correct") or 0)
                / int(values.get("n") or 1),
            }
            for subject, values in (official_metrics.get("by_subject") or {}).items()
        }

        _expect(research_metrics.get("label"), "research_all36", "research label")
        research_overall = research_metrics.get("overall") or {}
        _expect(
            (research_overall.get("new_correct"), research_overall.get("n")),
            (251, 274),
            "research_all36 metrics",
        )
        research = ResearchWaveResult(
            arm_id="research_all36",
            correct=251,
            rows=274,
            accuracy=251 / 274,
            classification="research_evaluation_only",
            eligible_for_official_headline=False,
            license_status="unverified",
            metrics_sha256=research_metrics_hash,
        )
        return SourceWaveSummary(
            model=MODEL,
            correct=249,
            rows=274,
            math_correct=117,
            math_rows=139,
            english_correct=9,
            english_rows=9,
            deterministic_correct=158,
            deterministic_rows=177,
            image_correct=91,
            image_rows=97,
            previous_correct=240,
            fixes=len(fixes),
            regressions=len(regressions),
            fix_task_ids=fixes,
            answer_changed_task_ids=answer_changed,
            target_task_ids=OFFICIAL_TARGET_TASK_IDS,
            tasks=tasks,
            by_subject=by_subject,
            freeze_sha256=PINNED_SHA256[FREEZE],
            amendment_sha256=PINNED_SHA256[AMENDMENT],
            completion_sha256=PINNED_SHA256[COMPLETION],
            official_metrics_sha256=PINNED_SHA256[OFFICIAL_METRICS],
            official_solver_sha256=solver_sha,
            official_image_judge_sha256=image_sha,
            research_all36=research,
            verified_files=(
                freeze_path,
                amendment_path,
                completion_path,
                official_metrics_path,
                base_metrics_path,
                solver_path,
                image_path,
                base_solver_path,
                research_metrics_path,
            ),
        )

    def validation_report(self) -> dict[str, Any]:
        summary = self.load()
        return {
            "status": "ok",
            "role": "active official all-9B development headline",
            "model": summary.model,
            "correct": summary.correct,
            "rows": summary.rows,
            "accuracy_exact": summary.accuracy,
            "math": [summary.math_correct, summary.math_rows],
            "english": [summary.english_correct, summary.english_rows],
            "deterministic": [summary.deterministic_correct, summary.deterministic_rows],
            "image_judge": [summary.image_correct, summary.image_rows],
            "fixes_vs_240": list(summary.fix_task_ids),
            "regressions_vs_240": summary.regressions,
            "integrity": {
                "freeze_sha256": summary.freeze_sha256,
                "amendment_sha256": summary.amendment_sha256,
                "completion_sha256": summary.completion_sha256,
                "official_metrics_sha256": summary.official_metrics_sha256,
            },
            "research_evaluation_only": {
                "arm_id": summary.research_all36.arm_id,
                "correct": summary.research_all36.correct,
                "rows": summary.research_all36.rows,
                "eligible_for_official_headline": False,
                "license_status": "unverified",
            },
        }


def build_active_source_wave_dataset(
    active_selector: TraceDataset,
    selector: SelectorWaveSummary,
    source_wave: SourceWaveSummary,
) -> TraceDataset:
    """Overlay the pinned official16 solver/outcomes onto the audited 240 trace."""

    if (active_selector.summary.correct, active_selector.summary.rows) != (240, 274):
        raise ArtifactError("source-wave projection requires active selector=240/274")
    if (selector.correct, selector.rows) != (240, 274):
        raise ArtifactError("source-wave projection requires the audited selector summary")
    if (source_wave.correct, source_wave.rows, source_wave.fixes, source_wave.regressions) != (
        249,
        274,
        9,
        0,
    ):
        raise ArtifactError("source-wave projection requires audited official16=249/274")
    wave_tasks = {task.task_id: task for task in source_wave.tasks}
    if set(wave_tasks) != {task.task_id for task in active_selector.tasks}:
        raise ArtifactError("source-wave and selector task sets differ")

    projected_tasks: list[TaskTrace] = []
    for task in active_selector.tasks:
        wave = wave_tasks[task.task_id]
        if task.subject != wave.subject:
            raise ArtifactError(f"{task.task_id}: source-wave subject mismatch")
        solver_row = wave.solver_row
        source_trace = {
            "arm_id": "official16",
            "targeted": task.task_id in source_wave.target_task_ids,
            "answer_changed_vs_240": wave.answer_changed_vs_240,
            "correctness_fix_vs_240": wave.correctness_fix_vs_240,
            "selector_v1_2_answer": task.final_answer,
            "selector_v1_2_correct": task.correct,
            "official16_final_answer": wave.final_answer,
            "official16_correct": wave.correct,
            "freeze_sha256": source_wave.freeze_sha256,
            "completion_sha256": source_wave.completion_sha256,
            "metrics_sha256": source_wave.official_metrics_sha256,
            "claim_boundary": (
                "official16 is the active hash-bound development result; "
                "research/license-unverified arms are excluded"
            ),
        }
        kwargs: dict[str, Any] = {
            "final_answer": wave.final_answer,
            "correct": wave.correct,
            "score_source": f"official16 frozen {wave.score_source}",
            "score_method": wave.score_method,
            "transition": (
                "wrong->correct vs selector 240"
                if wave.correctness_fix_vs_240
                else "official16 outcome"
            ),
            "raw": {**task.raw, "source_wave_v1_1": source_trace},
        }
        if task.task_id in source_wave.target_task_ids:
            kwargs.update(
                challenger_answer=wave.final_answer,
                final_origin="Official16 · verified source expansion wave",
                decision_action=(
                    "source_wave_replace"
                    if wave.answer_changed_vs_240
                    else "source_wave_confirm"
                ),
                decision_reason=(
                    "frozen official-source successor changed the selector answer"
                    if wave.answer_changed_vs_240
                    else "frozen official-source successor confirmed the selector answer"
                ),
            )
            reasoning = str(solver_row.get("reasoning") or "")
            if reasoning:
                kwargs.update(
                    reasoning=reasoning,
                    solution_steps=(
                        split_solution_steps(solver_row.get("solution_steps"))
                        or split_solution_steps(reasoning)
                    ),
                    reasoning_origin="frozen official16 solver artifact",
                    prompt_version=str(solver_row.get("prompt_version") or task.prompt_version),
                )
        projected_tasks.append(replace(task, **kwargs))

    summary = replace(
        active_selector.summary,
        label="Official16 · audited all-9B source expansion result",
        correct=source_wave.correct,
        accuracy=source_wave.accuracy,
        math_correct=source_wave.math_correct,
        math_accuracy=source_wave.math_correct / source_wave.math_rows,
        pipeline_provenance="9B official16 source wave over audited selector v1.2",
        by_subject=source_wave.by_subject,
        limitations=active_selector.summary.limitations
        + (
            "Active official headline is 249/274 from the frozen official16 development arm.",
            "research_all36=251/274 is license-unverified research-only and excluded from the headline.",
        ),
    )
    dataset = TraceDataset(
        summary=summary,
        tasks=tuple(projected_tasks),
        artifact_root=active_selector.artifact_root,
        source_files=active_selector.source_files + source_wave.verified_files,
    )
    dataset.validate()
    return dataset
