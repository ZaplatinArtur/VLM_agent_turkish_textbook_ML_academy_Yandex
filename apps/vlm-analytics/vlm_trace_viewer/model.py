from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Any, Iterable


@dataclass(frozen=True)
class AttentionRegion:
    bbox: tuple[float, float, float, float]
    label: str
    text: str
    image_width: int
    image_height: int


@dataclass(frozen=True)
class SourceEvidence:
    status: str = "absent"
    strength: str = "none"
    verifier: str = ""
    document_id: str = ""
    document_name: str = ""
    public_locator: str = ""
    matched_page: int | None = None
    key_page: int | None = None
    question_number: int | None = None
    record_id: str = ""
    key_bbox: tuple[float, float, float, float] | None = None
    checks: tuple[tuple[str, bool], ...] = ()
    page_coverage: float | None = None
    page_margin: float | None = None
    trace_fingerprint: str = ""
    pdf_sha256: str = ""

    @property
    def accepted(self) -> bool:
        return self.status == "pass" and self.strength == "strong"


@dataclass(frozen=True)
class PipelineStage:
    title: str
    subtitle: str
    state: str  # pass, active, skipped, fail, neutral


@dataclass(frozen=True)
class TaskTrace:
    task_id: str
    subject: str
    grade: str
    answer_type: str
    question_text: str
    question_image: Path | None
    final_answer: str
    anchor_answer: str
    challenger_answer: str
    correct: bool
    baseline_correct: bool | None
    score_source: str
    score_method: str
    transition: str
    reasoning: str
    solution_steps: tuple[str, ...]
    base_row_model: str
    final_origin: str
    reasoning_origin: str
    usage_origin: str
    prompt_version: str
    latency_s: float | None
    input_tokens: int | None
    output_tokens: int | None
    decision_action: str
    decision_reason: str
    source: SourceEvidence
    attention_regions: tuple[AttentionRegion, ...]
    candidates: tuple[dict[str, Any], ...]
    raw: dict[str, Any]

    @property
    def has_certificate(self) -> bool:
        return self.source.accepted

    @property
    def pipeline(self) -> tuple[PipelineStage, ...]:
        selector = self.raw.get("selector_v1_2")
        if not isinstance(selector, dict):
            selector = None
        source_wave = self.raw.get("source_wave_v1_1")
        if not isinstance(source_wave, dict):
            source_wave = None
        source_state = "pass" if self.has_certificate else "skipped"
        source_v7_action = (
            str(selector.get("source_v7_decision_action") or "")
            if selector
            else self.decision_action
        )
        source_v7_origin = (
            str(selector.get("source_v7_final_origin") or "")
            if selector
            else self.final_origin
        )
        composer_state = "active" if source_v7_action == "replace_anchor" else "pass"
        decision_label = {
            "replace_anchor": "замена разрешена",
            "keep_anchor": "anchor сохранён",
        }.get(source_v7_action, source_v7_action or "anchor сохранён")
        stages = [
            PipelineStage(
                "Reasoning anchor",
                f"META row · {self.base_row_model or 'model metadata absent'}",
                "pass",
            ),
            PipelineStage("Router", self.subject or "предмет определён", "pass"),
            PipelineStage(
                "Exact-source lookup",
                self.source.document_name or "точный источник не принят",
                source_state,
            ),
            PipelineStage(
                "PDF / page / key binding",
                (
                    f"стр. {self.source.matched_page} → ключ {self.source.key_page}"
                    if self.has_certificate
                    else "fail-closed"
                ),
                source_state,
            ),
            PipelineStage(
                "Source certificate",
                self.source.strength if self.has_certificate else "нет достаточного доказательства",
                source_state,
            ),
            PipelineStage(
                "Deterministic composer",
                f"{decision_label} · {source_v7_origin}",
                composer_state,
            ),
        ]
        if selector:
            stages.append(
                PipelineStage(
                    "Baseline Selector v1.2",
                    "three frozen groups unanimous · answer replaced",
                    "active",
                )
            )
        if source_wave:
            targeted = bool(source_wave.get("targeted"))
            changed = bool(source_wave.get("answer_changed_vs_240"))
            stages.append(
                PipelineStage(
                    "Official16 source wave",
                    (
                        "official source answer replaced selector output"
                        if changed
                        else "official source target confirmed selector output"
                        if targeted
                        else "frozen official16 passthrough"
                    ),
                    "active" if targeted else "pass",
                )
            )
        stages.append(
            PipelineStage(
                "Evaluation",
                f"{self.score_source} · {'correct' if self.correct else 'incorrect'}",
                "pass" if self.correct else "fail",
            ),
        )
        return tuple(stages)


@dataclass(frozen=True)
class RunSummary:
    label: str
    rows: int
    correct: int
    accuracy: float
    math_rows: int
    math_correct: int
    math_accuracy: float
    baseline_accuracy: float
    source_certificates: int
    answer_overrides: int
    direct_gain_vs_v6: int
    evaluator_corrections_vs_v6: int
    latency_median_s: float | None
    latency_p95_s: float | None
    latency_max_s: float | None
    pipeline_provenance: str
    base_row_models: tuple[str, ...]
    recorded_usage_scope: str
    source_adjudicated_image_rows: int = 0
    original_9b_judge_rows: int = 0
    by_subject: dict[str, dict[str, Any]] = field(default_factory=dict)
    limitations: tuple[str, ...] = ()
    source_shortcuts: int = 0
    anchor_fallbacks: int = 0
    source_shortcut_rate: float | None = None
    answer_equivalent_shortcuts: int = 0
    avoidable_recorded_latency_fraction: float | None = None
    avoidable_input_tokens_fraction: float | None = None
    avoidable_output_tokens_fraction: float | None = None
    speed_online_wall_clock_measured: bool = False
    speed_source_lookup_cost_included: bool = False


@dataclass(frozen=True)
class TraceDataset:
    summary: RunSummary
    tasks: tuple[TaskTrace, ...]
    artifact_root: Path
    source_files: tuple[Path, ...]

    def validate(self) -> None:
        ids = [task.task_id for task in self.tasks]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate task_id in trace dataset")
        if self.summary.rows != len(self.tasks):
            raise ValueError(
                f"summary rows={self.summary.rows}, loaded tasks={len(self.tasks)}"
            )
        actual_correct = sum(task.correct for task in self.tasks)
        if actual_correct != self.summary.correct:
            raise ValueError(
                f"summary correct={self.summary.correct}, task outcomes={actual_correct}"
            )


def split_solution_steps(value: str | Iterable[str] | None) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        raw = value.replace("\r\n", "\n").split("\n")
    else:
        raw = [str(item) for item in value]
    steps: list[str] = []
    for line in raw:
        line = line.strip()
        if not line:
            continue
        # Keep the model's wording, only remove common list markers.
        stripped = line.lstrip("-*• ")
        if len(stripped) > 2 and stripped[0].isdigit():
            prefix, sep, tail = stripped.partition(".")
            if sep and prefix.isdigit():
                stripped = tail.strip()
        steps.append(stripped or line)
    return tuple(steps)


def percentile_nearest(values: Iterable[float], percentile: float) -> float | None:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    index = max(0, min(len(ordered) - 1, int((len(ordered) * percentile) + 0.999999) - 1))
    return ordered[index]


def latency_summary(values: Iterable[float]) -> tuple[float | None, float | None, float | None]:
    prepared = [float(value) for value in values]
    if not prepared:
        return None, None, None
    return median(prepared), percentile_nearest(prepared, 0.95), max(prepared)
