from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Literal


AnswerType = Literal[
    "multiple_choice",
    "numeric",
    "short_text",
    "multi_answer",
    "open_ended",
    "unknown",
]

SetupName = Literal["no_tools", "web_search", "textbook_retrieval", "unknown"]


@dataclass(slots=True)
class BenchmarkTask:
    """A task and its gold reference before any candidate response is attached."""

    task_id: str
    subject: str = "unknown"
    grade: str | int | None = None
    answer_type: AnswerType = "unknown"
    question_text: str | None = None
    question_image_url: str | None = None
    reference_answer: str | None = None
    reference_image_url: str | None = None
    acceptable_answers: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self, *, allow_unresolved_assets: bool = False) -> None:
        if not self.task_id.strip():
            raise ValueError("task_id must not be empty")
        if not allow_unresolved_assets and not self.question_text and not self.question_image_url:
            raise ValueError("question_text or question_image_url is required")
        if not allow_unresolved_assets and not self.reference_answer and not self.reference_image_url:
            raise ValueError("reference_answer or reference_image_url is required")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class EvaluationItem:
    """One candidate response to one benchmark task."""

    task_id: str
    candidate_answer: str
    subject: str = "unknown"
    grade: str | int | None = None
    answer_type: AnswerType = "unknown"
    setup: SetupName = "unknown"
    question_text: str | None = None
    question_image_url: str | None = None
    reference_answer: str | None = None
    reference_image_url: str | None = None
    acceptable_answers: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "EvaluationItem":
        item = cls(
            task_id=str(value["task_id"]),
            candidate_answer=str(value["candidate_answer"]),
            subject=str(value.get("subject", "unknown")),
            grade=value.get("grade"),
            answer_type=value.get("answer_type", "unknown"),
            setup=value.get("setup", "unknown"),
            question_text=value.get("question_text"),
            question_image_url=value.get("question_image_url"),
            reference_answer=value.get("reference_answer"),
            reference_image_url=value.get("reference_image_url"),
            acceptable_answers=[str(x) for x in value.get("acceptable_answers", [])],
            metadata=dict(value.get("metadata", {})),
        )
        item.validate()
        return item

    def validate(self) -> None:
        if not self.task_id.strip():
            raise ValueError("task_id must not be empty")
        if not self.candidate_answer.strip():
            raise ValueError("candidate_answer must not be empty")
        if not self.question_text and not self.question_image_url:
            raise ValueError("question_text or question_image_url is required")
        if not self.reference_answer and not self.reference_image_url:
            raise ValueError("reference_answer or reference_image_url is required")

    def to_dict(self, *, blind_setup: bool = False) -> dict[str, Any]:
        result = asdict(self)
        if blind_setup:
            result.pop("setup", None)
        return result


VerdictLabel = Literal[
    "fully_correct",
    "mostly_correct",
    "partially_correct",
    "incorrect",
    "unjudgeable",
]

_VERDICT_KEYS = {
    "label",
    "score",
    "strict_correct",
    "final_answer_correct",
    "reasoning_correct",
    "complete",
    "confidence",
    "error_types",
    "rationale",
    "reference_quality_issue",
}
_SCORES_BY_LABEL = {
    "fully_correct": {4},
    "mostly_correct": {3},
    "partially_correct": {1, 2},
    "incorrect": {0},
    "unjudgeable": {0},
}


@dataclass(slots=True)
class JudgeVerdict:
    """Strict, auditable output expected from an LLM judge."""

    label: VerdictLabel
    score: int
    strict_correct: bool
    final_answer_correct: bool | None
    reasoning_correct: bool | None
    complete: bool | None
    confidence: float
    error_types: list[str] = field(default_factory=list)
    rationale: str = ""
    reference_quality_issue: bool = False

    def validate(self) -> None:
        if self.label not in _SCORES_BY_LABEL:
            raise ValueError(f"invalid verdict label: {self.label}")
        if not 0 <= self.score <= 4:
            raise ValueError("score must be between 0 and 4")
        if self.score not in _SCORES_BY_LABEL[self.label]:
            raise ValueError(f"score {self.score} is inconsistent with label {self.label}")
        if not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if self.strict_correct != (self.label == "fully_correct"):
            raise ValueError("strict_correct must be true only for fully_correct")
        if self.label == "fully_correct":
            if self.final_answer_correct is not True or self.complete is not True:
                raise ValueError("fully_correct requires a correct final answer and complete response")
            if self.reasoning_correct is False:
                raise ValueError("fully_correct cannot contain incorrect reasoning")
        if self.label == "mostly_correct" and self.final_answer_correct is False:
            raise ValueError("mostly_correct cannot have an explicitly incorrect final answer")
        if self.label == "unjudgeable" and any(
            value is not None
            for value in (self.final_answer_correct, self.reasoning_correct, self.complete)
        ):
            raise ValueError("unjudgeable assessment fields must be null")
        if len(self.rationale) > 1200:
            raise ValueError("rationale is too long")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "JudgeVerdict":
        keys = set(value)
        missing = sorted(_VERDICT_KEYS - keys)
        extra = sorted(keys - _VERDICT_KEYS)
        if missing or extra:
            raise ValueError(f"verdict keys mismatch; missing={missing}, extra={extra}")
        if not isinstance(value["label"], str):
            raise ValueError("label must be a string")
        if isinstance(value["score"], bool) or not isinstance(value["score"], int):
            raise ValueError("score must be an integer")
        if not isinstance(value["strict_correct"], bool):
            raise ValueError("strict_correct must be a boolean")
        for field_name in ("final_answer_correct", "reasoning_correct", "complete"):
            if value[field_name] is not None and not isinstance(value[field_name], bool):
                raise ValueError(f"{field_name} must be a boolean or null")
        if isinstance(value["confidence"], bool) or not isinstance(value["confidence"], (int, float)):
            raise ValueError("confidence must be numeric")
        if not isinstance(value["error_types"], list) or not all(
            isinstance(item, str) for item in value["error_types"]
        ):
            raise ValueError("error_types must be an array of strings")
        if not isinstance(value["rationale"], str):
            raise ValueError("rationale must be a string")
        if not isinstance(value["reference_quality_issue"], bool):
            raise ValueError("reference_quality_issue must be a boolean")
        verdict = cls(
            label=value["label"],
            score=value["score"],
            strict_correct=value["strict_correct"],
            final_answer_correct=value.get("final_answer_correct"),
            reasoning_correct=value.get("reasoning_correct"),
            complete=value.get("complete"),
            confidence=float(value["confidence"]),
            error_types=list(value["error_types"]),
            rationale=value["rationale"],
            reference_quality_issue=value["reference_quality_issue"],
        )
        verdict.validate()
        return verdict

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
