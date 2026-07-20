from __future__ import annotations

import json
from dataclasses import dataclass

from .schema import EvaluationItem


_SAFE_JUDGE_METADATA_KEYS = {
    "required_subanswers",
    "reference_notes",
    "gold_quality",
}


def _blind_payload(item: EvaluationItem) -> dict:
    payload = item.to_dict(blind_setup=True)
    # IDs are irrelevant to correctness and may encode source, split, setup, or
    # synthetic expected labels (for example "wrong_concise").
    payload.pop("task_id", None)
    metadata = payload.get("metadata")
    if isinstance(metadata, dict):
        safe_metadata = {
            key: metadata[key]
            for key in _SAFE_JUDGE_METADATA_KEYS
            if metadata.get(key) not in (None, "", [], {})
        }
        if safe_metadata:
            payload["metadata"] = safe_metadata
        else:
            payload.pop("metadata", None)
    return payload


SYSTEM_PROMPT = """You are an impartial evaluator of answers to school homework tasks.
Judge correctness against the supplied reference answer or annotated reference image.

Rules:
1. Evaluate substance, not writing style, verbosity, confidence, or formatting.
2. Do not infer which agent setup produced the answer and do not reward mentions of tools or sources.
3. Inspect every sub-question. A response that misses a required sub-answer is not fully correct.
4. Equivalent mathematical forms, units, notation, and languages are acceptable when semantically equivalent.
5. A correct final answer supported by materially false reasoning is at most mostly_correct.
6. If the reference is visibly ambiguous, incomplete, or likely wrong, set reference_quality_issue=true.
7. If the task or reference cannot be read, return unjudgeable instead of guessing.

Score rubric:
4 = fully correct and complete.
3 = correct core/final result with only a minor omission or minor non-consequential reasoning issue.
2 = meaningful partial progress, but a material error or missing required part.
1 = relevant attempt that is mostly incorrect.
0 = incorrect, irrelevant, or unsupported answer.

Label/score mapping is strict: fully_correct=4, mostly_correct=3,
partially_correct=1 or 2, incorrect=0. For unjudgeable use score=0,
strict_correct=false, and null for fields that cannot be assessed.

Return one JSON object only, with exactly these keys:
label, score, strict_correct, final_answer_correct, reasoning_correct, complete,
confidence, error_types, rationale, reference_quality_issue.

Allowed labels: fully_correct, mostly_correct, partially_correct, incorrect, unjudgeable.
strict_correct must be true if and only if label is fully_correct.
For fully_correct, final_answer_correct=true, complete=true, and reasoning_correct cannot be false.
For mostly_correct, final_answer_correct cannot be false.
Use null for final_answer_correct, reasoning_correct, or complete when that field cannot be assessed.
confidence is a number from 0 to 1. Keep rationale under 120 words."""


@dataclass(frozen=True, slots=True)
class JudgeRequest:
    system_prompt: str
    user_prompt: str
    image_urls: tuple[str, ...]
    image_labels: tuple[str, ...] = ()


def build_judge_request(item: EvaluationItem) -> JudgeRequest:
    item.validate()
    payload = _blind_payload(item)
    image_urls: list[str] = []
    image_labels: list[str] = []
    if item.question_image_url is not None:
        image_urls.append(item.question_image_url)
        image_labels.append("question image")
    if item.reference_image_url is not None:
        image_urls.append(item.reference_image_url)
        image_labels.append("annotated reference-answer image")
    payload.pop("question_image_url", None)
    payload.pop("reference_image_url", None)
    attachment_description = (
        "; ".join(f"image {index}: {label}" for index, label in enumerate(image_labels, start=1))
        if image_labels
        else "no attached images"
    )
    user_prompt = (
        f"Evaluate the candidate answer. Attachments: {attachment_description}.\n\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )
    return JudgeRequest(SYSTEM_PROMPT, user_prompt, tuple(image_urls), tuple(image_labels))


PAIRWISE_SYSTEM_PROMPT = """You are an impartial evaluator comparing two candidate answers to the same school task.
Use the question and reference, ignore style and verbosity, and do not guess how either answer was produced.
Return JSON only with keys winner, confidence, rationale, error_types_a, error_types_b.
winner must be A, B, tie, or unjudgeable. Prefer tie when both answers are substantively equivalent."""


def build_pairwise_prompt(item_a: EvaluationItem, item_b: EvaluationItem) -> str:
    if item_a.task_id != item_b.task_id:
        raise ValueError("pairwise items must share task_id")
    shared = _blind_payload(item_a)
    shared.pop("candidate_answer", None)
    shared.pop("question_image_url", None)
    shared.pop("reference_image_url", None)
    return json.dumps(
        {
            "task": shared,
            "candidate_A": item_a.candidate_answer,
            "candidate_B": item_b.candidate_answer,
        },
        ensure_ascii=False,
        indent=2,
    )
