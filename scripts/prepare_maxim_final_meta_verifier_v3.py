#!/usr/bin/env python3
"""Prepare the separate frozen274 twelve-candidate meta-verifier v3 queue.

V3 leaves v1/v2 immutable.  It freezes exactly twelve full274 candidate
slots, a new deterministic task-level blinding seed, and a stricter rendered
candidate-content budget before either new candidate is generated or scored.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Mapping, Sequence


BASE_SCRIPT = Path(__file__).with_name("prepare_maxim_final_meta_verifier_v1.py")
SPEC = importlib.util.spec_from_file_location("maxim_final_meta_prepare_v3_base", BASE_SCRIPT)
if SPEC is None or SPEC.loader is None:  # pragma: no cover - deployment failure
    raise RuntimeError(f"cannot load frozen v1 preparation engine: {BASE_SCRIPT}")
implementation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(implementation)

implementation.SCHEMA_VERSION = "maxim-final-meta-verifier-preparation-v3"
implementation.QUEUE_SCHEMA_VERSION = "maxim-final-meta-verifier-queue-v3"
implementation.PROFILE_SCHEMA_VERSION = "maxim-final-meta-verifier-profile-v3"
implementation.DEFAULT_BLINDING_SEED = (
    "maxim-final-meta-verifier-order-20260803-v3-12way"
)
implementation.REQUIRED_CANDIDATE_SLOTS = (
    "subject_router",
    "raw_verifier",
    "structural_rag",
    "mi_rag",
    "active_vision",
    "tiled_vision",
    "native_thinking_v4",
    "budgeted_thinking_v5",
    "stronger_27b_hard86_composite",
    "stronger_27b_direct",
    "paired_rag_norag_semantic_support_on_pinned_structural_context_v1",
    "literal_parallel8_lowconf_v1",
)
implementation.ROUTER_SLOT = "subject_router"
implementation.OPAQUE_IDS = tuple(
    f"C{index}" for index in range(1, len(implementation.REQUIRED_CANDIDATE_SLOTS) + 1)
)

# The dynamic candidate text is at most 800 characters per candidate.  The
# fixed labels, opaque ID, separators, and newlines are conservatively granted
# another 100 characters, so all twelve rendered candidate blocks are capped
# at 10,800 characters (strictly below the preregistered 10,900 limit).
FINAL_ANSWER_MAX_CHARS = 120
DECISIVE_REASONING_MAX_CHARS = 500
EVIDENCE_MAX_ITEMS = 2
EVIDENCE_ITEM_MAX_CHARS = 90
DYNAMIC_CHARS_PER_CANDIDATE = (
    FINAL_ANSWER_MAX_CHARS
    + DECISIVE_REASONING_MAX_CHARS
    + EVIDENCE_MAX_ITEMS * EVIDENCE_ITEM_MAX_CHARS
)
RENDERED_CHARS_PER_CANDIDATE_CAP = 900
TOTAL_RENDERED_CANDIDATE_CHARS_CAP = (
    len(implementation.REQUIRED_CANDIDATE_SLOTS) * RENDERED_CHARS_PER_CANDIDATE_CAP
)
_base_validate_profile = implementation.validate_profile


def bounded_candidate_payload(row: Mapping[str, Any], slot: str) -> dict[str, Any]:
    """Keep answer plus decisive reasoning inside the frozen v3 text budget."""

    final_answer = implementation._compact_text(
        row.get("final_answer"), FINAL_ANSWER_MAX_CHARS
    )
    if not final_answer:
        raise implementation.PreparationError(
            f"{slot}: solver row has empty final_answer"
        )

    aliases = [
        *implementation.REQUIRED_CANDIDATE_SLOTS,
        *(value.replace("_", " ") for value in implementation.REQUIRED_CANDIDATE_SLOTS),
        str(row.get("condition") or ""),
        str(row.get("model") or ""),
        str(row.get("prompt_version") or ""),
    ]

    # Explicit decisive reasoning, when present, is placed first and therefore
    # survives truncation.  Generic reasoning and compact solution steps fill
    # the remaining allowance without exposing source/model identities.
    reasoning_parts: list[str] = []
    for label, key in (
        ("Decisive", "decisive_reasoning"),
        ("Reasoning", "reasoning"),
        ("Check", "solution_steps"),
    ):
        value = implementation._compact_text(row.get(key), 500)
        if value:
            reasoning_parts.append(f"{label}: {value}")
    reasoning = implementation._compact_text(
        "\n".join(reasoning_parts), DECISIVE_REASONING_MAX_CHARS
    )
    reasoning = implementation._compact_text(
        implementation._redact_source_aliases(reasoning, aliases),
        DECISIVE_REASONING_MAX_CHARS,
    )
    if not reasoning:
        reasoning = "No bounded reasoning was emitted; verify from the original image."

    evidence_values: list[str] = []
    generation = row.get("generation")
    containers = [row]
    if isinstance(generation, Mapping):
        containers.append(generation)
    for container in containers:
        for key in (
            "decisive_evidence",
            "visual_facts",
            "transcribed_facts",
            "visible_evidence",
            "evidence_citations",
        ):
            evidence_values.extend(implementation._flatten_evidence(container.get(key)))

    evidence: list[str] = []
    for item in evidence_values:
        compact = implementation._compact_text(item, EVIDENCE_ITEM_MAX_CHARS)
        compact = implementation._compact_text(
            implementation._redact_source_aliases(compact, aliases),
            EVIDENCE_ITEM_MAX_CHARS,
        )
        if compact and compact not in evidence:
            evidence.append(compact)
        if len(evidence) >= EVIDENCE_MAX_ITEMS:
            break
    if not evidence:
        evidence = ["No separate evidence list; inspect the original image."]

    payload = {
        "final_answer": final_answer,
        "bounded_reasoning": reasoning,
        "bounded_evidence": evidence,
    }
    dynamic_chars = (
        len(final_answer) + len(reasoning) + sum(len(item) for item in evidence)
    )
    if dynamic_chars > DYNAMIC_CHARS_PER_CANDIDATE:  # defensive invariant
        raise implementation.PreparationError(
            f"{slot}: bounded payload exceeds {DYNAMIC_CHARS_PER_CANDIDATE} chars"
        )
    return payload


implementation.bounded_candidate_payload = bounded_candidate_payload


def validate_profile(profile: Mapping[str, Any]) -> None:
    """Bind every behavior-affecting v3 profile field, not only its schema."""

    _base_validate_profile(profile)
    if profile.get("frozen_before_candidate_generation_or_score_use") is not True:
        raise implementation.PreparationError("profile must be frozen before generation/score")
    if profile.get("score_or_judge_inputs_allowed") is not False:
        raise implementation.PreparationError("profile must prohibit score/judge inputs")

    prompt = profile.get("prompt_policy")
    expected_prompt = {
        "original_question_and_original_images_first": True,
        "candidate_source_identities_hidden": True,
        "candidate_order_deterministic_per_task": True,
        "candidate_final_answer_max_chars": FINAL_ANSWER_MAX_CHARS,
        "candidate_decisive_reasoning_max_chars": DECISIVE_REASONING_MAX_CHARS,
        "candidate_evidence_max_items": EVIDENCE_MAX_ITEMS,
        "candidate_evidence_item_max_chars": EVIDENCE_ITEM_MAX_CHARS,
        "maximum_dynamic_candidate_payload_chars": (
            len(implementation.REQUIRED_CANDIDATE_SLOTS)
            * DYNAMIC_CHARS_PER_CANDIDATE
        ),
        "maximum_rendered_candidate_content_chars_including_fixed_labels": (
            TOTAL_RENDERED_CANDIDATE_CHARS_CAP
        ),
        "absolute_rendered_candidate_content_limit_chars": 10900,
        "target_context_window_tokens": 16384,
        "independent_solution_required": True,
        "majority_vote_prohibited": True,
        "new_answer_not_in_candidates_allowed": True,
    }
    if not isinstance(prompt, Mapping) or any(
        prompt.get(key) != value for key, value in expected_prompt.items()
    ):
        raise implementation.PreparationError("profile prompt/content budget mismatch")

    generation = profile.get("generation")
    expected_generation = {
        "model": "Qwen/Qwen3.5-27B",
        "model_revision": "fc05daec18b0a78c049392ed2e771dde82bdf654",
        "enable_thinking": False,
        "temperature": 0.0,
        "top_p": 0.95,
        "seed": 20260803,
        "max_tokens": 3072,
        "structured_mode": "strict_json_schema",
        "transport_retries_per_semantic_attempt": 1,
        "semantic_attempts": 2,
    }
    if not isinstance(generation, Mapping) or any(
        generation.get(key) != value for key, value in expected_generation.items()
    ):
        raise implementation.PreparationError("profile generation policy mismatch")

    selection = profile.get("selection_policy")
    expected_selection = {
        "valid_verdict_action": "use_meta_verifier_final_answer",
        "min_confidence": 0.7,
        "min_decisive_evidence": 2,
        "require_answer_format_verified": True,
        "require_nonabstain": True,
        "abstain_action": "copy_exact_subject_router_row",
        "schema_or_transport_error_action": "copy_exact_subject_router_row",
        "confidence_or_evidence_gate_failure_action": "copy_exact_subject_router_row",
        "gold_or_score_conditioned_routing": False,
    }
    if not isinstance(selection, Mapping) or any(
        selection.get(key) != value for key, value in expected_selection.items()
    ):
        raise implementation.PreparationError("profile selection/fallback policy mismatch")


implementation.validate_profile = validate_profile

# Re-export the isolated preparation API used by the v3 runner/tests.  The
# inherited functions retain the isolated module globals rebound above.
SCHEMA_VERSION = implementation.SCHEMA_VERSION
QUEUE_SCHEMA_VERSION = implementation.QUEUE_SCHEMA_VERSION
PROFILE_SCHEMA_VERSION = implementation.PROFILE_SCHEMA_VERSION
FROZEN_BENCHMARK_SHA256 = implementation.FROZEN_BENCHMARK_SHA256
EXPECTED_ROWS = implementation.EXPECTED_ROWS
DEFAULT_BLINDING_SEED = implementation.DEFAULT_BLINDING_SEED
ROUTER_SLOT = implementation.ROUTER_SLOT
REQUIRED_CANDIDATE_SLOTS = implementation.REQUIRED_CANDIDATE_SLOTS
OPAQUE_IDS = implementation.OPAQUE_IDS
PreparationError = implementation.PreparationError
stable_json = implementation.stable_json
stable_sha256 = implementation.stable_sha256
sha256_file = implementation.sha256_file
load_json = implementation.load_json
load_jsonl = implementation.load_jsonl
index_unique = implementation.index_unique
write_json = implementation.write_json
write_jsonl = implementation.write_jsonl
parse_candidate_paths = implementation.parse_candidate_paths
blind_order = implementation.blind_order
audit_gold_free = implementation.audit_gold_free
audit_source_slots_hidden = implementation.audit_source_slots_hidden
audit_candidate_solver_row = implementation.audit_candidate_solver_row
prepare_queue = implementation.prepare_queue
build_parser = implementation.build_parser


def main(argv: Sequence[str] | None = None) -> int:
    implementation.__doc__ = __doc__
    return implementation.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
