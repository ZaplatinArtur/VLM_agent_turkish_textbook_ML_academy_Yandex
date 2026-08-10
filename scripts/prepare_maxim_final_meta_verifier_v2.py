#!/usr/bin/env python3
"""Prepare the separate 10-candidate frozen274 meta-verifier v2 queue.

This module deliberately reuses the already-preregistered v1 preparation
engine without modifying it.  It binds a new schema/seed, exactly ten source
slots, and tighter candidate text budgets sized for a 16k multimodal context.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Mapping, Sequence


BASE_SCRIPT = Path(__file__).with_name("prepare_maxim_final_meta_verifier_v1.py")
SPEC = importlib.util.spec_from_file_location("maxim_final_meta_prepare_v2_base", BASE_SCRIPT)
if SPEC is None or SPEC.loader is None:  # pragma: no cover - deployment failure
    raise RuntimeError(f"cannot load frozen v1 preparation engine: {BASE_SCRIPT}")
implementation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(implementation)

implementation.SCHEMA_VERSION = "maxim-final-meta-verifier-preparation-v2"
implementation.QUEUE_SCHEMA_VERSION = "maxim-final-meta-verifier-queue-v2"
implementation.PROFILE_SCHEMA_VERSION = "maxim-final-meta-verifier-profile-v2"
implementation.DEFAULT_BLINDING_SEED = "maxim-final-meta-verifier-order-20260803-v2"
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
)
implementation.ROUTER_SLOT = "subject_router"
implementation.OPAQUE_IDS = tuple(
    f"C{index}" for index in range(1, len(implementation.REQUIRED_CANDIDATE_SLOTS) + 1)
)

FINAL_ANSWER_MAX_CHARS = 120
REASONING_MAX_CHARS = 650
EVIDENCE_MAX_ITEMS = 2
EVIDENCE_ITEM_MAX_CHARS = 160


def bounded_candidate_payload(row: Mapping[str, Any], slot: str) -> dict[str, Any]:
    """Expose at most ~1.1k characters per anonymous candidate."""

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
    parts: list[str] = []
    for label, key in (("Reasoning", "reasoning"), ("Check", "solution_steps")):
        value = implementation._compact_text(row.get(key), 400)
        if value:
            parts.append(f"{label}: {value}")
    reasoning = implementation._compact_text("\n".join(parts), REASONING_MAX_CHARS)
    reasoning = implementation._compact_text(
        implementation._redact_source_aliases(reasoning, aliases),
        REASONING_MAX_CHARS,
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
    return {
        "final_answer": final_answer,
        "bounded_reasoning": reasoning,
        "bounded_evidence": evidence,
    }


implementation.bounded_candidate_payload = bounded_candidate_payload

# Re-export the preparation API required by the v2 runner and tests.  Functions
# retain the v1 engine's module globals, which above are rebound only inside
# this isolated module instance.
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
validate_profile = implementation.validate_profile
prepare_queue = implementation.prepare_queue
build_parser = implementation.build_parser


def main(argv: Sequence[str] | None = None) -> int:
    implementation.__doc__ = __doc__
    return implementation.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
