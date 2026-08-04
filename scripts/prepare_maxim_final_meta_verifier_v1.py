#!/usr/bin/env python3
"""Prepare a source-blind queue for the frozen274 final meta-verifier.

Only benchmark-visible task fields and bounded candidate answer/evidence fields
enter the public queue.  Source identities and the deterministic shuffle map
are written to a separate private key which the model runner never needs.
No score, judge outcome, reference answer, or gold-derived field is loaded.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "maxim-final-meta-verifier-preparation-v1"
QUEUE_SCHEMA_VERSION = "maxim-final-meta-verifier-queue-v1"
PROFILE_SCHEMA_VERSION = "maxim-final-meta-verifier-profile-v1"
FROZEN_BENCHMARK_SHA256 = (
    "5a6a38ccae7835f0d015f6e5979834208347b8e6e7a8d6884e4af97605f51ed9"
)
EXPECTED_ROWS = 274
DEFAULT_BLINDING_SEED = "maxim-final-meta-verifier-order-20260803-v1"
ROUTER_SLOT = "subject_router"
REQUIRED_CANDIDATE_SLOTS = (
    ROUTER_SLOT,
    "structural_rag",
    "active_vision",
    "tiled_vision",
    "native_thinking_v4",
    "budgeted_thinking_v5",
    "stronger_27b_direct",
)
OPAQUE_IDS = tuple(f"C{index}" for index in range(1, len(REQUIRED_CANDIDATE_SLOTS) + 1))

FORBIDDEN_QUEUE_KEYS = frozenset(
    {
        "reference_answer",
        "reference_solution",
        "gold_answer",
        "gold_solution",
        "acceptable_answers",
        "score",
        "scores",
        "accuracy",
        "correct",
        "strict_correct",
        "judge",
        "judge_verdict",
        "reward",
        "condition",
        "model",
        "prompt_version",
        "source",
        "source_id",
        "source_slot",
        "system_id",
        "candidate_name",
        "default_candidate_id",
        "router_candidate_id",
    }
)


class PreparationError(RuntimeError):
    pass


def stable_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def stable_sha256(value: Any) -> str:
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PreparationError(f"cannot load JSON {path}: {exc}") from exc


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8-sig") as source:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise PreparationError(
                        f"invalid JSONL {path}:{line_number}: {exc}"
                    ) from exc
                if not isinstance(value, dict):
                    raise PreparationError(
                        f"JSONL row is not an object: {path}:{line_number}"
                    )
                rows.append(value)
    except OSError as exc:
        raise PreparationError(f"cannot load JSONL {path}: {exc}") from exc
    return rows


def index_unique(
    rows: Iterable[Mapping[str, Any]], source_name: str
) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        task_id = str(row.get("task_id") or "")
        if not task_id:
            raise PreparationError(f"{source_name}: row without task_id")
        if task_id in indexed:
            raise PreparationError(f"{source_name}: duplicate task_id {task_id}")
        indexed[task_id] = row
    return indexed


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="\n") as sink:
        for row in rows:
            sink.write(stable_json(row) + "\n")
    os.replace(temporary, path)


def parse_candidate_paths(values: Sequence[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise PreparationError(f"--candidate expects SLOT=PATH, got {value!r}")
        slot, raw_path = value.split("=", 1)
        slot = slot.strip()
        if slot not in REQUIRED_CANDIDATE_SLOTS:
            raise PreparationError(f"unknown candidate slot: {slot!r}")
        if slot in result:
            raise PreparationError(f"duplicate candidate slot: {slot}")
        result[slot] = Path(raw_path).expanduser().resolve()
    missing = sorted(set(REQUIRED_CANDIDATE_SLOTS) - set(result))
    if missing:
        raise PreparationError(f"missing required candidate slots: {missing}")
    return result


def blind_order(task_id: str, seed: str) -> list[str]:
    return sorted(
        REQUIRED_CANDIDATE_SLOTS,
        key=lambda slot: hashlib.sha256(
            f"{seed}\0{task_id}\0{slot}".encode("utf-8")
        ).hexdigest(),
    )


def _flatten_evidence(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        return [stable_json(dict(value))]
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        result: list[str] = []
        for item in value:
            if isinstance(item, str):
                result.append(item)
            elif isinstance(item, Mapping):
                result.append(stable_json(dict(item)))
            elif item is not None:
                result.append(str(item))
        return result
    return [str(value)]


def _compact_text(value: Any, limit: int) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= limit:
        return text
    marker = " [truncated]"
    return text[: limit - len(marker)].rstrip() + marker


def _redact_source_aliases(text: str, aliases: Sequence[str]) -> str:
    result = text
    for alias in sorted({value for value in aliases if len(value.strip()) >= 3}, key=len, reverse=True):
        result = re.sub(re.escape(alias), "[source-redacted]", result, flags=re.IGNORECASE)
    return result


def bounded_candidate_payload(row: Mapping[str, Any], slot: str) -> dict[str, Any]:
    final_answer = _compact_text(row.get("final_answer"), 160)
    if not final_answer:
        raise PreparationError(f"{slot}: solver row has empty final_answer")

    aliases = [
        *(candidate_slot for candidate_slot in REQUIRED_CANDIDATE_SLOTS),
        *(candidate_slot.replace("_", " ") for candidate_slot in REQUIRED_CANDIDATE_SLOTS),
        slot,
        slot.replace("_", " "),
        str(row.get("condition") or ""),
        str(row.get("model") or ""),
        str(row.get("prompt_version") or ""),
    ]
    reasoning_parts = []
    for label, key in (("Reasoning", "reasoning"), ("Check", "solution_steps")):
        value = _compact_text(row.get(key), 800)
        if value:
            reasoning_parts.append(f"{label}: {value}")
    reasoning = _compact_text("\n".join(reasoning_parts), 1200)
    reasoning = _compact_text(_redact_source_aliases(reasoning, aliases), 1200)
    if not reasoning:
        reasoning = "No bounded reasoning was emitted by this anonymous candidate."

    evidence_values: list[str] = []
    generation = row.get("generation")
    safe_containers = [row]
    if isinstance(generation, Mapping):
        safe_containers.append(generation)
    for container in safe_containers:
        for key in (
            "visual_facts",
            "transcribed_facts",
            "visible_evidence",
            "evidence_citations",
        ):
            evidence_values.extend(_flatten_evidence(container.get(key)))
    evidence: list[str] = []
    for item in evidence_values:
        compact = _compact_text(item, 240)
        compact = _compact_text(_redact_source_aliases(compact, aliases), 240)
        if compact and compact not in evidence:
            evidence.append(compact)
        if len(evidence) >= 4:
            break
    if not evidence:
        evidence = ["No separate evidence list was emitted; inspect the original image."]
    return {
        "final_answer": final_answer,
        "bounded_reasoning": reasoning,
        "bounded_evidence": evidence,
    }


def audit_gold_free(value: Any, location: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            folded = str(key).casefold()
            if folded in FORBIDDEN_QUEUE_KEYS or "reference_answer" in folded:
                raise PreparationError(f"forbidden queue key at {location}.{key}")
            audit_gold_free(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            audit_gold_free(child, f"{location}[{index}]")


def audit_source_slots_hidden(value: Any) -> None:
    serialized = stable_json(value).casefold()
    for slot in REQUIRED_CANDIDATE_SLOTS:
        for alias in (slot.casefold(), slot.replace("_", " ").casefold()):
            if alias in serialized:
                raise PreparationError(
                    f"candidate source alias leaked into public queue: {alias!r}"
                )


def audit_candidate_solver_row(
    row: Mapping[str, Any], *, slot: str, task_id: str
) -> None:
    """Reject scored/gold-bearing solver rows before deriving public payloads."""

    forbidden_fragments = (
        "reference",
        "gold_answer",
        "gold_solution",
        "judge",
        "score",
        "accuracy",
        "strict_correct",
        "baseline_correct",
        "new_correct",
    )

    def visit(value: Any, location: str) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                folded = str(key).casefold()
                if folded != "gold_access" and any(
                    fragment in folded for fragment in forbidden_fragments
                ):
                    raise PreparationError(
                        f"{slot}:{task_id}: forbidden scored/gold key at {location}.{key}"
                    )
                if folded == "gold_access" and child is not False:
                    raise PreparationError(
                        f"{slot}:{task_id}: gold_access must be false at {location}.{key}"
                    )
                visit(child, f"{location}.{key}")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, f"{location}[{index}]")

    visit(row, "$")
    generation = row.get("generation")
    if not isinstance(generation, Mapping) or generation.get("gold_access") is not False:
        raise PreparationError(
            f"{slot}:{task_id}: solver row must explicitly bind generation.gold_access=false"
        )


def _question_images(task: Mapping[str, Any], task_id: str) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for image in task.get("question_images") or []:
        if not isinstance(image, Mapping):
            continue
        data = str(image.get("data") or "")
        if not Path(data).name:
            continue
        result.append(
            {
                "data": data,
                "mime_type": str(image.get("mime_type") or "image/png"),
            }
        )
    if not result:
        raise PreparationError(f"{task_id}: no usable original question image")
    return result


def validate_profile(profile: Mapping[str, Any]) -> None:
    if profile.get("schema_version") != PROFILE_SCHEMA_VERSION:
        raise PreparationError("profile schema_version mismatch")
    if profile.get("benchmark_sha256") != FROZEN_BENCHMARK_SHA256:
        raise PreparationError("profile benchmark pin mismatch")
    if tuple(profile.get("required_candidate_slots") or ()) != REQUIRED_CANDIDATE_SLOTS:
        raise PreparationError("profile candidate slots/order mismatch")
    if profile.get("router_fallback_slot") != ROUTER_SLOT:
        raise PreparationError("profile router fallback mismatch")
    if profile.get("blinding_seed") != DEFAULT_BLINDING_SEED:
        raise PreparationError("profile blinding seed mismatch")
    if profile.get("gold_access") is not False:
        raise PreparationError("profile must bind gold_access=false")


def prepare_queue(
    *,
    benchmark_path: Path,
    candidate_paths: Mapping[str, Path],
    profile_path: Path,
    queue_path: Path,
    private_key_path: Path,
    manifest_path: Path,
    enforce_frozen: bool = True,
) -> dict[str, Any]:
    benchmark_path = benchmark_path.resolve()
    profile_path = profile_path.resolve()
    profile = load_json(profile_path)
    if not isinstance(profile, Mapping):
        raise PreparationError("profile must be a JSON object")
    validate_profile(profile)

    benchmark_sha = sha256_file(benchmark_path)
    if enforce_frozen and benchmark_sha != FROZEN_BENCHMARK_SHA256:
        raise PreparationError(f"benchmark SHA256 mismatch: {benchmark_sha}")
    benchmark = load_jsonl(benchmark_path)
    benchmark_index = index_unique(benchmark, "benchmark")
    task_ids = [str(row["task_id"]) for row in benchmark]
    if enforce_frozen and len(task_ids) != EXPECTED_ROWS:
        raise PreparationError(f"expected {EXPECTED_ROWS} rows, got {len(task_ids)}")

    solvers: dict[str, dict[str, Mapping[str, Any]]] = {}
    source_manifest: dict[str, Any] = {}
    for slot in REQUIRED_CANDIDATE_SLOTS:
        path = candidate_paths[slot].resolve()
        rows = load_jsonl(path)
        indexed = index_unique(rows, slot)
        if set(indexed) != set(task_ids):
            missing = sorted(set(task_ids) - set(indexed))[:5]
            extra = sorted(set(indexed) - set(task_ids))[:5]
            raise PreparationError(
                f"{slot}: task set mismatch; missing={missing}, extra={extra}"
            )
        for task_id, row in indexed.items():
            audit_candidate_solver_row(row, slot=slot, task_id=task_id)
            if not str(row.get("final_answer") or "").strip():
                raise PreparationError(f"{slot}:{task_id}: empty final_answer")
            if row.get("error"):
                raise PreparationError(f"{slot}:{task_id}: unresolved solver error")
        solvers[slot] = dict(indexed)
        source_manifest[slot] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "rows": len(rows),
        }

    queue_rows: list[dict[str, Any]] = []
    key_rows: list[dict[str, Any]] = []
    opaque_position_counts = {slot: {opaque: 0 for opaque in OPAQUE_IDS} for slot in REQUIRED_CANDIDATE_SLOTS}
    for queue_index, task_id in enumerate(task_ids):
        task = benchmark_index[task_id]
        order = blind_order(task_id, DEFAULT_BLINDING_SEED)
        mapping: dict[str, str] = {}
        candidates: list[dict[str, Any]] = []
        for index, slot in enumerate(order):
            opaque_id = OPAQUE_IDS[index]
            mapping[opaque_id] = slot
            opaque_position_counts[slot][opaque_id] += 1
            candidates.append(
                {
                    "candidate_id": opaque_id,
                    **bounded_candidate_payload(solvers[slot][task_id], slot),
                }
            )
        queue_payload = {
            "schema_version": QUEUE_SCHEMA_VERSION,
            "queue_index": queue_index,
            "task_id": task_id,
            "subject": task.get("subject"),
            "grade": task.get("grade"),
            "question": task.get("question"),
            "answer_type": str(task.get("answer_type") or "unknown"),
            "question_images": _question_images(task, task_id),
            "candidates": candidates,
        }
        audit_gold_free(queue_payload)
        audit_source_slots_hidden(queue_payload)
        request_sha = stable_sha256(queue_payload)
        queue_rows.append({**queue_payload, "request_sha256": request_sha})
        key_rows.append(
            {
                "task_id": task_id,
                "request_sha256": request_sha,
                "opaque_to_source_slot": mapping,
                "router_opaque_id": next(
                    opaque for opaque, slot in mapping.items() if slot == ROUTER_SLOT
                ),
            }
        )

    write_jsonl(queue_path, queue_rows)
    write_jsonl(private_key_path, key_rows)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "benchmark": {
            "path": str(benchmark_path),
            "sha256": benchmark_sha,
            "rows": len(task_ids),
        },
        "profile": {
            "path": str(profile_path),
            "sha256": sha256_file(profile_path),
        },
        "generation_gold_access": False,
        "candidate_scores_loaded": False,
        "judge_artifacts_loaded": False,
        "queue_gold_free_audit": True,
        "candidate_source_identities_hidden_from_queue": True,
        "original_question_and_images_are_primary": True,
        "blinding_seed": DEFAULT_BLINDING_SEED,
        "sources_private": source_manifest,
        "queue_public": {
            "path": str(queue_path.resolve()),
            "sha256": sha256_file(queue_path),
            "rows": len(queue_rows),
        },
        "private_routing_key": {
            "path": str(private_key_path.resolve()),
            "sha256": sha256_file(private_key_path),
            "rows": len(key_rows),
            "must_not_be_mounted_into_or_loaded_by_model_runner": True,
        },
        "router_fallback": {
            "slot": ROUTER_SLOT,
            "solver_sha256": source_manifest[ROUTER_SLOT]["sha256"],
            "policy": "exact same-task row on verifier error, abstention, or frozen confidence gate",
        },
        "shuffle_audit": opaque_position_counts,
    }
    write_json(manifest_path, manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument(
        "--candidate",
        action="append",
        default=[],
        metavar="SLOT=PATH",
        help="repeat once for every preregistered candidate slot",
    )
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--private-key", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--skip-frozen-sha-check", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = prepare_queue(
        benchmark_path=args.benchmark,
        candidate_paths=parse_candidate_paths(args.candidate),
        profile_path=args.profile,
        queue_path=args.queue,
        private_key_path=args.private_key,
        manifest_path=args.manifest,
        enforce_frozen=not args.skip_frozen_sha_check,
    )
    print(stable_json(manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
