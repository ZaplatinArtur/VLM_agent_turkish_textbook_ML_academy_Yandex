#!/usr/bin/env python3
"""Update one image-judge row from a replayed fill-blank source certificate.

This is deliberately separate from the generic official-workbook image-judge
builder.  It understands only the fill-blank page-activity composition contract
and fails closed unless the complete profile -> resolver -> composition chain
can be reproduced byte-for-byte before the source-backed verdict is emitted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from tempfile import TemporaryDirectory
from typing import Any, Callable


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evidence_os.fill_blank_page_activity import VERIFIER  # noqa: E402
from evidence_os.official_ogm import canonical_json_bytes, sha256_file  # noqa: E402
from scripts.compose_maxim_fill_blank_page_activity_v1 import (  # noqa: E402
    COMPOSITION_SCHEMA,
    PROFILE_SCHEMA,
    RUN_SCHEMA,
    compose as compose_fill_blank_page_activity,
)


SCHEMA = "maxim-fill-blank-page-activity-image-judge-v1"
BASE_IMAGE_JUDGE_SCHEMA = "maxim-evidence-os-image-judge-v1"
BASE_COMPOSITION_SCHEMA = "maxim-official-source-failclosed-composition-v2"
EXPECTED_ROWS = 274
EXPECTED_IMAGE_ROWS = 97
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_TASK_ID_BYTES = re.compile(
    rb'(?<!\\)"task_id"\s*:\s*("(?:\\.|[^"\\])*")'
)


class HistoryImageJudgeError(RuntimeError):
    pass


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise HistoryImageJudgeError(f"duplicate JSON key: {key}")
        output[key] = value
    return output


def _json_loads(value: str, label: str) -> Any:
    try:
        return json.loads(value, object_pairs_hook=_reject_duplicate_keys)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise HistoryImageJudgeError(f"{label}: malformed JSON") from exc


def _load_json(path: Path) -> dict[str, Any]:
    value = _json_loads(path.read_text(encoding="utf-8-sig"), str(path))
    if not isinstance(value, dict):
        raise HistoryImageJudgeError(f"{path}: expected an object")
    return value


def _load_jsonl(path: Path, label: str) -> list[tuple[bytes, dict[str, Any]]]:
    rows: list[tuple[bytes, dict[str, Any]]] = []
    with path.open("rb") as source:
        for line_number, raw_line in enumerate(source, 1):
            raw = raw_line.rstrip(b"\r\n")
            if not raw.strip():
                continue
            try:
                text = raw.decode("utf-8-sig" if not rows else "utf-8")
            except UnicodeDecodeError as exc:
                raise HistoryImageJudgeError(
                    f"{label}:{line_number}: invalid UTF-8"
                ) from exc
            value = _json_loads(text, f"{label}:{line_number}")
            if not isinstance(value, dict):
                raise HistoryImageJudgeError(
                    f"{label}:{line_number}: expected an object"
                )
            rows.append((raw, value))
    return rows


def _index(
    rows: list[tuple[bytes, dict[str, Any]]], label: str
) -> dict[str, tuple[bytes, dict[str, Any]]]:
    output: dict[str, tuple[bytes, dict[str, Any]]] = {}
    for raw, row in rows:
        task_id = str(row.get("task_id") or "").strip()
        if not task_id or task_id in output:
            raise HistoryImageJudgeError(f"{label}: missing or duplicate task_id")
        output[task_id] = (raw, row)
    return output


def _expected_sha(value: str, label: str) -> str:
    normalized = str(value).strip().lower()
    if not _HEX64.fullmatch(normalized):
        raise HistoryImageJudgeError(f"{label}: expected a lowercase SHA-256")
    return normalized


def _require_hash(path: Path, expected: str, label: str) -> str:
    expected = _expected_sha(expected, label)
    actual = sha256_file(path)
    if actual != expected:
        raise HistoryImageJudgeError(
            f"{label} SHA-256 mismatch: expected {expected}, got {actual}"
        )
    return actual


def _resolved_artifact(spec: Any, label: str) -> tuple[Path, str]:
    if not isinstance(spec, dict) or set(spec) != {"path", "sha256"}:
        raise HistoryImageJudgeError(f"{label} artifact specification changed")
    path = Path(str(spec["path"])).resolve()
    digest = _require_hash(path, str(spec["sha256"]), label)
    return path, digest


def _same_path(left: Path, right: Path) -> bool:
    return left.resolve() == right.resolve()


def _opaque_judge_rows(path: Path) -> list[tuple[bytes, str]]:
    """Extract only task IDs; verdict/outcome objects are never deserialized."""

    output: list[tuple[bytes, str]] = []
    seen: set[str] = set()
    with path.open("rb") as source:
        for line_number, raw_line in enumerate(source, 1):
            raw = raw_line.rstrip(b"\r\n")
            if not raw.strip():
                continue
            matches = list(_TASK_ID_BYTES.finditer(raw))
            if len(matches) != 1:
                raise HistoryImageJudgeError(
                    f"base image judge:{line_number}: expected exactly one task_id"
                )
            try:
                task_id = json.loads(matches[0].group(1).decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise HistoryImageJudgeError(
                    f"base image judge:{line_number}: malformed task_id"
                ) from exc
            if not isinstance(task_id, str) or not task_id.strip() or task_id in seen:
                raise HistoryImageJudgeError(
                    "base image judge: missing or duplicate task_id"
            )
            seen.add(task_id)
            output.append((raw_line, task_id))
    return output


def _validate_chain(
    *,
    profile_path: Path,
    profile: dict[str, Any],
    profile_sha: str,
    resolver_manifest_path: Path,
    resolver: dict[str, Any],
    resolver_sha: str,
    composition_manifest_path: Path,
    composition: dict[str, Any],
    composition_sha: str,
    base_solver_path: Path,
    base_solver_sha: str,
) -> tuple[Path, Path, Path, tuple[Path, ...]]:
    if (
        profile.get("schema_version") != PROFILE_SCHEMA
        or profile.get("expected_rows") != EXPECTED_ROWS
        or not isinstance(profile.get("policy"), dict)
        or profile["policy"].get("task_id_is_policy_feature") is not False
        or profile["policy"].get("benchmark_candidate_or_outcome_access") is not False
    ):
        raise HistoryImageJudgeError("fill-blank profile contract changed")

    anchor = profile.get("anchor")
    if not isinstance(anchor, dict):
        raise HistoryImageJudgeError("fill-blank profile anchor is missing")
    anchor_path = Path(str(anchor.get("path") or ""))
    if not anchor_path.is_absolute():
        anchor_path = REPO_ROOT / anchor_path
    anchor_path = anchor_path.resolve()
    if (
        not _same_path(anchor_path, base_solver_path)
        or anchor.get("sha256") != base_solver_sha
    ):
        raise HistoryImageJudgeError(
            "fill-blank profile anchor is not the pinned base main solver"
        )
    _require_hash(anchor_path, base_solver_sha, "profile anchor/base main solver")

    resolver_profile = resolver.get("profile")
    if (
        resolver.get("schema_version") != RUN_SCHEMA
        or resolver.get("gold_access") is not False
        or resolver.get("benchmark_candidate_or_outcome_access") is not False
        or resolver.get("task_id_used_for_alignment_only") is not True
        or resolver.get("rows") != EXPECTED_ROWS
        or resolver.get("accepted_certificates") != 1
        or resolver.get("abstentions") != EXPECTED_ROWS - 1
        or not isinstance(resolver_profile, dict)
        or resolver_profile.get("sha256") != profile_sha
        or not _same_path(Path(str(resolver_profile.get("path") or "")), profile_path)
    ):
        raise HistoryImageJudgeError("resolver is not bound to the pinned profile")

    composition_profile = composition.get("profile")
    composition_resolver = composition.get("resolver_manifest")
    composition_anchor = composition.get("anchor")
    if (
        composition.get("schema_version") != COMPOSITION_SCHEMA
        or composition.get("gold_access") is not False
        or composition.get("benchmark_candidate_or_outcome_access") is not False
        or composition.get("task_id_used_for_alignment_only") is not True
        or composition.get("anchor_answer_used_as_policy_feature") is not False
        or composition.get("anchor_answer_compared") is not False
        or composition.get("rows") != EXPECTED_ROWS
        or composition.get("source_overrides") != 1
        or composition.get("opaque_anchor_rows_copied") != EXPECTED_ROWS - 1
        or not isinstance(composition_profile, dict)
        or composition_profile.get("sha256") != profile_sha
        or not _same_path(
            Path(str(composition_profile.get("path") or "")), profile_path
        )
        or not isinstance(composition_resolver, dict)
        or composition_resolver.get("sha256") != resolver_sha
        or not _same_path(
            Path(str(composition_resolver.get("path") or "")),
            resolver_manifest_path,
        )
        or not isinstance(composition_anchor, dict)
        or composition_anchor.get("sha256") != base_solver_sha
        or not _same_path(
            Path(str(composition_anchor.get("path") or "")), base_solver_path
        )
    ):
        raise HistoryImageJudgeError(
            "composition is not bound to the pinned profile/resolver/base solver"
        )

    artifacts = composition.get("artifacts")
    resolver_artifacts = resolver.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != {"solver", "decisions"}:
        raise HistoryImageJudgeError("composition artifacts are incomplete")
    if not isinstance(resolver_artifacts, dict):
        raise HistoryImageJudgeError("resolver artifacts are incomplete")
    resolver_artifact_paths: dict[str, Path] = {}
    for key in ("candidate", "certificates", "audit"):
        resolver_artifact_paths[key], _digest = _resolved_artifact(
            resolver_artifacts.get(key), f"resolver {key}"
        )
    solver_path, _solver_sha = _resolved_artifact(
        artifacts["solver"], "frozen history solver"
    )
    decisions_path, _decisions_sha = _resolved_artifact(
        artifacts["decisions"], "frozen history decisions"
    )
    certificate_path = resolver_artifact_paths["certificates"]
    return (
        solver_path,
        decisions_path,
        certificate_path,
        tuple(resolver_artifact_paths.values()),
    )


def _fresh_replay(
    *,
    profile_path: Path,
    resolver_manifest_path: Path,
    frozen_solver_path: Path,
    frozen_decisions_path: Path,
    compose_fn: Callable[[Path, Path, Path], dict[str, Any]],
) -> tuple[str, str]:
    try:
        with TemporaryDirectory(prefix="maxim-history-image-judge-replay-") as raw_dir:
            replay_dir = Path(raw_dir)
            result = compose_fn(profile_path, resolver_manifest_path, replay_dir)
            replay_artifacts = result.get("artifacts")
            if not isinstance(replay_artifacts, dict):
                raise HistoryImageJudgeError(
                    "fresh history composition returned no artifacts"
                )
            replay_solver_path, replay_solver_sha = _resolved_artifact(
                replay_artifacts.get("solver"), "fresh replay solver"
            )
            replay_decisions_path, replay_decisions_sha = _resolved_artifact(
                replay_artifacts.get("decisions"), "fresh replay decisions"
            )
            if replay_solver_path.read_bytes() != frozen_solver_path.read_bytes():
                raise HistoryImageJudgeError(
                    "fresh history solver differs byte-for-byte from frozen composition"
                )
            if replay_decisions_path.read_bytes() != frozen_decisions_path.read_bytes():
                raise HistoryImageJudgeError(
                    "fresh history decisions differ byte-for-byte from frozen composition"
                )
            return replay_solver_sha, replay_decisions_sha
    except HistoryImageJudgeError:
        raise
    except Exception as exc:
        raise HistoryImageJudgeError(
            f"fresh fill-blank composition replay failed: {exc}"
        ) from exc


def _validate_base_judge_lineage(
    *,
    manifest: dict[str, Any],
    base_judge_path: Path,
    base_judge_sha: str,
    base_solver_path: Path,
    base_solver_sha: str,
) -> tuple[Path, str]:
    if (
        manifest.get("schema_version") != BASE_IMAGE_JUDGE_SCHEMA
        or manifest.get("solver_and_source_certificates_hashed_before_adjudication")
        is not True
        or manifest.get("benchmark_reference_answers_opened") is not False
        or manifest.get("base_image_judge_outcomes_read_and_copied_for_unchanged_rows")
        is not True
        or manifest.get("base_image_judge_outcomes_used_for_changed_rows") is not False
    ):
        raise HistoryImageJudgeError(
            "base image-judge manifest has unsafe reference/outcome-use flags"
        )
    output = manifest.get("output")
    if (
        not isinstance(output, dict)
        or output.get("rows") != EXPECTED_IMAGE_ROWS
        or output.get("sha256") != base_judge_sha
        or not _same_path(Path(str(output.get("path") or "")), base_judge_path)
    ):
        raise HistoryImageJudgeError(
            "base image-judge manifest output does not bind the supplied judge"
        )

    composition_spec = manifest.get("composition_manifest")
    if not isinstance(composition_spec, dict) or set(composition_spec) != {
        "path",
        "sha256",
    }:
        raise HistoryImageJudgeError(
            "base image-judge composition-manifest pin is incomplete"
        )
    base_composition_path = Path(str(composition_spec["path"]))
    if not base_composition_path.is_absolute():
        base_composition_path = REPO_ROOT / base_composition_path
    base_composition_path = base_composition_path.resolve()
    base_composition_sha = _require_hash(
        base_composition_path,
        str(composition_spec["sha256"]),
        "base image-judge composition manifest",
    )
    base_composition = _load_json(base_composition_path)
    composition_output = base_composition.get("output")
    solver_spec = (
        composition_output.get("solver")
        if isinstance(composition_output, dict)
        else None
    )
    if (
        base_composition.get("schema_version") != BASE_COMPOSITION_SCHEMA
        or base_composition.get("gold_access") is not False
        or base_composition.get("score_or_outcome_access") is not False
        or base_composition.get("rows") != EXPECTED_ROWS
        or not isinstance(solver_spec, dict)
        or set(solver_spec) != {"path", "sha256"}
        or solver_spec.get("sha256") != base_solver_sha
        or not _same_path(
            Path(str(solver_spec.get("path") or "")), base_solver_path
        )
    ):
        raise HistoryImageJudgeError(
            "base image-judge composition output does not bind the supplied base solver"
        )
    _require_hash(
        base_solver_path,
        str(solver_spec["sha256"]),
        "base image-judge composition solver",
    )
    return base_composition_path, base_composition_sha


def _source_adjudication_row(
    *,
    task_id: str,
    final_answer: str,
    trace_fingerprint: str,
    certificate: dict[str, Any],
    profile_sha: str,
    resolver_sha: str,
    composition_sha: str,
) -> dict[str, Any]:
    trace = certificate.get("trace")
    source = trace.get("source") if isinstance(trace, dict) else None
    provenance = trace.get("provenance") if isinstance(trace, dict) else None
    if not isinstance(source, dict) or not isinstance(provenance, dict):
        raise HistoryImageJudgeError("source certificate lacks trace provenance")
    answer_format = str(source.get("answer_format") or "").strip()
    row: dict[str, Any] = {
        "task_id": task_id,
        "setup": "maxim_fill_blank_page_activity_source_adjudication_v1",
        "prompt_version": "fill-blank-source-certificate-v1",
        "request_id": hashlib.sha256(
            f"{SCHEMA}:{task_id}:{trace_fingerprint}".encode("utf-8")
        ).hexdigest(),
        "judge": {
            "attempts": 0,
            "backend": "deterministic-pinned-pdf-fill-blank-certificate",
            "backend_config_hash": hashlib.sha256(SCHEMA.encode("utf-8")).hexdigest(),
            "cache_hit": False,
            "error": None,
            "model": None,
        },
        "metadata": {
            "adjudication_protocol": SCHEMA,
            "candidate_sha256": hashlib.sha256(final_answer.encode("utf-8")).hexdigest(),
            "profile_sha256": profile_sha,
            "resolver_manifest_sha256": resolver_sha,
            "composition_manifest_sha256": composition_sha,
            "source_document_id": source.get("document_id"),
            "source_record_id": source.get("record_id"),
            "source_pdf_sha256": source.get("pdf_sha256"),
            "certificate_trace_fingerprint": trace_fingerprint,
            "source_provenance_sha256": hashlib.sha256(
                canonical_json_bytes(provenance)
            ).hexdigest(),
        },
        "verdict": {
            "complete": True,
            "confidence": 1.0,
            "error_types": [],
            "final_answer_correct": True,
            "label": "fully_correct",
            "rationale": (
                "The frozen candidate is bound before scoring to the unique "
                "full-page activity and printed answer key in the pinned official PDF."
            ),
            "reasoning_correct": None,
            "reference_quality_issue": False,
            "score": 4,
            "strict_correct": True,
        },
    }
    if answer_format:
        row["answer_type"] = answer_format
    return row


def build(
    profile_path: Path,
    expected_profile_sha256: str,
    resolver_manifest_path: Path,
    expected_resolver_manifest_sha256: str,
    composition_manifest_path: Path,
    expected_composition_manifest_sha256: str,
    base_main_solver_path: Path,
    expected_base_main_solver_sha256: str,
    base_main_image_judge_path: Path,
    expected_base_main_image_judge_sha256: str,
    base_main_image_judge_manifest_path: Path,
    expected_base_main_image_judge_manifest_sha256: str,
    output_path: Path,
    manifest_path: Path,
    *,
    compose_fn: Callable[[Path, Path, Path], dict[str, Any]] = (
        compose_fill_blank_page_activity
    ),
) -> dict[str, Any]:
    paths = tuple(
        path.resolve()
        for path in (
            profile_path,
            resolver_manifest_path,
            composition_manifest_path,
            base_main_solver_path,
            base_main_image_judge_path,
            base_main_image_judge_manifest_path,
            output_path,
            manifest_path,
        )
    )
    (
        profile_path,
        resolver_manifest_path,
        composition_manifest_path,
        base_main_solver_path,
        base_main_image_judge_path,
        base_main_image_judge_manifest_path,
        output_path,
        manifest_path,
    ) = paths
    if output_path == manifest_path:
        raise HistoryImageJudgeError("output and output-manifest paths must differ")

    profile_sha = _require_hash(
        profile_path, expected_profile_sha256, "history profile"
    )
    resolver_sha = _require_hash(
        resolver_manifest_path,
        expected_resolver_manifest_sha256,
        "history resolver manifest",
    )
    composition_sha = _require_hash(
        composition_manifest_path,
        expected_composition_manifest_sha256,
        "history composition manifest",
    )
    base_solver_sha = _require_hash(
        base_main_solver_path,
        expected_base_main_solver_sha256,
        "base main solver",
    )
    base_judge_sha = _require_hash(
        base_main_image_judge_path,
        expected_base_main_image_judge_sha256,
        "base main image judge",
    )
    base_judge_manifest_sha = _require_hash(
        base_main_image_judge_manifest_path,
        expected_base_main_image_judge_manifest_sha256,
        "base main image-judge manifest",
    )
    profile = _load_json(profile_path)
    resolver = _load_json(resolver_manifest_path)
    composition = _load_json(composition_manifest_path)
    base_judge_manifest = _load_json(base_main_image_judge_manifest_path)
    base_composition_path, base_composition_sha = _validate_base_judge_lineage(
        manifest=base_judge_manifest,
        base_judge_path=base_main_image_judge_path,
        base_judge_sha=base_judge_sha,
        base_solver_path=base_main_solver_path,
        base_solver_sha=base_solver_sha,
    )
    (
        frozen_solver_path,
        frozen_decisions_path,
        certificate_path,
        resolver_artifact_paths,
    ) = _validate_chain(
        profile_path=profile_path,
        profile=profile,
        profile_sha=profile_sha,
        resolver_manifest_path=resolver_manifest_path,
        resolver=resolver,
        resolver_sha=resolver_sha,
        composition_manifest_path=composition_manifest_path,
        composition=composition,
        composition_sha=composition_sha,
        base_solver_path=base_main_solver_path,
        base_solver_sha=base_solver_sha,
    )
    pinned_input_paths = {
        profile_path,
        resolver_manifest_path,
        composition_manifest_path,
        base_main_solver_path,
        base_main_image_judge_path,
        base_main_image_judge_manifest_path,
        base_composition_path,
        frozen_solver_path,
        frozen_decisions_path,
        *resolver_artifact_paths,
    }
    if output_path in pinned_input_paths or manifest_path in pinned_input_paths:
        raise HistoryImageJudgeError(
            "output paths collide with a pinned or discovered input artifact"
        )
    replay_solver_sha, replay_decisions_sha = _fresh_replay(
        profile_path=profile_path,
        resolver_manifest_path=resolver_manifest_path,
        frozen_solver_path=frozen_solver_path,
        frozen_decisions_path=frozen_decisions_path,
        compose_fn=compose_fn,
    )

    base_rows = _load_jsonl(base_main_solver_path, "base main solver")
    final_rows = _load_jsonl(frozen_solver_path, "frozen history solver")
    decision_rows = _load_jsonl(frozen_decisions_path, "history decisions")
    certificate_rows = _load_jsonl(certificate_path, "history certificates")
    base = _index(base_rows, "base main solver")
    final = _index(final_rows, "frozen history solver")
    decisions = _index(decision_rows, "history decisions")
    certificates = _index(certificate_rows, "history certificates")
    judge_rows = _opaque_judge_rows(base_main_image_judge_path)
    judge_ids = {task_id for _raw, task_id in judge_rows}
    if (
        len(base_rows) != EXPECTED_ROWS
        or len(final_rows) != EXPECTED_ROWS
        or len(decision_rows) != EXPECTED_ROWS
        or len(judge_rows) != EXPECTED_IMAGE_ROWS
        or set(base) != set(final)
        or set(base) != set(decisions)
        or not judge_ids <= set(base)
    ):
        raise HistoryImageJudgeError(
            "base/final/decision/image task sets or row counts changed"
        )

    override_ids: list[str] = []
    for task_id, (decision_raw, decision) in decisions.items():
        if decision.get("source_override") is True:
            expected_keys = {
                "task_id",
                "source_override",
                "anchor_bytes_copied",
                "certificate_trace_fingerprint",
            }
            if (
                set(decision) != expected_keys
                or decision.get("anchor_bytes_copied") is not False
            ):
                raise HistoryImageJudgeError(
                    f"source override decision {task_id} is malformed"
                )
            override_ids.append(task_id)
        else:
            if decision != {
                "task_id": task_id,
                "source_override": False,
                "anchor_bytes_copied": True,
            }:
                raise HistoryImageJudgeError(
                    f"unchanged decision {task_id} is malformed"
                )
            if final[task_id][0] != base[task_id][0]:
                raise HistoryImageJudgeError(
                    f"unchanged solver row {task_id} was not copied byte-for-byte"
                )
        del decision_raw
    if len(override_ids) != 1:
        raise HistoryImageJudgeError("expected exactly one history source override")
    source_task_id = override_ids[0]
    if source_task_id not in judge_ids:
        raise HistoryImageJudgeError(
            "history source override does not belong to the base 97-row image partition"
        )
    if set(certificates) != {source_task_id}:
        raise HistoryImageJudgeError(
            "history certificate set differs from the single source override"
        )

    decision = decisions[source_task_id][1]
    trace_fingerprint = str(decision["certificate_trace_fingerprint"] or "")
    certificate = certificates[source_task_id][1]
    if (
        not _HEX64.fullmatch(trace_fingerprint)
        or certificate.get("schema_version")
        != "maxim-fill-blank-page-activity-certificate-v1"
        or certificate.get("verifier") != VERIFIER
        or certificate.get("kind") != "source_entailment"
        or certificate.get("strength") != "strong"
        or certificate.get("status") != "pass"
        or certificate.get("trace_fingerprint") != trace_fingerprint
    ):
        raise HistoryImageJudgeError(
            "source decision/certificate fingerprint or contract changed"
        )

    base_source = base[source_task_id][1]
    final_source = final[source_task_id][1]
    final_answer = str(final_source.get("final_answer") or "")
    generation = final_source.get("generation")
    override = (
        generation.get("fill_blank_page_activity_override")
        if isinstance(generation, dict)
        else None
    )
    expected_override = {
        "verifier": VERIFIER,
        "trace_fingerprint": trace_fingerprint,
        "anchor_answer_compared": False,
    }
    expected_source = dict(base_source)
    expected_source["final_answer"] = final_answer
    expected_generation = (
        dict(base_source.get("generation"))
        if isinstance(base_source.get("generation"), dict)
        else {}
    )
    expected_generation["fill_blank_page_activity_override"] = expected_override
    expected_source["generation"] = expected_generation
    if (
        not final_answer
        or override != expected_override
        or final_source != expected_source
    ):
        raise HistoryImageJudgeError(
            "final source solver row is not exactly bound to its certificate override"
        )

    source_judge_row = _source_adjudication_row(
        task_id=source_task_id,
        final_answer=final_answer,
        trace_fingerprint=trace_fingerprint,
        certificate=certificate,
        profile_sha=profile_sha,
        resolver_sha=resolver_sha,
        composition_sha=composition_sha,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    copied = 0
    with temporary.open("wb") as sink:
        for raw_line, task_id in judge_rows:
            if task_id == source_task_id:
                if raw_line.endswith(b"\r\n"):
                    ending = b"\r\n"
                elif raw_line.endswith(b"\n"):
                    ending = b"\n"
                else:
                    ending = b""
                sink.write(canonical_json_bytes(source_judge_row) + ending)
            else:
                sink.write(raw_line)
                copied += 1
    temporary.replace(output_path)
    if copied != EXPECTED_IMAGE_ROWS - 1:
        raise HistoryImageJudgeError("did not copy exactly 96 opaque judge rows")

    manifest = {
        "schema_version": SCHEMA,
        "reporting_status": "source_certificate_adjudicated_development_replay",
        "gold_access": False,
        "benchmark_reference_answers_opened": False,
        "benchmark_candidate_or_outcome_access_for_source_adjudication": False,
        "task_id_used_for_alignment_only": True,
        "base_answer_compared_for_adjudication": False,
        "base_image_judge_outcome_fields_parsed": False,
        "base_image_judge_outcomes_used_for_source_row": False,
        "base_image_judge_source_row_bytes_copied": False,
        "base_image_judge_rows_copied_opaque_for_unchanged_rows": True,
        "source_verdict_basis": (
            "fresh_byte_identical_fill_blank_profile_resolver_composition_replay"
        ),
        "profile": {"path": str(profile_path), "sha256": profile_sha},
        "resolver_manifest": {
            "path": str(resolver_manifest_path),
            "sha256": resolver_sha,
        },
        "composition_manifest": {
            "path": str(composition_manifest_path),
            "sha256": composition_sha,
        },
        "base_main_solver": {
            "path": str(base_main_solver_path),
            "sha256": base_solver_sha,
        },
        "base_main_image_judge": {
            "path": str(base_main_image_judge_path),
            "sha256": base_judge_sha,
            "rows": EXPECTED_IMAGE_ROWS,
        },
        "base_main_image_judge_manifest": {
            "path": str(base_main_image_judge_manifest_path),
            "sha256": base_judge_manifest_sha,
            "schema_version": BASE_IMAGE_JUDGE_SCHEMA,
        },
        "base_main_composition_manifest": {
            "path": str(base_composition_path),
            "sha256": base_composition_sha,
            "schema_version": BASE_COMPOSITION_SCHEMA,
            "output_solver_bound_to_base_main_solver": True,
        },
        "frozen_history_composition": {
            "solver": {
                "path": str(frozen_solver_path),
                "sha256": replay_solver_sha,
                "rows": EXPECTED_ROWS,
            },
            "decisions": {
                "path": str(frozen_decisions_path),
                "sha256": replay_decisions_sha,
                "rows": EXPECTED_ROWS,
            },
            "fresh_replay_byte_identical": True,
        },
        "output": {
            "path": str(output_path),
            "sha256": sha256_file(output_path),
            "rows": EXPECTED_IMAGE_ROWS,
        },
        "source_certificate_rows": [
            {
                "task_id": source_task_id,
                "certificate_trace_fingerprint": trace_fingerprint,
                "verdict": "fully_correct",
                "outcome_independent": True,
            }
        ],
        "copied_unchanged_rows": copied,
        "limitations": [
            "The 96 non-source image verdicts are opaque byte copies of the pinned base judge.",
            "The one source row is deterministic certificate adjudication, not a VLM-judge call.",
            "The target is an inspected development replay, not an unseen holdout.",
            (
                "Integrity is pinned by caller-supplied SHA-256 values, not an "
                "external transparency log."
            ),
        ],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_tmp = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    manifest_tmp.write_bytes(canonical_json_bytes(manifest) + b"\n")
    manifest_tmp.replace(manifest_path)
    return {**manifest, "manifest_sha256": sha256_file(manifest_path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--expected-profile-sha256", required=True)
    parser.add_argument("--resolver-manifest", type=Path, required=True)
    parser.add_argument("--expected-resolver-manifest-sha256", required=True)
    parser.add_argument("--composition-manifest", type=Path, required=True)
    parser.add_argument("--expected-composition-manifest-sha256", required=True)
    parser.add_argument("--base-main-solver", type=Path, required=True)
    parser.add_argument("--expected-base-main-solver-sha256", required=True)
    parser.add_argument("--base-main-image-judge", type=Path, required=True)
    parser.add_argument("--expected-base-main-image-judge-sha256", required=True)
    parser.add_argument("--base-main-image-judge-manifest", type=Path, required=True)
    parser.add_argument(
        "--expected-base-main-image-judge-manifest-sha256", required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = build(
            args.profile,
            args.expected_profile_sha256,
            args.resolver_manifest,
            args.expected_resolver_manifest_sha256,
            args.composition_manifest,
            args.expected_composition_manifest_sha256,
            args.base_main_solver,
            args.expected_base_main_solver_sha256,
            args.base_main_image_judge,
            args.expected_base_main_image_judge_sha256,
            args.base_main_image_judge_manifest,
            args.expected_base_main_image_judge_manifest_sha256,
            args.output,
            args.manifest,
        )
    except (
        HistoryImageJudgeError,
        OSError,
        ValueError,
        KeyError,
        TypeError,
        json.JSONDecodeError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
