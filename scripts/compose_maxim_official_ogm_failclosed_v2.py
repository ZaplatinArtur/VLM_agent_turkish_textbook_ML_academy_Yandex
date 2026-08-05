#!/usr/bin/env python3
"""Overlay independently certified official-source keys on a pinned anchor.

The historical filename is retained for compatibility; the composer accepts
both the OGM API/PDF resolver and reviewed direct-PDF resolver profiles.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evidence_os.adapters import certificate_from_record  # noqa: E402
from evidence_os.contracts import (  # noqa: E402
    CandidateEnvelope,
    CertificateKind,
    DecisionAction,
    FrozenProfile,
    InferenceBundle,
    ProblemInput,
)
from evidence_os.official_ogm import (  # noqa: E402
    VERIFIER as OGM_VERIFIER,
    OfficialSourceError,
    PageMatcher,
    observed_source_question_marker,
    parser_observation,
    parser_observation_allow_missing_number,
    parser_observation_primary_layout_number,
    problem_for,
    sha256_file,
)
from evidence_os.official_pdf import VERIFIER as DIRECT_PDF_VERIFIER  # noqa: E402
from evidence_os.image_only_activity import (  # noqa: E402
    IMAGE_ONLY_ACTIVITY_ARTIFACT_ROLE,
    IMAGE_ONLY_ACTIVITY_TRACE_CHECKS,
    OBSERVATION_KIND as IMAGE_ONLY_ACTIVITY_OBSERVATION_KIND,
    RECORD_SELECTION_POLICY as IMAGE_ONLY_ACTIVITY_RECORD_SELECTION_POLICY,
    ImageOnlyActivityError,
    load_image_only_activity_visual_artifact_json,
    problem_for_image_only_activity,
    project_image_only_activity_observation,
    resolve_image_only_activity_question,
    verified_image_only_activity_bindings_from_artifact,
)
from evidence_os.official_workbook import (  # noqa: E402
    VERIFIER as WORKBOOK_VERIFIER,
    WorkbookThresholds,
    activity_label_projection_enabled,
    document_for_source,
    observed_coordinate_question_binding,
    observed_inline_question_binding,
    parse_workbook_index,
    resolve_workbook_question,
    validate_fail_closed_workbook_policy,
    verify_workbook_index_pdf,
)
from evidence_os.policy import EvidencePolicy  # noqa: E402
from evidence_os.visual_coordinate_binding import (  # noqa: E402
    ActivityVisualObservationRef,
    ActivityVisualRecordRef,
    VisualBindingThresholds,
    VisualCoordinateBindingError,
    load_activity_visual_artifact_json,
    verified_activity_bindings_from_artifact,
)


SCHEMA = "maxim-official-source-failclosed-composition-v2"
_PROFILE_CONTRACTS = {
    "maxim-official-ogm-exact-source-profile-v2": (
        "maxim-official-ogm-exact-source-run-v2",
        OGM_VERIFIER,
        10,
    ),
    "maxim-official-direct-pdf-profile-v2": (
        "maxim-official-direct-pdf-run-v2",
        DIRECT_PDF_VERIFIER,
        5,
    ),
    "maxim-public-workbook-profile-v1": (
        "maxim-public-workbook-run-v1",
        WORKBOOK_VERIFIER,
        8,
    ),
}
_WORKBOOK_TRACE_CHECKS = frozenset(
    {
        "strict_public_document_identity",
        "unique_content_page",
        "unique_source_question_record",
        "question_binding",
        "printed_number_visible_on_page",
        "reviewed_embedded_key",
        "valid_source_answer",
        "source_address_not_task_id",
    }
)


class CompositionError(RuntimeError):
    pass


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise CompositionError(f"{path}: expected object")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise CompositionError(f"{path}:{line_number}: expected object")
            rows.append(value)
    return rows


def _index(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        task_id = str(row.get("task_id") or "").strip()
        if not task_id or task_id in output:
            raise CompositionError(f"{label}: missing/duplicate task_id")
        output[task_id] = row
    return output


def _path(configured: str) -> Path:
    raw = Path(configured)
    return raw.resolve() if raw.is_absolute() else (REPO_ROOT / raw).resolve()


def _require_hash(path: Path, expected: str, label: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise CompositionError(f"{label} SHA-256 mismatch: expected {expected}, got {actual}")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as sink:
        for row in rows:
            sink.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                + "\n"
            )
    temporary.replace(path)


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _fresh_rebuild_activity_visual_artifact(
    *,
    profile_path: Path,
    parser_path: Path,
    locator_path: Path,
    source_index_path: Path,
    visual_path: Path,
    expected_visual_sha256: str,
    visual_payload: dict[str, Any],
    document_pdf_paths: dict[str, Path],
    generator_filename: str = "build_maxim_activity_visual_binding_v1.py",
    replay_mode: str = "fresh_source_only_poppler_sift_exact_bytes_v1",
    temp_prefix: str = "maxim_activity_visual_exact_replay_",
) -> dict[str, Any]:
    """Rebuild Poppler/SIFT evidence in a clean subprocess and require exact bytes."""

    runtime = visual_payload.get("runtime")
    inputs = visual_payload.get("inputs")
    if not isinstance(runtime, dict) or not isinstance(inputs, dict):
        raise CompositionError("activity visual replay lacks frozen runtime inputs")
    python_spec = runtime.get("python")
    poppler = runtime.get("poppler")
    if not isinstance(python_spec, dict) or not isinstance(poppler, dict):
        raise CompositionError("activity visual replay runtime is malformed")
    frozen_python = Path(str(python_spec.get("executable") or "")).resolve()
    current_python = Path(sys.executable).resolve()
    if frozen_python != current_python or not current_python.is_file():
        raise CompositionError(
            "activity visual replay must use the frozen current Python executable"
        )
    python_sha256 = sha256_file(current_python)

    tool_paths: dict[str, Path] = {}
    tool_hashes: dict[str, str] = {}
    for name in ("pdftoppm", "pdfinfo"):
        tool_spec = poppler.get(name)
        if not isinstance(tool_spec, dict):
            raise CompositionError(f"activity visual replay lacks {name}")
        tool_path = Path(str(tool_spec.get("path") or "")).resolve()
        expected_tool_sha = str(tool_spec.get("sha256") or "")
        _require_hash(tool_path, expected_tool_sha, f"activity visual {name}")
        tool_paths[name] = tool_path
        tool_hashes[name] = expected_tool_sha

    package_root = _path(str(runtime.get("package_root") or ""))
    task_image_dir = _path(str(inputs.get("task_image_dir") or ""))
    repo_root = REPO_ROOT.resolve()
    for candidate, label, require_directory in (
        (package_root, "activity visual package root", True),
        (task_image_dir, "activity visual task-image directory", True),
    ):
        try:
            candidate.relative_to(repo_root)
        except ValueError as exc:
            raise CompositionError(f"{label} escapes the repository") from exc
        if require_directory and not candidate.is_dir():
            raise CompositionError(f"{label} is unavailable")

    raw_documents = inputs.get("documents")
    if not isinstance(raw_documents, dict) or not raw_documents:
        raise CompositionError("activity visual replay has no document inventory")
    activity_document_paths: dict[str, Path] = {}
    for document_id in sorted(raw_documents):
        pdf_path = document_pdf_paths.get(document_id)
        if pdf_path is None:
            raise CompositionError(
                f"activity visual replay lacks verified PDF {document_id}"
            )
        activity_document_paths[document_id] = pdf_path.resolve()

    generator_path = (REPO_ROOT / "scripts" / generator_filename).resolve()
    if not generator_path.is_file():
        raise CompositionError("activity visual source-only generator is unavailable")
    generator_sha256 = sha256_file(generator_path)
    frozen_sha256 = sha256_file(visual_path)
    if frozen_sha256 != expected_visual_sha256:
        raise CompositionError("activity visual frozen artifact changed before replay")

    controlled_environment = {
        "PYTHONPATH": str(package_root),
        "PYTHONNOUSERSITE": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONIOENCODING": "utf-8",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
    }
    command_projection = {
        "generator": str(generator_path),
        "profile": str(profile_path),
        "parser_jsonl": str(parser_path),
        "source_locators": str(locator_path),
        "source_index": str(source_index_path),
        "task_image_dir": str(task_image_dir),
        "documents": {
            document_id: str(pdf_path)
            for document_id, pdf_path in activity_document_paths.items()
        },
        "pdftoppm": str(tool_paths["pdftoppm"]),
        "pdfinfo": str(tool_paths["pdfinfo"]),
        "environment": controlled_environment,
    }
    command_projection_sha256 = _canonical_sha256(command_projection)

    temp_parent = (REPO_ROOT / "tmp").resolve()
    temp_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=temp_prefix,
        dir=temp_parent,
    ) as raw_temp:
        rebuilt_path = Path(raw_temp) / "rebuilt.json"
        command = [
            str(current_python),
            str(generator_path),
            "--profile",
            str(profile_path),
            "--parser-jsonl",
            str(parser_path),
            "--source-locators",
            str(locator_path),
            "--source-index",
            str(source_index_path),
            "--task-image-dir",
            str(task_image_dir),
        ]
        for document_id, pdf_path in activity_document_paths.items():
            command.extend(("--document", f"{document_id}={pdf_path}"))
        command.extend(
            (
                "--pdftoppm",
                str(tool_paths["pdftoppm"]),
                "--pdfinfo",
                str(tool_paths["pdfinfo"]),
                "--output-json",
                str(rebuilt_path),
            )
        )
        environment = os.environ.copy()
        environment.update(controlled_environment)
        try:
            completed = subprocess.run(
                command,
                cwd=REPO_ROOT,
                env=environment,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
                shell=False,
                timeout=1_800,
            )
        except subprocess.TimeoutExpired as exc:
            raise CompositionError(
                "activity visual exact rebuild exceeded 1800 seconds"
            ) from exc
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip().splitlines()
            reason = detail[-1] if detail else f"exit {completed.returncode}"
            raise CompositionError(
                f"activity visual exact rebuild failed: {reason}"
            )
        try:
            report = json.loads(completed.stdout.strip())
        except json.JSONDecodeError as exc:
            raise CompositionError(
                "activity visual exact rebuild returned malformed JSON"
            ) from exc
        if (
            not isinstance(report, dict)
            or set(report) != {"output_json", "output_sha256", "summary"}
            or Path(str(report.get("output_json") or "")).resolve() != rebuilt_path.resolve()
            or not rebuilt_path.is_file()
        ):
            raise CompositionError("activity visual exact rebuild report is incomplete")
        rebuilt_sha256 = sha256_file(rebuilt_path)
        if report.get("output_sha256") != rebuilt_sha256:
            raise CompositionError(
                "activity visual generator report disagrees with rebuilt bytes"
            )
        if rebuilt_sha256 != expected_visual_sha256:
            raise CompositionError(
                "activity visual exact rebuild differs from the frozen profile"
            )
        frozen_bytes = visual_path.read_bytes()
        rebuilt_bytes = rebuilt_path.read_bytes()
        if rebuilt_bytes != frozen_bytes:
            raise CompositionError(
                "activity visual exact rebuild is not byte-identical"
            )

    return {
        "mode": replay_mode,
        "generator": {
            "path": str(generator_path),
            "sha256": generator_sha256,
        },
        "frozen_artifact": {
            "path": str(visual_path),
            "sha256": frozen_sha256,
            "size_bytes": len(frozen_bytes),
        },
        "reproduced_artifact": {
            "sha256": rebuilt_sha256,
            "size_bytes": len(rebuilt_bytes),
        },
        "exact_byte_identity": True,
        "command_projection_sha256": command_projection_sha256,
        "runtime": {
            "python_executable_sha256": python_sha256,
            "pdftoppm_sha256": tool_hashes["pdftoppm"],
            "pdfinfo_sha256": tool_hashes["pdfinfo"],
            "runtime_projection_sha256": _canonical_sha256(runtime),
        },
        "summary": report["summary"],
        "benchmark_answer_candidate_outcome_artifacts_read": False,
    }


def _fresh_rebuild_image_only_activity_visual_artifact(
    *,
    parser_path: Path,
    locator_path: Path,
    visual_path: Path,
    expected_visual_sha256: str,
    visual_payload: dict[str, Any],
    document_pdf_paths: dict[str, Path],
) -> dict[str, Any]:
    """Rebuild the image-only artifact from its isolated source-only profile."""

    inputs = visual_payload.get("inputs")
    if not isinstance(inputs, dict):
        raise CompositionError("image-only visual replay lacks frozen inputs")
    pinned_paths: dict[str, Path] = {}
    for key in ("profile", "parser_observations", "source_locators", "source_index"):
        spec = inputs.get(key)
        if not isinstance(spec, dict) or not isinstance(spec.get("sha256"), str):
            raise CompositionError(f"image-only visual replay lacks {key}")
        path = _path(str(spec.get("path") or ""))
        _require_hash(path, str(spec["sha256"]), f"image-only visual {key}")
        pinned_paths[key] = path
    if (
        pinned_paths["parser_observations"] != parser_path.resolve()
        or pinned_paths["source_locators"] != locator_path.resolve()
    ):
        raise CompositionError(
            "image-only visual replay parser/locator inputs differ from runtime"
        )
    return _fresh_rebuild_activity_visual_artifact(
        profile_path=pinned_paths["profile"],
        parser_path=parser_path,
        locator_path=locator_path,
        source_index_path=pinned_paths["source_index"],
        visual_path=visual_path,
        expected_visual_sha256=expected_visual_sha256,
        visual_payload=visual_payload,
        document_pdf_paths=document_pdf_paths,
        generator_filename="build_maxim_image_only_activity_visual_binding_v1.py",
        replay_mode=(
            "fresh_source_only_poppler_sift_image_only_activity_exact_bytes_v1"
        ),
        temp_prefix="maxim_image_only_activity_visual_exact_replay_",
    )


def _validate_workbook_certificate_artifact(
    *,
    task_id: str,
    raw_candidate: dict[str, Any],
    raw_certificate: dict[str, Any],
    trace: dict[str, Any],
    profile: dict[str, Any],
    profile_sha: str,
) -> None:
    generation = raw_candidate.get("generation")
    if (
        raw_candidate.get("abstain") is not False
        or raw_candidate.get("error") is not None
        or not isinstance(generation, dict)
        or generation.get("gold_access") is not False
        or generation.get("resolver") != WORKBOOK_VERIFIER
        or generation.get("source_certificate") is not True
    ):
        raise CompositionError(f"workbook candidate {task_id} violates resolver invariants")
    if (
        raw_certificate.get("schema_version")
        != "maxim-public-workbook-certificate-v1"
        or raw_certificate.get("verifier") != WORKBOOK_VERIFIER
        or trace.get("schema_version") != "public-workbook-source-trace-v1"
        or trace.get("verifier") != WORKBOOK_VERIFIER
        or trace.get("accepted") is not True
    ):
        raise CompositionError(f"workbook certificate {task_id} has an invalid contract")
    trace_checks = trace.get("checks")
    deterministic_checks = raw_certificate.get("deterministic_checks")
    trace_source_for_checks = trace.get("source")
    is_image_only_activity = isinstance(
        trace.get("image_only_activity_binding"), dict
    )
    if is_image_only_activity:
        expected_trace_checks = set(IMAGE_ONLY_ACTIVITY_TRACE_CHECKS)
    else:
        expected_trace_checks = set(_WORKBOOK_TRACE_CHECKS)
        if (
            isinstance(trace_source_for_checks, dict)
            and trace_source_for_checks.get("key_binding_kind")
            in {"coordinate_choice_answer_key", "inline_solution_projected"}
        ):
            expected_trace_checks.add("observed_question_text_matches_source")
        if isinstance(trace.get("visual_page_binding"), dict):
            expected_trace_checks.add("visual_page_binding")
    if (
        not isinstance(trace_checks, dict)
        or set(trace_checks) != expected_trace_checks
        or any(value is not True for value in trace_checks.values())
        or not isinstance(deterministic_checks, list)
        or len(deterministic_checks) != len(expected_trace_checks)
        or any(value is not True for value in deterministic_checks)
    ):
        raise CompositionError(f"workbook certificate {task_id} has inconsistent checks")
    inputs = profile.get("inputs")
    runtime = profile.get("runtime")
    documents = profile.get("documents")
    provenance = trace.get("provenance")
    trace_source = trace.get("source")
    if not all(
        isinstance(value, expected)
        for value, expected in (
            (inputs, dict),
            (runtime, dict),
            (documents, list),
            (provenance, dict),
            (trace_source, dict),
        )
    ):
        raise CompositionError(f"workbook certificate {task_id} lacks frozen provenance")
    document_id = str(trace_source.get("document_id") or "")
    frozen_documents = {
        str(item.get("document_id") or ""): item
        for item in documents
        if isinstance(item, dict)
    }
    frozen_document = frozen_documents.get(document_id)
    if frozen_document is None:
        raise CompositionError(f"workbook certificate {task_id} names an unknown document")
    expected_provenance = {
        "profile_sha256": profile_sha,
        "parser_observations_sha256": str(inputs["parser_observations"]["sha256"]),
        "source_locators_sha256": str(inputs["source_locators"]["sha256"]),
        "source_index_sha256": str(inputs["source_index"]["sha256"]),
        "workbook_pdf_sha256": str(frozen_document["pdf_sha256"]),
        "pypdf_version": str(runtime["pypdf_version"]),
        "pdfplumber_version": str(runtime["pdfplumber_version"]),
    }
    visual_spec = inputs.get("activity_visual_evidence")
    if isinstance(visual_spec, dict):
        expected_provenance["activity_visual_evidence_sha256"] = str(
            visual_spec.get("sha256") or ""
        )
    image_only_visual_spec = inputs.get("image_only_activity_visual_evidence")
    if isinstance(image_only_visual_spec, dict):
        expected_provenance[
            "image_only_activity_visual_evidence_sha256"
        ] = str(image_only_visual_spec.get("sha256") or "")
    if (
        set(provenance) != set(expected_provenance) | {"source_verification"}
        or any(provenance.get(key) != value for key, value in expected_provenance.items())
    ):
        raise CompositionError(f"workbook certificate {task_id} provenance was altered")
    source_verification = provenance.get("source_verification")
    content_marker_counts = (
        source_verification.get("content_marker_counts")
        if isinstance(source_verification, dict)
        else None
    )
    answer_format = trace_source.get("answer_format")
    key_binding_kind = trace_source.get("key_binding_kind")
    projection_sha = str(trace_source.get("key_projection_sha256") or "")
    content_projection_sha = str(
        trace_source.get("content_projection_sha256") or ""
    )
    binding_projection_sha = str(
        trace_source.get("binding_projection_sha256") or ""
    )
    question_marker_kind = str(
        trace_source.get("question_marker_kind") or "numbered_item"
    )
    key_context_page_number = trace_source.get("key_context_page_number")
    binding_is_supported = (
        answer_format == "choice"
        and key_binding_kind in {
            "inline_solution",
            "answer_key_table",
            "answer_key_list",
        }
        and not projection_sha
        and not content_projection_sha
        and question_marker_kind == "numbered_item"
    ) or (
        answer_format == "choice"
        and key_binding_kind == "inline_solution_projected"
        and question_marker_kind == "numbered_item"
        and not projection_sha
        and len(content_projection_sha) == 64
        and all(
            character in "0123456789abcdef"
            for character in content_projection_sha
        )
        and not binding_projection_sha
        and isinstance(key_context_page_number, int)
        and not isinstance(key_context_page_number, bool)
        and key_context_page_number >= 1
    ) or (
        answer_format == "choice"
        and key_binding_kind == "coordinate_choice_answer_key"
        and question_marker_kind == "numbered_item"
        and isinstance(key_context_page_number, int)
        and not isinstance(key_context_page_number, bool)
        and key_context_page_number >= 1
        and not binding_projection_sha
        and all(
            len(value) == 64
            and all(character in "0123456789abcdef" for character in value)
            for value in (projection_sha, content_projection_sha)
        )
        and isinstance(trace_source.get("content_section"), str)
        and bool(trace_source.get("content_section"))
        and isinstance(trace_source.get("section"), str)
        and bool(trace_source.get("section"))
        and isinstance(trace_source.get("test_variant"), str)
        and bool(trace_source.get("test_variant"))
    ) or (
        answer_format == "short_text"
        and key_binding_kind == "coordinate_answer_key"
        and len(projection_sha) == 64
        and all(character in "0123456789abcdef" for character in projection_sha)
        and not content_projection_sha
        and question_marker_kind == "numbered_item"
    ) or (
        answer_format == "short_text"
        and key_binding_kind == "coordinate_table_answer_key"
        and question_marker_kind in {"numbered_item", "example_label"}
        and isinstance(key_context_page_number, int)
        and not isinstance(key_context_page_number, bool)
        and key_context_page_number >= 1
        and len(projection_sha) == 64
        and all(character in "0123456789abcdef" for character in projection_sha)
        and len(content_projection_sha) == 64
        and all(
            character in "0123456789abcdef"
            for character in content_projection_sha
        )
    ) or (
        answer_format == "short_text"
        and key_binding_kind == "activity_answer_key"
        and question_marker_kind == "activity_label"
        and isinstance(key_context_page_number, int)
        and not isinstance(key_context_page_number, bool)
        and key_context_page_number >= 1
        and all(
            len(value) == 64
            and all(character in "0123456789abcdef" for character in value)
            for value in (
                projection_sha,
                content_projection_sha,
                binding_projection_sha,
            )
        )
        and isinstance(trace_source.get("source_unit_number"), int)
        and not isinstance(trace_source.get("source_unit_number"), bool)
        and trace_source.get("source_unit_number") >= 1
        and trace_source.get("source_answer_format")
        in {"labelled_short_text", "numbered_short_text", "scalar_exit"}
        and isinstance(trace_source.get("test_variant"), str)
        and bool(trace_source.get("test_variant"))
    )
    if (
        trace_source.get("pdf_sha256") != frozen_document.get("pdf_sha256")
        or not binding_is_supported
        or not isinstance(source_verification, dict)
        or int(source_verification.get("records", 0)) < 1
        or source_verification.get("verified_records")
        != source_verification.get("records")
        or not isinstance(content_marker_counts, dict)
        or content_marker_counts.get(str(trace_source.get("record_id") or "")) != 1
    ):
        raise CompositionError(f"workbook certificate {task_id} lacks PDF-bound evidence")


def _validate_workbook_trace_against_source(
    *,
    task_id: str,
    candidate_answer: str,
    observation: Any,
    trace: dict[str, Any],
    source_records: dict[str, tuple[Any, Any]],
    allow_example_label_marker: bool,
    allow_activity_label_marker: bool = False,
    thresholds: WorkbookThresholds | None = None,
) -> None:
    trace_source = trace.get("source")
    trace_observation = trace.get("observation")
    trace_match = trace.get("match")
    if not all(
        isinstance(value, dict)
        for value in (trace_source, trace_observation, trace_match)
    ):
        raise CompositionError(f"workbook certificate {task_id} lacks source trace")
    record_id = str(trace_source.get("record_id") or "")
    source_entry = source_records.get(record_id)
    if source_entry is None:
        raise CompositionError(
            f"workbook certificate {task_id} does not name a frozen source record"
        )
    document, question = source_entry
    expected_content_bbox = (
        list(question.content_bbox) if question.content_bbox is not None else None
    )
    expected_source = {
        "document_id": document.document_id,
        "public_locator": document.identity.public_locator,
        "name": document.identity.name,
        "pdf_sha256": document.pdf_sha256,
        "matched_page_number": question.content_page_number,
        "question_number": question.question_number,
        "question_marker_kind": question.question_marker_kind,
        "record_id": question.record_id,
        "answer_format": question.answer_format,
        "key_binding_kind": question.key_binding_kind,
        "key_page_number": question.key_page_number,
        "key_context_page_number": question.key_context_page_number,
        "key_bbox": list(question.key_bbox),
        "content_bbox": expected_content_bbox,
        "key_projection_sha256": question.key_projection_sha256,
        "content_projection_sha256": question.content_projection_sha256,
    }
    if question.key_binding_kind == "activity_answer_key":
        expected_source.update(
            {
                "binding_projection_sha256": question.binding_projection_sha256,
                "source_answer_format": question.source_answer_format,
                "source_unit_number": question.source_unit_number,
                "test_variant": question.test_variant,
            }
        )
    elif question.key_binding_kind == "coordinate_choice_answer_key":
        expected_source.update(
            {
                "content_section": question.content_section,
                "section": question.section,
                "test_variant": question.test_variant,
            }
        )
    actual_source = {
        key: trace_source.get(key) for key in expected_source
    }
    if question.key_binding_kind not in {
        "coordinate_table_answer_key",
        "activity_answer_key",
        "coordinate_choice_answer_key",
        "inline_solution_projected",
    }:
        actual_source["question_marker_kind"] = str(
            trace_source.get("question_marker_kind") or "numbered_item"
        )
        actual_source["key_context_page_number"] = trace_source.get(
            "key_context_page_number", question.key_page_number
        )
        actual_source["key_projection_sha256"] = str(
            trace_source.get("key_projection_sha256") or ""
        )
        actual_source["content_projection_sha256"] = str(
            trace_source.get("content_projection_sha256") or ""
        )
    if any(actual_source[key] != value for key, value in expected_source.items()):
        raise CompositionError(
            f"workbook certificate {task_id} source trace differs from frozen index"
        )
    if candidate_answer != question.answer.strip():
        raise CompositionError(
            f"workbook certificate {task_id} answer differs from frozen source"
        )
    marker_kind, marker_number = observed_source_question_marker(observation)
    if (
        marker_kind != question.question_marker_kind
        or marker_number != question.question_number
        or (
            marker_kind == "example_label"
            and not allow_example_label_marker
        )
        or (
            marker_kind == "activity_label"
            and not allow_activity_label_marker
        )
    ):
        raise CompositionError(
            f"workbook certificate {task_id} lacks one matching observed source marker"
        )
    expected_observation = {
        "image_sha256": observation.image_sha256,
        "image_size": [observation.width, observation.height],
        "observed_question_number": observation.question_number,
        "observed_source_marker_kind": marker_kind,
        "observed_source_marker_number": marker_number,
        "parser_identity": observation.parser_identity,
    }
    actual_observation = {
        key: trace_observation.get(key) for key in expected_observation
    }
    if (
        question.key_binding_kind not in {
            "coordinate_table_answer_key",
            "activity_answer_key",
            "coordinate_choice_answer_key",
        }
        and question.question_marker_kind == "numbered_item"
    ):
        actual_observation["observed_source_marker_kind"] = str(
            trace_observation.get("observed_source_marker_kind") or "numbered_item"
        )
        actual_observation["observed_source_marker_number"] = trace_observation.get(
            "observed_source_marker_number", observation.question_number
        )
    if any(
        actual_observation[key] != value
        for key, value in expected_observation.items()
    ):
        raise CompositionError(
            f"workbook certificate {task_id} observation trace was altered"
        )
    expected_binding_method = {
        "numbered_item": "printed_number",
        "example_label": "source_visible_example_label",
        "activity_label": "source_visible_activity_label",
    }.get(marker_kind)
    if trace_match.get("question_binding_method") != expected_binding_method:
        raise CompositionError(
            f"workbook certificate {task_id} binding method conflicts with its marker"
        )
    if question.key_binding_kind == "coordinate_choice_answer_key":
        if thresholds is None:
            raise CompositionError(
                f"workbook certificate {task_id} lacks coordinate-choice thresholds"
            )
        page_questions = [
            candidate
            for candidate in document.questions
            if candidate.content_page_number == question.content_page_number
        ]
        expected_question_match = observed_coordinate_question_binding(
            observation,
            question,
            page_questions,
            thresholds,
        )
        if (
            trace_match.get("observed_question_text") != expected_question_match
            or expected_question_match.get("passed") is not True
        ):
            raise CompositionError(
                f"workbook certificate {task_id} lacks exact observed-to-source "
                "question binding"
            )
    elif question.key_binding_kind == "inline_solution_projected":
        if thresholds is None:
            raise CompositionError(
                f"workbook certificate {task_id} lacks inline-source thresholds"
            )
        page_questions = [
            candidate
            for candidate in document.questions
            if candidate.content_page_number == question.content_page_number
        ]
        expected_question_match = observed_inline_question_binding(
            observation,
            question,
            page_questions,
            thresholds,
        )
        if (
            trace_match.get("observed_question_text") != expected_question_match
            or expected_question_match.get("passed") is not True
        ):
            raise CompositionError(
                f"workbook certificate {task_id} lacks unique observed-to-inline "
                "source binding"
            )


def compose(profile_path: Path, resolver_manifest_path: Path, output_dir: Path) -> dict[str, Any]:
    profile = _load_json(profile_path)
    activity_visual_reproduction: dict[str, Any] | None = None
    image_only_activity_visual_reproduction: dict[str, Any] | None = None
    profile_schema = str(profile.get("schema_version") or "")
    contract = _PROFILE_CONTRACTS.get(profile_schema)
    if contract is None:
        raise CompositionError("unsupported profile schema")
    resolver_schema, verifier, min_checks = contract
    if profile_schema == "maxim-public-workbook-profile-v1":
        policy = profile.get("policy")
        if not isinstance(policy, dict):
            raise CompositionError("workbook profile policy is missing")
        number_projection, allow_example_label_marker = (
            validate_fail_closed_workbook_policy(policy)
        )
        allow_activity_label_marker = activity_label_projection_enabled(policy)
        workbook_thresholds = WorkbookThresholds(
            min_page_coverage=float(policy["min_page_coverage"]),
            min_page_matched_tokens=int(policy["min_page_matched_tokens"]),
            min_page_margin=float(policy["min_page_margin"]),
            min_numberless_question_coverage=float(
                policy["min_numberless_question_coverage"]
            ),
            min_numberless_question_matched_tokens=int(
                policy["min_numberless_question_matched_tokens"]
            ),
            min_numberless_question_margin=float(
                policy["min_numberless_question_margin"]
            ),
            min_coordinate_question_similarity=float(
                policy.get("min_coordinate_question_similarity", 0.90)
            ),
            min_coordinate_question_source_tokens=int(
                policy.get("min_coordinate_question_source_tokens", 8)
            ),
            min_coordinate_question_margin=float(
                policy.get("min_coordinate_question_margin", 0.25)
            ),
            min_inline_question_coverage=float(
                policy.get("min_inline_question_coverage", 0.85)
            ),
            min_inline_question_matched_tokens=int(
                policy.get("min_inline_question_matched_tokens", 8)
            ),
            min_inline_question_margin=float(
                policy.get("min_inline_question_margin", 0.25)
            ),
        )
        workbook_visual_thresholds = VisualBindingThresholds(
            min_good_matches=int(policy.get("visual_min_good_matches", 50)),
            min_inliers=int(policy.get("visual_min_inliers", 40)),
            min_inlier_ratio=float(policy.get("visual_min_inlier_ratio", 0.65)),
            min_task_hull_fraction=float(
                policy.get("visual_min_task_hull_fraction", 0.30)
            ),
            max_median_reprojection_error=float(
                policy.get("visual_max_median_reprojection_error", 1.0)
            ),
            min_mapped_inside_fraction=float(
                policy.get("visual_min_mapped_inside_fraction", 0.98)
            ),
            max_scale_anisotropy=float(
                policy.get("visual_max_scale_anisotropy", 1.15)
            ),
            min_rank_score_margin=float(
                policy.get("visual_min_rank_score_margin", 10.0)
            ),
            min_rank_score_ratio=float(
                policy.get("visual_min_rank_score_ratio", 5.0)
            ),
        )
        identity_projection = str(
            policy.get("yandex_public_identity_projection")
            or "url_name_plus_required_numeric_nosw_v1"
        )
        if identity_projection not in {
            "url_name_plus_required_numeric_nosw_v1",
            "url_name_plus_optional_numeric_nosw_v2",
        }:
            raise CompositionError("unsupported Yandex public-identity projection")
        allow_missing_nosw = (
            identity_projection == "url_name_plus_optional_numeric_nosw_v2"
        )
        if number_projection == "unique_block_markers_v1":
            observation_loader = parser_observation_allow_missing_number
        elif number_projection == "primary_layout_then_unique_v1":
            observation_loader = parser_observation_primary_layout_number
        else:
            raise CompositionError("unsupported workbook question-number projection")
    else:
        observation_loader = parser_observation
        allow_example_label_marker = False
        allow_activity_label_marker = False
        workbook_thresholds = None
        workbook_visual_thresholds = None
        allow_missing_nosw = False
    profile_sha = sha256_file(profile_path)
    expected_rows = int(profile.get("expected_rows", 0))
    anchor_spec = profile.get("anchor")
    inputs = profile.get("inputs")
    if not isinstance(anchor_spec, dict) or not isinstance(inputs, dict):
        raise CompositionError("profile anchor and inputs are required")
    anchor_path = _path(str(anchor_spec.get("path") or ""))
    parser_path = _path(str(inputs["parser_observations"]["path"]))
    locator_path = _path(str(inputs["source_locators"]["path"]))
    _require_hash(anchor_path, str(anchor_spec.get("sha256") or ""), "anchor")
    _require_hash(
        parser_path, str(inputs["parser_observations"].get("sha256") or ""), "parser"
    )
    _require_hash(
        locator_path, str(inputs["source_locators"].get("sha256") or ""), "source locators"
    )
    if profile_schema == "maxim-public-workbook-profile-v1":
        source_index_path = _path(str(inputs["source_index"]["path"]))
        _require_hash(
            source_index_path,
            str(inputs["source_index"].get("sha256") or ""),
            "workbook source index",
        )
        workbook_index = parse_workbook_index(_load_json(source_index_path))
        workbook_documents_by_id = {
            document.document_id: document
            for document in workbook_index.documents
        }
        workbook_source_records = {
            question.record_id: (document, question)
            for document in workbook_index.documents
            for question in document.questions
        }
        visual_spec = inputs.get("activity_visual_evidence")
        if visual_spec is not None:
            if (
                not isinstance(visual_spec, dict)
                or set(visual_spec) != {"path", "sha256", "allowed_role"}
                or visual_spec.get("allowed_role")
                != "answer_free_sift_ransac_page_binding_fallback_only"
                or policy.get("visual_page_binding_mode")
                != "fallback_only_after_text_page_gate_failure_v1"
            ):
                raise CompositionError("workbook visual fallback role is not fail-closed")
            visual_path = _path(str(visual_spec.get("path") or ""))
            _require_hash(
                visual_path,
                str(visual_spec.get("sha256") or ""),
                "activity visual evidence",
            )
        else:
            visual_path = None
        image_only_visual_spec = inputs.get(
            "image_only_activity_visual_evidence"
        )
        if image_only_visual_spec is not None:
            if (
                not isinstance(image_only_visual_spec, dict)
                or set(image_only_visual_spec)
                != {"path", "sha256", "allowed_role"}
                or image_only_visual_spec.get("allowed_role")
                != IMAGE_ONLY_ACTIVITY_ARTIFACT_ROLE
                or policy.get("image_only_activity_observation_projection")
                != IMAGE_ONLY_ACTIVITY_OBSERVATION_KIND
                or policy.get("image_only_activity_record_projection")
                != IMAGE_ONLY_ACTIVITY_RECORD_SELECTION_POLICY
                or policy.get("task_id_is_policy_feature") is not False
                or policy.get("benchmark_candidate_or_outcome_access") is not False
            ):
                raise CompositionError(
                    "workbook image-only activity role is not fail-closed"
                )
            image_only_visual_path = _path(
                str(image_only_visual_spec.get("path") or "")
            )
            _require_hash(
                image_only_visual_path,
                str(image_only_visual_spec.get("sha256") or ""),
                "image-only activity visual evidence",
            )
        else:
            image_only_visual_path = None
        workbook_visual_records = tuple(
            ActivityVisualRecordRef(
                document_id=document.document_id,
                record_id=question.record_id,
                content_page_number=question.content_page_number,
                activity_number=question.question_number,
                key_projection_sha256=question.key_projection_sha256,
                content_projection_sha256=question.content_projection_sha256,
                binding_projection_sha256=question.binding_projection_sha256,
                visually_checked=question.visually_checked,
                content_bbox=question.content_bbox,
            )
            for document in workbook_index.documents
            for question in document.questions
            if question.key_binding_kind == "activity_answer_key"
            and question.question_marker_kind == "activity_label"
            and question.key_projection_sha256 is not None
            and question.content_projection_sha256 is not None
            and question.binding_projection_sha256 is not None
            and question.content_bbox is not None
        )
    else:
        workbook_source_records = {}
        workbook_documents_by_id = {}
        workbook_visual_records = ()
        visual_path = None
        image_only_visual_path = None
        image_only_visual_spec = None
    resolver_manifest = _load_json(resolver_manifest_path)
    if resolver_manifest.get("schema_version") != resolver_schema:
        raise CompositionError("resolver manifest schema is invalid")
    if resolver_manifest.get("gold_access") is not False or resolver_manifest.get(
        "benchmark_candidate_or_outcome_access"
    ) is not False:
        raise CompositionError("resolver manifest lacks strict blind attestations")
    manifest_profile = resolver_manifest.get("profile")
    artifacts = resolver_manifest.get("artifacts")
    if not isinstance(manifest_profile, dict) or not isinstance(artifacts, dict):
        raise CompositionError("resolver manifest is incomplete")
    if manifest_profile.get("sha256") != profile_sha:
        raise CompositionError("resolver was produced under a different frozen profile")
    if profile_schema == "maxim-public-workbook-profile-v1":
        manifest_inputs = resolver_manifest.get("inputs")
        if not isinstance(manifest_inputs, dict):
            raise CompositionError("workbook resolver manifest has no pinned inputs")
        manifest_input_keys = ["parser_observations", "source_locators", "source_index"]
        if visual_path is not None:
            manifest_input_keys.append("activity_visual_evidence")
        if image_only_visual_path is not None:
            manifest_input_keys.append("image_only_activity_visual_evidence")
        for key in manifest_input_keys:
            manifest_item = manifest_inputs.get(key)
            if (
                not isinstance(manifest_item, dict)
                or manifest_item.get("sha256") != inputs[key].get("sha256")
            ):
                raise CompositionError(f"workbook resolver {key} provenance changed")
        manifest_documents = manifest_inputs.get("documents")
        frozen_documents = profile.get("documents")
        if not isinstance(manifest_documents, dict) or not isinstance(frozen_documents, list):
            raise CompositionError("workbook resolver document provenance is incomplete")
        frozen_document_hashes = {
            str(item.get("document_id") or ""): str(item.get("pdf_sha256") or "")
            for item in frozen_documents
            if isinstance(item, dict)
        }
        if set(manifest_documents) != set(frozen_document_hashes) or any(
            not isinstance(manifest_documents[document_id], dict)
            or manifest_documents[document_id].get("sha256") != expected_hash
            for document_id, expected_hash in frozen_document_hashes.items()
        ):
            raise CompositionError("workbook resolver PDF provenance changed")
        try:
            import pdfplumber
            import pypdf
            from pypdf import PdfReader
        except ImportError as exc:  # pragma: no cover
            raise CompositionError(
                "workbook composition replay requires the pinned PDF runtime"
            ) from exc
        runtime = profile.get("runtime")
        if (
            not isinstance(runtime, dict)
            or str(runtime.get("pypdf_version") or "")
            != str(pypdf.__version__)
            or str(runtime.get("pdfplumber_version") or "")
            != str(pdfplumber.__version__)
        ):
            raise CompositionError(
                "workbook composition PDF runtime differs from the frozen profile"
            )
        workbook_replay_caches: dict[str, dict[str, Any]] = {}
        for document_id, document in workbook_documents_by_id.items():
            document_spec = manifest_documents.get(document_id)
            if not isinstance(document_spec, dict):
                raise CompositionError(
                    f"workbook resolver lacks PDF path for {document_id}"
                )
            pdf_path = Path(str(document_spec.get("path") or "")).resolve()
            _require_hash(pdf_path, document.pdf_sha256, f"workbook PDF {document_id}")
            reader = PdfReader(str(pdf_path))
            page_texts = [page.extract_text() or "" for page in reader.pages]
            if len(page_texts) != document.page_count:
                raise CompositionError(
                    f"workbook PDF page count changed for {document_id}"
                )
            workbook_replay_caches[document_id] = {
                "path": pdf_path,
                "page_texts": page_texts,
                "matcher": PageMatcher(page_texts),
                "source_verification": verify_workbook_index_pdf(
                    pdf_path, document
                ),
            }
    else:
        workbook_replay_caches = {}
    candidate_path = Path(str(artifacts["candidate"]["path"])).resolve()
    certificate_path = Path(str(artifacts["certificates"]["path"])).resolve()
    _require_hash(candidate_path, str(artifacts["candidate"]["sha256"]), "resolver candidate")
    _require_hash(
        certificate_path,
        str(artifacts["certificates"]["sha256"]),
        "resolver certificates",
    )

    anchor_rows = _load_jsonl(anchor_path)
    parser_rows = _load_jsonl(parser_path)
    locator_rows = _load_jsonl(locator_path)
    candidate_rows = _load_jsonl(candidate_path)
    certificate_rows = _load_jsonl(certificate_path)
    if len(anchor_rows) != expected_rows or len(parser_rows) != expected_rows:
        raise CompositionError(f"anchor/parser must contain exactly {expected_rows} rows")
    anchor = _index(anchor_rows, "anchor")
    parser = _index(parser_rows, "parser")
    locators = _index(locator_rows, "source locators")
    candidates = _index(candidate_rows, "resolver candidate")
    certificates = _index(certificate_rows, "resolver certificates")
    if profile_schema == "maxim-public-workbook-profile-v1" and (
        resolver_manifest.get("accepted_certificates") != len(certificates)
        or resolver_manifest.get("rows") != expected_rows
        or resolver_manifest.get("task_id_used_for_alignment_only") is not True
        or resolver_manifest.get("viewer_nosw_used_as_policy_feature") is not False
    ):
        raise CompositionError("workbook resolver manifest counters or invariants changed")
    if profile_schema == "maxim-public-workbook-profile-v1" and visual_path is not None:
        visual_certificate_count = sum(
            isinstance(
                certificate.get("trace", {}).get("visual_page_binding"), dict
            )
            for certificate in certificates.values()
            if isinstance(certificate.get("trace"), dict)
        )
        if resolver_manifest.get("visual_fallback_certificates") != visual_certificate_count:
            raise CompositionError("workbook visual fallback counter changed")
    if (
        profile_schema == "maxim-public-workbook-profile-v1"
        and image_only_visual_path is not None
    ):
        structural_image_only_count = 0
        for raw in parser.values():
            try:
                project_image_only_activity_observation(raw)
            except ImageOnlyActivityError:
                continue
            structural_image_only_count += 1
        image_only_certificate_count = sum(
            isinstance(
                certificate.get("trace", {}).get(
                    "image_only_activity_binding"
                ),
                dict,
            )
            for certificate in certificates.values()
            if isinstance(certificate.get("trace"), dict)
        )
        if (
            resolver_manifest.get("image_only_activity_observations")
            != structural_image_only_count
            or resolver_manifest.get("image_only_activity_certificates")
            != image_only_certificate_count
        ):
            raise CompositionError(
                "workbook image-only activity counters changed"
            )
    task_set = set(anchor)
    if set(parser) != task_set or set(candidates) != task_set or not task_set <= set(locators):
        raise CompositionError("resolver, parser, locator, and anchor task sets do not align")
    if not set(certificates) <= task_set:
        raise CompositionError("certificate artifact contains an unknown task")
    workbook_visual_bindings = {}
    if profile_schema == "maxim-public-workbook-profile-v1" and visual_path is not None:
        visual_observations: dict[str, ActivityVisualObservationRef] = {}
        for task_id, raw in parser.items():
            try:
                observation = observation_loader(raw)
            except (OfficialSourceError, ValueError, KeyError, TypeError):
                continue
            marker_kind, marker_number = observed_source_question_marker(observation)
            if marker_kind != "activity_label" or marker_number is None:
                continue
            source_url = str(locators[task_id].get("source_url") or "")
            try:
                visual_document = document_for_source(
                    workbook_index,
                    source_url,
                    allow_missing_nosw=allow_missing_nosw,
                )
            except OfficialSourceError:
                continue
            if visual_document is None:
                continue
            visual_observations[task_id] = ActivityVisualObservationRef(
                task_id=task_id,
                task_image_sha256=observation.image_sha256,
                width=observation.width,
                height=observation.height,
                parser_identity=observation.parser_identity,
                document_id=visual_document.document_id,
                pdf_sha256=visual_document.pdf_sha256,
                marker_kind=marker_kind,
                marker_number=marker_number,
            )
        assert workbook_visual_thresholds is not None
        try:
            visual_payload = load_activity_visual_artifact_json(visual_path)
            workbook_visual_bindings = verified_activity_bindings_from_artifact(
                visual_payload,
                repo_root=REPO_ROOT,
                expected_parser_sha256=str(
                    inputs["parser_observations"].get("sha256") or ""
                ),
                expected_source_locators_sha256=str(
                    inputs["source_locators"].get("sha256") or ""
                ),
                expected_source_index_sha256=str(
                    inputs["source_index"].get("sha256") or ""
                ),
                observations_by_task_id=visual_observations,
                records=workbook_visual_records,
                document_pdf_paths={
                    document_id: cache["path"]
                    for document_id, cache in workbook_replay_caches.items()
                },
                thresholds=workbook_visual_thresholds,
            )
        except VisualCoordinateBindingError as exc:
            raise CompositionError(
                f"activity visual evidence failed independent replay: {exc}"
            ) from exc
        activity_visual_reproduction = _fresh_rebuild_activity_visual_artifact(
            profile_path=profile_path,
            parser_path=parser_path,
            locator_path=locator_path,
            source_index_path=source_index_path,
            visual_path=visual_path,
            expected_visual_sha256=str(visual_spec.get("sha256") or ""),
            visual_payload=visual_payload,
            document_pdf_paths={
                document_id: cache["path"]
                for document_id, cache in workbook_replay_caches.items()
            },
        )

    workbook_image_only_observations = {}
    workbook_image_only_bindings = {}
    if (
        profile_schema == "maxim-public-workbook-profile-v1"
        and image_only_visual_path is not None
    ):
        for task_id, raw in parser.items():
            try:
                image_only_observation = (
                    project_image_only_activity_observation(raw)
                )
            except ImageOnlyActivityError:
                continue
            try:
                observation_loader(raw)
            except (OfficialSourceError, ValueError, KeyError, TypeError):
                pass
            else:
                raise CompositionError(
                    f"parser row {task_id} is ambiguous between text and image-only routes"
                )
            workbook_image_only_observations[task_id] = image_only_observation
        try:
            image_only_visual_payload = (
                load_image_only_activity_visual_artifact_json(
                    image_only_visual_path
                )
            )
            workbook_image_only_bindings = (
                verified_image_only_activity_bindings_from_artifact(
                    image_only_visual_payload,
                    repo_root=REPO_ROOT,
                    expected_parser_sha256=str(
                        inputs["parser_observations"].get("sha256") or ""
                    ),
                    expected_source_locators_sha256=str(
                        inputs["source_locators"].get("sha256") or ""
                    ),
                    observations_by_task_id=workbook_image_only_observations,
                    source_urls_by_task_id={
                        task_id: str(locators[task_id].get("source_url") or "")
                        for task_id in workbook_image_only_observations
                    },
                    documents_by_id=workbook_documents_by_id,
                    records=workbook_visual_records,
                    document_pdf_paths={
                        document_id: cache["path"]
                        for document_id, cache in workbook_replay_caches.items()
                    },
                    thresholds=VisualBindingThresholds(),
                )
            )
        except ImageOnlyActivityError as exc:
            raise CompositionError(
                "image-only activity evidence failed independent replay: "
                f"{exc}"
            ) from exc
        image_only_activity_visual_reproduction = (
            _fresh_rebuild_image_only_activity_visual_artifact(
                parser_path=parser_path,
                locator_path=locator_path,
                visual_path=image_only_visual_path,
                expected_visual_sha256=str(
                    image_only_visual_spec.get("sha256") or ""
                ),
                visual_payload=image_only_visual_payload,
                document_pdf_paths={
                    document_id: cache["path"]
                    for document_id, cache in workbook_replay_caches.items()
                },
            )
        )

    evidence_policy = EvidencePolicy()
    frozen_policy = FrozenProfile(
        name=str(profile.get("profile_name") or "maxim-official-ogm-v2"),
        allowed_strong_kinds=frozenset({CertificateKind.SOURCE_ENTAILMENT}),
        min_claim_coverage=1.0,
        min_deterministic_checks=min_checks,
        min_independent_verifiers=1,
    )
    output_rows: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    overrides = 0
    certified_equal_anchor = 0
    for raw_anchor in anchor_rows:
        task_id = str(raw_anchor["task_id"])
        anchor_answer = str(raw_anchor.get("final_answer") or "").strip()
        if not anchor_answer:
            raise CompositionError(f"anchor {task_id} has no final_answer")
        anchor_candidate = CandidateEnvelope(source="fail-closed-anchor", final_answer=anchor_answer)
        raw_candidate = candidates[task_id]
        candidate_answer = str(raw_candidate.get("final_answer") or "").strip()
        challenger_tuple: tuple[CandidateEnvelope, ...] = ()
        certificate_fingerprint = None
        if task_id in certificates:
            if not candidate_answer:
                raise CompositionError(f"certified candidate {task_id} has no answer")
            generation = raw_candidate.get("generation")
            if not isinstance(generation, dict) or generation.get("gold_access") is not False:
                raise CompositionError(f"candidate {task_id} lacks a strict blind attestation")
            image_only_observation = workbook_image_only_observations.get(task_id)
            observation = (
                image_only_observation
                if image_only_observation is not None
                else observation_loader(parser[task_id])
            )
            source_url = str(locators[task_id].get("source_url") or "")
            trace = certificates[task_id].get("trace")
            if not isinstance(trace, dict) or trace.get("accepted") is not True:
                raise CompositionError(f"certificate {task_id} trace is not accepted")
            if profile_schema == "maxim-public-workbook-profile-v1":
                _validate_workbook_certificate_artifact(
                    task_id=task_id,
                    raw_candidate=raw_candidate,
                    raw_certificate=certificates[task_id],
                    trace=trace,
                    profile=profile,
                    profile_sha=profile_sha,
                )
                if image_only_observation is None:
                    _validate_workbook_trace_against_source(
                        task_id=task_id,
                        candidate_answer=candidate_answer,
                        observation=observation,
                        trace=trace,
                        source_records=workbook_source_records,
                        allow_example_label_marker=allow_example_label_marker,
                        allow_activity_label_marker=allow_activity_label_marker,
                        thresholds=workbook_thresholds,
                    )
                elif not isinstance(
                    trace.get("image_only_activity_binding"), dict
                ):
                    raise CompositionError(
                        f"image-only certificate {task_id} lacks its distinct binding trace"
                    )
                try:
                    replay_document = document_for_source(
                        workbook_index,
                        source_url,
                        allow_missing_nosw=allow_missing_nosw,
                    )
                    if replay_document is None:
                        raise OfficialSourceError(
                            "source is absent from the frozen workbook index"
                        )
                    replay_cache = workbook_replay_caches[
                        replay_document.document_id
                    ]
                    if image_only_observation is not None:
                        image_only_binding = workbook_image_only_bindings.get(
                            image_only_observation.image_sha256
                        )
                        if image_only_binding is None:
                            raise ImageOnlyActivityError(
                                "image-only observation lacks a replayed binding"
                            )
                        replay_result = resolve_image_only_activity_question(
                            image_only_observation,
                            source_url,
                            replay_document,
                            image_only_binding,
                            verified_content_marker_counts=replay_cache[
                                "source_verification"
                            ]["content_marker_counts"],
                            allow_missing_nosw=allow_missing_nosw,
                        )
                    else:
                        assert workbook_thresholds is not None
                        replay_result = resolve_workbook_question(
                            observation,
                            source_url,
                            replay_document,
                            replay_cache["matcher"],
                            replay_cache["page_texts"],
                            workbook_thresholds,
                            allow_missing_nosw=allow_missing_nosw,
                            allow_example_label_marker=allow_example_label_marker,
                            allow_activity_label_marker=allow_activity_label_marker,
                            verified_content_marker_counts=replay_cache[
                                "source_verification"
                            ]["content_marker_counts"],
                            verified_activity_visual_binding=workbook_visual_bindings.get(
                                observation.image_sha256
                            ),
                            activity_visual_thresholds=workbook_visual_thresholds,
                        )
                except (
                    ImageOnlyActivityError,
                    OfficialSourceError,
                    KeyError,
                    ValueError,
                    TypeError,
                ) as exc:
                    raise CompositionError(
                        f"workbook certificate {task_id} cannot be replayed: {exc}"
                    ) from exc
                expected_trace_keys = set(replay_result.trace) | {"provenance"}
                replayed_trace = {
                    key: trace.get(key) for key in replay_result.trace
                }
                if (
                    not replay_result.accepted
                    or replay_result.answer != candidate_answer
                    or set(trace) != expected_trace_keys
                    or replayed_trace != replay_result.trace
                ):
                    raise CompositionError(
                        f"workbook certificate {task_id} differs from a complete "
                        "resolver/PDF replay"
                    )
                trace_source = trace.get("source")
                answer_format = (
                    str(trace_source.get("answer_format") or "")
                    if isinstance(trace_source, dict)
                    else ""
                )
                if answer_format not in {"choice", "short_text"}:
                    raise CompositionError(
                        f"certificate {task_id} has no valid source answer format"
                    )
            else:
                answer_format = "choice"
            if image_only_observation is not None:
                problem = problem_for_image_only_activity(
                    image_only_observation,
                    source_url,
                    answer_format=answer_format,
                )
            else:
                problem = problem_for(
                    observation,
                    source_url,
                    answer_format=answer_format,
                )
            bare = CandidateEnvelope(source=verifier, final_answer=candidate_answer)
            certificate = certificate_from_record(
                problem,
                bare,
                certificates[task_id],
                allowed_verifiers=frozenset({verifier}),
                allowed_kinds=frozenset({CertificateKind.SOURCE_ENTAILMENT}),
                require_inline_trace=True,
            )
            provenance = trace.get("provenance")
            if not isinstance(provenance, dict) or provenance.get("profile_sha256") != profile_sha:
                raise CompositionError(f"certificate {task_id} is not profile-bound")
            challenger_tuple = (
                CandidateEnvelope(
                    source=verifier,
                    final_answer=candidate_answer,
                    certificates=(certificate,),
                ),
            )
            certificate_fingerprint = certificate.trace_fingerprint
            bundle = InferenceBundle(
                problem=problem,
                anchor=anchor_candidate,
                candidates=challenger_tuple,
            )
            decision = evidence_policy.decide(bundle, frozen_policy)
        else:
            # No certificate means no policy feature is consulted.  Avoid even
            # parsing OCR for an ineligible row; it cannot affect the anchor.
            problem = ProblemInput(statement="No certified challenger was admitted.")
            decision = evidence_policy.decide(
                InferenceBundle(problem=problem, anchor=anchor_candidate), frozen_policy
            )
        reasons[decision.reason.value] += 1
        output = dict(raw_anchor)
        if decision.action is DecisionAction.REPLACE_ANCHOR:
            overrides += 1
            output["final_answer"] = decision.selected.final_answer
            generation = output.get("generation")
            copied_generation = dict(generation) if isinstance(generation, dict) else {}
            copied_generation["official_source_override"] = {
                "verifier": verifier,
                "trace_fingerprint": certificate_fingerprint,
                "profile_sha256": profile_sha,
            }
            output["generation"] = copied_generation
        elif challenger_tuple and candidate_answer == anchor_answer:
            certified_equal_anchor += 1
        output_rows.append(output)
        decisions.append(
            {
                "task_id": task_id,
                "action": decision.action.value,
                "reason": decision.reason.value,
                "anchor_answer": anchor_answer,
                "selected_answer": str(output["final_answer"]),
                "certificate_trace_fingerprint": certificate_fingerprint,
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    solver_path = output_dir / "solver.jsonl"
    decisions_path = output_dir / "decisions.jsonl"
    _write_jsonl(solver_path, output_rows)
    _write_jsonl(decisions_path, decisions)
    manifest = {
        "schema_version": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows": expected_rows,
        "overrides": overrides,
        "certified_equal_anchor": certified_equal_anchor,
        "certificates_seen": len(certificates),
        "decision_reasons": dict(sorted(reasons.items())),
        "gold_access": False,
        "score_or_outcome_access": False,
        "profile": {"path": str(profile_path), "sha256": profile_sha},
        "resolver_manifest": {
            "path": str(resolver_manifest_path),
            "sha256": sha256_file(resolver_manifest_path),
        },
        "anchor": {"path": str(anchor_path), "sha256": sha256_file(anchor_path)},
        "output": {
            "solver": {"path": str(solver_path), "sha256": sha256_file(solver_path)},
            "decisions": {"path": str(decisions_path), "sha256": sha256_file(decisions_path)},
        },
    }
    if activity_visual_reproduction is not None:
        manifest["activity_visual_reproduction"] = activity_visual_reproduction
    if image_only_activity_visual_reproduction is not None:
        manifest["image_only_activity_visual_reproduction"] = (
            image_only_activity_visual_reproduction
        )
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_bytes(
        (json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode(
            "utf-8"
        )
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-json", type=Path, required=True)
    parser.add_argument("--resolver-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = compose(
            args.profile_json.resolve(),
            args.resolver_manifest.resolve(),
            args.output_dir.resolve(),
        )
    except (
        CompositionError,
        OfficialSourceError,
        OSError,
        ValueError,
        KeyError,
        json.JSONDecodeError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
