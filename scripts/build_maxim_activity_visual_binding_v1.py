#!/usr/bin/env python3
"""Build source-only SIFT evidence for indexed workbook activity pages.

The generator projects only parser-observed ``activity_label`` markers and
source-index records whose key binding is ``activity_answer_key``.  It never
loads a solver, candidate, scorer, evaluation, or task outcome.  Source-index
answer values are deliberately left outside the allowlisted record projection.

The primary output key is the parser-pinned task-image SHA-256.  ``task_id`` is
retained only in an alignment-audit object used to join the parser row to its
public source locator; it is never a page or record selection feature.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evidence_os.official_ogm import (  # noqa: E402
    canonical_json_bytes,
    canonical_json_sha256,
    observed_source_question_marker,
    parser_observation_primary_layout_number,
    sha256_file,
)
from src.evidence_os.official_workbook import (  # noqa: E402
    OfficialSourceError,
    strict_public_document_identity,
)
from src.evidence_os.visual_coordinate_binding import (  # noqa: E402
    SiftRuntimeProfile,
    VisualPageEvidence,
    compute_sift_page_evidence,
)


SCHEMA = "maxim-activity-visual-binding-source-evidence-v1"
GENERATOR = "build-maxim-activity-visual-binding-v1"
PROFILE_SCHEMA = "maxim-public-workbook-profile-v1"
SOURCE_INDEX_SCHEMA = "public-workbook-source-index-v1"
EXPECTED_OPENCV_VERSION = "5.0.0"
EXPECTED_NUMPY_VERSION = "2.5.1"
EXPECTED_POPPLER_VERSION = "26.05.0"
EXPECTED_ACTIVITY_POLICY = "primary_paragraph_title_order_one_v1"
SIFT_PROCESS_WORKERS = 4
PINNED_PACKAGE_ROOT = (
    REPO_ROOT / "tmp" / "portfolio_official_sources" / "python_pkgs"
).resolve()
HEX64 = re.compile(r"^[0-9a-f]{64}$")
DOCUMENT_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{2,80}$")
DOCUMENT_FIELDS = frozenset(
    {
        "document_id",
        "locator",
        "pdf_sha256",
        "page_count",
        "content_page_ranges",
        "questions",
    }
)
LOCATOR_FIELDS = frozenset({"kind", "public_locator", "name"})
SOURCE_LOCATOR_FIELDS = frozenset({"source_url", "task_id"})


class ActivityVisualBindingBuildError(RuntimeError):
    """The source-only evidence build cannot be certified."""


@dataclass(frozen=True, slots=True)
class ActivityRecord:
    document_id: str
    record_id: str
    content_page_number: int
    activity_number: int
    question_marker_kind: str
    key_binding_kind: str
    content_bbox: tuple[float, float, float, float]
    key_projection_sha256: str
    content_projection_sha256: str
    binding_projection_sha256: str
    visually_checked: bool


@dataclass(frozen=True, slots=True)
class ActivityDocument:
    document_id: str
    locator_kind: str
    public_locator: str
    locator_name: str
    pdf_sha256: str
    page_count: int
    content_page_ranges: tuple[tuple[int, int], ...]

    @property
    def identity_key(self) -> tuple[str, str, str]:
        return (self.locator_kind, self.public_locator, self.locator_name)


@dataclass(frozen=True, slots=True)
class ActivityObservation:
    task_id: str
    image_basename: str
    image_sha256: str
    width: int
    height: int
    parser_identity: str
    activity_number: int
    document_id: str
    parser_projection_sha256: str
    source_identity_projection_sha256: str
    task_image_path: Path


def _positive_integer(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise ActivityVisualBindingBuildError(f"{label} must be a positive integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ActivityVisualBindingBuildError(
            f"{label} must be a positive integer"
        ) from exc
    if result < 1 or result != value:
        raise ActivityVisualBindingBuildError(f"{label} must be a positive integer")
    return result


def _sha256(value: Any, label: str) -> str:
    result = str(value or "").strip()
    if HEX64.fullmatch(result) is None:
        raise ActivityVisualBindingBuildError(f"{label} is not a SHA-256 pin")
    return result


def _bbox(value: Any, label: str) -> tuple[float, float, float, float]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes, bytearray))
        or len(value) != 4
        or any(
            not isinstance(item, (int, float))
            or isinstance(item, bool)
            or not math.isfinite(float(item))
            for item in value
        )
    ):
        raise ActivityVisualBindingBuildError(f"{label} is not a finite four-value box")
    result = tuple(float(item) for item in value)
    if not (0.0 <= result[0] < result[2] and 0.0 <= result[1] < result[3]):
        raise ActivityVisualBindingBuildError(f"{label} is not an ordered box")
    return result  # type: ignore[return-value]


def _content_page_ranges(
    value: Any,
    *,
    document_id: str,
    page_count: int,
) -> tuple[tuple[int, int], ...]:
    if not isinstance(value, Sequence) or isinstance(
        value, (str, bytes, bytearray)
    ):
        raise ActivityVisualBindingBuildError(
            f"content_page_ranges are missing for {document_id}"
        )
    ranges: list[tuple[int, int]] = []
    previous_end = 0
    for index, raw_range in enumerate(value):
        if (
            not isinstance(raw_range, Sequence)
            or isinstance(raw_range, (str, bytes, bytearray))
            or len(raw_range) != 2
        ):
            raise ActivityVisualBindingBuildError(
                f"content_page_ranges[{index}] is malformed for {document_id}"
            )
        start = _positive_integer(
            raw_range[0], f"content_page_ranges[{index}] start"
        )
        end = _positive_integer(raw_range[1], f"content_page_ranges[{index}] end")
        if start > end or end > page_count or start <= previous_end:
            raise ActivityVisualBindingBuildError(
                f"content_page_ranges are unordered, overlapping, or out of PDF for {document_id}"
            )
        ranges.append((start, end))
        previous_end = end
    if not ranges:
        raise ActivityVisualBindingBuildError(
            f"content_page_ranges are empty for {document_id}"
        )
    return tuple(ranges)


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ActivityVisualBindingBuildError(f"cannot load {label}: {path}") from exc
    if not isinstance(value, dict):
        raise ActivityVisualBindingBuildError(f"{label} must be a JSON object")
    return value


def _load_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8-sig") as source:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ActivityVisualBindingBuildError(
                        f"{label}:{line_number} must be a JSON object"
                    )
                rows.append(value)
    except ActivityVisualBindingBuildError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ActivityVisualBindingBuildError(f"cannot load {label}: {path}") from exc
    return rows


def _require_file_hash(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise ActivityVisualBindingBuildError(f"{label} is not a file: {path}")
    actual = sha256_file(path)
    if actual != _sha256(expected, f"{label} expected hash"):
        raise ActivityVisualBindingBuildError(
            f"{label} SHA-256 mismatch: expected {expected}, got {actual}"
        )
    return actual


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _is_inside(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def _runtime_pins() -> dict[str, Any]:
    if sys.version_info[:2] != (3, 12):
        raise ActivityVisualBindingBuildError(
            "this generator requires the pinned Python 3.12 runtime"
        )
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
    except ImportError as exc:
        raise ActivityVisualBindingBuildError(
            "the pinned OpenCV/NumPy package directory is unavailable"
        ) from exc
    if cv2.__version__ != EXPECTED_OPENCV_VERSION:
        raise ActivityVisualBindingBuildError(
            f"OpenCV must be {EXPECTED_OPENCV_VERSION}, got {cv2.__version__}"
        )
    if np.__version__ != EXPECTED_NUMPY_VERSION:
        raise ActivityVisualBindingBuildError(
            f"NumPy must be {EXPECTED_NUMPY_VERSION}, got {np.__version__}"
        )
    cv2_path = Path(cv2.__file__).resolve()
    numpy_path = Path(np.__file__).resolve()
    if not _is_inside(cv2_path, PINNED_PACKAGE_ROOT) or not _is_inside(
        numpy_path, PINNED_PACKAGE_ROOT
    ):
        raise ActivityVisualBindingBuildError(
            "OpenCV/NumPy were not imported from tmp/portfolio_official_sources/python_pkgs"
        )
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    return {
        "python": {
            "executable": str(Path(sys.executable).resolve()),
            "version": ".".join(str(item) for item in sys.version_info[:3]),
        },
        "opencv": {
            "module_path": _display_path(cv2_path),
            "version": str(cv2.__version__),
            "threads": 1,
            "opencl_enabled": False,
        },
        "numpy": {
            "module_path": _display_path(numpy_path),
            "version": str(np.__version__),
        },
        "package_root": _display_path(PINNED_PACKAGE_ROOT),
    }


def _profile_input(profile: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    inputs = profile.get("inputs")
    if not isinstance(inputs, Mapping):
        raise ActivityVisualBindingBuildError("profile inputs are missing")
    spec = inputs.get(key)
    if not isinstance(spec, Mapping):
        raise ActivityVisualBindingBuildError(f"profile input {key} is missing")
    return spec


def _validate_profile(
    profile: Mapping[str, Any],
    *,
    parser_path: Path,
    locator_path: Path,
    source_index_path: Path,
) -> tuple[int, dict[str, dict[str, Any]], dict[str, str]]:
    if profile.get("schema_version") != PROFILE_SCHEMA:
        raise ActivityVisualBindingBuildError("unsupported public-workbook profile")
    expected_rows = _positive_integer(profile.get("expected_rows"), "profile expected_rows")
    policy = profile.get("policy")
    if not isinstance(policy, Mapping):
        raise ActivityVisualBindingBuildError("profile policy is missing")
    if policy.get("activity_label_projection") != EXPECTED_ACTIVITY_POLICY:
        raise ActivityVisualBindingBuildError("activity-label projection is not enabled")
    if policy.get("question_number_projection") != "primary_layout_then_unique_v1":
        raise ActivityVisualBindingBuildError("parser observation projection is not pinned")
    if policy.get("task_id_is_policy_feature") is not False:
        raise ActivityVisualBindingBuildError("task_id must not be a policy feature")
    if policy.get("benchmark_candidate_or_outcome_access") is not False:
        raise ActivityVisualBindingBuildError(
            "profile does not forbid benchmark candidate/outcome access"
        )
    identity_projection = str(policy.get("yandex_public_identity_projection") or "")
    if identity_projection not in {
        "url_name_plus_required_numeric_nosw_v1",
        "url_name_plus_optional_numeric_nosw_v2",
    }:
        raise ActivityVisualBindingBuildError("unsupported public identity projection")

    supplied = {
        "parser_observations": parser_path,
        "source_locators": locator_path,
        "source_index": source_index_path,
    }
    hashes: dict[str, str] = {}
    for key, path in supplied.items():
        spec = _profile_input(profile, key)
        hashes[key] = _require_file_hash(path, str(spec.get("sha256") or ""), key)

    raw_documents = profile.get("documents")
    if not isinstance(raw_documents, Sequence) or isinstance(
        raw_documents, (str, bytes, bytearray)
    ):
        raise ActivityVisualBindingBuildError("profile documents are missing")
    documents: dict[str, dict[str, Any]] = {}
    for raw in raw_documents:
        if not isinstance(raw, Mapping):
            raise ActivityVisualBindingBuildError("profile document is malformed")
        document_id = str(raw.get("document_id") or "").strip()
        if not document_id or document_id in documents:
            raise ActivityVisualBindingBuildError("profile document id is missing/duplicate")
        documents[document_id] = {
            "pdf_sha256": _sha256(raw.get("pdf_sha256"), "profile PDF hash"),
            "page_count": _positive_integer(raw.get("page_count"), "profile page_count"),
        }
    return expected_rows, documents, hashes


def _safe_activity_projection(
    payload: Mapping[str, Any],
) -> tuple[dict[str, ActivityDocument], list[ActivityRecord]]:
    if set(payload) != {"schema_version", "documents"}:
        raise ActivityVisualBindingBuildError("source-index root fields changed")
    if payload.get("schema_version") != SOURCE_INDEX_SCHEMA:
        raise ActivityVisualBindingBuildError("unsupported source-index schema")
    raw_documents = payload.get("documents")
    if not isinstance(raw_documents, Sequence) or isinstance(
        raw_documents, (str, bytes, bytearray)
    ):
        raise ActivityVisualBindingBuildError("source-index documents are missing")

    documents: dict[str, ActivityDocument] = {}
    records: list[ActivityRecord] = []
    record_ids: set[str] = set()
    source_addresses: set[tuple[str, int, int]] = set()
    for raw_document in raw_documents:
        if not isinstance(raw_document, Mapping) or set(raw_document) != DOCUMENT_FIELDS:
            raise ActivityVisualBindingBuildError("source-index document fields changed")
        document_id = str(raw_document.get("document_id") or "").strip()
        pdf_sha256 = _sha256(raw_document.get("pdf_sha256"), "source-index PDF hash")
        page_count = _positive_integer(raw_document.get("page_count"), "source-index page_count")
        if (
            DOCUMENT_ID.fullmatch(document_id) is None
            or not document_id.endswith(pdf_sha256[:12])
        ):
            raise ActivityVisualBindingBuildError("source-index document id is malformed")
        locator = raw_document.get("locator")
        if not isinstance(locator, Mapping) or set(locator) != LOCATOR_FIELDS:
            raise ActivityVisualBindingBuildError("source-index document locator changed")
        locator_kind = str(locator.get("kind") or "")
        public_locator = str(locator.get("public_locator") or "")
        locator_name = str(locator.get("name") or "")
        if locator_kind not in {"yandex_public", "direct_https"}:
            raise ActivityVisualBindingBuildError("source-index locator kind is unsupported")
        if not public_locator or not locator_name:
            raise ActivityVisualBindingBuildError("source-index locator is incomplete")
        content_page_ranges = _content_page_ranges(
            raw_document.get("content_page_ranges"),
            document_id=document_id,
            page_count=page_count,
        )
        content_pages = {
            page
            for start, end in content_page_ranges
            for page in range(start, end + 1)
        }

        raw_questions = raw_document.get("questions")
        if not isinstance(raw_questions, Sequence) or isinstance(
            raw_questions, (str, bytes, bytearray)
        ):
            raise ActivityVisualBindingBuildError("source-index questions are missing")
        document_records: list[ActivityRecord] = []
        for raw_question in raw_questions:
            if not isinstance(raw_question, Mapping):
                raise ActivityVisualBindingBuildError("source-index question is malformed")
            if raw_question.get("key_binding_kind") != "activity_answer_key":
                continue
            if raw_question.get("question_marker_kind") != "activity_label":
                raise ActivityVisualBindingBuildError(
                    "activity key record does not use an activity-label marker"
                )
            page_number = _positive_integer(
                raw_question.get("content_page_number"), "activity content page"
            )
            activity_number = _positive_integer(
                raw_question.get("question_number"), "activity number"
            )
            if page_number not in content_pages or activity_number > 999:
                raise ActivityVisualBindingBuildError("activity source address is out of range")
            record_id = str(raw_question.get("record_id") or "").strip()
            expected_record_id = f"{document_id}:p{page_number}:q{activity_number}"
            if record_id != expected_record_id:
                raise ActivityVisualBindingBuildError(
                    "activity record id is not its exact source address"
                )
            source_address = (document_id, page_number, activity_number)
            if record_id in record_ids or source_address in source_addresses:
                raise ActivityVisualBindingBuildError("activity source address is duplicated")
            visually_checked = raw_question.get("visually_checked")
            if visually_checked is not True:
                raise ActivityVisualBindingBuildError("activity record is not visually checked")
            record = ActivityRecord(
                document_id=document_id,
                record_id=record_id,
                content_page_number=page_number,
                activity_number=activity_number,
                question_marker_kind="activity_label",
                key_binding_kind="activity_answer_key",
                content_bbox=_bbox(
                    raw_question.get("content_bbox"), "activity content bbox"
                ),
                key_projection_sha256=_sha256(
                    raw_question.get("key_projection_sha256"),
                    "activity key projection",
                ),
                content_projection_sha256=_sha256(
                    raw_question.get("content_projection_sha256"),
                    "activity content projection",
                ),
                binding_projection_sha256=_sha256(
                    raw_question.get("binding_projection_sha256"),
                    "activity binding projection",
                ),
                visually_checked=True,
            )
            document_records.append(record)
            records.append(record)
            record_ids.add(record_id)
            source_addresses.add(source_address)
        if document_records:
            if document_id in documents:
                raise ActivityVisualBindingBuildError("activity document is duplicated")
            documents[document_id] = ActivityDocument(
                document_id=document_id,
                locator_kind=locator_kind,
                public_locator=public_locator,
                locator_name=locator_name,
                pdf_sha256=pdf_sha256,
                page_count=page_count,
                content_page_ranges=content_page_ranges,
            )
    if not records:
        raise ActivityVisualBindingBuildError("source index contains no activity records")
    return documents, sorted(
        records,
        key=lambda item: (item.document_id, item.content_page_number, item.activity_number),
    )


def _index_source_locators(rows: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    result: dict[str, str] = {}
    for row in rows:
        if set(row) != SOURCE_LOCATOR_FIELDS:
            raise ActivityVisualBindingBuildError("source-locator fields changed")
        task_id = str(row.get("task_id") or "").strip()
        source_url = str(row.get("source_url") or "").strip()
        if not task_id or not source_url or task_id in result:
            raise ActivityVisualBindingBuildError(
                "source locator has a missing/duplicate alignment key"
            )
        result[task_id] = source_url
    return result


def _indexed_identity_for_source(
    source_url: str,
    *,
    activity_documents: Mapping[str, ActivityDocument],
    allow_missing_nosw: bool,
) -> str:
    try:
        identity = strict_public_document_identity(
            source_url,
            allow_missing_nosw=allow_missing_nosw,
        )
    except OfficialSourceError as exc:
        raise ActivityVisualBindingBuildError("activity source locator is malformed") from exc
    identity_key = (identity.kind, identity.public_locator, identity.name)
    matches = [
        item.document_id
        for item in activity_documents.values()
        if item.identity_key == identity_key
    ]
    if len(matches) != 1:
        raise ActivityVisualBindingBuildError(
            "activity observation does not map to exactly one indexed activity document"
        )
    return matches[0]


def _activity_observations(
    parser_rows: Sequence[Mapping[str, Any]],
    locator_index: Mapping[str, str],
    activity_documents: Mapping[str, ActivityDocument],
    *,
    task_image_dir: Path,
    allow_missing_nosw: bool,
) -> list[ActivityObservation]:
    if not task_image_dir.is_dir():
        raise ActivityVisualBindingBuildError(
            f"task image directory is missing: {task_image_dir}"
        )
    observations: list[ActivityObservation] = []
    image_hashes: set[str] = set()
    audit_task_ids: set[str] = set()
    for raw in parser_rows:
        try:
            observation = parser_observation_primary_layout_number(raw)
        except (OfficialSourceError, KeyError, TypeError, ValueError):
            # A malformed row is not an observable activity marker and remains
            # outside this narrowly scoped evidence build.  Rows admitted below
            # still pass the complete frozen parser projection.
            continue
        marker_kind, marker_number = observed_source_question_marker(observation)
        if marker_kind != "activity_label":
            continue
        if marker_number is None:
            raise ActivityVisualBindingBuildError("activity observation has no marker number")
        source_url = locator_index.get(observation.task_id)
        if source_url is None:
            raise ActivityVisualBindingBuildError(
                "activity observation has no public source locator"
            )
        document_id = _indexed_identity_for_source(
            source_url,
            activity_documents=activity_documents,
            allow_missing_nosw=allow_missing_nosw,
        )

        raw_images = raw.get("images")
        if not isinstance(raw_images, Sequence) or isinstance(
            raw_images, (str, bytes, bytearray)
        ) or len(raw_images) != 1 or not isinstance(raw_images[0], Mapping):
            raise ActivityVisualBindingBuildError("activity parser image metadata changed")
        raw_image = raw_images[0]
        image_basename = str(raw_image.get("image_basename") or "").strip()
        if (
            not image_basename
            or Path(image_basename).name != image_basename
            or Path(image_basename).suffix.casefold() != ".png"
        ):
            raise ActivityVisualBindingBuildError("task image basename is unsafe")
        if (
            str(raw_image.get("image_sha256") or "") != observation.image_sha256
            or raw_image.get("width") != observation.width
            or raw_image.get("height") != observation.height
        ):
            raise ActivityVisualBindingBuildError("parser image projection is inconsistent")
        task_image_path = (task_image_dir / image_basename).resolve()
        if task_image_path.parent != task_image_dir.resolve() or not task_image_path.is_file():
            raise ActivityVisualBindingBuildError("task image is outside the pinned directory")
        if sha256_file(task_image_path) != observation.image_sha256:
            raise ActivityVisualBindingBuildError("task image bytes changed")
        try:
            import cv2  # type: ignore
        except ImportError as exc:  # pragma: no cover - checked before this function
            raise ActivityVisualBindingBuildError("OpenCV runtime disappeared") from exc
        decoded = cv2.imread(str(task_image_path), cv2.IMREAD_GRAYSCALE)
        if decoded is None or decoded.shape[:2] != (observation.height, observation.width):
            raise ActivityVisualBindingBuildError("task image dimensions changed")
        if observation.image_sha256 in image_hashes or observation.task_id in audit_task_ids:
            raise ActivityVisualBindingBuildError("activity observation is duplicated")
        observations.append(
            ActivityObservation(
                task_id=observation.task_id,
                image_basename=image_basename,
                image_sha256=observation.image_sha256,
                width=observation.width,
                height=observation.height,
                parser_identity=observation.parser_identity,
                activity_number=int(marker_number),
                document_id=document_id,
                parser_projection_sha256=canonical_json_sha256(
                    {
                        "image_sha256": observation.image_sha256,
                        "width": observation.width,
                        "height": observation.height,
                        "parser_identity": observation.parser_identity,
                        "observed_source_marker": {
                            "kind": "activity_label",
                            "number": int(marker_number),
                        },
                    }
                ),
                source_identity_projection_sha256=canonical_json_sha256(
                    {
                        "document_id": document_id,
                        "kind": activity_documents[document_id].locator_kind,
                        "public_locator": activity_documents[
                            document_id
                        ].public_locator,
                        "name": activity_documents[document_id].locator_name,
                    }
                ),
                task_image_path=task_image_path,
            )
        )
        image_hashes.add(observation.image_sha256)
        audit_task_ids.add(observation.task_id)
    if not observations:
        raise ActivityVisualBindingBuildError("parser contains no activity-label observations")
    return sorted(observations, key=lambda item: item.image_sha256)


def _parse_document_mappings(values: Sequence[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        document_id, separator, raw_path = value.partition("=")
        document_id = document_id.strip()
        raw_path = raw_path.strip()
        if not separator or not document_id or not raw_path or document_id in result:
            raise ActivityVisualBindingBuildError(
                "--document must be a unique DOCUMENT_ID=PDF_PATH pair"
            )
        result[document_id] = Path(raw_path).resolve()
    return result


def _discover_executable(explicit: str | None, name: str) -> Path:
    raw = explicit or shutil.which(f"{name}.exe") or shutil.which(name)
    if not raw:
        raise ActivityVisualBindingBuildError(f"cannot discover {name}")
    path = Path(raw).resolve()
    if not path.is_file() or path.suffix.casefold() != ".exe":
        raise ActivityVisualBindingBuildError(
            f"{name} must resolve to a concrete .exe for a reproducible build"
        )
    return path


def _run_tool(tool: Path, arguments: Sequence[str], label: str) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        [str(tool), *arguments],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        shell=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip().splitlines()
        reason = detail[0] if detail else f"exit {completed.returncode}"
        raise ActivityVisualBindingBuildError(f"{label} failed: {reason}")
    return completed


def _tool_pin(tool: Path, name: str) -> dict[str, str]:
    completed = _run_tool(tool, ["-v"], f"{name} version probe")
    lines = [
        line.strip()
        for line in (completed.stdout + "\n" + completed.stderr).splitlines()
        if line.strip()
    ]
    prefix = f"{name} version "
    version_lines = [line for line in lines if line.startswith(prefix)]
    if len(version_lines) != 1:
        raise ActivityVisualBindingBuildError(f"cannot parse {name} version")
    version = version_lines[0][len(prefix) :].strip()
    if version != EXPECTED_POPPLER_VERSION:
        raise ActivityVisualBindingBuildError(
            f"{name} must be Poppler {EXPECTED_POPPLER_VERSION}, got {version}"
        )
    return {
        "path": str(tool),
        "sha256": sha256_file(tool),
        "version": version,
        "version_line": version_lines[0],
    }


def _pdf_page_count(pdfinfo: Path, pdf_path: Path) -> int:
    completed = _run_tool(pdfinfo, [str(pdf_path)], "pdfinfo")
    pages = [
        line.split(":", 1)[1].strip()
        for line in completed.stdout.splitlines()
        if line.startswith("Pages:") and ":" in line
    ]
    if len(pages) != 1 or not pages[0].isdigit():
        raise ActivityVisualBindingBuildError("pdfinfo did not expose one page count")
    return int(pages[0])


def _render_page(
    pdftoppm: Path,
    pdf_path: Path,
    page_number: int,
    render_dir: Path,
    *,
    dpi: int,
) -> Path:
    prefix = render_dir / f"page_{page_number:04d}"
    output = prefix.with_suffix(".png")
    _run_tool(
        pdftoppm,
        [
            "-f",
            str(page_number),
            "-l",
            str(page_number),
            "-r",
            str(dpi),
            "-png",
            "-singlefile",
            str(pdf_path),
            str(prefix),
        ],
        f"render page {page_number}",
    )
    if not output.is_file():
        raise ActivityVisualBindingBuildError(
            f"pdftoppm did not render indexed page {page_number}"
        )
    return output


def _evidence_projection(evidence: VisualPageEvidence) -> dict[str, Any]:
    value = asdict(evidence)
    value["mapped_polygon"] = (
        [list(point) for point in evidence.mapped_polygon]
        if evidence.mapped_polygon is not None
        else None
    )
    return value


def _compute_evidence_job(
    job: tuple[str, str, str, str, str, int],
) -> tuple[str, VisualPageEvidence]:
    (
        task_image_path,
        rendered_page_path,
        task_image_sha256,
        document_id,
        pdf_sha256,
        page_number,
    ) = job
    try:
        import cv2  # type: ignore
    except ImportError as exc:  # pragma: no cover - parent validates the runtime
        raise ActivityVisualBindingBuildError("OpenCV runtime disappeared") from exc
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    evidence = compute_sift_page_evidence(
        Path(task_image_path),
        Path(rendered_page_path),
        task_image_sha256=task_image_sha256,
        document_id=document_id,
        pdf_sha256=pdf_sha256,
        page_number=page_number,
        profile=SiftRuntimeProfile(
            render_dpi=144,
            expected_opencv_version=EXPECTED_OPENCV_VERSION,
        ),
    )
    return task_image_sha256, evidence


def _build(args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    runtime = _runtime_pins()
    profile_path = Path(args.profile).resolve()
    parser_path = Path(args.parser_jsonl).resolve()
    locator_path = Path(args.source_locators).resolve()
    source_index_path = Path(args.source_index).resolve()
    task_image_dir = Path(args.task_image_dir).resolve()
    output_path = Path(args.output_json).resolve()
    profile = _load_json(profile_path, "profile")
    expected_rows, profile_documents, input_hashes = _validate_profile(
        profile,
        parser_path=parser_path,
        locator_path=locator_path,
        source_index_path=source_index_path,
    )

    source_payload = _load_json(source_index_path, "source index")
    activity_documents, activity_records = _safe_activity_projection(source_payload)
    for document_id, document in activity_documents.items():
        frozen = profile_documents.get(document_id)
        if (
            frozen is None
            or frozen["pdf_sha256"] != document.pdf_sha256
            or frozen["page_count"] != document.page_count
        ):
            raise ActivityVisualBindingBuildError(
                f"profile/source-index activity document mismatch: {document_id}"
            )

    parser_rows = _load_jsonl(parser_path, "parser observations")
    if len(parser_rows) != expected_rows:
        raise ActivityVisualBindingBuildError(
            f"parser row count changed: expected {expected_rows}, got {len(parser_rows)}"
        )
    source_locator_rows = _load_jsonl(locator_path, "source locators")
    locator_index = _index_source_locators(source_locator_rows)
    policy = profile["policy"]
    allow_missing_nosw = (
        policy.get("yandex_public_identity_projection")
        == "url_name_plus_optional_numeric_nosw_v2"
    )
    observations = _activity_observations(
        parser_rows,
        locator_index,
        activity_documents,
        task_image_dir=task_image_dir,
        allow_missing_nosw=allow_missing_nosw,
    )

    document_paths = _parse_document_mappings(args.document)
    if set(document_paths) != set(activity_documents):
        raise ActivityVisualBindingBuildError(
            "--document mappings must equal the indexed activity-document set"
        )
    pdftoppm = _discover_executable(args.pdftoppm, "pdftoppm")
    pdfinfo = _discover_executable(args.pdfinfo, "pdfinfo")
    poppler = {
        "pdftoppm": _tool_pin(pdftoppm, "pdftoppm"),
        "pdfinfo": _tool_pin(pdfinfo, "pdfinfo"),
    }

    sift_profile = SiftRuntimeProfile(
        render_dpi=144,
        expected_opencv_version=EXPECTED_OPENCV_VERSION,
    )
    observations_by_document: dict[str, list[ActivityObservation]] = {
        document_id: [
            observation
            for observation in observations
            if observation.document_id == document_id
        ]
        for document_id in activity_documents
    }

    temp_parent = REPO_ROOT / "tmp" / "pdfs"
    temp_parent.mkdir(parents=True, exist_ok=True)
    document_output: dict[str, Any] = {}
    evidence_by_image: dict[str, list[VisualPageEvidence]] = {
        observation.image_sha256: [] for observation in observations
    }
    with tempfile.TemporaryDirectory(
        prefix="maxim_activity_visual_binding_v1_", dir=temp_parent
    ) as raw_temp:
        temp_root = Path(raw_temp)
        for document_id in sorted(activity_documents):
            document = activity_documents[document_id]
            pdf_path = document_paths[document_id]
            pdf_sha256 = _require_file_hash(
                pdf_path, document.pdf_sha256, f"workbook PDF {document_id}"
            )
            actual_page_count = _pdf_page_count(pdfinfo, pdf_path)
            if actual_page_count != document.page_count:
                raise ActivityVisualBindingBuildError(
                    f"PDF page count changed for {document_id}"
                )
            candidate_content_pages = [
                page
                for start, end in document.content_page_ranges
                for page in range(start, end + 1)
            ]
            document_render_dir = temp_root / document_id
            document_render_dir.mkdir()
            rendered_pages: dict[int, Path] = {}
            rendered_output: dict[str, Any] = {}
            for render_index, page_number in enumerate(candidate_content_pages, 1):
                rendered = _render_page(
                    pdftoppm,
                    pdf_path,
                    page_number,
                    document_render_dir,
                    dpi=sift_profile.render_dpi,
                )
                try:
                    import cv2  # type: ignore
                except ImportError as exc:  # pragma: no cover
                    raise ActivityVisualBindingBuildError("OpenCV runtime disappeared") from exc
                page_image = cv2.imread(str(rendered), cv2.IMREAD_GRAYSCALE)
                if page_image is None:
                    raise ActivityVisualBindingBuildError("rendered page cannot be decoded")
                rendered_pages[page_number] = rendered
                rendered_output[str(page_number)] = {
                    "rendered_page_sha256": sha256_file(rendered),
                    "width": int(page_image.shape[1]),
                    "height": int(page_image.shape[0]),
                }
                if render_index == 1 or render_index % 25 == 0 or render_index == len(
                    candidate_content_pages
                ):
                    print(
                        "PROGRESS "
                        f"rendered_pages={render_index}/{len(candidate_content_pages)} "
                        f"document_id={document_id}",
                        file=sys.stderr,
                        flush=True,
                    )
            jobs = [
                (
                    str(observation.task_image_path),
                    str(rendered_pages[page_number]),
                    observation.image_sha256,
                    document_id,
                    pdf_sha256,
                    page_number,
                )
                for observation in observations_by_document[document_id]
                for page_number in candidate_content_pages
            ]
            print(
                f"PROGRESS sift_pairs=0/{len(jobs)} document_id={document_id}",
                file=sys.stderr,
                flush=True,
            )
            with ProcessPoolExecutor(max_workers=SIFT_PROCESS_WORKERS) as executor:
                futures = [executor.submit(_compute_evidence_job, job) for job in jobs]
                for completed_count, future in enumerate(as_completed(futures), 1):
                    image_sha256, evidence = future.result()
                    evidence_by_image[image_sha256].append(evidence)
                    if (
                        completed_count == 1
                        or completed_count % 25 == 0
                        or completed_count == len(jobs)
                    ):
                        print(
                            "PROGRESS "
                            f"sift_pairs={completed_count}/{len(jobs)} "
                            f"document_id={document_id}",
                            file=sys.stderr,
                            flush=True,
                        )
            document_output[document_id] = {
                "pdf_path": _display_path(pdf_path),
                "pdf_sha256": pdf_sha256,
                "page_count": document.page_count,
                "content_page_ranges": [
                    list(page_range) for page_range in document.content_page_ranges
                ],
                "candidate_content_pages": candidate_content_pages,
                "rendered_pages": rendered_output,
            }

    bindings: dict[str, Any] = {}
    for observation in observations:
        evidences = sorted(
            evidence_by_image[observation.image_sha256],
            key=lambda item: item.page_number,
        )
        bindings[observation.image_sha256] = {
            "alignment_audit": {
                "task_id": observation.task_id,
                "task_id_role": "parser_to_public_source_locator_alignment_only",
                "task_id_used_as_page_or_record_feature": False,
            },
            "task_image": {
                "path": _display_path(observation.task_image_path),
                "image_basename": observation.image_basename,
                "image_sha256": observation.image_sha256,
                "width": observation.width,
                "height": observation.height,
            },
            "parser_identity": observation.parser_identity,
            "source_pins": {
                "parser_artifact_sha256": input_hashes["parser_observations"],
                "parser_projection_sha256": observation.parser_projection_sha256,
                "source_locators_artifact_sha256": input_hashes["source_locators"],
                "source_identity_projection_sha256": observation.source_identity_projection_sha256,
                "source_index_sha256": input_hashes["source_index"],
                "pdf_sha256": activity_documents[
                    observation.document_id
                ].pdf_sha256,
            },
            "observed_source_marker": {
                "kind": "activity_label",
                "number": observation.activity_number,
            },
            "document_id": observation.document_id,
            "raw_page_evidences": [_evidence_projection(item) for item in evidences],
        }

    output = {
        "schema_version": SCHEMA,
        "generator": GENERATOR,
        "source_only_guards": {
            "source_index_record_filter": "key_binding_kind=activity_answer_key AND question_marker_kind=activity_label",
            "parser_observation_filter": "observed_source_question_marker=activity_label",
            "render_scope": "all indexed physical content pages expanded from each activity document content_page_ranges",
            "task_id_role": "alignment_audit_only",
            "task_id_is_policy_feature": False,
            "source_answer_value_access": False,
            "benchmark_answer_candidate_outcome_artifacts_read": False,
        },
        "inputs": {
            "parser_observations": {
                "path": _display_path(parser_path),
                "sha256": input_hashes["parser_observations"],
                "rows": len(parser_rows),
            },
            "source_locators": {
                "path": _display_path(locator_path),
                "sha256": input_hashes["source_locators"],
                "rows": len(source_locator_rows),
            },
            "source_index": {
                "path": _display_path(source_index_path),
                "sha256": input_hashes["source_index"],
            },
            "task_image_dir": _display_path(task_image_dir),
            "documents": document_output,
        },
        "runtime": {
            **runtime,
            "poppler": poppler,
            "sift_process_workers": SIFT_PROCESS_WORKERS,
        },
        "profiles": {
            "sift": asdict(sift_profile),
        },
        "activity_records": [asdict(record) for record in activity_records],
        "bindings_by_task_image_sha256": bindings,
        "summary": {
            "activity_documents": len(activity_documents),
            "activity_records": len(activity_records),
            "activity_observations": len(observations),
            "raw_page_evidences": sum(
                len(binding["raw_page_evidences"]) for binding in bindings.values()
            ),
            "decision_layer": "not_applied_raw_evidence_only",
        },
    }
    return output, output_path


def _write_output(payload: Mapping[str, Any], output_path: Path) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    encoded = canonical_json_bytes(payload) + b"\n"
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        with temporary.open("wb") as sink:
            sink.write(encoded)
            sink.flush()
            os.fsync(sink.fileno())
        os.replace(temporary, output_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return sha256_file(output_path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--parser-jsonl", required=True)
    parser.add_argument("--source-locators", required=True)
    parser.add_argument("--source-index", required=True)
    parser.add_argument("--task-image-dir", required=True)
    parser.add_argument(
        "--document",
        action="append",
        default=[],
        metavar="DOCUMENT_ID=PDF_PATH",
        help="exact PDF mapping; repeat once per indexed activity document",
    )
    parser.add_argument("--pdftoppm", help="absolute path to pinned pdftoppm.exe")
    parser.add_argument("--pdfinfo", help="absolute path to pinned pdfinfo.exe")
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def main() -> int:
    try:
        payload, output_path = _build(_parse_args())
        output_sha256 = _write_output(payload, output_path)
    except (
        ActivityVisualBindingBuildError,
        OfficialSourceError,
        OSError,
        TypeError,
        ValueError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(
        canonical_json_bytes(
            {
                "output_json": str(output_path),
                "output_sha256": output_sha256,
                "summary": payload["summary"],
            }
        ).decode("utf-8")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
