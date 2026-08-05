from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import pytest

pdfplumber = pytest.importorskip("pdfplumber")

from evidence_os.activity_answer_key import activity_marker_inventory
from evidence_os.image_only_activity import (
    ImageOnlyActivityError,
    load_image_only_activity_visual_artifact_json,
    project_image_only_activity_observation,
    resolve_image_only_activity_question,
    verified_image_only_activity_bindings_from_artifact,
)
from evidence_os.official_workbook import (
    parse_workbook_index,
    verify_workbook_index_pdf,
)
from evidence_os.visual_coordinate_binding import (
    ActivityVisualRecordRef,
    VisualBindingThresholds,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from compose_maxim_official_ogm_failclosed_v2 import (  # noqa: E402
    _validate_workbook_certificate_artifact,
)
from run_maxim_public_workbook_v1 import _certificate_record  # noqa: E402

ARTIFACT_PATH = (
    ROOT
    / "reports"
    / "maxim_official_exact_source_v2_20260805"
    / "frozen"
    / "activity_visual_binding_biology_activity4_imageonly_candidate_v1.json"
)


def _repo_path(value: str) -> Path:
    return (ROOT / value).resolve()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


@pytest.fixture(scope="module")
def replay_context() -> dict[str, Any]:
    artifact = load_image_only_activity_visual_artifact_json(ARTIFACT_PATH)
    inputs = artifact["inputs"]
    parser_rows = _load_jsonl(_repo_path(inputs["parser_observations"]["path"]))
    locator_rows = _load_jsonl(_repo_path(inputs["source_locators"]["path"]))
    locators = {row["task_id"]: row for row in locator_rows}
    observations = {}
    raw_by_task_id = {}
    for raw in parser_rows:
        try:
            observation = project_image_only_activity_observation(raw)
        except ImageOnlyActivityError:
            continue
        observations[observation.task_id] = observation
        raw_by_task_id[observation.task_id] = raw
    index = parse_workbook_index(
        json.loads(
            _repo_path(inputs["source_index"]["path"]).read_text(
                encoding="utf-8-sig"
            )
        )
    )
    documents = {document.document_id: document for document in index.documents}
    records = tuple(
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
        for document in index.documents
        for question in document.questions
    )
    pdf_paths = {
        document_id: _repo_path(spec["pdf_path"])
        for document_id, spec in inputs["documents"].items()
    }
    call = {
        "repo_root": ROOT,
        "expected_parser_sha256": inputs["parser_observations"]["sha256"],
        "expected_source_locators_sha256": inputs["source_locators"]["sha256"],
        "observations_by_task_id": observations,
        "source_urls_by_task_id": {
            task_id: str(locators[task_id]["source_url"])
            for task_id in observations
        },
        "documents_by_id": documents,
        "records": records,
        "document_pdf_paths": pdf_paths,
        "thresholds": VisualBindingThresholds(),
    }
    bindings = verified_image_only_activity_bindings_from_artifact(
        artifact,
        **call,
    )
    return {
        "artifact": artifact,
        "bindings": bindings,
        "call": call,
        "raw_by_task_id": raw_by_task_id,
        "pdf_paths": pdf_paths,
        "documents": documents,
    }


def test_strict_runtime_replays_full_page_set_and_pdf_marker_inventory(
    replay_context: dict[str, Any],
) -> None:
    artifact = replay_context["artifact"]
    binding = next(iter(replay_context["bindings"].values()))
    document_spec = artifact["inputs"]["documents"][binding.document_id]

    assert document_spec["candidate_content_pages"] == list(range(1, 179))
    assert set(document_spec["rendered_pages"]) == {
        str(page) for page in range(1, 179)
    }
    assert len(binding.evidences) == 178
    assert {item.page_number for item in binding.evidences} == set(range(1, 179))
    assert binding.decision.selected_page_number == 75
    assert binding.marker_inventory == (4,)

    with pdfplumber.open(replay_context["pdf_paths"][binding.document_id]) as pdf:
        assert len(pdf.pages) == 183
        assert activity_marker_inventory(pdf.pages[74]) == (4,)


def test_task_id_rename_cannot_change_page_or_record_decision(
    replay_context: dict[str, Any],
) -> None:
    artifact = copy.deepcopy(replay_context["artifact"])
    original_binding = next(iter(replay_context["bindings"].values()))
    renamed_id = "renamed-alignment-only-id"
    raw = copy.deepcopy(
        replay_context["raw_by_task_id"][original_binding.alignment_task_id]
    )
    raw["task_id"] = renamed_id
    renamed_observation = project_image_only_activity_observation(raw)
    raw_artifact_binding = next(
        iter(artifact["bindings_by_task_image_sha256"].values())
    )
    raw_artifact_binding["alignment_audit"]["task_id"] = renamed_id
    call = dict(replay_context["call"])
    call["observations_by_task_id"] = {renamed_id: renamed_observation}
    source_url = replay_context["call"]["source_urls_by_task_id"][
        original_binding.alignment_task_id
    ]
    call["source_urls_by_task_id"] = {renamed_id: source_url}

    renamed = next(
        iter(
            verified_image_only_activity_bindings_from_artifact(
                artifact,
                **call,
            ).values()
        )
    )

    assert renamed.alignment_task_id == renamed_id
    assert renamed.observation.parser_projection_sha256 == (
        original_binding.observation.parser_projection_sha256
    )
    assert renamed.trace() == original_binding.trace()


def test_strict_runtime_rejects_artifact_guard_tampering(
    replay_context: dict[str, Any],
) -> None:
    artifact = copy.deepcopy(replay_context["artifact"])
    artifact["source_only_guards"]["task_id_is_policy_feature"] = True

    with pytest.raises(ImageOnlyActivityError, match="guards changed"):
        verified_image_only_activity_bindings_from_artifact(
            artifact,
            **replay_context["call"],
        )


def test_strict_runtime_rejects_selected_pdf_marker_inventory_tampering(
    replay_context: dict[str, Any],
) -> None:
    artifact = copy.deepcopy(replay_context["artifact"])
    raw_binding = next(
        iter(artifact["bindings_by_task_image_sha256"].values())
    )
    raw_binding["source_page_activity_marker_inventory"][
        "canonical_activity_marker_numbers"
    ] = [3, 4]

    with pytest.raises(ImageOnlyActivityError, match="marker inventory changed"):
        verified_image_only_activity_bindings_from_artifact(
            artifact,
            **replay_context["call"],
        )


def test_strict_runtime_rejects_extra_live_record_on_selected_page(
    replay_context: dict[str, Any],
) -> None:
    artifact_record = replay_context["call"]["records"][0]
    extra_activity_number = artifact_record.activity_number + 1
    extra = replace(
        artifact_record,
        record_id=(
            f"{artifact_record.document_id}:p"
            f"{artifact_record.content_page_number}:q{extra_activity_number}"
        ),
        activity_number=extra_activity_number,
    )
    call = dict(replay_context["call"])
    call["records"] = tuple(call["records"]) + (extra,)

    with pytest.raises(ImageOnlyActivityError, match="selected page record set"):
        verified_image_only_activity_bindings_from_artifact(
            replay_context["artifact"],
            **call,
        )


def test_image_only_only_certificate_contract_replays_without_normal_visual(
    replay_context: dict[str, Any],
) -> None:
    import pypdf

    artifact = replay_context["artifact"]
    inputs = artifact["inputs"]
    binding = next(iter(replay_context["bindings"].values()))
    observation = binding.observation
    source_url = replay_context["call"]["source_urls_by_task_id"][
        binding.alignment_task_id
    ]
    document = replay_context["documents"][binding.document_id]
    source_verification = verify_workbook_index_pdf(
        replay_context["pdf_paths"][binding.document_id],
        document,
    )
    result = resolve_image_only_activity_question(
        observation,
        source_url,
        document,
        binding,
        verified_content_marker_counts=source_verification[
            "content_marker_counts"
        ],
        allow_missing_nosw=True,
    )
    assert result.accepted is True
    assert result.answer
    assert result.problem.statement == ""
    assert "observed_question_number" not in result.trace["observation"]
    assert result.trace["observation"]["observed_source_marker_kind"] is None
    assert result.trace["observation"]["observed_source_marker_number"] is None

    artifact_sha = hashlib.sha256(ARTIFACT_PATH.read_bytes()).hexdigest()
    profile_sha = "f" * 64
    profile = {
        "inputs": {
            "parser_observations": {
                "sha256": inputs["parser_observations"]["sha256"]
            },
            "source_locators": {
                "sha256": inputs["source_locators"]["sha256"]
            },
            "source_index": {"sha256": inputs["source_index"]["sha256"]},
            "image_only_activity_visual_evidence": {
                "sha256": artifact_sha
            },
        },
        "runtime": {
            "pypdf_version": str(pypdf.__version__),
            "pdfplumber_version": str(pdfplumber.__version__),
        },
        "documents": [
            {
                "document_id": document.document_id,
                "pdf_sha256": document.pdf_sha256,
                "page_count": document.page_count,
            }
        ],
    }
    result.trace["provenance"] = {
        "profile_sha256": profile_sha,
        "parser_observations_sha256": inputs["parser_observations"][
            "sha256"
        ],
        "source_locators_sha256": inputs["source_locators"]["sha256"],
        "source_index_sha256": inputs["source_index"]["sha256"],
        "workbook_pdf_sha256": document.pdf_sha256,
        "pypdf_version": str(pypdf.__version__),
        "pdfplumber_version": str(pdfplumber.__version__),
        "image_only_activity_visual_evidence_sha256": artifact_sha,
        "source_verification": source_verification,
    }
    certificate = _certificate_record(
        binding.alignment_task_id,
        result,
        result.answer,
    )
    candidate = {
        "task_id": binding.alignment_task_id,
        "final_answer": result.answer,
        "abstain": False,
        "error": None,
        "generation": {
            "gold_access": False,
            "resolver": "public-workbook-ocr-page-key-binding-v1",
            "source_certificate": True,
        },
    }

    _validate_workbook_certificate_artifact(
        task_id=binding.alignment_task_id,
        raw_candidate=candidate,
        raw_certificate=certificate,
        trace=result.trace,
        profile=profile,
        profile_sha=profile_sha,
    )
