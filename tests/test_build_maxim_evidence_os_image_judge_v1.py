from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from evidence_os.official_ogm import (
    PageMatcher,
    canonical_json_bytes,
    parser_observation_allow_missing_number,
    sha256_file,
)
from evidence_os.official_workbook import (
    WorkbookThresholds,
    document_for_source,
    parse_workbook_index,
    resolve_workbook_question,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

SCRIPT = SCRIPTS / "build_maxim_evidence_os_image_judge_v1.py"
SPEC = importlib.util.spec_from_file_location("maxim_evidence_os_image_judge", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
image_judge = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(image_judge)

from run_maxim_public_workbook_v1 import _certificate_record  # noqa: E402
import compose_maxim_official_ogm_failclosed_v2 as composer  # noqa: E402


TASK_IDS = tuple(f"task_{index:03d}" for index in range(274))
IMAGE_IDS = frozenset(TASK_IDS[:97])
CHANGED_ID = TASK_IDS[0]
UNCHANGED_ID = TASK_IDS[1]
TARGET_TEXT = (
    "algebra triangle orchard compass lantern isotope fraction theorem polygon "
    "velocity cylinder matrix radius quotient symmetry tangent integer prism equation"
)
DISTRACTOR_TEXT = (
    "history empire treaty archive dynasty republic parliament chronology museum "
    "geography migration monument citizenship reform document manuscript"
)
PAGE_TEXTS = (f"1. {TARGET_TEXT}", f"1. {DISTRACTOR_TEXT}")
SOURCE_URL = (
    "https://docs.yandex.ru/docs/view?"
    "url=ya-disk-public%3A%2F%2Fsynthetic-public-key"
    "&name=synthetic-book.pdf&nosw=17"
)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as sink:
        for row in rows:
            sink.write(canonical_json_bytes(row).decode("utf-8") + "\n")


def _parser_row(task_id: str) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "parser": {
            "gold_access": False,
            "pipeline_version": "synthetic-v1",
            "layout_model": "synthetic-layout",
            "recognition_model": "synthetic-recognition",
        },
        "images": [
            {
                "image_sha256": "f" * 64,
                "width": 900,
                "height": 600,
                "parsing_res_list": [
                    {"block_label": "text", "block_content": f"1. {TARGET_TEXT}"}
                ],
            }
        ],
    }


def _anchor_row(task_id: str) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "final_answer": "A",
        "generation": {"gold_access": False, "resolver": "synthetic-anchor"},
    }


def _base_judge_row(task_id: str) -> dict[str, Any]:
    row = {
        "task_id": task_id,
        "subject": "synthetic-math",
        "grade": "7",
        "answer_type": "choice",
        "verdict": {
            "label": "incorrect",
            "score": 0,
            "strict_correct": False,
            "final_answer_correct": False,
            "reasoning_correct": None,
            "complete": True,
            "confidence": 1.0,
            "error_types": [],
            "rationale": "Synthetic pre-existing verdict.",
            "reference_quality_issue": False,
        },
    }
    if task_id == CHANGED_ID:
        # These canaries model fields which must never be copied into a newly
        # certificate-adjudicated row.
        row["gold"] = "must-not-propagate"
        row["reference_answer"] = "must-not-propagate"
        row["outcome"] = {"strict_correct": False}
    return row


@dataclass
class SyntheticBundle:
    root: Path
    profile_path: Path
    resolver_manifest_path: Path
    composition_manifest_path: Path
    base_solver_path: Path
    base_judge_path: Path
    output_path: Path
    output_manifest_path: Path
    candidate_path: Path
    certificate_path: Path
    solver_path: Path
    decisions_path: Path
    profile: dict[str, Any]
    candidates: dict[str, dict[str, Any]]
    certificates: dict[str, dict[str, Any]]
    solver: dict[str, dict[str, Any]]
    decisions: dict[str, dict[str, Any]]
    base_solver: dict[str, dict[str, Any]]
    base_judge: dict[str, dict[str, Any]]
    source_index: dict[str, Any]
    source_index_path: Path
    pdf_path: Path
    original_result: Any

    @property
    def document_id(self) -> str:
        return str(self.source_index["documents"][0]["document_id"])

    @property
    def source_questions(self) -> list[dict[str, Any]]:
        return self.source_index["documents"][0]["questions"]

    def seal(self) -> dict[str, Any]:
        _write_jsonl(self.candidate_path, [self.candidates[task] for task in TASK_IDS])
        _write_jsonl(
            self.certificate_path,
            [self.certificates[task] for task in sorted(self.certificates)],
        )
        _write_jsonl(self.solver_path, [self.solver[task] for task in TASK_IDS])
        _write_jsonl(self.decisions_path, [self.decisions[task] for task in self.decisions])
        _write_jsonl(self.base_solver_path, [self.base_solver[task] for task in TASK_IDS])
        _write_jsonl(self.base_judge_path, [self.base_judge[task] for task in TASK_IDS[:97]])

        profile_sha = sha256_file(self.profile_path)
        resolver_manifest = {
            "schema_version": "maxim-public-workbook-run-v1",
            "rows": 274,
            "accepted_certificates": len(self.certificates),
            "gold_access": False,
            "benchmark_candidate_or_outcome_access": False,
            "task_id_used_for_alignment_only": True,
            "viewer_nosw_used_as_policy_feature": False,
            "profile": {"path": str(self.profile_path), "sha256": profile_sha},
            "inputs": {
                "documents": {
                    self.document_id: {
                        "path": str(self.pdf_path),
                        "sha256": sha256_file(self.pdf_path),
                    }
                }
            },
            "artifacts": {
                "candidate": {
                    "path": str(self.candidate_path),
                    "sha256": sha256_file(self.candidate_path),
                },
                "certificates": {
                    "path": str(self.certificate_path),
                    "sha256": sha256_file(self.certificate_path),
                },
            },
        }
        _write_json(self.resolver_manifest_path, resolver_manifest)
        resolver_sha = sha256_file(self.resolver_manifest_path)
        composition_manifest = {
            "schema_version": "maxim-official-source-failclosed-composition-v2",
            "rows": 274,
            "overrides": sum(
                decision["action"] == "replace_anchor"
                for decision in self.decisions.values()
            ),
            "gold_access": False,
            "score_or_outcome_access": False,
            "profile": {"path": str(self.profile_path), "sha256": profile_sha},
            "resolver_manifest": {
                "path": str(self.resolver_manifest_path),
                "sha256": resolver_sha,
            },
            "output": {
                "solver": {
                    "path": str(self.solver_path),
                    "sha256": sha256_file(self.solver_path),
                },
                "decisions": {
                    "path": str(self.decisions_path),
                    "sha256": sha256_file(self.decisions_path),
                },
            },
        }
        _write_json(self.composition_manifest_path, composition_manifest)
        return {
            "profile_path": self.profile_path,
            "resolver_manifest_path": self.resolver_manifest_path,
            "expected_resolver_manifest_sha256": resolver_sha,
            "composition_manifest_path": self.composition_manifest_path,
            "expected_composition_manifest_sha256": sha256_file(
                self.composition_manifest_path
            ),
            "base_solver_path": self.base_solver_path,
            "expected_base_solver_sha256": sha256_file(self.base_solver_path),
            "base_judge_path": self.base_judge_path,
            "expected_base_judge_sha256": sha256_file(self.base_judge_path),
            "output_path": self.output_path,
            "manifest_path": self.output_manifest_path,
        }


class _SyntheticPage:
    def __init__(self, text: str) -> None:
        self._text = text

    def extract_text(self) -> str:
        return self._text


class _SyntheticPdfReader:
    def __init__(self, _path: str) -> None:
        self.pages = [_SyntheticPage(text) for text in PAGE_TEXTS]


@pytest.fixture
def synthetic_bundle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SyntheticBundle:
    fake_pypdf = ModuleType("pypdf")
    fake_pypdf.__version__ = "synthetic-pypdf-1"  # type: ignore[attr-defined]
    fake_pypdf.PdfReader = _SyntheticPdfReader  # type: ignore[attr-defined]
    fake_pdfplumber = ModuleType("pdfplumber")
    fake_pdfplumber.__version__ = "synthetic-pdfplumber-1"  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pypdf", fake_pypdf)
    monkeypatch.setitem(sys.modules, "pdfplumber", fake_pdfplumber)
    monkeypatch.setattr(
        image_judge,
        "verify_workbook_index_pdf",
        lambda _path, document: {
            "records": len(document.questions),
            "verified_records": len(document.questions),
            "content_marker_counts": {
                question.record_id: 1 for question in document.questions
            },
        },
    )

    pdf_path = tmp_path / "synthetic-book.pdf"
    pdf_path.write_bytes(b"synthetic pinned public workbook")
    pdf_sha = sha256_file(pdf_path)
    document_id = f"synthetic_book_{pdf_sha[:12]}"
    source_index = {
        "schema_version": "public-workbook-source-index-v1",
        "documents": [
            {
                "document_id": document_id,
                "locator": {
                    "kind": "yandex_public",
                    "public_locator": "ya-disk-public://synthetic-public-key",
                    "name": "synthetic-book.pdf",
                },
                "pdf_sha256": pdf_sha,
                "page_count": 2,
                "content_page_ranges": [[1, 2]],
                "questions": [
                    {
                        "record_id": f"{document_id}:p1:q1",
                        "content_page_number": 1,
                        "question_number": 1,
                        "question_text": f"1. {TARGET_TEXT}",
                        "answer": "B",
                        "answer_format": "choice",
                        "key_binding_kind": "inline_solution",
                        "key_page_number": 1,
                        "key_bbox": [10, 10, 20, 20],
                        "visually_checked": True,
                    },
                    {
                        "record_id": f"{document_id}:p2:q1",
                        "content_page_number": 2,
                        "question_number": 1,
                        "question_text": f"1. {DISTRACTOR_TEXT}",
                        "answer": "C",
                        "answer_format": "choice",
                        "key_binding_kind": "inline_solution",
                        "key_page_number": 2,
                        "key_bbox": [30, 30, 40, 40],
                        "visually_checked": True,
                    },
                ],
            }
        ],
    }
    source_index_path = tmp_path / "source-index.json"
    _write_json(source_index_path, source_index)

    parser_path = tmp_path / "parser.jsonl"
    locator_path = tmp_path / "locators.jsonl"
    anchor_path = tmp_path / "anchor.jsonl"
    parser_rows = [_parser_row(task_id) for task_id in TASK_IDS]
    locator_rows = [
        {"task_id": task_id, "source_url": SOURCE_URL} for task_id in TASK_IDS
    ]
    anchor_rows = [_anchor_row(task_id) for task_id in TASK_IDS]
    _write_jsonl(parser_path, parser_rows)
    _write_jsonl(locator_path, locator_rows)
    _write_jsonl(anchor_path, anchor_rows)

    profile_path = tmp_path / "profile.json"
    profile = {
        "schema_version": "maxim-public-workbook-profile-v1",
        "profile_name": "synthetic-workbook-profile",
        "expected_rows": 274,
        "anchor": {"path": str(anchor_path), "sha256": sha256_file(anchor_path)},
        "inputs": {
            "parser_observations": {
                "path": str(parser_path),
                "sha256": sha256_file(parser_path),
            },
            "source_locators": {
                "path": str(locator_path),
                "sha256": sha256_file(locator_path),
            },
            "source_index": {
                "path": str(source_index_path),
                "sha256": sha256_file(source_index_path),
            },
        },
        "documents": [
            {"document_id": document_id, "pdf_sha256": pdf_sha, "page_count": 2}
        ],
        "runtime": {
            "pypdf_version": "synthetic-pypdf-1",
            "pdfplumber_version": "synthetic-pdfplumber-1",
        },
        "policy": {
            "min_page_coverage": 0.65,
            "min_page_matched_tokens": 10,
            "min_page_margin": 0.12,
            "min_numberless_question_coverage": 1.0,
            "min_numberless_question_matched_tokens": 999,
            "min_numberless_question_margin": 1.0,
            "require_observed_question_number": True,
            "allow_numberless_question_binding": False,
            "require_unique_printed_number_on_page": True,
            "require_pdf_bound_key_context": True,
            "question_number_projection": "unique_block_markers_v1",
        },
    }
    _write_json(profile_path, profile)
    profile_sha = sha256_file(profile_path)

    index = parse_workbook_index(source_index)
    document = document_for_source(index, SOURCE_URL)
    assert document is not None
    observation = parser_observation_allow_missing_number(parser_rows[0])
    thresholds = WorkbookThresholds(
        min_page_coverage=0.65,
        min_page_matched_tokens=10,
        min_page_margin=0.12,
        min_numberless_question_coverage=1.0,
        min_numberless_question_matched_tokens=999,
        min_numberless_question_margin=1.0,
    )
    original_result = resolve_workbook_question(
        observation,
        SOURCE_URL,
        document,
        PageMatcher(PAGE_TEXTS),
        PAGE_TEXTS,
        thresholds,
    )
    assert original_result.accepted and original_result.answer == "B"
    original_result.trace["provenance"] = {
        "profile_sha256": profile_sha,
        "parser_observations_sha256": sha256_file(parser_path),
        "source_locators_sha256": sha256_file(locator_path),
        "source_index_sha256": sha256_file(source_index_path),
        "workbook_pdf_sha256": pdf_sha,
        "pypdf_version": "synthetic-pypdf-1",
        "pdfplumber_version": "synthetic-pdfplumber-1",
        "source_verification": {
            "records": 2,
            "verified_records": 2,
            "content_marker_counts": {
                question.record_id: 1 for question in document.questions
            },
        },
    }
    changed_certificate = _certificate_record(CHANGED_ID, original_result, "B")
    trace_fingerprint = str(changed_certificate["trace_fingerprint"])

    candidates = {
        task_id: {
            "task_id": task_id,
            "final_answer": "B" if task_id == CHANGED_ID else "",
            "abstain": task_id != CHANGED_ID,
            "error": None if task_id == CHANGED_ID else "synthetic_abstain",
            "generation": {
                "gold_access": False,
                "resolver": image_judge.WORKBOOK_VERIFIER,
                "source_certificate": task_id == CHANGED_ID,
            },
        }
        for task_id in TASK_IDS
    }
    certificates = {CHANGED_ID: changed_certificate}
    solver = {task_id: _anchor_row(task_id) for task_id in TASK_IDS}
    solver[CHANGED_ID] = {
        "task_id": CHANGED_ID,
        "final_answer": "B",
        "generation": {
            "gold_access": False,
            "resolver": "synthetic-anchor",
            "official_source_override": {
                "verifier": image_judge.WORKBOOK_VERIFIER,
                "trace_fingerprint": trace_fingerprint,
                "profile_sha256": profile_sha,
            },
        },
    }
    decisions = {
        task_id: {
            "task_id": task_id,
            "action": "replace_anchor" if task_id == CHANGED_ID else "keep_anchor",
            "reason": (
                "strongly_verified_challenger"
                if task_id == CHANGED_ID
                else "no_challengers"
            ),
            "anchor_answer": "A",
            "selected_answer": "B" if task_id == CHANGED_ID else "A",
            "certificate_trace_fingerprint": (
                trace_fingerprint if task_id == CHANGED_ID else None
            ),
        }
        for task_id in TASK_IDS
    }
    base_solver = {
        task_id: {
            "task_id": task_id,
            # Only the image partition is tied to this legacy solver/judge.
            # Non-image answers intentionally differ from the profile anchor.
            "final_answer": "A" if task_id in IMAGE_IDS else "LEGACY",
        }
        for task_id in TASK_IDS
    }
    base_judge = {task_id: _base_judge_row(task_id) for task_id in TASK_IDS[:97]}

    bundle = SyntheticBundle(
        root=tmp_path,
        profile_path=profile_path,
        resolver_manifest_path=tmp_path / "resolver-manifest.json",
        composition_manifest_path=tmp_path / "composition-manifest.json",
        base_solver_path=tmp_path / "base-solver.jsonl",
        base_judge_path=tmp_path / "base-judge.jsonl",
        output_path=tmp_path / "image-judge.jsonl",
        output_manifest_path=tmp_path / "image-judge-manifest.json",
        candidate_path=tmp_path / "candidate.jsonl",
        certificate_path=tmp_path / "certificates.jsonl",
        solver_path=tmp_path / "composed-solver.jsonl",
        decisions_path=tmp_path / "decisions.jsonl",
        profile=profile,
        candidates=candidates,
        certificates=certificates,
        solver=solver,
        decisions=decisions,
        base_solver=base_solver,
        base_judge=base_judge,
        source_index=source_index,
        source_index_path=source_index_path,
        pdf_path=pdf_path,
        original_result=original_result,
    )
    bundle.seal()
    return bundle


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _configure_certified_equal_anchor(
    bundle: SyntheticBundle,
) -> None:
    """Turn the synthetic changed row into a source-confirmed anchor answer."""

    anchor_path = Path(str(bundle.profile["anchor"]["path"]))
    anchor_rows = _read_jsonl(anchor_path)
    anchor_row = next(row for row in anchor_rows if row["task_id"] == CHANGED_ID)
    anchor_row["final_answer"] = "B"
    _write_jsonl(anchor_path, anchor_rows)
    bundle.profile["anchor"]["sha256"] = sha256_file(anchor_path)
    _write_json(bundle.profile_path, bundle.profile)

    trace = deepcopy(bundle.original_result.trace)
    trace["provenance"]["profile_sha256"] = sha256_file(bundle.profile_path)
    result = SimpleNamespace(
        problem=bundle.original_result.problem,
        checks=bundle.original_result.checks,
        trace=trace,
    )
    certificate = _certificate_record(CHANGED_ID, result, "B")
    fingerprint = str(certificate["trace_fingerprint"])
    bundle.certificates[CHANGED_ID] = certificate
    bundle.solver[CHANGED_ID] = deepcopy(anchor_row)
    bundle.base_solver[CHANGED_ID]["final_answer"] = "B"
    bundle.decisions[CHANGED_ID] = {
        "task_id": CHANGED_ID,
        "action": "keep_anchor",
        "reason": "equivalent_to_anchor",
        "anchor_answer": "B",
        "selected_answer": "B",
        "certificate_trace_fingerprint": fingerprint,
    }


def test_valid_bundle_uses_profile_anchor_and_strictly_projects_changed_row(
    synthetic_bundle: SyntheticBundle,
) -> None:
    manifest = image_judge.build(**synthetic_bundle.seal())

    output = {row["task_id"]: row for row in _read_jsonl(synthetic_bundle.output_path)}
    changed = output[CHANGED_ID]
    assert changed["verdict"]["strict_correct"] is True
    assert changed["verdict"]["label"] == "fully_correct"
    assert changed["verdict"]["reasoning_correct"] is None
    assert changed["metadata"]["composition_action"] == "replace_anchor"
    assert changed["metadata"]["answer_changed_from_anchor"] is True
    assert "gold" not in changed
    assert "reference_answer" not in changed
    assert "outcome" not in changed
    assert manifest["benchmark_reference_answers_opened"] is False
    assert manifest["base_image_judge_outcomes_read_and_copied_for_unchanged_rows"] is True
    assert manifest["base_image_judge_outcomes_used_for_changed_rows"] is False


def test_certified_equal_anchor_gets_deterministic_fully_correct_verdict(
    synthetic_bundle: SyntheticBundle,
) -> None:
    _configure_certified_equal_anchor(synthetic_bundle)
    original = deepcopy(synthetic_bundle.base_judge[CHANGED_ID])
    assert original["verdict"]["strict_correct"] is False

    manifest = image_judge.build(**synthetic_bundle.seal())

    output = {row["task_id"]: row for row in _read_jsonl(synthetic_bundle.output_path)}
    confirmed = output[CHANGED_ID]
    assert confirmed["setup"] == "maxim_evidence_os_official_source_adjudication_v1"
    assert confirmed["judge"]["backend"] == "deterministic-pinned-pdf-certificate"
    assert confirmed["verdict"]["label"] == "fully_correct"
    assert confirmed["verdict"]["strict_correct"] is True
    assert confirmed["metadata"]["composition_action"] == "keep_anchor"
    assert confirmed["metadata"]["answer_changed_from_anchor"] is False
    assert confirmed != original
    assert manifest["copied_nonadjudicated_rows"] == 96
    assert manifest["official_certificate_rows"] == [
        {
            "task_id": CHANGED_ID,
            "source_record_id": synthetic_bundle.source_questions[0]["record_id"],
            "certificate_trace_fingerprint": synthetic_bundle.certificates[CHANGED_ID][
                "trace_fingerprint"
            ],
            "composition_action": "keep_anchor",
            "answer_changed_from_anchor": False,
        }
    ]


def test_certified_equal_anchor_verdict_is_independent_of_base_verdict(
    synthetic_bundle: SyntheticBundle,
) -> None:
    _configure_certified_equal_anchor(synthetic_bundle)
    image_judge.build(**synthetic_bundle.seal())
    first = {
        row["task_id"]: row for row in _read_jsonl(synthetic_bundle.output_path)
    }[CHANGED_ID]

    synthetic_bundle.base_judge[CHANGED_ID]["verdict"] = {
        "label": "fully_correct",
        "score": 4,
        "strict_correct": True,
        "final_answer_correct": True,
        "reasoning_correct": True,
        "complete": True,
        "confidence": 0.125,
        "error_types": ["synthetic-equal-anchor-canary"],
        "rationale": "A contradictory synthetic base verdict.",
        "reference_quality_issue": True,
    }
    image_judge.build(**synthetic_bundle.seal())
    second = {
        row["task_id"]: row for row in _read_jsonl(synthetic_bundle.output_path)
    }[CHANGED_ID]

    assert second == first


def test_noncertified_equal_anchor_row_is_byte_copied(
    synthetic_bundle: SyntheticBundle,
) -> None:
    expected = deepcopy(synthetic_bundle.base_judge[UNCHANGED_ID])

    image_judge.build(**synthetic_bundle.seal())

    base_lines = synthetic_bundle.base_judge_path.read_bytes().splitlines(keepends=True)
    output_lines = synthetic_bundle.output_path.read_bytes().splitlines(keepends=True)
    output = {row["task_id"]: row for row in _read_jsonl(synthetic_bundle.output_path)}
    assert output[UNCHANGED_ID] == expected
    assert canonical_json_bytes(output[UNCHANGED_ID]) == canonical_json_bytes(expected)
    assert output_lines[1] == base_lines[1]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("reason", "no_challengers"),
        ("certificate_trace_fingerprint", "0" * 64),
    ],
)
def test_certified_equal_anchor_forged_selection_fails_closed(
    synthetic_bundle: SyntheticBundle,
    field: str,
    value: str,
) -> None:
    _configure_certified_equal_anchor(synthetic_bundle)
    synthetic_bundle.decisions[CHANGED_ID][field] = value

    with pytest.raises(
        image_judge.JudgeBuildError,
        match="source certificate that is not composition-selected",
    ):
        image_judge.build(**synthetic_bundle.seal())


def test_changed_verdict_is_independent_of_the_base_judge_outcome(
    synthetic_bundle: SyntheticBundle,
) -> None:
    image_judge.build(**synthetic_bundle.seal())
    first = {
        row["task_id"]: row for row in _read_jsonl(synthetic_bundle.output_path)
    }[CHANGED_ID]

    synthetic_bundle.base_judge[CHANGED_ID]["verdict"] = {
        "label": "fully_correct",
        "score": 4,
        "strict_correct": True,
        "final_answer_correct": True,
        "reasoning_correct": True,
        "complete": True,
        "confidence": 0.123,
        "error_types": ["synthetic-outcome-canary"],
        "rationale": "A contradictory synthetic base outcome.",
        "reference_quality_issue": True,
    }
    image_judge.build(**synthetic_bundle.seal())
    second = {
        row["task_id"]: row for row in _read_jsonl(synthetic_bundle.output_path)
    }[CHANGED_ID]

    assert second == first


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("status", "fail"),
        ("strength", "weak"),
        ("claim_coverage", 0.0),
        ("contradiction_count", 1),
    ],
)
def test_non_strong_certificate_cannot_create_a_correct_verdict(
    synthetic_bundle: SyntheticBundle,
    field: str,
    value: Any,
) -> None:
    synthetic_bundle.certificates[CHANGED_ID][field] = value

    with pytest.raises(image_judge.JudgeBuildError, match="strong passing proof"):
        image_judge.build(**synthetic_bundle.seal())


def test_answer_fingerprint_tamper_is_rejected(
    synthetic_bundle: SyntheticBundle,
) -> None:
    synthetic_bundle.certificates[CHANGED_ID]["answer_fingerprint"] = "0" * 64

    with pytest.raises(image_judge.JudgeBuildError, match="input/answer bound"):
        image_judge.build(**synthetic_bundle.seal())


def test_different_same_number_source_record_fails_repeated_resolution(
    synthetic_bundle: SyntheticBundle,
) -> None:
    forged_trace = deepcopy(synthetic_bundle.original_result.trace)
    second = synthetic_bundle.source_questions[1]
    forged_trace["source"].update(
        {
            "matched_page_number": second["content_page_number"],
            "runner_up_page_number": 1,
            "record_id": second["record_id"],
            "question_number": second["question_number"],
            "question_marker_kind": second.get(
                "question_marker_kind", "numbered_item"
            ),
            "answer_format": second["answer_format"],
            "key_binding_kind": second["key_binding_kind"],
            "key_page_number": second["key_page_number"],
            "key_context_page_number": second.get(
                "key_context_page_number", second["key_page_number"]
            ),
            "key_bbox": second["key_bbox"],
            "content_bbox": None,
            "key_projection_sha256": None,
            "content_projection_sha256": None,
        }
    )
    forged_result = SimpleNamespace(
        problem=synthetic_bundle.original_result.problem,
        checks=synthetic_bundle.original_result.checks,
        trace=forged_trace,
    )
    forged_certificate = _certificate_record(CHANGED_ID, forged_result, "C")
    fingerprint = str(forged_certificate["trace_fingerprint"])
    synthetic_bundle.certificates[CHANGED_ID] = forged_certificate
    synthetic_bundle.candidates[CHANGED_ID]["final_answer"] = "C"
    synthetic_bundle.solver[CHANGED_ID]["final_answer"] = "C"
    synthetic_bundle.solver[CHANGED_ID]["generation"]["official_source_override"][
        "trace_fingerprint"
    ] = fingerprint
    synthetic_bundle.decisions[CHANGED_ID]["selected_answer"] = "C"
    synthetic_bundle.decisions[CHANGED_ID][
        "certificate_trace_fingerprint"
    ] = fingerprint

    with pytest.raises(image_judge.JudgeBuildError, match="repeated OCR-to-source"):
        image_judge.build(**synthetic_bundle.seal())


def test_composer_cross_checks_frozen_source_context_and_observed_marker(
    synthetic_bundle: SyntheticBundle,
) -> None:
    index = parse_workbook_index(synthetic_bundle.source_index)
    source_records = {
        question.record_id: (document, question)
        for document in index.documents
        for question in document.questions
    }
    observation = parser_observation_allow_missing_number(_parser_row(CHANGED_ID))
    trace = deepcopy(synthetic_bundle.original_result.trace)
    composer._validate_workbook_trace_against_source(
        task_id=CHANGED_ID,
        candidate_answer="B",
        observation=observation,
        trace=trace,
        source_records=source_records,
        allow_example_label_marker=False,
    )

    wrong_context = deepcopy(trace)
    wrong_context["source"]["key_context_page_number"] = 2
    with pytest.raises(composer.CompositionError, match="frozen index"):
        composer._validate_workbook_trace_against_source(
            task_id=CHANGED_ID,
            candidate_answer="B",
            observation=observation,
            trace=wrong_context,
            source_records=source_records,
            allow_example_label_marker=False,
        )

    conflict = replace(observation, primary_example_label_number=24)
    with pytest.raises(composer.CompositionError, match="observed source marker"):
        composer._validate_workbook_trace_against_source(
            task_id=CHANGED_ID,
            candidate_answer="B",
            observation=conflict,
            trace=trace,
            source_records=source_records,
            allow_example_label_marker=True,
        )


def test_incomplete_decision_artifact_is_rejected(
    synthetic_bundle: SyntheticBundle,
) -> None:
    synthetic_bundle.decisions.pop(TASK_IDS[-1])

    with pytest.raises(image_judge.JudgeBuildError, match="task sets do not align"):
        image_judge.build(**synthetic_bundle.seal())
