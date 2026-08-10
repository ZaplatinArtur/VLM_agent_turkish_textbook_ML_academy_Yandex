from __future__ import annotations

from argparse import Namespace
import hashlib
import json
from pathlib import Path

import pytest

from evidence_os.adapters import problem_from_public_payload
from evidence_os.certificates import answer_fingerprint, input_fingerprint
from evidence_os.contracts import CandidateEnvelope
from scripts.compose_maxim_evidence_os_v1 import CompositionError, compose
from scripts.project_maxim_evidence_os_inputs_v1 import ProjectionError, project


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _args(
    *,
    anchor_raw: Path,
    anchor_public: Path,
    public_tasks: Path,
    output_dir: Path,
    candidate_public: Path | None = None,
    candidate_raw: Path | None = None,
    certificate: Path | None = None,
    profile_json: Path | None = None,
) -> Namespace:
    candidates = [f"challenger={candidate_public}"] if candidate_public else []
    raw_candidates = [f"challenger={candidate_raw}"] if candidate_raw else []
    certificates = [f"challenger={certificate}"] if certificate else []
    return Namespace(
        anchor_solver=anchor_raw,
        anchor_public=anchor_public,
        profile_json=profile_json,
        anchor_sha256=hashlib.sha256(anchor_raw.read_bytes()).hexdigest(),
        public_tasks=public_tasks,
        candidate=candidates,
        raw_candidate=raw_candidates,
        certificate=certificates,
        image_root=None,
        output_dir=output_dir,
        expected_rows=1,
        profile_name="synthetic-evidence-os-v1",
        min_claim_coverage=1.0,
        min_deterministic_checks=1,
        min_independent_verifiers=1,
    )


def _stage(source: Path, destination: Path) -> None:
    project(source, destination, destination.with_suffix(".manifest.json"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_profile(
    path: Path,
    *,
    anchor_raw: Path,
    anchor_public: Path,
    public_tasks: Path,
    candidate_public: Path,
    candidate_raw: Path,
    certificate: Path | None = None,
) -> None:
    verifier = "synthetic-independent-calculator-v1"
    profile = {
        "schema_version": "maxim-evidence-os-frozen-profile-v1",
        "profile_name": "synthetic-evidence-os-v1",
        "expected_rows": 1,
        "anchor": {
            "sha256": _sha256(anchor_raw),
            "public_projection_sha256": _sha256(anchor_public),
        },
        "public_tasks": {"sha256": _sha256(public_tasks)},
        "policy": {
            "min_claim_coverage": 1.0,
            "min_deterministic_checks": 1,
            "min_independent_verifiers": 1,
            "task_id_source_url_sha_or_order_features_allowed": False,
            "external_certificates_require_profile_binding": True,
            "certificate_trace_content_required": True,
        },
        "allowed_strong_certificate_kinds": ["executable_check"],
        "legacy_modules": {
            "challenger": {
                "sha256": _sha256(candidate_raw),
                "public_projection_sha256": _sha256(candidate_public),
                "mode": "evidence_gated" if certificate else "shadow",
            }
        },
        "certificate_inputs": (
            {
                "challenger": {
                    "sha256": _sha256(certificate),
                    "allowed_verifiers": [verifier],
                    "allowed_kinds": ["executable_check"],
                    "require_inline_trace": True,
                }
            }
            if certificate
            else {}
        ),
    }
    path.write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")


def test_projection_strips_legacy_negative_attestations(tmp_path: Path) -> None:
    raw = tmp_path / "legacy.jsonl"
    public = tmp_path / "public.jsonl"
    _write_jsonl(
        raw,
        [
            {
                "task_id": "opaque-1",
                "final_answer": "C",
                "generation": {"gold_access": False},
                "offline_provenance": {"gold_access": False, "private_note": "ignored"},
                "reasoning": "also ignored",
            }
        ],
    )

    _stage(raw, public)
    row = json.loads(public.read_text(encoding="utf-8"))

    assert row["task_id"] == "opaque-1"
    assert row["final_answer"] == "C"
    assert row["generation"] == {"gold_access": False}
    assert "offline_provenance" not in row
    assert "reasoning" not in row


@pytest.mark.parametrize(
    "leak",
    [
        {"reference_answer": "B"},
        {"generation": {"gold_access": True}},
        {"trace": {"judge_score": 1.0}},
    ],
)
def test_projection_rejects_evaluation_fields(tmp_path: Path, leak: dict[str, object]) -> None:
    raw = tmp_path / "legacy.jsonl"
    row: dict[str, object] = {"task_id": "opaque-1", "final_answer": "C"}
    row.update(leak)
    _write_jsonl(raw, [row])

    with pytest.raises(ProjectionError):
        _stage(raw, tmp_path / "public.jsonl")


def test_composer_keeps_exact_anchor_without_strong_certificate(tmp_path: Path) -> None:
    anchor_raw = tmp_path / "anchor.raw.jsonl"
    anchor_public = tmp_path / "anchor.public.jsonl"
    challenger_raw = tmp_path / "challenger.raw.jsonl"
    challenger_public = tmp_path / "challenger.public.jsonl"
    public_tasks = tmp_path / "tasks.public.jsonl"
    anchor_bytes = (
        b'{"task_id":"opaque-1", "final_answer":"C", '
        b'"generation":{"gold_access":false}, '
        b'"offline_provenance":{"gold_access":false}}\n'
    )
    anchor_raw.write_bytes(anchor_bytes)
    _write_jsonl(
        challenger_raw,
        [{"task_id": "opaque-1", "final_answer": "B", "generation": {"gold_access": False}}],
    )
    _write_jsonl(
        public_tasks,
        [
            {
                "task_id": "opaque-1",
                "question": "120 sayisinin yuzde 15'i kactir?",
                "question_images": [],
                "answer_type": "choice",
                "subject": "Math",
            }
        ],
    )
    _stage(anchor_raw, anchor_public)
    _stage(challenger_raw, challenger_public)
    profile_json = tmp_path / "profile.json"
    _write_profile(
        profile_json,
        anchor_raw=anchor_raw,
        anchor_public=anchor_public,
        public_tasks=public_tasks,
        candidate_public=challenger_public,
        candidate_raw=challenger_raw,
    )

    manifest = compose(
        _args(
            anchor_raw=anchor_raw,
            anchor_public=anchor_public,
            public_tasks=public_tasks,
            output_dir=tmp_path / "output",
            candidate_public=challenger_public,
            profile_json=profile_json,
        )
    )

    assert manifest["overrides"] == 0
    assert manifest["anchor_exact_copy"] is True
    assert (tmp_path / "output" / "solver.jsonl").read_bytes() == anchor_bytes


def test_composer_accepts_only_exactly_bound_strong_certificate(tmp_path: Path) -> None:
    anchor_raw = tmp_path / "anchor.raw.jsonl"
    anchor_public = tmp_path / "anchor.public.jsonl"
    challenger_raw = tmp_path / "challenger.raw.jsonl"
    challenger_public = tmp_path / "challenger.public.jsonl"
    public_tasks = tmp_path / "tasks.public.jsonl"
    certificate_path = tmp_path / "certificate.public.jsonl"
    _write_jsonl(
        anchor_raw,
        [{"task_id": "opaque-1", "final_answer": "C", "generation": {"gold_access": False}}],
    )
    _write_jsonl(
        challenger_raw,
        [{"task_id": "opaque-1", "final_answer": "B", "generation": {"gold_access": False}}],
    )
    task = {
        "task_id": "opaque-1",
        "question": "120 sayisinin yuzde 15'i kactir?",
        "question_images": [],
        "answer_type": "choice",
        "subject": "Math",
    }
    _write_jsonl(public_tasks, [task])
    _stage(anchor_raw, anchor_public)
    _stage(challenger_raw, challenger_public)
    problem, _ = problem_from_public_payload(task)
    challenger = CandidateEnvelope(source="challenger", final_answer="B")
    trace = "120 * 0.15 = 18 -> B"
    _write_jsonl(
        certificate_path,
        [
            {
                "task_id": "opaque-1",
                "kind": "executable_check",
                "strength": "strong",
                "status": "pass",
                "input_fingerprint": input_fingerprint(problem),
                "answer_fingerprint": answer_fingerprint(challenger),
                "input_bound": True,
                "answer_bound": True,
                "claim_coverage": 1.0,
                "contradiction_count": 0,
                "deterministic_checks": [True, True],
                "verifier": "synthetic-independent-calculator-v1",
                "trace": trace,
                "trace_fingerprint": hashlib.sha256(trace.encode("utf-8")).hexdigest(),
            }
        ],
    )

    with pytest.raises(CompositionError, match="profile-json"):
        compose(
            _args(
                anchor_raw=anchor_raw,
                anchor_public=anchor_public,
                public_tasks=public_tasks,
                output_dir=tmp_path / "unprofiled-output",
                candidate_public=challenger_public,
                candidate_raw=challenger_raw,
                certificate=certificate_path,
            )
        )

    profile_json = tmp_path / "profile.json"
    _write_profile(
        profile_json,
        anchor_raw=anchor_raw,
        anchor_public=anchor_public,
        public_tasks=public_tasks,
        candidate_public=challenger_public,
        candidate_raw=challenger_raw,
        certificate=certificate_path,
    )

    manifest = compose(
        _args(
            anchor_raw=anchor_raw,
            anchor_public=anchor_public,
            public_tasks=public_tasks,
            output_dir=tmp_path / "output",
            candidate_public=challenger_public,
            candidate_raw=challenger_raw,
            certificate=certificate_path,
            profile_json=profile_json,
        )
    )

    assert manifest["overrides"] == 1
    selected = json.loads((tmp_path / "output" / "solver.jsonl").read_text(encoding="utf-8"))
    assert selected["final_answer"] == "B"

    certificate_row = json.loads(certificate_path.read_text(encoding="utf-8"))
    certificate_row["trace"] = "tampered trace"
    _write_jsonl(certificate_path, [certificate_row])
    _write_profile(
        profile_json,
        anchor_raw=anchor_raw,
        anchor_public=anchor_public,
        public_tasks=public_tasks,
        candidate_public=challenger_public,
        candidate_raw=challenger_raw,
        certificate=certificate_path,
    )
    with pytest.raises(ValueError, match="trace fingerprint"):
        compose(
            _args(
                anchor_raw=anchor_raw,
                anchor_public=anchor_public,
                public_tasks=public_tasks,
                output_dir=tmp_path / "tampered-trace-output",
                candidate_public=challenger_public,
                candidate_raw=challenger_raw,
                certificate=certificate_path,
                profile_json=profile_json,
            )
        )
    certificate_row["trace"] = trace
    _write_jsonl(certificate_path, [certificate_row])

    certificate_row["verifier"] = "self-declared-untrusted-verifier"
    _write_jsonl(certificate_path, [certificate_row])
    _write_profile(
        profile_json,
        anchor_raw=anchor_raw,
        anchor_public=anchor_public,
        public_tasks=public_tasks,
        candidate_public=challenger_public,
        candidate_raw=challenger_raw,
        certificate=certificate_path,
    )
    with pytest.raises(ValueError, match="not profile-authorized"):
        compose(
            _args(
                anchor_raw=anchor_raw,
                anchor_public=anchor_public,
                public_tasks=public_tasks,
                output_dir=tmp_path / "untrusted-verifier-output",
                candidate_public=challenger_public,
                candidate_raw=challenger_raw,
                certificate=certificate_path,
                profile_json=profile_json,
            )
        )
    certificate_row["verifier"] = "synthetic-independent-calculator-v1"
    _write_jsonl(certificate_path, [certificate_row])

    # Even a profile-pinned raw artifact cannot carry an answer different from
    # the public candidate whose exact answer fingerprint was certified.
    _write_jsonl(
        challenger_raw,
        [{"task_id": "opaque-1", "final_answer": "D", "generation": {"gold_access": False}}],
    )
    _write_profile(
        profile_json,
        anchor_raw=anchor_raw,
        anchor_public=anchor_public,
        public_tasks=public_tasks,
        candidate_public=challenger_public,
        candidate_raw=challenger_raw,
        certificate=certificate_path,
    )
    with pytest.raises(CompositionError, match="raw/public output mismatch"):
        compose(
            _args(
                anchor_raw=anchor_raw,
                anchor_public=anchor_public,
                public_tasks=public_tasks,
                output_dir=tmp_path / "stale-raw-output",
                candidate_public=challenger_public,
                candidate_raw=challenger_raw,
                certificate=certificate_path,
                profile_json=profile_json,
            )
        )

    _write_jsonl(
        challenger_raw,
        [{"task_id": "opaque-1", "prediction": "B", "generation": {"gold_access": False}}],
    )
    _write_profile(
        profile_json,
        anchor_raw=anchor_raw,
        anchor_public=anchor_public,
        public_tasks=public_tasks,
        candidate_public=challenger_public,
        candidate_raw=challenger_raw,
        certificate=certificate_path,
    )
    with pytest.raises(CompositionError, match="scorer-visible final_answer is missing"):
        compose(
            _args(
                anchor_raw=anchor_raw,
                anchor_public=anchor_public,
                public_tasks=public_tasks,
                output_dir=tmp_path / "prediction-only-output",
                candidate_public=challenger_public,
                candidate_raw=challenger_raw,
                certificate=certificate_path,
                profile_json=profile_json,
            )
        )
