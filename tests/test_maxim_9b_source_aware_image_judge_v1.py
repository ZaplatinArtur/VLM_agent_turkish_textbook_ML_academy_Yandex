from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.build_maxim_9b_source_aware_image_judge_v1 import (
    BuildError,
    _index,
    _is_source_adjudicated_judge_row,
    _source_rows,
    _stage_answer_action,
)


def _write_certificate(path: Path, *, task_id: str = "val_0001") -> str:
    row = {
        "task_id": task_id,
        "kind": "source_entailment",
        "strength": "strong",
        "status": "pass",
        "trace_fingerprint": "a" * 64,
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8", newline="\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_source_rows_accepts_only_composition_selected_certificate(tmp_path: Path) -> None:
    certificates = tmp_path / "certificates.jsonl"
    digest = _write_certificate(certificates)
    resolver = {
        "artifacts": {
            "certificates": {"path": str(certificates), "sha256": digest}
        }
    }
    decisions = _index(
        [
            (
                b"{}",
                {
                    "task_id": "val_0001",
                    "action": "replace_anchor",
                    "reason": "strongly_verified_challenger",
                    "certificate_trace_fingerprint": "a" * 64,
                },
            )
        ],
        "decisions",
    )
    selected = _source_rows(
        profile={"schema_version": "maxim-public-workbook-profile-v1"},
        resolver=resolver,
        decisions=decisions,
    )
    assert set(selected) == {"val_0001"}


def test_source_rows_rejects_certificate_not_selected_by_composition(tmp_path: Path) -> None:
    certificates = tmp_path / "certificates.jsonl"
    digest = _write_certificate(certificates)
    resolver = {
        "artifacts": {
            "certificates": {"path": str(certificates), "sha256": digest}
        }
    }
    decisions = _index(
        [
            (
                b"{}",
                {
                    "task_id": "val_0001",
                    "action": "keep_anchor",
                    "reason": "no_challengers",
                    "certificate_trace_fingerprint": None,
                },
            )
        ],
        "decisions",
    )
    with pytest.raises(BuildError, match="not composition-selected"):
        _source_rows(
            profile={"schema_version": "maxim-public-workbook-profile-v1"},
            resolver=resolver,
            decisions=decisions,
        )


def test_index_rejects_duplicate_task_ids() -> None:
    rows = [
        (b"{}", {"task_id": "val_0001"}),
        (b"{}", {"task_id": "val_0001"}),
    ]
    with pytest.raises(BuildError, match="duplicate"):
        _index(rows, "rows")


def test_source_adjudicated_lineage_requires_deterministic_backend_and_verdict_origin() -> None:
    row = {
        "judge": {
            "backend": "deterministic-official-source-certificate",
            "model": None,
        },
        "metadata": {
            "verdict_origin": "deterministic_official_source_adjudication",
            "stage_answer_action": "keep_immediate_base_confirmed_by_source",
        },
    }
    assert _is_source_adjudicated_judge_row(row)
    row["judge"]["model"] = "Qwen/Qwen3.5-9B"
    assert not _is_source_adjudicated_judge_row(row)


@pytest.mark.parametrize(
    ("profile", "decision", "expected"),
    [
        (
            {"schema_version": "maxim-public-workbook-profile-v1"},
            {"action": "keep_anchor", "reason": "equivalent_to_anchor"},
            "keep_immediate_base_confirmed_by_source",
        ),
        (
            {"schema_version": "maxim-public-workbook-profile-v1"},
            {"action": "replace_anchor", "reason": "strongly_verified_challenger"},
            "replace_immediate_base_with_source",
        ),
        (
            {"schema_version": "maxim-fill-blank-page-activity-profile-v1"},
            {"source_override": True},
            "replace_immediate_base_with_source",
        ),
    ],
)
def test_stage_answer_action_is_explicit_and_exact(
    profile: dict, decision: dict, expected: str
) -> None:
    assert _stage_answer_action(profile, decision) == expected
