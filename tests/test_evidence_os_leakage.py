from __future__ import annotations

import ast
import json
from pathlib import Path
import re
from uuid import uuid4

import pytest

from evidence_os.ingest import (
    ForbiddenEvidenceError,
    LineageRejectedError,
    load_candidate_jsonl,
    validate_lineage,
)


def _write_row(path: Path, **updates: object) -> None:
    row: dict[str, object] = {
        "task_id": "opaque-alignment-key",
        "question": "Observable question",
        "final_answer": "B",
        "generation": {"call_count": 1, "gold_access": False},
    }
    row.update(updates)
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")


def _safe_source(tmp_path: Path) -> Path:
    # The lineage guard intentionally scans the complete path.  Pytest derives
    # ``tmp_path`` from test names, some of which mention denied fields, so put
    # the actual artifact in the neutral per-session parent directory.
    return tmp_path.parent / f"candidate_{uuid4().hex}.jsonl"


def test_exact_generation_gold_access_false_is_accepted_and_projected_away(
    tmp_path: Path,
) -> None:
    source = _safe_source(tmp_path)
    _write_row(source)

    run = load_candidate_jsonl(source, name="solver")
    payload = run._payload_by_task_id["opaque-alignment-key"]

    assert "gold_access" not in payload["generation"]
    assert payload["generation"]["call_count"] == 1


@pytest.mark.parametrize(
    "generation",
    [
        {"gold_access": True},
        {"gold_access": None},
        {"gold_access": 0},
        {"trace": {"gold_access": False}},
        [{"gold_access": False}],
    ],
)
def test_gold_access_attestation_is_allowed_only_at_exact_path_and_value(
    tmp_path: Path,
    generation: object,
) -> None:
    source = _safe_source(tmp_path)
    _write_row(source, generation=generation)

    with pytest.raises(ForbiddenEvidenceError, match="gold_access"):
        load_candidate_jsonl(source, name="solver")


def test_top_level_gold_access_false_is_not_a_general_escape_hatch(tmp_path: Path) -> None:
    source = _safe_source(tmp_path)
    _write_row(source, gold_access=False)

    with pytest.raises(ForbiddenEvidenceError, match="gold_access"):
        load_candidate_jsonl(source, name="solver")


@pytest.mark.parametrize(
    "nested_payload",
    [
        {"generation": {"trace": {"gold": "B"}}},
        {"generation": {"events": [{"judge_score": 1.0}]}},
        {"tool_calls": [{"result": {"referenceAnswer": "B"}}]},
        {"usage": {"details": {"isCorrect": True}}},
        {"raw_response": {"analysis": {"oracle_outcome": "pass"}}},
    ],
)
def test_forbidden_evaluation_keys_are_rejected_recursively(
    tmp_path: Path,
    nested_payload: dict[str, object],
) -> None:
    source = _safe_source(tmp_path)
    _write_row(source, **nested_payload)

    with pytest.raises(ForbiddenEvidenceError):
        load_candidate_jsonl(source, name="solver")


def test_nested_task_id_is_rejected_even_though_root_id_is_for_alignment(
    tmp_path: Path,
) -> None:
    source = _safe_source(tmp_path)
    _write_row(source, generation={"trace": {"task_id": "leak"}})

    with pytest.raises(ForbiddenEvidenceError, match="nested task_id"):
        load_candidate_jsonl(source, name="solver")


@pytest.mark.parametrize(
    "declared_lineage",
    [
        "frozen judge outputs",
        "posthoc-oracle-selection",
        "evaluation/scores/v3",
        "human_adjudications",
        "gold labels export",
    ],
)
def test_declared_lineage_denylist_rejects_evaluation_artifacts(
    declared_lineage: str,
) -> None:
    with pytest.raises(LineageRejectedError):
        validate_lineage("solver-only/candidate.jsonl", declared_lineage=declared_lineage)


def test_path_lineage_denylist_runs_before_file_loading(tmp_path: Path) -> None:
    unsafe = tmp_path / "judge" / "candidate.jsonl"

    with pytest.raises(LineageRejectedError, match="judge"):
        load_candidate_jsonl(unsafe, name="candidate")


def test_solver_only_lineage_is_accepted() -> None:
    validate_lineage(
        "reports/solver-only/candidate.jsonl",
        declared_lineage="frozen solver generation v1",
    )


def test_production_policy_has_no_evaluator_imports_or_task_maps() -> None:
    policy_path = Path(__file__).resolve().parents[1] / "src" / "evidence_os" / "policy.py"
    source = policy_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported: list[str] = []
    string_literals: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            string_literals.append(node.value)

    assert not any(name.startswith("vlm_judge") for name in imported)
    assert not any("score" in name or "judge" in name for name in imported)
    joined = "\n".join(string_literals)
    assert "strict_correct" not in joined
    assert re.search(r"\bval_\d+\b", joined) is None
    assert "task_id" not in source
