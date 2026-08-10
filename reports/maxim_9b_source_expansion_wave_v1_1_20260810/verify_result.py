from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any


REPORT_DIR = Path(__file__).resolve().parent
REPO = REPORT_DIR.parents[1]
RESULT_PATH = REPORT_DIR / "RESULT.json"

ARM_ORDER = (
    "base240",
    "math5_v11",
    "english5",
    "meb7_6",
    "official16",
    "bs11_8_research",
    "research_bs24",
    "fenomen12_research",
    "research_fenomen28",
    "research_all36",
)
RESEARCH_ARMS = (
    "bs11_8_research",
    "research_bs24",
    "fenomen12_research",
    "research_fenomen28",
    "research_all36",
)
FIXED_IDS = (
    "val_0048",
    "val_0050",
    "val_0051",
    "val_0054",
    "val_0055",
    "val_0056",
    "val_0057",
    "val_0058",
    "val_0182",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AssertionError(f"expected JSON object: {path}")
    return value


def repo_path(value: str) -> Path:
    path = (REPO / value).resolve()
    path.relative_to(REPO)
    if not path.is_file():
        raise AssertionError(f"missing artifact: {path}")
    return path


def assert_close(actual: float, expected: float) -> None:
    if not math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=5e-7):
        raise AssertionError(f"numeric mismatch: {actual!r} != {expected!r}")


def main() -> None:
    result = read_json(RESULT_PATH)
    assert result["schema_version"] == "maxim-9b-source-expansion-wave-v1.1-release-result-v1"
    assert result["status"] == "audited_final"
    assert result["local_artifact_verification"] == "pass"
    assert result["independent_postscore_audit"] == {
        "status": "pass",
        "reported_via_parent_coordination": True,
        "separate_evidence_artifact_pinned": False,
    }
    assert result["model_lineage"] == {
        "answer_producing_model": "Qwen/Qwen3.5-9B",
        "all_9b": True,
        "qwen_27b_answers_present": False,
    }

    wave_artifacts = result["wave_artifacts"]
    for descriptor in wave_artifacts.values():
        path = repo_path(descriptor["path"])
        assert sha256(path) == descriptor["sha256"], path

    freeze = read_json(repo_path(wave_artifacts["freeze"]["path"]))
    amendment = read_json(repo_path(wave_artifacts["independent_audit_amendment"]["path"]))
    attempt = read_json(repo_path(wave_artifacts["attempt_marker"]["path"]))
    completion_path = repo_path(wave_artifacts["completion"]["path"])
    completion = read_json(completion_path)
    completion_sidecar = completion_path.with_name("WAVE_COMPLETION_SHA256.txt")
    assert completion_sidecar.read_text(encoding="ascii") == (
        f"{sha256(completion_path)}  WAVE_COMPLETION.json\n"
    )

    assert freeze["official_headline_arm"] == "official16"
    assert freeze["arm_ids"] == list(ARM_ORDER)
    assert freeze["research_evaluation_only_arm_ids"] == list(RESEARCH_ARMS)
    assert amendment["audit_status"] == "pass"
    assert amendment["evaluation_authorized"] is True
    assert amendment["wave_freeze_sha256"] == wave_artifacts["freeze"]["sha256"]
    assert attempt["status"] == "attempt_started_before_any_scorer"
    assert attempt["arm_ids"] == list(ARM_ORDER)
    assert attempt["independent_audit_amendment_sha256"] == wave_artifacts["independent_audit_amendment"]["sha256"]
    assert completion["status"] == "all_ten_completed_outputs_hash_frozen"
    assert completion["arm_ids"] == list(ARM_ORDER)
    assert completion["all_returncodes_zero"] is True
    assert completion["all_ten_completed_before_manifest"] is True
    assert completion["all_ten_started_via_one_shared_barrier"] is True
    assert completion["individual_output_content_never_parsed_or_printed_by_launcher"] is True
    assert completion["research_outputs_separate"] is True
    assert completion["official_headline_arm"] == "official16"
    assert completion["research_evaluation_only_arm_ids"] == list(RESEARCH_ARMS)
    assert all(item["returncode"] == 0 for item in completion["process_metadata"])

    metrics: dict[str, dict[str, Any]] = {}
    for arm_id in ARM_ORDER:
        descriptor = result["arms"][arm_id]
        path = repo_path(descriptor["metrics_json_path"])
        assert sha256(path) == descriptor["metrics_json_sha256"]
        completion_descriptor = completion["output_artifacts"][arm_id]["json"]
        assert sha256(path) == completion_descriptor["sha256"]
        assert path.stat().st_size == completion_descriptor["size_bytes"]
        metric = read_json(path)
        metrics[arm_id] = metric
        overall = metric["overall"]
        assert overall["new_correct"] == descriptor["correct"]
        assert overall["n"] == descriptor["total"] == 274
        assert_close(overall["new_accuracy"], descriptor["accuracy"])
        assert overall["solver_errors"] == 0
        assert overall["missing_answers"] == 0

    base = {row["task_id"]: row for row in metrics["base240"]["task_outcomes"]}
    official = {row["task_id"]: row for row in metrics["official16"]["task_outcomes"]}
    assert set(base) == set(official) and len(base) == 274
    fixed = tuple(
        task_id
        for task_id in sorted(base)
        if not base[task_id]["new_correct"] and official[task_id]["new_correct"]
    )
    regressed = tuple(
        task_id
        for task_id in sorted(base)
        if base[task_id]["new_correct"] and not official[task_id]["new_correct"]
    )
    assert fixed == FIXED_IDS
    assert regressed == ()
    assert tuple(result["official"]["fixed_task_ids_vs_base"]) == FIXED_IDS
    assert result["official"]["regressed_task_ids_vs_base"] == []
    assert result["official"]["new_correct"] - result["official"]["base_correct"] == 9
    assert metrics["official16"]["by_subject"]["Math"]["new_correct"] == 117
    assert metrics["official16"]["by_source"]["deterministic"]["new_correct"] == 158
    assert metrics["official16"]["by_source"]["image_judge"]["new_correct"] == 91

    for arm_id in RESEARCH_ARMS:
        arm_result = result["arms"][arm_id]
        arm_freeze = freeze["arms"][arm_id]
        arm_manifest = read_json(repo_path("experiments/maxim_9b_source_expansion_wave_v1_1/final_wave/" + arm_freeze["manifest"]["path"]))
        assert arm_result["classification"] == "research_evaluation_only"
        assert arm_result["production_eligible"] is False
        assert arm_result["license_verified"] is False
        assert arm_freeze["classification"] == "research_evaluation_only"
        assert arm_freeze["eligible_for_official_headline"] is False
        assert arm_manifest["classification"] == "research_evaluation_only"
        assert arm_manifest["production_eligible"] is False
        assert arm_manifest["license_verified"] is False
        assert arm_manifest["source_asset_redistribution_allowed"] is False
    assert result["arms"]["research_all36"]["correct"] == 251
    assert result["research_governance"]["copied_pdf_or_crop_assets"] == []

    bge = result["external_retrieval_evidence"]
    assert bge["separate_from_project_benchmark_score"] is True
    assert bge["project_score_attribution"] is False
    assert sha256(repo_path(bge["dataset"]["manifest_path"])) == bge["dataset"]["manifest_sha256"]
    assert sha256(repo_path(bge["preregistration"]["path"])) == bge["preregistration"]["sha256"]
    for condition in bge["conditions"].values():
        path = repo_path(condition["result_path"])
        assert sha256(path) == condition["result_sha256"]
        payload = read_json(path)
        assert payload["project_benchmark_access"] is False
        assert payload["project_benchmark_score_computed"] is False
        assert_close(payload["metrics"]["Recall@1"], condition["recall_at_1"])
        assert_close(payload["metrics"]["Recall@5"], condition["recall_at_5"])
        assert_close(payload["metrics"]["MRR"], condition["mrr"])
    bge_payload = read_json(repo_path(bge["conditions"]["bge_m3"]["result_path"]))
    assert bge_payload["condition"]["model_id"] == "BAAI/bge-m3"
    assert bge_payload["condition"]["revision"] == "5617a9f61b028005a4858fdac845db406aefb181"
    assert bge_payload["condition"]["license"] == "mit"
    assert bge_payload["condition"]["embedding_dimension"] == 1024

    forbidden_suffixes = {".pdf", ".png", ".jpg", ".jpeg", ".webp"}
    assert not [path for path in REPORT_DIR.rglob("*") if path.suffix.lower() in forbidden_suffixes]
    print("PASS: source wave release artifacts, +9/0 delta, research governance, and separate BGE evidence verified")


if __name__ == "__main__":
    main()
