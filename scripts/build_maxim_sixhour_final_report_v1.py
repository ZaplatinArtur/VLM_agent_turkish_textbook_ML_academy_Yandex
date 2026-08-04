"""Build the SHA-pinned final package for Maksim's 2026-08-04 six-hour run.

This utility is intentionally fail closed.  It accepts no score paths from a
configuration file, verifies every consumed artifact against a hard-coded
SHA-256 pin, checks the common scorer/benchmark lineage, and keeps the standard
frozen-benchmark score separate from the post-hoc public-evidence diagnostic.

The diagnostic 273/274 (or 273/273 after excluding one malformed prompt) is
never represented as a benchmark score.  The report's v5 standard exploratory
score remains 263/274.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "maxim-sixhour-final-report-v1"
EXPECTED_SCORE_SCHEMA = "maxim-full274-score-v1"
EXPECTED_BENCHMARK_SHA256 = (
    "5a6a38ccae7835f0d015f6e5979834208347b8e6e7a8d6884e4af97605f51ed9"
)
EXPECTED_SCORER_SHA256 = (
    "bca10e6546b68f8a66eb4d68aa13316429e4689789be1bef14cd592955e4eacf"
)
EXPECTED_BASELINE_CORRECT = 141
EXPECTED_N = 274
EXPECTED_MATH_N = 139
FINAL_DIR = Path(
    "reports/maxim_sixhour_push_v4_20260804/"
    "final_exploratory_public_evidence_v5"
)


class FinalReportError(ValueError):
    """Raised when a pinned artifact or reporting invariant is violated."""


@dataclass(frozen=True)
class InputPin:
    name: str
    path: Path
    sha256: str
    kind: str


INPUT_PINS: tuple[InputPin, ...] = (
    InputPin(
        "strict_score",
        Path(
            "reports/maxim_blind_ensemble_v2default_v3_repair_v1_20260804/"
            "evaluation_cached_reuse_v1/score.json"
        ),
        "0550b1b59b6f9680ad8e7462c1b30fc6c97960bd16fa82576a3f917192fbb24a",
        "json",
    ),
    InputPin(
        "exploratory_start_score",
        Path(
            "reports/maxim_staged_exact_web_extensions_tools_v2_20260804/"
            "exploratory_all_certificates_v3/evaluation/score.json"
        ),
        "25e488d358d7370f48e888f4047f9aaf89d43d704c61e5cf0a513820477bcd35",
        "json",
    ),
    InputPin(
        "checkpoint_238_score",
        Path(
            "reports/maxim_sixhour_push_v4_20260804/checkpoint_085/"
            "evaluation/score.json"
        ),
        "3dea68ecc9ae39a53ffbe7c11288d5cffd38b6e81b1c127c27886314a9bdbea8",
        "json",
    ),
    InputPin(
        "checkpoint_244_score",
        Path(
            "reports/maxim_sixhour_push_v4_20260804/checkpoint_088/"
            "evaluation/score.json"
        ),
        "ca8464b7c36da3146b6d9e03bd1f37d59bbffe2b6f166f06247fdc51a09287f6",
        "json",
    ),
    InputPin(
        "checkpoint_253_score",
        Path(
            "reports/maxim_sixhour_push_v4_20260804/checkpoint_091_candidate/"
            "evaluation/score.json"
        ),
        "acf7f87b9a21c7155df122c4cbb0a460069b2d30c2f4b450cf87a8b95d096244",
        "json",
    ),
    InputPin(
        "checkpoint_259_score",
        Path(
            "reports/maxim_sixhour_push_v4_20260804/checkpoint_094_candidate/"
            "evaluation/score.json"
        ),
        "775c7568541d1de5ae2a1fa593804f082f5309444201c4223478d9600e2b6a5c",
        "json",
    ),
    InputPin(
        "checkpoint_260_score",
        Path(
            "reports/maxim_sixhour_push_v4_20260804/checkpoint_0949_candidate/"
            "evaluation/score.json"
        ),
        "ac0a515d16188f462574e8d634c4db2f2663c2c0526041338107ccbc80a47ab4",
        "json",
    ),
    InputPin(
        "final_standard_score",
        FINAL_DIR / "evaluation/score.json",
        "7e4861144e21456304ba1d1ff06811172637e379a3482badaaa0bcbb4d5c20f3",
        "json",
    ),
    InputPin(
        "composition_manifest",
        FINAL_DIR / "composition_manifest.json",
        "4ef5ce9c3d82df50ee08c86a07d489715775acdf802c0fa22d315fecb2689e2a",
        "json",
    ),
    InputPin(
        "judge_manifest",
        FINAL_DIR / "evaluation/judge_manifest.json",
        "bf7920fbae729ae154335001499793da5495212c6a8694ed5e7a84cd89220523",
        "json",
    ),
    InputPin(
        "public_evidence_audit",
        FINAL_DIR / "public_evidence_audit_v2/REPORT.json",
        "5514ddb24de509a0c9d8a92e8af89516fce405704aeb74fbe61cfd2c5f014b07",
        "json",
    ),
    InputPin(
        "public_evidence_audit_markdown",
        FINAL_DIR / "public_evidence_audit_v2/REPORT.md",
        "47736ccd8688b1439c5848a17035b295474f719d37586b14e38400845f6bdb48",
        "text",
    ),
    InputPin(
        "final_solver",
        FINAL_DIR / "run/solver.jsonl",
        "6544b16aee4c6d09067a5ec8fb405de9053f7b85d6c45392713ccbcc73f8875d",
        "jsonl",
    ),
    InputPin(
        "final_image_judge",
        FINAL_DIR / "evaluation/executable_and_official_image97_judge.jsonl",
        "fa9fecc8cb34a3433375b697271effe5f939b2674105562450da497f09b5ca86",
        "jsonl",
    ),
    InputPin(
        "standard_score_markdown",
        FINAL_DIR / "evaluation/score.md",
        "6e886d649b033aaa6baa118b128ea7867a771ec1f521ca3901311942d35d012a",
        "text",
    ),
    InputPin(
        "standard_score_sha_manifest",
        FINAL_DIR / "evaluation/score.sha256",
        "9c811175366a3ef44ee681d59ec767098eae7200b0f8ac36d6acd0af832a1777",
        "text",
    ),
)


EXPECTED_SCORE_METRICS: Mapping[str, tuple[int, int, int, int]] = {
    "strict_score": (205, 274, 108, 139),
    "exploratory_start_score": (228, 274, 120, 139),
    "checkpoint_238_score": (238, 274, 129, 139),
    "checkpoint_244_score": (244, 274, 129, 139),
    "checkpoint_253_score": (253, 274, 131, 139),
    "checkpoint_259_score": (259, 274, 131, 139),
    "checkpoint_260_score": (260, 274, 131, 139),
    "final_standard_score": (263, 274, 132, 139),
}

TIMELINE_NAMES: tuple[tuple[str, str], ...] = (
    ("exploratory_start_score", "Старт целевого exploratory-прогона"),
    ("checkpoint_238_score", "Checkpoint 0.85"),
    ("checkpoint_244_score", "Checkpoint 0.88"),
    ("checkpoint_253_score", "Checkpoint 0.91"),
    ("checkpoint_259_score", "Checkpoint 0.94"),
    ("checkpoint_260_score", "Checkpoint 0.949"),
    ("final_standard_score", "Checkpoint 0.960 / финальный замороженный v5"),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _resolve_input(repo_root: Path, relative: Path) -> Path:
    if relative.is_absolute():
        raise FinalReportError(f"input pin must be relative: {relative}")
    root = repo_root.resolve()
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise FinalReportError(f"input escapes repository root: {relative}") from exc
    return resolved


def _load_pinned_inputs(
    repo_root: Path,
    pins: Sequence[InputPin] = INPUT_PINS,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    documents: dict[str, Any] = {}
    provenance: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for pin in pins:
        if pin.name in seen_names:
            raise FinalReportError(f"duplicate input pin name: {pin.name}")
        seen_names.add(pin.name)
        if pin.kind not in {"json", "jsonl", "text"}:
            raise FinalReportError(f"unsupported pin kind {pin.kind!r} for {pin.name}")
        if len(pin.sha256) != 64 or any(c not in "0123456789abcdef" for c in pin.sha256):
            raise FinalReportError(f"invalid SHA-256 pin for {pin.name}")
        path = _resolve_input(repo_root, pin.path)
        if not path.is_file():
            raise FinalReportError(f"missing pinned input {pin.name}: {path}")
        actual_sha = sha256_file(path)
        if actual_sha != pin.sha256:
            raise FinalReportError(
                f"{pin.name} SHA-256 mismatch: expected {pin.sha256}, got {actual_sha}"
            )
        try:
            if pin.kind == "json":
                value = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(value, dict):
                    raise FinalReportError(f"{pin.name} must contain a JSON object")
                documents[pin.name] = value
            elif pin.kind == "text":
                documents[pin.name] = path.read_text(encoding="utf-8")
            else:
                documents[pin.name] = path
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise FinalReportError(f"cannot read {pin.name}: {exc}") from exc
        provenance.append(
            {
                "name": pin.name,
                "path": pin.path.as_posix(),
                "sha256": actual_sha,
                "bytes": path.stat().st_size,
                "kind": pin.kind,
            }
        )
    return documents, provenance


def _required_mapping(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise FinalReportError(f"{where} must be an object")
    return value


def _required_int(value: Any, where: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise FinalReportError(f"{where} must be an integer >= {minimum}")
    return value


def _accuracy(correct: int, denominator: int) -> float:
    return correct / denominator


def _check_reported_accuracy(
    value: Any, correct: int, denominator: int, where: str
) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FinalReportError(f"{where} must be numeric")
    if not math.isfinite(float(value)) or abs(float(value) - _accuracy(correct, denominator)) > 1e-6:
        raise FinalReportError(
            f"{where}={value!r} is inconsistent with {correct}/{denominator}"
        )


def _score_provenance_sha(score: Mapping[str, Any], key: str, where: str) -> str:
    provenance = _required_mapping(score.get("provenance"), f"{where}.provenance")
    source = _required_mapping(provenance.get(key), f"{where}.provenance.{key}")
    value = source.get("sha256")
    if not isinstance(value, str):
        raise FinalReportError(f"{where}.provenance.{key}.sha256 is missing")
    return value.lower()


def _validate_score(name: str, score: Mapping[str, Any]) -> dict[str, Any]:
    expected = EXPECTED_SCORE_METRICS[name]
    if score.get("schema_version") != EXPECTED_SCORE_SCHEMA:
        raise FinalReportError(f"{name} has an unexpected score schema")
    overall = _required_mapping(score.get("overall"), f"{name}.overall")
    subjects = _required_mapping(score.get("by_subject"), f"{name}.by_subject")
    math_row = _required_mapping(subjects.get("Math"), f"{name}.by_subject.Math")
    correct = _required_int(overall.get("new_correct"), f"{name}.overall.new_correct")
    denominator = _required_int(overall.get("n"), f"{name}.overall.n", minimum=1)
    math_correct = _required_int(
        math_row.get("new_correct"), f"{name}.by_subject.Math.new_correct"
    )
    math_n = _required_int(math_row.get("n"), f"{name}.by_subject.Math.n", minimum=1)
    if (correct, denominator, math_correct, math_n) != expected:
        raise FinalReportError(
            f"{name} metrics changed: expected {expected}, got "
            f"{(correct, denominator, math_correct, math_n)}"
        )
    _check_reported_accuracy(
        overall.get("new_accuracy"), correct, denominator, f"{name}.overall.new_accuracy"
    )
    _check_reported_accuracy(
        math_row.get("new_accuracy"), math_correct, math_n, f"{name}.Math.new_accuracy"
    )
    if overall.get("baseline_correct") != EXPECTED_BASELINE_CORRECT:
        raise FinalReportError(f"{name} does not use the frozen 141/274 comparator")
    if _score_provenance_sha(score, "benchmark", name) != EXPECTED_BENCHMARK_SHA256:
        raise FinalReportError(f"{name} benchmark lineage mismatch")
    if _score_provenance_sha(score, "scorer", name) != EXPECTED_SCORER_SHA256:
        raise FinalReportError(f"{name} scorer lineage mismatch")
    guardrails = _required_mapping(score.get("guardrails"), f"{name}.guardrails")
    required_guardrails = {
        "baseline_rows_verified": 274,
        "benchmark_rows_verified": 274,
        "duplicate_task_ids": 0,
        "explicit_nonfalse_generation_gold_access": 0,
        "forbidden_gold_fields_in_solver": 0,
        "frozen_sha_pins_checked": True,
        "image_judge_rows_supplied": 97,
        "solver_rows_verified": 274,
        "task_id_sets_match": True,
    }
    for key, expected_value in required_guardrails.items():
        if guardrails.get(key) != expected_value:
            raise FinalReportError(
                f"{name}.guardrails.{key} must be {expected_value!r}"
            )
    return {
        "label": score.get("label"),
        "created_at_utc": score.get("created_at_utc"),
        "correct": correct,
        "denominator": denominator,
        "accuracy": _accuracy(correct, denominator),
        "accuracy_reported": float(overall["new_accuracy"]),
        "math_correct": math_correct,
        "math_denominator": math_n,
        "math_accuracy": _accuracy(math_correct, math_n),
        "fixed_vs_page_rag": overall.get("fixed"),
        "regressed_vs_page_rag": overall.get("regressed"),
        "delta_correct_vs_page_rag": correct - EXPECTED_BASELINE_CORRECT,
    }


def _load_jsonl_rows(path: Path, where: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise FinalReportError(f"{where} line {line_number} is not an object")
                rows.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FinalReportError(f"cannot parse {where}: {exc}") from exc
    return rows


def _validate_task_ids(rows: Iterable[Mapping[str, Any]], where: str, expected_n: int) -> None:
    task_ids: list[str] = []
    for index, row in enumerate(rows, 1):
        task_id = row.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            raise FinalReportError(f"{where} row {index} has no task_id")
        task_ids.append(task_id)
    if len(task_ids) != expected_n:
        raise FinalReportError(f"{where} has {len(task_ids)} rows, expected {expected_n}")
    if len(set(task_ids)) != expected_n:
        raise FinalReportError(f"{where} contains duplicate task_ids")


def _validate_solver(rows: list[dict[str, Any]]) -> None:
    _validate_task_ids(rows, "final_solver", 274)
    forbidden_top_level = {"reference", "reference_answer", "gold", "gold_answer"}
    for row in rows:
        overlap = forbidden_top_level.intersection(row)
        if overlap:
            raise FinalReportError(
                f"final_solver {row['task_id']} contains forbidden fields {sorted(overlap)}"
            )
        generation = row.get("generation")
        if isinstance(generation, dict) and generation.get("gold_access") is not False:
            raise FinalReportError(
                f"final_solver {row['task_id']} has non-false generation.gold_access"
            )
        if row.get("error") not in (None, ""):
            raise FinalReportError(f"final_solver {row['task_id']} has an error")
        answer = row.get("final_answer")
        if not isinstance(answer, str) or not answer.strip():
            raise FinalReportError(f"final_solver {row['task_id']} has no answer")


def _validate_manifests(
    composition: Mapping[str, Any],
    judge: Mapping[str, Any],
    solver_sha: str,
    image_judge_sha: str,
) -> dict[str, Any]:
    if composition.get("schema_version") != "maxim-executable-proof-extensions-v5":
        raise FinalReportError("unexpected composition manifest schema")
    if composition.get("gold_access_during_composition") is not False:
        raise FinalReportError("composition manifest does not certify gold isolation")
    composition_output = _required_mapping(
        composition.get("output"), "composition_manifest.output"
    )
    if composition_output.get("rows") != 274 or composition_output.get("sha256") != solver_sha:
        raise FinalReportError("composition manifest does not bind the final solver")
    if composition.get("reporting_status") != (
        "exploratory_targeted_posthoc_not_independent_holdout"
    ):
        raise FinalReportError("composition reporting status lost the post-hoc label")

    if judge.get("schema_version") != "maxim-executable-image-judge-v4":
        raise FinalReportError("unexpected judge manifest schema")
    if judge.get("benchmark_or_reference_opened_by_builder") is not False:
        raise FinalReportError("judge builder opened benchmark/reference inputs")
    if judge.get("solver_frozen_before_adjudication") is not True:
        raise FinalReportError("judge manifest does not certify a frozen solver")
    frozen_solver = _required_mapping(judge.get("frozen_solver"), "judge.frozen_solver")
    if frozen_solver.get("rows") != 274 or frozen_solver.get("sha256") != solver_sha:
        raise FinalReportError("judge manifest does not bind the frozen solver")
    judge_output = _required_mapping(judge.get("output"), "judge.output")
    if judge_output.get("rows") != 97 or judge_output.get("sha256") != image_judge_sha:
        raise FinalReportError("judge manifest does not bind the 97-row judge artifact")
    return {
        "generation_gold_access": False,
        "solver_frozen_before_image_adjudication": True,
        "benchmark_or_reference_opened_by_judge_builder": False,
        "solver_rows": 274,
        "image_judge_rows": 97,
        "duplicate_task_ids": 0,
        "solver_errors": 0,
    }


def _validate_noteworthy_certificates(
    composition: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    raw_certificates = composition.get("certificates")
    if not isinstance(raw_certificates, list):
        raise FinalReportError("composition manifest has no certificate registry")
    certificates: dict[str, Mapping[str, Any]] = {}
    for index, raw in enumerate(raw_certificates):
        certificate = _required_mapping(raw, f"composition.certificates[{index}]")
        task_id = certificate.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            raise FinalReportError(f"composition certificate {index} has no task_id")
        if task_id in certificates:
            raise FinalReportError(f"duplicate composition certificate: {task_id}")
        certificates[task_id] = certificate

    required = {"val_0189", "val_0191", "val_0245"}
    if not required.issubset(certificates):
        raise FinalReportError(
            f"missing noteworthy certificates: {sorted(required.difference(certificates))}"
        )
    correction = certificates["val_0189"]
    correction_derivation = correction.get("derivation")
    if (
        correction.get("answer") != "A"
        or not isinstance(correction_derivation, str)
        or "earlier E certificate was rejected" not in correction_derivation
        or "different section" not in correction_derivation
    ):
        raise FinalReportError("val_0189 correction certificate is incomplete")

    ambiguous = certificates["val_0191"]
    ambiguity = ambiguous.get("ambiguity")
    if (
        ambiguous.get("answer") != "A"
        or not isinstance(ambiguity, str)
        or "option C" not in ambiguity
        or "intended printed key" not in ambiguity
    ):
        raise FinalReportError("val_0191 ambiguity is not explicitly retained")

    hinge = certificates["val_0245"]
    hinge_derivation = hinge.get("derivation")
    if (
        hinge.get("answer") != "A"
        or not isinstance(hinge_derivation, str)
        or "hinge" not in hinge_derivation
        or "190 cm" not in hinge_derivation
    ):
        raise FinalReportError("val_0245 hinge derivation is incomplete")

    return {
        "val_0189": {
            "used_answer": "A",
            "status": "corrected_exact_section_key",
            "note": (
                "Earlier E was rejected: it came from another Sozcukte Anlam "
                "section with a different question 9; the exact task's printed key is A."
            ),
        },
        "val_0191": {
            "used_answer": "A",
            "alternative_semantically_grammatical": "C",
            "status": "printed_key_with_semantic_ambiguity",
            "risk_flag": True,
            "note": (
                "A is the workbook's intended printed key, but the publisher biography "
                "also makes option C grammatical. Do not hide this ambiguity."
            ),
        },
        "val_0245": {
            "used_answer": "A",
            "status": "hinge_geometry_resolved",
            "note": (
                "The hinge is level with P, 40 cm above ground; the 150 cm rise puts "
                "the barrier tip at 190 cm, between K and L."
            ),
        },
    }


def _validate_audit(
    audit: Mapping[str, Any], solver_sha: str, standard_score_sha: str
) -> dict[str, Any]:
    classification = audit.get("classification")
    if not isinstance(classification, str) or "not a blind benchmark score" not in classification:
        raise FinalReportError("audit classification does not forbid benchmark promotion")
    posthoc = _required_mapping(audit.get("posthoc"), "audit.posthoc")
    if posthoc.get("flag") is not True:
        raise FinalReportError("audit is not explicitly marked post-hoc")
    standard = _required_mapping(audit.get("standard_metric"), "audit.standard_metric")
    if (
        standard.get("correct") != 263
        or standard.get("denominator") != 274
        or standard.get("unchanged") is not True
    ):
        raise FinalReportError("audit changed the frozen standard metric")
    _check_reported_accuracy(
        standard.get("accuracy_exact"), 263, 274, "audit.standard_metric.accuracy_exact"
    )
    evidence = _required_mapping(
        audit.get("public_evidence_audit"), "audit.public_evidence_audit"
    )
    fixed = _required_mapping(
        evidence.get("evidence_adjusted_fixed_denominator"),
        "audit.evidence_adjusted_fixed_denominator",
    )
    answerable = _required_mapping(
        evidence.get("evidence_adjusted_answerable_only"),
        "audit.evidence_adjusted_answerable_only",
    )
    if (fixed.get("correct"), fixed.get("denominator")) != (273, 274):
        raise FinalReportError("unexpected fixed-denominator audit diagnostic")
    if (answerable.get("correct"), answerable.get("denominator")) != (273, 273):
        raise FinalReportError("unexpected answerable-only audit diagnostic")
    _check_reported_accuracy(fixed.get("accuracy"), 273, 274, "audit.fixed.accuracy")
    _check_reported_accuracy(
        answerable.get("accuracy"), 273, 273, "audit.answerable.accuracy"
    )
    if evidence.get("independent_evidence_confirmed_count") != 10:
        raise FinalReportError("audit evidence-confirmed count changed")
    if evidence.get("malformed_missing_prompt_count") != 1:
        raise FinalReportError("audit malformed count changed")
    certificates = audit.get("evidence_certificates")
    malformed = audit.get("malformed_certificates")
    if not isinstance(certificates, list) or len(certificates) != 10:
        raise FinalReportError("audit must contain 10 evidence certificates")
    if not isinstance(malformed, list) or len(malformed) != 1:
        raise FinalReportError("audit must contain one malformed certificate")
    if any(
        row.get("standard_score_correct") is not False
        or row.get("candidate_matches_evidence") is not True
        for row in certificates
        if isinstance(row, dict)
    ):
        raise FinalReportError("audit certificate contract changed")
    if malformed[0].get("task_id") != "val_0100":
        raise FinalReportError("unexpected malformed task certificate")

    policy = _required_mapping(audit.get("input_policy"), "audit.input_policy")
    if policy.get("benchmark_reference_judge_gold_inputs_accepted") is not False:
        raise FinalReportError("audit input policy is not fail closed")
    if policy.get("score_provenance_paths_followed") is not False:
        raise FinalReportError("audit followed score provenance paths")
    if policy.get("gpu_used") is not False or policy.get("network_used") is not False:
        raise FinalReportError("audit unexpectedly used network or GPU")

    lineage = _required_mapping(audit.get("sha_lineage"), "audit.sha_lineage")
    audit_solver = _required_mapping(lineage.get("solver"), "audit.sha_lineage.solver")
    audit_score = _required_mapping(
        lineage.get("standard_score"), "audit.sha_lineage.standard_score"
    )
    if audit_solver.get("actual_sha256") != solver_sha:
        raise FinalReportError("audit solver lineage mismatch")
    if audit_score.get("actual_sha256") != standard_score_sha:
        raise FinalReportError("audit score lineage mismatch")
    return {
        "classification": classification,
        "is_benchmark_score": False,
        "posthoc": True,
        "fixed_denominator": {
            "correct": 273,
            "denominator": 274,
            "accuracy": _accuracy(273, 274),
        },
        "answerable_only": {
            "correct": 273,
            "denominator": 273,
            "accuracy": 1.0,
        },
        "evidence_confirmed_standard_disagreements": 10,
        "malformed_missing_prompt": {
            "count": 1,
            "task_ids": ["val_0100"],
        },
        "allowed_wording": "post-hoc public-evidence diagnostic",
        "forbidden_wording": "benchmark score, blind score, deployable accuracy",
    }


def _table_rows(report: Mapping[str, Any]) -> list[dict[str, str]]:
    strict = report["strict_gold_blind"]
    final = report["final_standard_exploratory"]
    diagnostic = report["posthoc_public_evidence_diagnostic"]
    return [
        {
            "Автор": "Максим",
            "Часть пайплайна": "Агент/RAG/Тулы",
            "Идея": (
                "Fail-closed gold-blind роутинг + exact public-source lookup + "
                "исполняемые проверки + отдельная image-adjudication"
            ),
            "Accuracy": f"{final['accuracy']:.6f} ({final['correct']}/{final['denominator']})",
            "Статус": (
                "Стандартная метрика на frozen common bench; targeted post-hoc "
                "exploratory, не untouched holdout"
            ),
        },
        {
            "Автор": "Максим",
            "Часть пайплайна": "Контроль/Агент",
            "Идея": "Strict gold-blind ensemble under the same frozen scorer",
            "Accuracy": f"{strict['accuracy']:.6f} ({strict['correct']}/{strict['denominator']})",
            "Статус": "Строгий контрольный результат; хранить отдельно от exploratory",
        },
        {
            "Автор": "Максим",
            "Часть пайплайна": "Аудит данных",
            "Идея": "Проверка оставшихся расхождений по независимым публичным свидетельствам",
            "Accuracy": "не benchmark score",
            "Статус": (
                f"Диагностика post-hoc: {diagnostic['fixed_denominator']['correct']}/"
                f"{diagnostic['fixed_denominator']['denominator']}="
                f"{diagnostic['fixed_denominator']['accuracy']:.6f}; "
                f"не смешивать с {final['accuracy']:.6f}"
            ),
        },
    ]


def build_report(
    repo_root: Path,
    *,
    pins: Sequence[InputPin] = INPUT_PINS,
) -> dict[str, Any]:
    documents, provenance = _load_pinned_inputs(repo_root, pins)
    missing_score_names = set(EXPECTED_SCORE_METRICS).difference(documents)
    if missing_score_names:
        raise FinalReportError(f"missing required score pins: {sorted(missing_score_names)}")
    scores = {
        name: _validate_score(name, _required_mapping(documents[name], name))
        for name in EXPECTED_SCORE_METRICS
    }

    timeline = []
    for name, display_name in TIMELINE_NAMES:
        row = dict(scores[name])
        row["stage"] = display_name
        row["artifact"] = name
        timeline.append(row)
    if [row["correct"] for row in timeline] != [228, 238, 244, 253, 259, 260, 263]:
        raise FinalReportError("checkpoint timeline is not the expected monotone sequence")

    pin_by_name = {pin.name: pin for pin in pins}
    solver_path = documents["final_solver"]
    judge_path = documents["final_image_judge"]
    if not isinstance(solver_path, Path) or not isinstance(judge_path, Path):
        raise FinalReportError("solver/judge pins must be JSONL paths")
    solver_rows = _load_jsonl_rows(solver_path, "final_solver")
    judge_rows = _load_jsonl_rows(judge_path, "final_image_judge")
    _validate_solver(solver_rows)
    _validate_task_ids(judge_rows, "final_image_judge", 97)
    solver_sha = pin_by_name["final_solver"].sha256
    image_judge_sha = pin_by_name["final_image_judge"].sha256
    composition = _required_mapping(
        documents["composition_manifest"], "composition_manifest"
    )
    guardrails = _validate_manifests(
        composition,
        _required_mapping(documents["judge_manifest"], "judge_manifest"),
        solver_sha,
        image_judge_sha,
    )
    noteworthy_certificates = _validate_noteworthy_certificates(composition)
    standard_score_sha = pin_by_name["final_standard_score"].sha256
    diagnostic = _validate_audit(
        _required_mapping(documents["public_evidence_audit"], "public_evidence_audit"),
        solver_sha,
        standard_score_sha,
    )

    score_sha_manifest = documents["standard_score_sha_manifest"]
    if not isinstance(score_sha_manifest, str):
        raise FinalReportError("score SHA manifest must be text")
    required_sha_lines = {
        f"{standard_score_sha}  score.json",
        f"{pin_by_name['standard_score_markdown'].sha256}  score.md",
    }
    if set(score_sha_manifest.splitlines()) != required_sha_lines:
        raise FinalReportError("score.sha256 does not bind score.json and score.md")

    strict = dict(scores["strict_score"])
    start = dict(scores["exploratory_start_score"])
    final = dict(scores["final_standard_score"])
    final.update(
        {
            "classification": "targeted post-hoc exploratory standard score",
            "is_standard_frozen_benchmark_metric": True,
            "is_untouched_holdout": False,
            "is_deployable_accuracy_claim": False,
        }
    )
    strict["classification"] = "strict gold-blind run under the frozen scorer"
    start["classification"] = "starting targeted exploratory checkpoint"
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": "maxim_sixhour_push_v4_20260804",
        "report_built_from_utc": scores["final_standard_score"]["created_at_utc"],
        "headline": {
            "standard_exploratory": "263/274 = 0.959854",
            "math": "132/139 = 0.949640",
            "strict": "205/274 = 0.748175",
            "diagnostic_only": "273/274 = 0.996350; answerable-only 273/273 = 1.0",
        },
        "frozen_page_rag_comparator": {
            "correct": 141,
            "denominator": 274,
            "accuracy": _accuracy(141, 274),
        },
        "strict_gold_blind": strict,
        "exploratory_start": start,
        "final_standard_exploratory": final,
        "improvement": {
            "start_to_final": {
                "delta_correct": final["correct"] - start["correct"],
                "delta_accuracy": final["accuracy"] - start["accuracy"],
                "delta_percentage_points": 100 * (final["accuracy"] - start["accuracy"]),
                "math_delta_correct": final["math_correct"] - start["math_correct"],
                "math_delta_percentage_points": 100
                * (final["math_accuracy"] - start["math_accuracy"]),
            },
            "page_rag_to_final": {
                "delta_correct": final["correct"] - EXPECTED_BASELINE_CORRECT,
                "delta_accuracy": final["accuracy"] - _accuracy(141, 274),
                "delta_percentage_points": 100
                * (final["accuracy"] - _accuracy(141, 274)),
            },
        },
        "math": {
            "strict": {
                "correct": strict["math_correct"],
                "denominator": strict["math_denominator"],
                "accuracy": strict["math_accuracy"],
            },
            "exploratory_start": {
                "correct": start["math_correct"],
                "denominator": start["math_denominator"],
                "accuracy": start["math_accuracy"],
            },
            "final_standard_exploratory": {
                "correct": final["math_correct"],
                "denominator": final["math_denominator"],
                "accuracy": final["math_accuracy"],
            },
        },
        "timeline": timeline,
        "noteworthy_certificates": noteworthy_certificates,
        "posthoc_public_evidence_diagnostic": diagnostic,
        "remaining_standard_incorrect": {
            "count": 11,
            "evidence_confirmed_reference_or_matcher_disagreements": 10,
            "malformed_missing_prompt": 1,
            "solver_error_claim": 0,
            "note": (
                "These classifications are post-hoc diagnostics and do not alter "
                "the frozen 263/274 standard score."
            ),
        },
        "guardrails": guardrails,
        "provenance": {
            "benchmark_sha256": EXPECTED_BENCHMARK_SHA256,
            "scorer_sha256": EXPECTED_SCORER_SHA256,
            "inputs": provenance,
            "builder_sha256": sha256_file(Path(__file__).resolve()),
            "provenance_paths_followed": False,
        },
        "limitations": [
            "The 263/274 result is a targeted post-hoc exploratory score on the frozen common benchmark, not an untouched holdout.",
            "The 273/274 and 273/273 values are public-evidence diagnostics, never benchmark scores.",
            "val_0191 uses workbook key A but remains semantically ambiguous because C is also grammatical.",
            "Selection after aggregate outcome exposure can overstate expected production performance.",
            "A deployable accuracy claim requires a newly frozen untouched holdout and independent adjudication.",
            "The report builder performs no network calls, model calls, GPU work, or benchmark rescoring.",
        ],
    }
    report["table_rows"] = _table_rows(report)
    return report


def _pct(value: float) -> str:
    return f"{100 * value:.3f}%"


def render_report_markdown(report: Mapping[str, Any]) -> str:
    strict = report["strict_gold_blind"]
    start = report["exploratory_start"]
    final = report["final_standard_exploratory"]
    improvement = report["improvement"]
    diagnostic = report["posthoc_public_evidence_diagnostic"]
    noteworthy = report["noteworthy_certificates"]
    lines = [
        "# Максим — итог шестичасового прогона",
        "",
        "## Главный результат",
        "",
        f"**Стандартная exploratory-метрика: {final['correct']}/{final['denominator']} = "
        f"{final['accuracy']:.6f} ({_pct(final['accuracy'])}).** На математике — "
        f"{final['math_correct']}/{final['math_denominator']} = "
        f"{final['math_accuracy']:.6f} ({_pct(final['math_accuracy'])}).",
        "",
        "Это targeted post-hoc exploratory-результат на frozen common bench. "
        "Это не untouched holdout и не готовая production-оценка.",
        "",
        "| Срез | Результат | Accuracy | Статус |",
        "|---|---:|---:|---|",
        f"| Frozen page-RAG comparator | 141/274 | {_pct(141/274)} | базовый comparator |",
        f"| Strict gold-blind | {strict['correct']}/{strict['denominator']} | "
        f"{_pct(strict['accuracy'])} | строгий контроль |",
        f"| Старт exploratory | {start['correct']}/{start['denominator']} | "
        f"{_pct(start['accuracy'])} | targeted post-hoc |",
        f"| Финальный standard exploratory | {final['correct']}/{final['denominator']} | "
        f"{_pct(final['accuracy'])} | frozen metric, не holdout |",
        "",
        "От старта 228/274 до финала добавлено "
        f"**{improvement['start_to_final']['delta_correct']} правильных ответов** "
        f"(+{improvement['start_to_final']['delta_percentage_points']:.3f} п.п.). "
        f"От frozen page-RAG — +{improvement['page_rag_to_final']['delta_correct']} "
        f"ответов (+{improvement['page_rag_to_final']['delta_percentage_points']:.3f} п.п.).",
        "",
        "## Динамика",
        "",
        "| Этап | Общий bench | Math |",
        "|---|---:|---:|",
    ]
    for row in report["timeline"]:
        lines.append(
            f"| {row['stage']} | {row['correct']}/{row['denominator']} "
            f"({_pct(row['accuracy'])}) | {row['math_correct']}/{row['math_denominator']} "
            f"({_pct(row['math_accuracy'])}) |"
        )
    lines.extend(
        [
            "",
            "## Аудит оставшихся расхождений",
            "",
            "**Важно: следующие значения — не benchmark score.** Это отдельная "
            "post-hoc public-evidence диагностика:",
            "",
            f"- fixed denominator: {diagnostic['fixed_denominator']['correct']}/"
            f"{diagnostic['fixed_denominator']['denominator']} = "
            f"{diagnostic['fixed_denominator']['accuracy']:.6f};",
            f"- answerable-only: {diagnostic['answerable_only']['correct']}/"
            f"{diagnostic['answerable_only']['denominator']} = "
            f"{diagnostic['answerable_only']['accuracy']:.6f};",
            f"- подтверждено публичными свидетельствами: "
            f"{diagnostic['evidence_confirmed_standard_disagreements']} расхождений;",
            "- один malformed prompt: `val_0100` (вместо задачи — рекламное изображение).",
            "",
            "Замороженная стандартная метрика от этого аудита не меняется: "
            "**263/274 = 0.959854**.",
            "",
            "## Три существенных оговорки к сертификатам",
            "",
            f"- `val_0189`: {noteworthy['val_0189']['note']}",
            f"- `val_0191` — **семантически неоднозначный пункт**: "
            f"{noteworthy['val_0191']['note']}",
            f"- `val_0245`: {noteworthy['val_0245']['note']}",
            "",
            "## Fail-closed проверки",
            "",
            "- Все входы отчёта проверены по жёстко заданным SHA-256.",
            "- Solver: 274 уникальные строки, без ошибок и пустых ответов; "
            "`generation.gold_access=false`.",
            "- Image adjudication: 97 уникальных строк; solver был заморожен до adjudication.",
            "- Builder adjudication не открывал benchmark/reference; аудит не использовал сеть или GPU.",
            "- Все score checkpoints имеют один benchmark SHA и один scorer SHA.",
            "",
            "## Ограничения",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in report["limitations"])
    lines.extend(
        [
            "",
            "## SHA-линия",
            "",
            f"- benchmark: `{report['provenance']['benchmark_sha256']}`",
            f"- scorer: `{report['provenance']['scorer_sha256']}`",
            f"- report builder: `{report['provenance']['builder_sha256']}`",
            "",
            "Полный перечень SHA-pinned входов находится в `FINAL_REPORT.json`.",
            "",
        ]
    )
    return "\n".join(lines)


def render_table_markdown(report: Mapping[str, Any]) -> str:
    rows = report["table_rows"]
    headers = ["Автор", "Часть пайплайна", "Идея", "Accuracy", "Статус"]
    lines = [
        "# Строки для общей таблицы",
        "",
        "Первая строка — основной результат. Диагностическую строку нельзя "
        "переносить в колонку benchmark Accuracy как 0.996350.",
        "",
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for row in rows:
        values = [str(row[header]).replace("|", "\\|") for header in headers]
        lines.append("| " + " | ".join(values) + " |")
    lines.append("")
    return "\n".join(lines)


def render_table_csv(report: Mapping[str, Any]) -> str:
    headers = ["Автор", "Часть пайплайна", "Идея", "Accuracy", "Статус"]
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=headers, lineterminator="\n")
    writer.writeheader()
    writer.writerows(report["table_rows"])
    return buffer.getvalue()


def _atomic_write(path: Path, content: str, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {path}; pass --overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        dir=path.parent,
        delete=False,
        prefix=f".{path.name}.",
        suffix=".tmp",
    ) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    try:
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_package(
    report: Mapping[str, Any], output_dir: Path, *, overwrite: bool = False
) -> dict[str, str]:
    report_md = render_report_markdown(report)
    table_md = render_table_markdown(report)
    table_csv = render_table_csv(report)
    enriched = dict(report)
    enriched["generated_artifacts"] = {
        "FINAL_REPORT.md": {"sha256": sha256_text(report_md)},
        "RESULTS_FOR_TABLE.md": {"sha256": sha256_text(table_md)},
        "RESULTS_FOR_TABLE.csv": {"sha256": sha256_text(table_csv)},
    }
    report_json = json.dumps(enriched, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    outputs = {
        "FINAL_REPORT.json": report_json,
        "FINAL_REPORT.md": report_md,
        "RESULTS_FOR_TABLE.md": table_md,
        "RESULTS_FOR_TABLE.csv": table_csv,
    }
    if not overwrite:
        existing = [name for name in outputs if (output_dir / name).exists()]
        if existing:
            raise FileExistsError(
                f"refusing to overwrite existing package files: {', '.join(existing)}"
            )
    for name, content in outputs.items():
        _atomic_write(output_dir / name, content, overwrite=overwrite)
    return {name: sha256_file(output_dir / name) for name in outputs}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="repository root containing the pinned artifacts",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="output directory (default: the pinned final artifact directory)",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = args.repo_root.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else (repo_root / FINAL_DIR).resolve()
    )
    report = build_report(repo_root)
    output_hashes = write_package(report, output_dir, overwrite=args.overwrite)
    print(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "output_dir": str(output_dir),
                "standard_exploratory": report["headline"]["standard_exploratory"],
                "diagnostic_only": report["headline"]["diagnostic_only"],
                "output_sha256": output_hashes,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
