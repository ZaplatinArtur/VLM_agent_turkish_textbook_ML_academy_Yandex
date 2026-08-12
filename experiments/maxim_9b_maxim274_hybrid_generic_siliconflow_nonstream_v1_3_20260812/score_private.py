"""Post-completion aggregate-only wrapper around the frozen Maxim274 scorer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from compose_hybrid import ComposeError, verify_exact_composition
from nonstream_protocol import exclusive_json, read_json, sha256_file

HERE = Path(__file__).resolve().parent
PRIVATE_FREEZE = HERE / "PRIVATE_SCORE_FREEZE.json"
PRIVATE_SIDECAR = HERE / "PRIVATE_SCORE_FREEZE_SHA256.txt"
EXECUTION_FREEZE = HERE / "EXECUTION_FREEZE.json"
COMPLETION = HERE / "COMPLETION.json"
COMPOSE_COMPLETION = HERE / "HYBRID_COMPOSE_COMPLETION.json"
COMPOSED_SOLVER = HERE / "runs" / "hybrid_solver_274.jsonl"
RESULT = HERE / "PRIVATE_RESULT.json"


class ScoreError(RuntimeError):
    pass


def validate_standard_score_provenance(
    score: Any,
    freeze: dict,
    compose: dict,
    expected_image_judge_sha256: str,
) -> tuple[int, int]:
    if type(score) is not dict or score.get("schema_version") != "maxim-full274-score-v1":
        raise ScoreError("standard score schema mismatch")
    overall = score.get("overall")
    provenance = score.get("provenance")
    guardrails = score.get("guardrails")
    artifacts = freeze.get("artifacts")
    if (
        type(overall) is not dict
        or type(provenance) is not dict
        or type(guardrails) is not dict
        or type(artifacts) is not dict
        or provenance.get("benchmark", {}).get("sha256") != artifacts["benchmark"]["sha256"]
        or provenance.get("frozen_page_rag_judge", {}).get("sha256") != artifacts["baseline_judge"]["sha256"]
        or provenance.get("scorer", {}).get("sha256") != artifacts["standard_scorer"]["sha256"]
        or provenance.get("solver_results", {}).get("sha256") != compose["composed_solver_sha256"]
        or provenance.get("image_judge", {}).get("sha256") != expected_image_judge_sha256
        or guardrails.get("benchmark_rows_verified") != 274
        or guardrails.get("solver_rows_verified") != 274
        or guardrails.get("baseline_rows_verified") != 274
        or guardrails.get("task_id_sets_match") is not True
        or guardrails.get("duplicate_task_ids") != 0
        or guardrails.get("forbidden_gold_fields_in_solver") != 0
        or guardrails.get("explicit_nonfalse_generation_gold_access") != 0
        or guardrails.get("frozen_sha_pins_checked") is not True
    ):
        raise ScoreError("standard score provenance/guardrail mismatch")
    correct, total = overall.get("new_correct"), overall.get("n")
    if type(correct) is not int or total != 274:
        raise ScoreError("aggregate denominator mismatch")
    return correct, total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-private-freeze-sha256", required=True)
    parser.add_argument("--expected-execution-freeze-sha256", required=True)
    parser.add_argument("--expected-completion-sha256", required=True)
    parser.add_argument("--expected-compose-completion-sha256", required=True)
    parser.add_argument(
        "--image-judge-jsonl",
        type=Path,
        required=True,
        help="Exact image-judge input supplied to the standard scorer after composition",
    )
    parser.add_argument("--expected-image-judge-sha256", required=True)
    parser.add_argument("--score-json", type=Path, required=True, help="Output of the pinned standard scorer, produced only after completion")
    parser.add_argument("--expected-score-json-sha256", required=True)
    args = parser.parse_args()
    if RESULT.exists():
        raise ScoreError("private aggregate result exists")
    if sha256_file(PRIVATE_FREEZE) != args.expected_private_freeze_sha256 or PRIVATE_SIDECAR.read_text(encoding="ascii").strip() != args.expected_private_freeze_sha256:
        raise ScoreError("private freeze pin mismatch")
    freeze = read_json(PRIVATE_FREEZE)
    if freeze.get("state") != "private_score_frozen_unexecuted" or freeze.get("execution_freeze_sha256") != args.expected_execution_freeze_sha256:
        raise ScoreError("private freeze closure mismatch")
    if sha256_file(EXECUTION_FREEZE) != args.expected_execution_freeze_sha256 or sha256_file(COMPLETION) != args.expected_completion_sha256:
        raise ScoreError("execution/completion pin mismatch")
    completion = read_json(COMPLETION)
    if completion.get("rows") != 256 or completion.get("gold_opened") is not False or completion.get("outcomes_opened") is not False:
        raise ScoreError("completion guard mismatch")
    if sha256_file(COMPOSE_COMPLETION) != args.expected_compose_completion_sha256:
        raise ScoreError("compose completion external pin mismatch")
    compose = read_json(COMPOSE_COMPLETION)
    if (
        compose.get("schema_version") != "maxim256-hybrid-compose-completion-v1"
        or compose.get("candidate_execution_freeze_sha256") != args.expected_execution_freeze_sha256
        or compose.get("candidate_completion_sha256") != args.expected_completion_sha256
        or compose.get("rows") != 274
        or compose.get("generic_rows") != 256
        or compose.get("certified_rows") != 18
        or compose.get("composed_solver_sha256") != sha256_file(COMPOSED_SOLVER)
        or compose.get("gold_opened") is not False
        or compose.get("outcomes_opened") is not False
    ):
        raise ScoreError("exact composition closure mismatch")
    try:
        verify_exact_composition(
            args.expected_execution_freeze_sha256,
            compose.get("candidate_independent_audit_sha256"),
            args.expected_completion_sha256,
            compose["composed_solver_sha256"],
        )
    except (ComposeError, KeyError, TypeError) as exc:
        raise ScoreError("independent exact composition recomputation failed") from exc
    if sha256_file(args.score_json) != args.expected_score_json_sha256:
        raise ScoreError("standard score JSON external pin mismatch")
    if sha256_file(args.image_judge_jsonl) != args.expected_image_judge_sha256:
        raise ScoreError("image judge external pin mismatch")
    score = read_json(args.score_json)
    correct, total = validate_standard_score_provenance(
        score,
        freeze,
        compose,
        args.expected_image_judge_sha256,
    )
    result = {"schema_version": "maxim256-hybrid-private-aggregate-v1", "execution_freeze_sha256": args.expected_execution_freeze_sha256, "private_score_freeze_sha256": args.expected_private_freeze_sha256, "completion_sha256": args.expected_completion_sha256, "compose_completion_sha256": args.expected_compose_completion_sha256, "composed_solver_sha256": compose["composed_solver_sha256"], "image_judge_sha256": args.expected_image_judge_sha256, "standard_score_sha256": args.expected_score_json_sha256, "correct": correct, "total": 274, "accuracy": correct / 274, "pre_outcome_threshold_correct_at_least": 240, "threshold_met": correct >= 240, "task_ids_present": False, "gold_answers_present": False, "per_row_outcomes_present": False}
    exclusive_json(RESULT, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
