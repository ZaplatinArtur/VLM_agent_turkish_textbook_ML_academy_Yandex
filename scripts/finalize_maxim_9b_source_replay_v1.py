#!/usr/bin/env python3
"""Freeze one materialized 9B-only source-replay milestone."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any


MODEL = "Qwen/Qwen3.5-9B"
BENCHMARK_SHA256 = "5a6a38ccae7835f0d015f6e5979834208347b8e6e7a8d6884e4af97605f51ed9"
SCHEMA = "maxim-9b-source-replay-aggregate-v1"
FREEZE_SCHEMA = "maxim-9b-source-replay-freeze-v1"
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
FORBIDDEN_SOLVER_FIELDS = {
    "reference_answer",
    "reference_solution",
    "gold_answer",
    "gold_solution",
    "correct",
    "is_correct",
    "strict_correct",
    "judge",
    "judge_verdict",
    "parsed_verdict",
    "verdict",
    "outcome",
}
# Byte pins for the known 27B-bearing solver chain that this replay replaces.
# Rejecting both the model bytes and these hashes prevents a renamed/copy-only
# artifact from silently re-entering a purported 9B-only chain.
KNOWN_27B_SOLVER_SHA256 = {
    "05c0acb048be4487abed51ade2e3260f52297781bf8473c5fd36efa8eb793cae",
    "aa76740913819b81e23f926e89be68e30501e6f6e14f36867afb3a9f122cc678",
    "96cb913202f63d14d5af1935247cf0477e7799041c9ecf61f3cafb6bf05bc09e",
    "b0b366c80de4cf45dec48f4cb77367cb632d31f824c08eb9d3474c0e860248ea",
    "01740f36989e19cec5f809936377bde964befee3c88b3b35f68972e3ee418d57",
    "48735253b140f23f126625fea06a1e09b6440f0dc8afd64753216bc94da53ea0",
}


class FinalizeError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def stable_projection(value: Any) -> Any:
    """Remove only volatile timestamps and filesystem locations recursively."""

    if isinstance(value, dict):
        return {
            key: stable_projection(item)
            for key, item in sorted(value.items())
            if key not in {"created_at_utc", "path"}
        }
    if isinstance(value, list):
        return [stable_projection(item) for item in value]
    return value


def validate_solver_rows(rows: dict[str, dict[str, Any]], label: str) -> None:
    for task_id, row in rows.items():
        forbidden = sorted(FORBIDDEN_SOLVER_FIELDS.intersection(row))
        if forbidden:
            raise FinalizeError(f"{label}:{task_id}: forbidden outcome fields {forbidden}")
        generation = row.get("generation")
        if isinstance(generation, dict) and generation.get("gold_access", False) is not False:
            raise FinalizeError(f"{label}:{task_id}: generation.gold_access is not false")


def guard_no_known_27b(value: Any, label: str) -> None:
    serialized = canonical(value)
    if b"Qwen/Qwen3.5-27B" in serialized:
        raise FinalizeError(f"{label}: 27B model bytes found")
    text_value = serialized.decode("utf-8")
    found = sorted(digest for digest in KNOWN_27B_SOLVER_SHA256 if digest in text_value)
    if found:
        raise FinalizeError(f"{label}: known 27B solver SHA found: {found}")


def compare_scores(
    *,
    label: str,
    role: str,
    solver_path: Path,
    score_path: Path,
    final_outcomes: dict[str, bool],
) -> dict[str, Any]:
    score = load_json(score_path)
    rows = score.get("task_outcomes")
    if not isinstance(rows, list):
        raise FinalizeError(f"comparison {label}: missing task_outcomes")
    baseline: dict[str, bool] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise FinalizeError(f"comparison {label}: malformed task outcome")
        task_id = str(row.get("task_id") or "")
        correct = row.get("new_correct")
        if not task_id or task_id in baseline or not isinstance(correct, bool):
            raise FinalizeError(f"comparison {label}: malformed/duplicate task outcome")
        baseline[task_id] = correct
    if set(baseline) != set(final_outcomes):
        raise FinalizeError(f"comparison {label}: task set differs")
    score_solver = score.get("provenance", {}).get("solver_results", {})
    if (
        score_solver.get("sha256") != sha256_file(solver_path)
        or Path(str(score_solver.get("path") or "")).resolve() != solver_path.resolve()
    ):
        raise FinalizeError(f"comparison {label}: score/solver binding failed")
    transitions = Counter(
        {"both_correct": 0, "fixed": 0, "regressed": 0, "both_wrong": 0}
    )
    for task_id, before in baseline.items():
        after = final_outcomes[task_id]
        if before and after:
            transitions["both_correct"] += 1
        elif not before and after:
            transitions["fixed"] += 1
        elif before and not after:
            transitions["regressed"] += 1
        else:
            transitions["both_wrong"] += 1
    before_correct = sum(baseline.values())
    after_correct = sum(final_outcomes.values())
    return {
        "label": label,
        "role": role,
        "solver": artifact(solver_path, rows=len(baseline)),
        "score": artifact(score_path),
        "before_correct": before_correct,
        "after_correct": after_correct,
        "delta_correct": after_correct - before_correct,
        "before_accuracy": round(before_correct / len(baseline), 6),
        "after_accuracy": round(after_correct / len(baseline), 6),
        **dict(sorted(transitions.items())),
    }


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise FinalizeError(f"{path}: expected object")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise FinalizeError(f"{path}:{number}: expected object")
        rows.append(value)
    return rows


def index(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        task_id = str(row.get("task_id") or "").strip()
        if not task_id or task_id in result:
            raise FinalizeError(f"{label}: missing or duplicate task_id")
        result[task_id] = row
    return result


def artifact(path: Path, *, rows: int | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {"path": str(path.resolve()), "sha256": sha256_file(path)}
    if rows is not None:
        result["rows"] = rows
    return result


def referenced(spec: Any, label: str) -> tuple[Path, str]:
    if not isinstance(spec, dict):
        raise FinalizeError(f"{label}: missing artifact")
    path = Path(str(spec.get("path") or "")).resolve()
    expected = str(spec.get("sha256") or "").lower()
    if _HEX64.fullmatch(expected) is None or sha256_file(path) != expected:
        raise FinalizeError(f"{label}: hash mismatch")
    return path, expected


def parse_stage(raw: str) -> tuple[str, Path, Path, Path]:
    name, sep, paths = raw.partition("=")
    items = paths.split(",") if sep else []
    if not name or len(items) != 3:
        raise FinalizeError("--stage must be NAME=RESOLVER,COMPOSITION,JUDGE_MANIFEST")
    return name, *(Path(item).resolve() for item in items)


def parse_comparison(raw: str) -> tuple[str, str, Path, Path]:
    identity, sep, paths = raw.partition("=")
    label, role_sep, role = identity.partition(":")
    items = paths.split(",") if sep else []
    if not label or not role_sep or role not in {"anchor", "adjacent"} or len(items) != 2:
        raise FinalizeError(
            "--comparison must be LABEL:anchor|adjacent=SOLVER,SCORE_JSON"
        )
    return label, role, Path(items[0]).resolve(), Path(items[1]).resolve()


def finalize(args: argparse.Namespace) -> dict[str, Any]:
    anchor_path = args.anchor_solver.resolve()
    final_solver_path = args.final_solver.resolve()
    final_judge_path = args.final_image_judge.resolve()
    score_path = args.score_json.resolve()
    anchor_rows = index(load_jsonl(anchor_path), "anchor")
    final_rows = index(load_jsonl(final_solver_path), "final solver")
    judge_rows = index(load_jsonl(final_judge_path), "final judge")
    score = load_json(score_path)
    validate_solver_rows(anchor_rows, "anchor")
    validate_solver_rows(final_rows, "final solver")
    if len(anchor_rows) != 274 or len(final_rows) != 274 or len(judge_rows) != 97:
        raise FinalizeError("expected anchor/final/judge rows 274/274/97")
    if set(anchor_rows) != set(final_rows) or not set(judge_rows) <= set(final_rows):
        raise FinalizeError("anchor/final/judge task sets differ")
    if any(row.get("model") != MODEL for row in anchor_rows.values()):
        raise FinalizeError("anchor is not pure Qwen3.5-9B")
    if any(row.get("model") != MODEL for row in final_rows.values()):
        raise FinalizeError("final solver model closure is not Qwen3.5-9B")
    if sha256_file(anchor_path) != args.expected_anchor_sha256.lower():
        raise FinalizeError("anchor SHA-256 mismatch")
    if sha256_file(anchor_path) in KNOWN_27B_SOLVER_SHA256:
        raise FinalizeError("anchor is a known 27B solver artifact")
    benchmark_spec = score.get("provenance", {}).get("benchmark")
    benchmark_path, benchmark_sha = referenced(benchmark_spec, "score benchmark")
    if benchmark_sha != BENCHMARK_SHA256:
        raise FinalizeError("score benchmark is not the frozen 274-row benchmark")
    benchmark_rows = index(load_jsonl(benchmark_path), "benchmark")
    if set(benchmark_rows) != set(anchor_rows) or len(benchmark_rows) != 274:
        raise FinalizeError("anchor/final task set does not exactly match benchmark")

    stages: list[dict[str, Any]] = []
    source_owner: dict[str, str] = {}
    source_answer: dict[str, str] = {}
    certificate_bundle: list[dict[str, Any]] = []
    current_solver = anchor_path
    current_sha = sha256_file(current_solver)
    for raw in args.stage:
        name, resolver_path, composition_path, judge_manifest_path = parse_stage(raw)
        resolver = load_json(resolver_path)
        composition = load_json(composition_path)
        judge_manifest = load_json(judge_manifest_path)
        guard_no_known_27b(profile := load_json(referenced(resolver.get("profile"), f"{name} profile")[0]), f"{name} profile")
        guard_no_known_27b(resolver, f"{name} resolver")
        guard_no_known_27b(composition, f"{name} composition")
        guard_no_known_27b(judge_manifest, f"{name} judge manifest")
        profile_path, profile_sha = referenced(resolver.get("profile"), f"{name} profile")
        profile_anchor_path, profile_anchor_sha = referenced(
            profile.get("anchor"), f"{name} profile anchor"
        )
        if profile_anchor_path != current_solver or profile_anchor_sha != current_sha:
            raise FinalizeError(f"{name}: profile does not continue the exact solver chain")
        if profile_anchor_sha in KNOWN_27B_SOLVER_SHA256:
            raise FinalizeError(f"{name}: profile anchor is a known 27B solver")
        if (
            resolver.get("gold_access") is not False
            or resolver.get("benchmark_candidate_or_outcome_access") is not False
            or composition.get("profile", {}).get("sha256") != profile_sha
            or composition.get("resolver_manifest", {}).get("sha256") != sha256_file(resolver_path)
        ):
            raise FinalizeError(f"{name}: source-only chain attestation failed")
        outputs = composition.get("artifacts") or composition.get("output")
        solver_path, solver_sha = referenced(outputs.get("solver"), f"{name} solver")
        decisions_path, decisions_sha = referenced(outputs.get("decisions"), f"{name} decisions")
        certificate_path, certificate_sha = referenced(
            resolver.get("artifacts", {}).get("certificates"), f"{name} certificates"
        )
        candidate_path, candidate_sha = referenced(
            resolver.get("artifacts", {}).get("candidate"), f"{name} candidates"
        )
        certificates = index(load_jsonl(certificate_path), f"{name} certificates")
        candidates = index(load_jsonl(candidate_path), f"{name} candidates")
        decisions = index(load_jsonl(decisions_path), f"{name} decisions")
        guard_no_known_27b(certificates, f"{name} certificates")
        guard_no_known_27b(candidates, f"{name} candidates")
        guard_no_known_27b(decisions, f"{name} decisions")
        before = index(load_jsonl(current_solver), f"{name} base")
        after = index(load_jsonl(solver_path), f"{name} output")
        validate_solver_rows(before, f"{name} base")
        validate_solver_rows(after, f"{name} output")
        if solver_sha in KNOWN_27B_SOLVER_SHA256:
            raise FinalizeError(f"{name}: output is a known 27B solver")
        if set(before) != set(after):
            raise FinalizeError(f"{name}: task set changed")
        if set(decisions) != set(before):
            raise FinalizeError(f"{name}: decision task set changed")
        replacements = sum(
            before[task_id].get("final_answer") != after[task_id].get("final_answer")
            for task_id in certificates
        )
        confirmations = len(certificates) - replacements
        for task_id, certificate in certificates.items():
            if (
                certificate.get("kind") != "source_entailment"
                or certificate.get("strength") != "strong"
                or certificate.get("status") != "pass"
            ):
                raise FinalizeError(f"{name}:{task_id}: certificate is not strong/pass")
            candidate_answer = str(candidates[task_id].get("final_answer") or "")
            old = source_answer.get(task_id)
            if old is not None and old != candidate_answer:
                raise FinalizeError(f"{name}:{task_id}: source answer conflict")
            source_answer[task_id] = candidate_answer
            source_owner[task_id] = name
        judge_output = judge_manifest.get("output")
        if not isinstance(judge_output, dict) or judge_output.get("sha256") != sha256_file(
            Path(str(judge_output.get("path") or ""))
        ):
            raise FinalizeError(f"{name}: judge manifest output hash mismatch")
        stages.append(
            {
                "name": name,
                "profile": artifact(profile_path),
                "resolver_manifest": artifact(resolver_path),
                "composition_manifest": artifact(composition_path),
                "judge_manifest": artifact(judge_manifest_path),
                "solver": artifact(solver_path, rows=274),
                "decisions": artifact(decisions_path, rows=274),
                "certificates": artifact(certificate_path, rows=len(certificates)),
                "candidates": artifact(candidate_path, rows=len(candidates)),
                "stage_certificate_count": len(certificates),
                "stage_answer_replacements": replacements,
                "stage_answer_confirmations": confirmations,
            }
        )
        certificate_bundle.append(
            {
                "stage": name,
                "path": str(certificate_path),
                "sha256": certificate_sha,
                "rows": len(certificates),
                "candidate_path": str(candidate_path),
                "candidate_sha256": candidate_sha,
            }
        )
        current_solver, current_sha = solver_path, solver_sha
    if current_solver != final_solver_path or current_sha != sha256_file(final_solver_path):
        raise FinalizeError("final solver is not the last stage output")
    if b"Qwen/Qwen3.5-27B" in final_solver_path.read_bytes() or b"Qwen/Qwen3.5-27B" in final_judge_path.read_bytes():
        raise FinalizeError("27B bytes found in final artifacts")

    projection = [
        {
            "task_id": task_id,
            "answer_sha256": hashlib.sha256(source_answer[task_id].encode()).hexdigest(),
            "owner_stage": source_owner[task_id],
        }
        for task_id in sorted(source_owner)
    ]
    final_origins = Counter()
    for task_id, final in final_rows.items():
        if task_id not in source_owner:
            final_origins["qwen35_9b_anchor_passthrough"] += 1
        elif final.get("final_answer") != anchor_rows[task_id].get("final_answer"):
            final_origins["deterministic_official_source_replacement"] += 1
        else:
            final_origins["official_source_confirmation_of_9b_anchor"] += 1
    overall = score.get("overall")
    if not isinstance(overall, dict) or overall.get("n") != 274:
        raise FinalizeError("score overall is malformed")
    score_solver_sha = score.get("provenance", {}).get("solver_results", {}).get("sha256")
    score_judge_sha = score.get("provenance", {}).get("image_judge", {}).get("sha256")
    if score_solver_sha != current_sha or score_judge_sha != sha256_file(final_judge_path):
        raise FinalizeError("score is not bound to the final solver/judge")
    outcome_rows = score.get("task_outcomes")
    if not isinstance(outcome_rows, list):
        raise FinalizeError("score has no task outcomes")
    final_outcomes: dict[str, bool] = {}
    for row in outcome_rows:
        if not isinstance(row, dict):
            raise FinalizeError("score has malformed task outcome")
        task_id = str(row.get("task_id") or "")
        correct = row.get("new_correct")
        if not task_id or task_id in final_outcomes or not isinstance(correct, bool):
            raise FinalizeError("score has malformed/duplicate task outcome")
        final_outcomes[task_id] = correct
    if set(final_outcomes) != set(anchor_rows):
        raise FinalizeError("score outcome task set differs from benchmark")
    comparisons = []
    for raw in args.comparison:
        label, role, comparison_solver, comparison_score = parse_comparison(raw)
        comparisons.append(
            compare_scores(
                label=label,
                role=role,
                solver_path=comparison_solver,
                score_path=comparison_score,
                final_outcomes=final_outcomes,
            )
        )
    anchor_comparisons = [item for item in comparisons if item["role"] == "anchor"]
    adjacent_comparisons = [item for item in comparisons if item["role"] == "adjacent"]
    if len(anchor_comparisons) != 1 or len(adjacent_comparisons) > 1:
        raise FinalizeError("exactly one anchor and at most one adjacent comparison required")
    if Path(anchor_comparisons[0]["solver"]["path"]).resolve() != anchor_path:
        raise FinalizeError("anchor comparison does not reference the frozen anchor")
    benchmark = score.get("provenance", {}).get("benchmark")
    scorer = score.get("provenance", {}).get("scorer")
    aggregate = {
        "schema_version": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "label": args.label,
        "reporting_status": args.reporting_status,
        "model_selection_status": args.model_selection_status,
        "gold_access_during_generation": False,
        "gold_access_during_postgeneration_score": True,
        "inherited_27b_outputs": False,
        "model_closure": [MODEL],
        "upstream_generation_model_closure": [MODEL],
        "answer_origin_closure": sorted(final_origins),
        "anchor": artifact(anchor_path, rows=274),
        "final_solver": artifact(final_solver_path, rows=274),
        "final_image_judge": artifact(final_judge_path, rows=97),
        "score": artifact(score_path),
        "benchmark": benchmark,
        "scorer": scorer,
        "overall": overall,
        "slices": score.get("by_subject"),
        "evaluator_split": score.get("by_source"),
        "comparison_vs_page_rag": {
            "authority": "score_maxim_full274.changes_vs_frozen_page_rag",
            **(score.get("changes_vs_frozen_page_rag") or {}),
        },
        "comparisons": comparisons,
        "comparison_vs_anchor": anchor_comparisons[0],
        "comparison_vs_adjacent": adjacent_comparisons[0] if adjacent_comparisons else None,
        "protocol": score.get("protocol"),
        "source_union": {
            "size": len(projection),
            "sha256": hashlib.sha256(canonical(projection)).hexdigest(),
            "latest_stage_owner_projection": projection,
            "answer_conflicts": 0,
        },
        "stage_counts": [
            {
                key: stage[key]
                for key in (
                    "name",
                    "stage_certificate_count",
                    "stage_answer_replacements",
                    "stage_answer_confirmations",
                )
            }
            for stage in stages
        ],
        "certificate_bundle": certificate_bundle,
        "final_origin_counts": dict(sorted(final_origins.items())),
        "stages": stages,
        "content_projection_contract": {
            "canonicalization": "utf8-json-sort-keys-compact",
            "recursively_excluded_keys": ["created_at_utc", "path"],
            "other_fields_excluded": [],
        },
    }
    aggregate_projection = stable_projection(aggregate)
    aggregate["content_projection"] = aggregate_projection
    aggregate["content_projection_sha256"] = hashlib.sha256(
        canonical(aggregate_projection)
    ).hexdigest()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    aggregate_path = args.output_dir / "aggregate.json"
    aggregate_path.write_bytes(canonical(aggregate) + b"\n")
    freeze = {
        "schema_version": FREEZE_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "label": args.label,
        "aggregate": artifact(aggregate_path),
        "anchor": aggregate["anchor"],
        "final_solver": aggregate["final_solver"],
        "final_image_judge": aggregate["final_image_judge"],
        "score": aggregate["score"],
        "benchmark": benchmark,
        "source_union": aggregate["source_union"],
        "model_closure": aggregate["model_closure"],
        "upstream_generation_model_closure": aggregate[
            "upstream_generation_model_closure"
        ],
        "answer_origin_closure": aggregate["answer_origin_closure"],
        "inherited_27b_outputs": False,
        "profile_bound_reissue": True,
        "content_projection_sha256": aggregate["content_projection_sha256"],
        "freeze_projection_contract": {
            "canonicalization": "utf8-json-sort-keys-compact",
            "recursively_excluded_keys": ["created_at_utc", "path"],
            "aggregate_binding": "content_projection_sha256",
        },
    }
    freeze_projection = stable_projection(freeze)
    # The byte SHA of aggregate.json changes with its intentionally excluded
    # created_at/path fields. Bind the freeze projection to the aggregate's
    # stable content projection instead of that volatile byte digest.
    freeze_projection["aggregate"] = {
        "content_projection_sha256": aggregate["content_projection_sha256"]
    }
    freeze["freeze_projection"] = freeze_projection
    freeze["freeze_projection_sha256"] = hashlib.sha256(
        canonical(freeze_projection)
    ).hexdigest()
    freeze_path = args.output_dir / "freeze_manifest.json"
    freeze_path.write_bytes(canonical(freeze) + b"\n")
    report = (
        f"# {args.label}\n\n"
        f"- результат: **{overall['new_correct']}/{overall['n']} = {overall['new_accuracy']:.6f}**\n"
        f"- anchor: `{args.model_selection_status}`\n"
        f"- source union: {len(projection)} задач; конфликтов ответов: 0\n"
        f"- model closure: только `{MODEL}`; замены имеют детерминированное source-origin\n"
        f"- inherited 27B outputs: `false`\n"
        f"- freeze SHA-256: `{sha256_file(freeze_path)}`\n"
    )
    (args.output_dir / "REPORT_RU.md").write_text(report, encoding="utf-8", newline="\n")
    return {**aggregate, "freeze": artifact(freeze_path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--reporting-status", required=True)
    parser.add_argument("--model-selection-status", required=True)
    parser.add_argument("--anchor-solver", type=Path, required=True)
    parser.add_argument("--expected-anchor-sha256", required=True)
    parser.add_argument("--stage", action="append", default=[], required=True)
    parser.add_argument(
        "--comparison",
        action="append",
        default=[],
        required=True,
        help="LABEL:anchor|adjacent=SOLVER,SCORE_JSON",
    )
    parser.add_argument("--final-solver", type=Path, required=True)
    parser.add_argument("--final-image-judge", type=Path, required=True)
    parser.add_argument("--score-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = finalize(args)
    except (FinalizeError, OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}")
        return 2
    print(json.dumps({"label": result["label"], "overall": result["overall"], "freeze": result["freeze"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
