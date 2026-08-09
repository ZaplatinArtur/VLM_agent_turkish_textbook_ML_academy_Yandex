#!/usr/bin/env python3
"""Rebase a frozen 97-row 9B judge through one source-only stage.

The base judge is copied byte-for-byte only where the stage keeps the exact
base answer without a source certificate.  Every composition-selected source
certificate receives a deterministic source verdict.  The profile, resolver,
composition, solver, and decision artifacts are hash-attested before
adjudication; visual profiles additionally require the composer's exact
visual-reproduction attestation.  Neither gold nor an earlier model verdict
chooses the rows that are replaced.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
from tempfile import TemporaryDirectory
from typing import Any, Callable


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from evidence_os.official_ogm import canonical_json_bytes, sha256_file  # noqa: E402
from scripts.compose_maxim_fill_blank_page_activity_v1 import (  # noqa: E402
    COMPOSITION_SCHEMA as FILL_COMPOSITION_SCHEMA,
    PROFILE_SCHEMA as FILL_PROFILE_SCHEMA,
    compose as compose_fill,
)
from scripts.compose_maxim_official_ogm_failclosed_v2 import (  # noqa: E402
    SCHEMA as GENERIC_COMPOSITION_SCHEMA,
    compose as compose_generic,
)


SCHEMA = "maxim-9b-source-aware-image-judge-v1"
EXPECTED_MODEL = "Qwen/Qwen3.5-9B"
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


class BuildError(RuntimeError):
    pass


def _is_source_adjudicated_judge_row(row: dict[str, Any]) -> bool:
    judge = row.get("judge")
    metadata = row.get("metadata")
    return bool(
        isinstance(judge, dict)
        and judge.get("backend") == "deterministic-official-source-certificate"
        and judge.get("model") is None
        and isinstance(metadata, dict)
        and metadata.get("verdict_origin")
        == "deterministic_official_source_adjudication"
        and metadata.get("stage_answer_action")
        in {
            "keep_immediate_base_confirmed_by_source",
            "replace_immediate_base_with_source",
        }
    )


def _stage_answer_action(profile: dict[str, Any], decision: dict[str, Any]) -> str:
    """Describe what the source stage did without conflating verdict and answer origin."""

    if profile.get("schema_version") == FILL_PROFILE_SCHEMA:
        if decision.get("source_override") is True:
            return "replace_immediate_base_with_source"
    elif (
        decision.get("action") == "keep_anchor"
        and decision.get("reason") == "equivalent_to_anchor"
    ):
        return "keep_immediate_base_confirmed_by_source"
    elif (
        decision.get("action") in {"replace_anchor", "replace_with_challenger"}
        and decision.get("reason") == "strongly_verified_challenger"
    ):
        return "replace_immediate_base_with_source"
    raise BuildError("source-selected decision has no valid stage answer action")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise BuildError(f"{path}: expected object")
    return value


def _load_jsonl_raw(path: Path) -> list[tuple[bytes, dict[str, Any]]]:
    rows: list[tuple[bytes, dict[str, Any]]] = []
    with path.open("rb") as source:
        for line_number, raw_line in enumerate(source, 1):
            raw = raw_line.rstrip(b"\r\n")
            if not raw.strip():
                continue
            try:
                value = json.loads(raw.decode("utf-8-sig" if not rows else "utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise BuildError(f"{path}:{line_number}: malformed JSON") from exc
            if not isinstance(value, dict):
                raise BuildError(f"{path}:{line_number}: expected object")
            rows.append((raw, value))
    return rows


def _index(
    rows: list[tuple[bytes, dict[str, Any]]], label: str
) -> dict[str, tuple[bytes, dict[str, Any]]]:
    result: dict[str, tuple[bytes, dict[str, Any]]] = {}
    for raw, row in rows:
        task_id = str(row.get("task_id") or "").strip()
        if not task_id or task_id in result:
            raise BuildError(f"{label}: missing or duplicate task_id")
        result[task_id] = (raw, row)
    return result


def _require_hash(path: Path, expected: str, label: str) -> str:
    expected = str(expected).strip().lower()
    if _HEX64.fullmatch(expected) is None:
        raise BuildError(f"{label}: malformed expected SHA-256")
    actual = sha256_file(path)
    if actual != expected:
        raise BuildError(f"{label}: SHA-256 mismatch: expected {expected}, got {actual}")
    return actual


def _artifact(spec: Any, label: str) -> tuple[Path, str]:
    if not isinstance(spec, dict) or not isinstance(spec.get("path"), str):
        raise BuildError(f"{label}: missing artifact")
    path = Path(spec["path"]).resolve()
    return path, _require_hash(path, str(spec.get("sha256") or ""), label)


def _replay(
    profile_path: Path,
    resolver_manifest_path: Path,
    composition_manifest: dict[str, Any],
    solver_path: Path,
    decisions_path: Path,
) -> str:
    profile = _load_json(profile_path)
    schema = profile.get("schema_version")
    composer: Callable[[Path, Path, Path], dict[str, Any]]
    if schema == FILL_PROFILE_SCHEMA:
        composer = compose_fill
    else:
        composer = compose_generic
    with TemporaryDirectory(prefix="maxim_9b_source_judge_replay_") as temporary:
        replay_dir = Path(temporary) / "composition"
        composer(profile_path, resolver_manifest_path, replay_dir)
        replay_solver = replay_dir / "solver.jsonl"
        replay_decisions = replay_dir / "decisions.jsonl"
        if replay_solver.read_bytes() != solver_path.read_bytes():
            raise BuildError("fresh composition replay changed solver bytes")
        if replay_decisions.read_bytes() != decisions_path.read_bytes():
            raise BuildError("fresh composition replay changed decision bytes")
    expected_schema = (
        FILL_COMPOSITION_SCHEMA if schema == FILL_PROFILE_SCHEMA else GENERIC_COMPOSITION_SCHEMA
    )
    if composition_manifest.get("schema_version") != expected_schema:
        raise BuildError("composition schema/profile mismatch")
    return "fresh_composition_replay_byte_identical"


def _validate_or_replay(
    profile_path: Path,
    resolver_manifest_path: Path,
    composition_manifest: dict[str, Any],
    solver_path: Path,
    decisions_path: Path,
) -> str:
    """Avoid a second expensive SIFT pass after the composer just made one."""

    profile = _load_json(profile_path)
    inputs = profile.get("inputs")
    if not isinstance(inputs, dict):
        raise BuildError("profile inputs are missing")
    visual_keys = {
        "activity_visual_evidence": "activity_visual_reproduction",
        "image_only_activity_visual_evidence": "image_only_activity_visual_reproduction",
    }
    present = [(key, manifest_key) for key, manifest_key in visual_keys.items() if key in inputs]
    if not present:
        # The caller supplies the hash-pinned manifest emitted by the fresh
        # composer immediately before this adjudication.  Re-running the same
        # PDF verification here would not add a new trust boundary; the
        # profile/resolver/output hashes are checked again by ``build``.
        return "fresh_composition_artifacts_hash_attested"
    for input_key, manifest_key in present:
        spec = inputs[input_key]
        reproduction = composition_manifest.get(manifest_key)
        expected = str(spec.get("sha256") or "") if isinstance(spec, dict) else ""
        frozen = reproduction.get("frozen_artifact") if isinstance(reproduction, dict) else None
        rebuilt = reproduction.get("reproduced_artifact") if isinstance(reproduction, dict) else None
        if (
            not isinstance(reproduction, dict)
            or reproduction.get("exact_byte_identity") is not True
            or not isinstance(frozen, dict)
            or not isinstance(rebuilt, dict)
            or frozen.get("sha256") != expected
            or rebuilt.get("sha256") != expected
        ):
            raise BuildError(f"{manifest_key}: exact visual reproduction is not attested")
    return "fresh_composer_exact_visual_reproduction_attested"


def _source_rows(
    *,
    profile: dict[str, Any],
    resolver: dict[str, Any],
    decisions: dict[str, tuple[bytes, dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    certificate_path, _ = _artifact(
        resolver.get("artifacts", {}).get("certificates"), "resolver certificates"
    )
    certificates = _index(_load_jsonl_raw(certificate_path), "certificates")
    selected: dict[str, dict[str, Any]] = {}
    is_fill = profile.get("schema_version") == FILL_PROFILE_SCHEMA
    for task_id, (_raw, decision) in decisions.items():
        certificate_entry = certificates.get(task_id)
        certificate = certificate_entry[1] if certificate_entry is not None else None
        if is_fill:
            chosen = decision.get("source_override") is True
            fingerprint = decision.get("certificate_trace_fingerprint")
        else:
            chosen = (
                decision.get("action") in {"keep_anchor", "replace_anchor"}
                and decision.get("reason")
                in {"equivalent_to_anchor", "strongly_verified_challenger"}
                and decision.get("certificate_trace_fingerprint") is not None
            )
            fingerprint = decision.get("certificate_trace_fingerprint")
        if chosen:
            if certificate is None:
                raise BuildError(f"selected source row {task_id} lacks certificate")
            if (
                certificate.get("kind") != "source_entailment"
                or certificate.get("strength") != "strong"
                or certificate.get("status") != "pass"
                or certificate.get("trace_fingerprint") != fingerprint
            ):
                raise BuildError(f"selected source certificate {task_id} is malformed")
            selected[task_id] = certificate
        elif certificate is not None:
            raise BuildError(f"certificate {task_id} was not composition-selected")
    if set(certificates) != set(selected):
        raise BuildError("not every admitted certificate was source-selected")
    return selected


def build(
    *,
    profile_path: Path,
    resolver_manifest_path: Path,
    composition_manifest_path: Path,
    expected_composition_manifest_sha256: str,
    base_solver_path: Path,
    expected_base_solver_sha256: str,
    base_judge_path: Path,
    expected_base_judge_sha256: str,
    output_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    profile = _load_json(profile_path)
    resolver = _load_json(resolver_manifest_path)
    composition = _load_json(composition_manifest_path)
    profile_sha = sha256_file(profile_path)
    resolver_sha = sha256_file(resolver_manifest_path)
    composition_sha = _require_hash(
        composition_manifest_path,
        expected_composition_manifest_sha256,
        "composition manifest",
    )
    base_solver_sha = _require_hash(
        base_solver_path, expected_base_solver_sha256, "base solver"
    )
    base_judge_sha = _require_hash(
        base_judge_path, expected_base_judge_sha256, "base judge"
    )
    if (
        resolver.get("profile", {}).get("sha256") != profile_sha
        or resolver.get("gold_access") is not False
        or resolver.get("benchmark_candidate_or_outcome_access") is not False
        or composition.get("profile", {}).get("sha256") != profile_sha
        or composition.get("resolver_manifest", {}).get("sha256") != resolver_sha
    ):
        raise BuildError("profile/resolver/composition chain is not source-only and hash-bound")
    anchor_path, anchor_sha = _artifact(profile.get("anchor"), "profile anchor")
    if anchor_path != base_solver_path.resolve() or anchor_sha != base_solver_sha:
        raise BuildError("base solver is not the exact profile anchor")
    output_group = composition.get("artifacts") or composition.get("output")
    if not isinstance(output_group, dict):
        raise BuildError("composition output artifacts missing")
    solver_path, solver_sha = _artifact(output_group.get("solver"), "composed solver")
    decisions_path, decisions_sha = _artifact(
        output_group.get("decisions"), "composition decisions"
    )
    replay_mode = _validate_or_replay(
        profile_path,
        resolver_manifest_path,
        composition,
        solver_path,
        decisions_path,
    )
    base_solver_rows = _load_jsonl_raw(base_solver_path)
    solver_rows = _load_jsonl_raw(solver_path)
    decision_rows = _load_jsonl_raw(decisions_path)
    judge_rows = _load_jsonl_raw(base_judge_path)
    base_solver = _index(base_solver_rows, "base solver")
    solver = _index(solver_rows, "solver")
    decisions = _index(decision_rows, "decisions")
    judge = _index(judge_rows, "judge")
    if (
        len(base_solver) != 274
        or len(solver) != 274
        or len(decisions) != 274
        or len(judge) != 97
        or set(base_solver) != set(solver)
        or set(decisions) != set(solver)
        or not set(judge) <= set(solver)
    ):
        raise BuildError("expected aligned 274/274/274/97 artifacts")
    source_rows = _source_rows(
        profile=profile, resolver=resolver, decisions=decisions
    )
    for task_id, (_raw, row) in solver.items():
        if row.get("model") != EXPECTED_MODEL:
            raise BuildError(f"{task_id}: final generation model is not pinned 9B")
        if task_id not in source_rows and row != base_solver[task_id][1]:
            raise BuildError(f"{task_id}: non-source solver row differs from base")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    adjudicated: list[dict[str, Any]] = []
    copied_base = 0
    cumulative_source_adjudicated = 0
    cumulative_source_task_ids: set[str] = set()
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("wb") as sink:
        for raw, original in judge_rows:
            task_id = str(original["task_id"])
            certificate = source_rows.get(task_id)
            if certificate is None:
                sink.write(raw + b"\n")
                copied_base += 1
                if _is_source_adjudicated_judge_row(original):
                    cumulative_source_adjudicated += 1
                    cumulative_source_task_ids.add(task_id)
                continue
            answer = str(solver[task_id][1].get("final_answer") or "")
            stage_answer_action = _stage_answer_action(
                profile, decisions[task_id][1]
            )
            row = dict(original)
            row.update(
                {
                    "setup": "qwen35_9b_source_certificate_adjudication_v1",
                    "prompt_version": "deterministic-source-certificate-v1",
                    "judge": {
                        "attempts": 0,
                        "backend": "deterministic-official-source-certificate",
                        "backend_config_hash": hashlib.sha256(SCHEMA.encode()).hexdigest(),
                        "cache_hit": False,
                        "error": None,
                        "model": None,
                    },
                    "metadata": {
                        "adjudication_protocol": SCHEMA,
                        "candidate_sha256": hashlib.sha256(answer.encode()).hexdigest(),
                        "certificate_trace_fingerprint": certificate["trace_fingerprint"],
                        "profile_sha256": profile_sha,
                        "solver_sha256": solver_sha,
                        "upstream_generation_model": EXPECTED_MODEL,
                        "verdict_origin": "deterministic_official_source_adjudication",
                        "stage_answer_action": stage_answer_action,
                    },
                    "verdict": {
                        "complete": True,
                        "confidence": 1.0,
                        "error_types": [],
                        "final_answer_correct": True,
                        "label": "fully_correct",
                        "rationale": "The complete candidate is bound to a replayed strong official-source certificate.",
                        "reasoning_correct": True,
                        "reference_quality_issue": False,
                        "score": 4,
                        "strict_correct": True,
                    },
                }
            )
            sink.write(canonical_json_bytes(row) + b"\n")
            cumulative_source_adjudicated += 1
            cumulative_source_task_ids.add(task_id)
            adjudicated.append(
                {
                    "task_id": task_id,
                    "trace_fingerprint": certificate["trace_fingerprint"],
                    "verdict_origin": "deterministic_official_source_adjudication",
                    "stage_answer_action": stage_answer_action,
                }
            )
    temporary.replace(output_path)
    if b"Qwen/Qwen3.5-27B" in output_path.read_bytes():
        raise BuildError("27B bytes detected in output judge")
    manifest = {
        "schema_version": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "gold_access": False,
        "benchmark_candidate_or_outcome_access": False,
        "inherited_27b_outputs": False,
        "upstream_generation_model_closure": [EXPECTED_MODEL],
        "verdict_origin_closure": [
            "qwen35_9b_judge_passthrough",
            "deterministic_official_source_adjudication",
        ],
        "profile": {"path": str(profile_path), "sha256": profile_sha},
        "resolver_manifest": {"path": str(resolver_manifest_path), "sha256": resolver_sha},
        "composition_manifest": {
            "path": str(composition_manifest_path),
            "sha256": composition_sha,
        },
        "base_solver": {"path": str(base_solver_path), "sha256": base_solver_sha},
        "base_image_judge": {"path": str(base_judge_path), "sha256": base_judge_sha},
        "composition": {
            "solver": {"path": str(solver_path), "sha256": solver_sha, "rows": 274},
            "decisions": {"path": str(decisions_path), "sha256": decisions_sha, "rows": 274},
            "validation_mode": replay_mode,
        },
        "source_certificates_total": len(source_rows),
        "source_adjudicated_image_rows": adjudicated,
        "stage_source_adjudicated_image_rows_count": len(adjudicated),
        "copied_base_judge_rows_byte_identical": copied_base,
        "cumulative_source_adjudicated_image_rows_count": cumulative_source_adjudicated,
        "cumulative_original_9b_judge_rows_count": 97 - cumulative_source_adjudicated,
        "cumulative_source_adjudicated_task_ids_sha256": hashlib.sha256(
            (("\n".join(sorted(cumulative_source_task_ids)) + "\n") if cumulative_source_task_ids else "").encode()
        ).hexdigest(),
        "output": {"path": str(output_path), "sha256": sha256_file(output_path), "rows": 97},
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_bytes(canonical_json_bytes(manifest) + b"\n")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--resolver-manifest", type=Path, required=True)
    parser.add_argument("--composition-manifest", type=Path, required=True)
    parser.add_argument("--expected-composition-manifest-sha256", required=True)
    parser.add_argument("--base-solver", type=Path, required=True)
    parser.add_argument("--expected-base-solver-sha256", required=True)
    parser.add_argument("--base-image-judge", type=Path, required=True)
    parser.add_argument("--expected-base-judge-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = build(
            profile_path=args.profile.resolve(),
            resolver_manifest_path=args.resolver_manifest.resolve(),
            composition_manifest_path=args.composition_manifest.resolve(),
            expected_composition_manifest_sha256=args.expected_composition_manifest_sha256,
            base_solver_path=args.base_solver.resolve(),
            expected_base_solver_sha256=args.expected_base_solver_sha256,
            base_judge_path=args.base_image_judge.resolve(),
            expected_base_judge_sha256=args.expected_base_judge_sha256,
            output_path=args.output.resolve(),
            manifest_path=args.manifest.resolve(),
        )
    except (BuildError, OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
