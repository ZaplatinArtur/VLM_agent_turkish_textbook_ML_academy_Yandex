#!/usr/bin/env python3
"""Compose a frozen, evidence-gated solver without inference-time labels.

The command has two deliberately separate phases:

1. candidate runs are aligned by ``task_id`` and projected to ID-free inputs;
2. decisions are made, after which IDs are reattached only for output/audit.

No score, judge, reference-answer, or outcome artifact is accepted by the
production boundary.  Cached legacy observations remain weak evidence; an
override requires an explicitly bound strong certificate run.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evidence_os.adapters import (  # noqa: E402
    adapt_legacy_solver_payload,
    certificate_from_record,
    problem_from_public_payload,
)
from evidence_os.certificates import (  # noqa: E402
    answer_fingerprint,
    certificate_fingerprint,
    input_fingerprint,
)
from evidence_os.contracts import CertificateKind, FrozenProfile, InferenceBundle  # noqa: E402
from evidence_os.ingest import (  # noqa: E402
    AlignmentError,
    CandidateRun,
    align_candidate_runs,
    load_candidate_jsonl,
)
from evidence_os.policy import EvidencePolicy  # noqa: E402


SCHEMA = "maxim-evidence-os-composition-v1"
LEDGER_SCHEMA = "maxim-evidence-os-ledger-v1"

SOLVER_POLICY_FIELDS = frozenset(
    {
        "answer",
        "condition",
        "error",
        "final_answer",
        "forced_answer",
        "generation",
        "model",
        "prediction",
        "prompt_version",
    }
)
PUBLIC_POLICY_FIELDS = frozenset(
    {
        "answer_format",
        "answer_type",
        "grade",
        "images",
        "question",
        "question_images",
        "statement",
        "subject",
    }
)
CERTIFICATE_POLICY_FIELDS = frozenset(
    {
        "answer_bound",
        "answer_fingerprint",
        "claim_coverage",
        "contradiction_count",
        "deterministic_checks",
        "input_bound",
        "input_fingerprint",
        "kind",
        "status",
        "strength",
        "trace",
        "trace_fingerprint",
        "verifier",
    }
)


class CompositionError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_named_paths(values: Sequence[str], option: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise CompositionError(f"{option} expects NAME=PATH, got {value!r}")
        name, raw_path = value.split("=", 1)
        name = name.strip()
        if not name or not raw_path.strip():
            raise CompositionError(f"{option} expects NAME=PATH, got {value!r}")
        if name in {"anchor", "__public__"} or name.startswith("__certificate__:"):
            raise CompositionError(f"reserved candidate name: {name!r}")
        if name in result:
            raise CompositionError(f"duplicate candidate name: {name!r}")
        result[name] = Path(raw_path).expanduser().resolve()
    return result


def _load_raw_rows(path: Path) -> tuple[tuple[str, ...], dict[str, dict[str, Any]], dict[str, bytes]]:
    order: list[str] = []
    rows: dict[str, dict[str, Any]] = {}
    lines: dict[str, bytes] = {}
    for line_number, raw_line in enumerate(path.read_bytes().splitlines(keepends=True), 1):
        if not raw_line.strip():
            continue
        try:
            row = json.loads(raw_line.decode("utf-8-sig"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CompositionError(f"invalid JSONL {path}:{line_number}: {exc}") from exc
        if not isinstance(row, dict):
            raise CompositionError(f"{path}:{line_number}: expected object")
        task_id = str(row.get("task_id") or "").strip()
        if not task_id or task_id in rows:
            raise CompositionError(f"{path}:{line_number}: missing or duplicate task_id")
        order.append(task_id)
        rows[task_id] = row
        lines[task_id] = raw_line if raw_line.endswith((b"\n", b"\r")) else raw_line + b"\n"
    if not order:
        raise CompositionError(f"empty solver artifact: {path}")
    return tuple(order), rows, lines


def _solver_output_signature(row: Mapping[str, Any]) -> tuple[str, str | None, bool]:
    """Return the output fields that can change frozen-scorer semantics."""

    # The frozen scorer reads only ``final_answer``.  A fallback value in
    # ``prediction``/``answer`` must never make a row eligible for override.
    answer = str(row.get("final_answer") or "").strip()
    error_value = row.get("error")
    error = str(error_value).strip() if error_value not in (None, "") else None
    return answer_fingerprint(answer), error, row.get("forced_answer") is True


def _validate_raw_public_binding(
    *,
    name: str,
    raw_rows: Mapping[str, Mapping[str, Any]],
    public_rows: Mapping[str, Mapping[str, Any]],
) -> None:
    """Prove that output material matches the candidate seen by the policy."""

    if set(raw_rows) != set(public_rows):
        raise CompositionError(f"raw/public task-set mismatch for {name!r}")
    for task_id in sorted(raw_rows):
        raw_final = str(raw_rows[task_id].get("final_answer") or "").strip()
        public_final = str(public_rows[task_id].get("final_answer") or "").strip()
        if not raw_final or not public_final:
            raise CompositionError(
                f"scorer-visible final_answer is missing for {name!r} at task {task_id!r}"
            )
        if _solver_output_signature(raw_rows[task_id]) != _solver_output_signature(
            public_rows[task_id]
        ):
            raise CompositionError(
                f"raw/public output mismatch for {name!r} at task {task_id!r}"
            )


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(value), ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as sink:
        for row in rows:
            sink.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    temporary.replace(path)


def _load_run(path: Path, name: str, fields: frozenset[str]) -> CandidateRun:
    return load_candidate_jsonl(path, name=name, policy_fields=fields)


def _validate_frozen_profile(
    path: Path,
    *,
    args: argparse.Namespace,
    anchor_raw_sha: str,
    anchor_public_sha: str,
    public_tasks_sha: str,
    candidate_public_shas: Mapping[str, str],
    raw_candidate_shas: Mapping[str, str],
    certificate_shas: Mapping[str, str],
) -> dict[str, Any]:
    try:
        profile = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CompositionError(f"invalid frozen profile JSON: {exc}") from exc
    if not isinstance(profile, dict):
        raise CompositionError("frozen profile must be one JSON object")
    if profile.get("schema_version") != "maxim-evidence-os-frozen-profile-v1":
        raise CompositionError("frozen profile schema mismatch")
    if profile.get("profile_name") != args.profile_name:
        raise CompositionError("frozen profile name mismatch")
    if profile.get("expected_rows") != args.expected_rows:
        raise CompositionError("frozen profile row-count mismatch")

    anchor = profile.get("anchor")
    public = profile.get("public_tasks")
    policy = profile.get("policy")
    modules = profile.get("legacy_modules")
    certificate_inputs = profile.get("certificate_inputs")
    if not all(
        isinstance(item, dict)
        for item in (anchor, public, policy, modules, certificate_inputs)
    ):
        raise CompositionError("frozen profile sections are missing")
    assert isinstance(anchor, dict) and isinstance(public, dict)
    assert isinstance(policy, dict) and isinstance(modules, dict)
    assert isinstance(certificate_inputs, dict)
    expected = {
        "anchor raw": (anchor.get("sha256"), anchor_raw_sha),
        "anchor public": (anchor.get("public_projection_sha256"), anchor_public_sha),
        "public tasks": (public.get("sha256"), public_tasks_sha),
    }
    for label, (declared, actual) in expected.items():
        if declared != actual:
            raise CompositionError(
                f"frozen profile {label} SHA mismatch: expected={declared}, actual={actual}"
            )
    if set(candidate_public_shas) != set(modules):
        raise CompositionError("candidate names differ from frozen profile modules")
    for name, actual in candidate_public_shas.items():
        module = modules.get(name)
        if not isinstance(module, dict):
            raise CompositionError(f"invalid frozen module profile: {name}")
        if module.get("public_projection_sha256") != actual:
            raise CompositionError(f"frozen profile candidate SHA mismatch: {name}")
        if module.get("mode") not in {"shadow", "perception_only", "evidence_gated"}:
            raise CompositionError(f"invalid frozen module mode: {name}")
    for name, actual in raw_candidate_shas.items():
        module = modules.get(name)
        if not isinstance(module, dict) or module.get("sha256") != actual:
            raise CompositionError(f"frozen profile raw candidate SHA mismatch: {name}")

    if set(certificate_shas) != set(certificate_inputs):
        raise CompositionError("certificate inputs differ from frozen profile bindings")
    allowed_profile_kinds_raw = profile.get("allowed_strong_certificate_kinds")
    if not isinstance(allowed_profile_kinds_raw, list) or not allowed_profile_kinds_raw:
        raise CompositionError("frozen profile has no allowed strong certificate kinds")
    try:
        allowed_profile_kinds = {
            CertificateKind(str(value)).value for value in allowed_profile_kinds_raw
        }
    except ValueError as exc:
        raise CompositionError(f"invalid strong certificate kind: {exc}") from exc
    for name, actual in certificate_shas.items():
        binding = certificate_inputs.get(name)
        module = modules.get(name)
        if not isinstance(binding, dict) or not isinstance(module, dict):
            raise CompositionError(f"invalid frozen certificate binding: {name}")
        if module.get("mode") != "evidence_gated":
            raise CompositionError(f"certificate module is not evidence-gated: {name}")
        if name not in raw_candidate_shas:
            raise CompositionError(f"certificate module has no bound raw output: {name}")
        if binding.get("sha256") != actual:
            raise CompositionError(f"frozen profile certificate SHA mismatch: {name}")
        verifiers = binding.get("allowed_verifiers")
        kinds = binding.get("allowed_kinds")
        if (
            not isinstance(verifiers, list)
            or not verifiers
            or any(not isinstance(value, str) or not value.strip() for value in verifiers)
        ):
            raise CompositionError(f"certificate verifier allowlist is invalid: {name}")
        if not isinstance(kinds, list) or not kinds:
            raise CompositionError(f"certificate kind allowlist is invalid: {name}")
        try:
            allowed_binding_kinds = {CertificateKind(str(value)).value for value in kinds}
        except ValueError as exc:
            raise CompositionError(f"invalid certificate kind for {name}: {exc}") from exc
        if not allowed_binding_kinds <= allowed_profile_kinds:
            raise CompositionError(f"certificate kind exceeds frozen profile: {name}")
        if binding.get("require_inline_trace") is not True:
            raise CompositionError(f"certificate inline trace is not required: {name}")
    thresholds = {
        "min_claim_coverage": args.min_claim_coverage,
        "min_deterministic_checks": args.min_deterministic_checks,
        "min_independent_verifiers": args.min_independent_verifiers,
    }
    for key, actual in thresholds.items():
        if policy.get(key) != actual:
            raise CompositionError(f"frozen profile policy mismatch: {key}")
    if policy.get("task_id_source_url_sha_or_order_features_allowed") is not False:
        raise CompositionError("frozen profile does not forbid identity features")
    if policy.get("external_certificates_require_profile_binding") is not True:
        raise CompositionError("frozen profile does not bind external certificates")
    if policy.get("certificate_trace_content_required") is not True:
        raise CompositionError("frozen profile does not require certificate trace content")
    return profile


def compose(args: argparse.Namespace) -> dict[str, Any]:
    anchor_path = args.anchor_solver.resolve()
    anchor_public_path = args.anchor_public.resolve()
    public_path = args.public_tasks.resolve()
    output_dir = args.output_dir.resolve()
    candidate_paths = _parse_named_paths(args.candidate, "--candidate")
    raw_candidate_paths = _parse_named_paths(args.raw_candidate, "--raw-candidate")
    certificate_paths = _parse_named_paths(args.certificate, "--certificate")
    unknown_raw_candidates = sorted(set(raw_candidate_paths) - set(candidate_paths))
    if unknown_raw_candidates:
        raise CompositionError(
            "raw candidate run(s) have no matching public candidate: "
            + ", ".join(unknown_raw_candidates)
        )
    unknown_certificates = sorted(set(certificate_paths) - set(candidate_paths))
    if unknown_certificates:
        raise CompositionError(
            "certificate run(s) have no matching candidate: " + ", ".join(unknown_certificates)
        )
    if not args.profile_json:
        raise CompositionError("--profile-json is required at the production boundary")
    missing_raw_for_certificates = sorted(set(certificate_paths) - set(raw_candidate_paths))
    if missing_raw_for_certificates:
        raise CompositionError(
            "certificate run(s) have no bound raw output: "
            + ", ".join(missing_raw_for_certificates)
        )

    actual_anchor_sha = _sha256(anchor_path)
    if args.anchor_sha256 and actual_anchor_sha != args.anchor_sha256:
        raise CompositionError(
            f"anchor SHA mismatch: expected={args.anchor_sha256}, actual={actual_anchor_sha}"
        )

    anchor_public_sha = _sha256(anchor_public_path)
    public_tasks_sha = _sha256(public_path)
    candidate_public_shas = {
        name: _sha256(path) for name, path in candidate_paths.items()
    }
    raw_candidate_shas = {
        name: _sha256(path) for name, path in raw_candidate_paths.items()
    }
    certificate_shas = {
        name: _sha256(path) for name, path in certificate_paths.items()
    }
    frozen_profile_path = args.profile_json.resolve()
    frozen_profile = _validate_frozen_profile(
        frozen_profile_path,
        args=args,
        anchor_raw_sha=actual_anchor_sha,
        anchor_public_sha=anchor_public_sha,
        public_tasks_sha=public_tasks_sha,
        candidate_public_shas=candidate_public_shas,
        raw_candidate_shas=raw_candidate_shas,
        certificate_shas=certificate_shas,
    )

    # The raw anchor is output material only.  Inference reads a separately
    # staged, hash-bound public projection with no provenance/reasoning fields.
    anchor_run = _load_run(anchor_public_path, "anchor", SOLVER_POLICY_FIELDS)
    public_run = _load_run(public_path, "__public__", PUBLIC_POLICY_FIELDS)
    candidate_runs = {
        name: _load_run(path, name, SOLVER_POLICY_FIELDS)
        for name, path in candidate_paths.items()
    }
    certificate_runs = {
        name: _load_run(
            path,
            f"__certificate__:{name}",
            CERTIFICATE_POLICY_FIELDS,
        )
        for name, path in certificate_paths.items()
    }
    runs = [anchor_run, public_run, *candidate_runs.values(), *certificate_runs.values()]
    batch = align_candidate_runs(runs)
    if len(batch.cases) != args.expected_rows:
        raise CompositionError(
            f"expected {args.expected_rows} aligned rows, received {len(batch.cases)}"
        )

    anchor_order, anchor_rows, anchor_lines = _load_raw_rows(anchor_path)
    _, anchor_public_rows, _ = _load_raw_rows(anchor_public_path)
    _validate_raw_public_binding(
        name="anchor",
        raw_rows=anchor_rows,
        public_rows=anchor_public_rows,
    )
    raw_sources = {"anchor": (anchor_order, anchor_lines)}
    for name, path in raw_candidate_paths.items():
        raw_order, raw_rows, raw_lines = _load_raw_rows(path)
        _, candidate_public_rows, _ = _load_raw_rows(candidate_paths[name])
        _validate_raw_public_binding(
            name=name,
            raw_rows=raw_rows,
            public_rows=candidate_public_rows,
        )
        raw_sources[name] = (raw_order, raw_lines)

    allowed_strong_kinds = frozenset(
        CertificateKind(str(value))
        for value in frozen_profile["allowed_strong_certificate_kinds"]
    )
    profile = FrozenProfile(
        name=args.profile_name,
        allowed_strong_kinds=allowed_strong_kinds,
        min_claim_coverage=args.min_claim_coverage,
        min_deterministic_checks=args.min_deterministic_checks,
        min_independent_verifiers=args.min_independent_verifiers,
    )
    policy = EvidencePolicy()
    policy_outputs: list[dict[str, Any]] = []
    problem_fingerprints: list[str] = []
    image_binding_complete_count = 0
    image_root = args.image_root.resolve() if args.image_root else None

    for case in batch.cases:
        public_payload = case.candidates["__public__"]
        problem, image_binding_complete = problem_from_public_payload(
            public_payload,
            image_root=image_root,
        )
        problem_fingerprints.append(input_fingerprint(problem))
        image_binding_complete_count += int(image_binding_complete)
        anchor_adapted = adapt_legacy_solver_payload(
            "anchor",
            case.candidates["anchor"],
            problem,
            image_binding_complete=image_binding_complete,
        )
        adapted = []
        for name in sorted(candidate_runs):
            result = adapt_legacy_solver_payload(
                name,
                case.candidates[name],
                problem,
                image_binding_complete=image_binding_complete,
            )
            if name in certificate_runs:
                certificate_payload = case.candidates[f"__certificate__:{name}"]
                trust = frozen_profile["certificate_inputs"][name]
                certificate = certificate_from_record(
                    problem,
                    result.candidate,
                    certificate_payload,
                    allowed_verifiers=frozenset(trust["allowed_verifiers"]),
                    allowed_kinds=frozenset(
                        CertificateKind(str(value)) for value in trust["allowed_kinds"]
                    ),
                    require_inline_trace=trust["require_inline_trace"] is True,
                )
                result = replace(
                    result,
                    candidate=replace(
                        result.candidate,
                        certificates=result.candidate.certificates + (certificate,),
                    ),
                )
            adapted.append(result)

        decision = policy.decide(
            InferenceBundle(
                problem=problem,
                anchor=anchor_adapted.candidate,
                candidates=tuple(item.candidate for item in adapted),
            ),
            profile,
        )
        policy_outputs.append(
            {
                "schema_version": LEDGER_SCHEMA,
                "action": decision.action.value,
                "reason": decision.reason.value,
                "selected_source": decision.selected.source,
                "image_binding_complete": image_binding_complete,
                "candidate_audits": [
                    {
                        "source": item.audit.source,
                        "certificate_observations": item.audit.certificate_observations,
                        "rejection_reasons": item.audit.rejection_reasons,
                    }
                    for item in adapted
                ],
                "admitted_certificates": [
                    {
                        "kind": certificate.kind.value,
                        "verifier": certificate.verifier,
                        "fingerprint": certificate_fingerprint(certificate),
                    }
                    for certificate in decision.admitted_certificates
                ],
                "task_id_used_for_alignment_only": True,
                "gold_access": False,
                "score_or_judge_access": False,
            }
        )

    attached = batch.attach_task_ids(policy_outputs)
    output_dir.mkdir(parents=True, exist_ok=True)
    solver_path = output_dir / "solver.jsonl"
    selected_lines: list[bytes] = []
    override_count = 0
    for ledger_row in attached:
        task_id = str(ledger_row["task_id"])
        selected_source = str(ledger_row["selected_source"])
        if selected_source not in raw_sources:
            raise CompositionError(f"policy selected unknown source {selected_source!r}")
        selected_lines.append(raw_sources[selected_source][1][task_id])
        override_count += int(selected_source != "anchor")

    if override_count == 0:
        # Preserve the exact frozen anchor bytes, including whitespace/newlines.
        shutil.copyfile(anchor_path, solver_path)
    else:
        temporary = solver_path.with_suffix(".jsonl.tmp")
        temporary.write_bytes(b"".join(selected_lines))
        temporary.replace(solver_path)

    ledger_path = output_dir / "evidence_ledger.jsonl"
    _write_jsonl(ledger_path, attached)
    reason_counts = Counter(str(row["reason"]) for row in attached)
    rejection_counts = Counter(
        reason
        for row in attached
        for audit in row["candidate_audits"]
        for reason in audit["rejection_reasons"]
    )
    input_bindings = {
        "anchor_raw_output_only": {"path": str(anchor_path), "sha256": actual_anchor_sha},
        "anchor_public_inference": {
            "path": str(anchor_public_path),
            "sha256": anchor_public_sha,
        },
        "public_tasks": {"path": str(public_path), "sha256": public_tasks_sha},
        **{
            f"candidate:{name}": {"path": str(path), "sha256": _sha256(path)}
            for name, path in sorted(candidate_paths.items())
        },
        **{
            f"candidate_raw_output_only:{name}": {
                "path": str(path),
                "sha256": _sha256(path),
            }
            for name, path in sorted(raw_candidate_paths.items())
        },
        **{
            f"certificate:{name}": {"path": str(path), "sha256": _sha256(path)}
            for name, path in sorted(certificate_paths.items())
        },
    }
    if frozen_profile_path:
        input_bindings["frozen_profile"] = {
            "path": str(frozen_profile_path),
            "sha256": _sha256(frozen_profile_path),
        }
    manifest = {
        "schema_version": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "profile": {
            "name": profile.name,
            "allowed_strong_kinds": sorted(kind.value for kind in profile.allowed_strong_kinds),
            "min_claim_coverage": profile.min_claim_coverage,
            "min_deterministic_checks": profile.min_deterministic_checks,
            "min_independent_verifiers": profile.min_independent_verifiers,
            "policy": "unique_strong_input_and_answer_bound_challenger_failclosed",
        },
        "rows": len(attached),
        "complete": len(attached) == args.expected_rows,
        "overrides": override_count,
        "reason_counts": dict(sorted(reason_counts.items())),
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "gold_access": False,
        "score_or_judge_access": False,
        "task_id_used_for_alignment_only": True,
        "source_or_sha_used_as_policy_feature": False,
        "observable_inputs": {
            "bundle_sha256": hashlib.sha256(
                json.dumps(
                    problem_fingerprints,
                    sort_keys=False,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
            "image_root": str(image_root) if image_root else None,
            "fully_image_bound_rows": image_binding_complete_count,
            "rows": len(problem_fingerprints),
        },
        "input_bindings": input_bindings,
        "outputs": {
            "solver": {"path": str(solver_path), "sha256": _sha256(solver_path)},
            "evidence_ledger": {"path": str(ledger_path), "sha256": _sha256(ledger_path)},
        },
        "anchor_exact_copy": _sha256(solver_path) == actual_anchor_sha,
    }
    manifest_path = output_dir / "composition_manifest.json"
    _write_json(manifest_path, manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor-solver", type=Path, required=True)
    parser.add_argument("--anchor-public", type=Path, required=True)
    parser.add_argument("--profile-json", type=Path, required=True)
    parser.add_argument("--anchor-sha256", default="")
    parser.add_argument("--public-tasks", type=Path, required=True)
    parser.add_argument("--candidate", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--raw-candidate", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--certificate", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, default=274)
    parser.add_argument("--profile-name", default="maxim-evidence-os-v1")
    parser.add_argument("--min-claim-coverage", type=float, default=1.0)
    parser.add_argument("--min-deterministic-checks", type=int, default=1)
    parser.add_argument("--min-independent-verifiers", type=int, default=1)
    return parser


def main() -> int:
    try:
        manifest = compose(build_parser().parse_args())
    except (CompositionError, AlignmentError, ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
