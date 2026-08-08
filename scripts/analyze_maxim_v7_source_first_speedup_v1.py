#!/usr/bin/env python3
"""Audit the quality-preserving source-first shortcut on frozen V7 artifacts.

This is an artifact-equivalence and recorded-usage analysis.  It does not read
gold answers or judge verdicts, and it does not claim an online latency
measurement.  Task IDs are used only to align already sealed rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
import unicodedata
from pathlib import Path
from typing import Any


SCHEMA = "maxim-v7-source-first-speedup-v1"
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_WHITESPACE = re.compile(r"\s+")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            task_id = str(row.get("task_id", "")).strip()
            if not task_id:
                raise ValueError(f"{path}:{line_number}: missing task_id")
            if task_id in rows:
                raise ValueError(f"{path}:{line_number}: duplicate task_id {task_id}")
            rows[task_id] = row
    return rows


def _repo_path(repo_root: Path, raw: str) -> Path:
    candidate = Path(raw)
    return candidate if candidate.is_absolute() else repo_root / candidate


def _artifact_from_manifest(
    manifest_path: Path,
    repo_root: Path,
    artifact_name: str,
) -> tuple[Path, dict[str, Any]]:
    manifest = _json(manifest_path)
    artifact = manifest.get("artifacts", {}).get(artifact_name)
    if not isinstance(artifact, dict):
        raise ValueError(f"{manifest_path}: missing artifacts.{artifact_name}")
    path = _repo_path(repo_root, str(artifact.get("path", "")))
    expected = str(artifact.get("sha256", ""))
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(
            f"SHA-256 mismatch for {artifact_name}: expected {expected}, got {actual}"
        )
    return path, manifest


def _usable_source(row: dict[str, Any]) -> bool:
    return bool(
        not row.get("abstain")
        and not row.get("error")
        and str(row.get("final_answer", "")).strip()
    )


def _answer_fingerprint(value: object) -> str:
    canonical = _WHITESPACE.sub(
        " ", unicodedata.normalize("NFKC", str(value or ""))
    ).strip()
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _stable_trace(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _validate_certificate(
    task_id: str,
    certificate: dict[str, Any],
    candidate: dict[str, Any],
) -> None:
    checks = certificate.get("deterministic_checks")
    trace = certificate.get("trace")
    trace_checks = trace.get("checks") if isinstance(trace, dict) else None
    expected_answer = _answer_fingerprint(candidate.get("final_answer"))
    declared_trace = str(certificate.get("trace_fingerprint") or "")
    actual_trace = hashlib.sha256(_stable_trace(trace)).hexdigest()
    conditions = {
        "status_pass": certificate.get("status") == "pass",
        "strength_strong": certificate.get("strength") == "strong",
        "input_bound": certificate.get("input_bound") is True,
        "answer_bound": certificate.get("answer_bound") is True,
        "input_fingerprint": bool(
            _HEX64.fullmatch(str(certificate.get("input_fingerprint") or ""))
        ),
        "answer_fingerprint": certificate.get("answer_fingerprint")
        == expected_answer,
        "full_claim_coverage": float(certificate.get("claim_coverage", 0.0))
        == 1.0,
        "no_contradictions": int(certificate.get("contradiction_count", -1))
        == 0,
        "deterministic_checks": isinstance(checks, list)
        and bool(checks)
        and all(value is True for value in checks),
        "verifier": bool(str(certificate.get("verifier") or "").strip()),
        "trace_accepted": isinstance(trace, dict)
        and trace.get("accepted") is True,
        "trace_checks": isinstance(trace_checks, dict)
        and bool(trace_checks)
        and all(value is True for value in trace_checks.values()),
        "trace_fingerprint": bool(_HEX64.fullmatch(declared_trace))
        and declared_trace == actual_trace,
    }
    failed = [name for name, passed in conditions.items() if not passed]
    if failed:
        raise ValueError(f"invalid source certificate for {task_id}: {failed}")


def _validated_stage(
    manifest_path: Path,
    repo_root: Path,
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, Any],
    Path,
    Path,
]:
    candidate_path, manifest = _artifact_from_manifest(
        manifest_path, repo_root, "candidate"
    )
    certificate_path, certificate_manifest = _artifact_from_manifest(
        manifest_path, repo_root, "certificates"
    )
    if certificate_manifest != manifest:
        raise ValueError("resolver manifest changed between artifact reads")
    candidate_rows = _jsonl(candidate_path)
    certificate_rows = _jsonl(certificate_path)
    usable = {
        task_id: row for task_id, row in candidate_rows.items() if _usable_source(row)
    }
    expected = int(manifest.get("accepted_certificates", -1))
    if len(certificate_rows) != expected or set(certificate_rows) != set(usable):
        raise ValueError(
            "certificate rows do not match usable source candidates or manifest count"
        )
    for task_id, certificate in certificate_rows.items():
        _validate_certificate(task_id, certificate, usable[task_id])
    return candidate_rows, certificate_rows, manifest, candidate_path, certificate_path


def _usage(row: dict[str, Any], key: str) -> float:
    value = row.get("usage", {}).get(key, 0)
    return float(value or 0)


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _markdown(result: dict[str, Any]) -> str:
    usage = result["recorded_anchor_usage"]
    return "\n".join(
        [
            "# Source-first cascade: V7 artifact replay",
            "",
            "Это проверка эквивалентности на уже замороженных артефактах V7, а не новый score и не online latency benchmark.",
            "",
            "## Результат",
            "",
            f"- строк: `{result['rows']}`;",
            f"- задач с единственным принятым source-answer: `{result['source_shortcuts']}` (`{result['source_shortcut_rate']:.2%}`);",
            f"- задач, где reasoning anchor всё ещё нужен: `{result['anchor_fallbacks']}`;",
            f"- source-answer совпал с финальным V7: `{result['answer_equivalent_rows']}/{result['source_shortcuts']}`;",
            f"- потенциально исключаемая записанная model latency: `{usage['avoidable_latency_s']:.3f}` из `{usage['total_latency_s']:.3f}` секунд (`{usage['avoidable_latency_fraction']:.2%}`);",
            f"- потенциально исключаемые input tokens: `{usage['avoidable_input_tokens']}` из `{usage['total_input_tokens']}` (`{usage['avoidable_input_fraction']:.2%}`);",
            f"- потенциально исключаемые output tokens: `{usage['avoidable_output_tokens']}` из `{usage['total_output_tokens']}` (`{usage['avoidable_output_fraction']:.2%}`).",
            "",
            "## Интерпретация",
            "",
            "В production source resolver запускается первым. Если найден ровно один сильный input-bound и answer-bound сертификат, его ответ можно вернуть без вызова reasoning-модели. При отсутствии сертификата или конфликте запускается прежний anchor и полный fail-closed composer.",
            "",
            "Качество в этом replay не меняется: все shortcut-ответы дословно совпадают с финальным V7. Реальная задержка source resolver здесь не измерена, поэтому проценты выше описывают устранённую model work, а не обещанное wall-clock ускорение сервиса.",
            "",
            "## Границы честного утверждения",
            "",
            "- gold answers, judge verdicts и correctness не читались;",
            "- task_id применялся только для выравнивания строк;",
            "- replay использует previously inspected development artifacts;",
            "- перед production нужен online benchmark с cold/warm cache, p50/p95 и стоимостью source lookup.",
            "",
        ]
    )


def analyze(
    profile_path: Path,
    output_json: Path,
    output_markdown: Path,
    repo_root: Path,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    profile_path = profile_path.resolve()
    profile = _json(profile_path)
    if profile.get("schema_version") != "maxim-composite-source-pipeline-v7":
        raise ValueError("expected the frozen V7 composite profile")

    main_manifest_path = _repo_path(
        repo_root, profile["main_stage"]["resolver_manifest"]["path"]
    )
    history_manifest_path = _repo_path(
        repo_root, profile["history_stage"]["resolver_manifest"]["path"]
    )
    final_solver_path = _repo_path(
        repo_root, profile["history_stage"]["solver"]["path"]
    )

    for path, expected in (
        (main_manifest_path, profile["main_stage"]["resolver_manifest"]["sha256"]),
        (history_manifest_path, profile["history_stage"]["resolver_manifest"]["sha256"]),
        (final_solver_path, profile["history_stage"]["solver"]["sha256"]),
    ):
        if _sha256(path) != expected:
            raise ValueError(f"frozen V7 hash mismatch: {path}")

    (
        main_rows,
        main_certificates,
        main_manifest,
        main_candidate_path,
        main_certificate_path,
    ) = _validated_stage(main_manifest_path, repo_root)
    (
        history_rows,
        history_certificates,
        history_manifest,
        history_candidate_path,
        history_certificate_path,
    ) = _validated_stage(history_manifest_path, repo_root)
    final_rows = _jsonl(final_solver_path)
    expected_rows = int(profile["expected_rows"])
    if not all(len(rows) == expected_rows for rows in (main_rows, history_rows, final_rows)):
        raise ValueError("V7 artifact row counts differ from the frozen profile")
    if not (main_rows.keys() == history_rows.keys() == final_rows.keys()):
        raise ValueError("V7 artifact task sets differ")

    shortcut_rows: dict[str, dict[str, Any]] = {
        task_id: row for task_id, row in main_rows.items() if _usable_source(row)
    }
    conflicts: list[str] = []
    for task_id, row in history_rows.items():
        if not _usable_source(row):
            continue
        previous = shortcut_rows.get(task_id)
        if previous and str(previous["final_answer"]).strip() != str(row["final_answer"]).strip():
            conflicts.append(task_id)
            shortcut_rows.pop(task_id, None)
            continue
        shortcut_rows[task_id] = row

    expected_certificates = int(main_manifest["accepted_certificates"]) + int(
        history_manifest["accepted_certificates"]
    )
    if len(shortcut_rows) + len(conflicts) != expected_certificates:
        raise ValueError("usable source rows do not match resolver certificate counts")

    mismatches = [
        task_id
        for task_id, source in shortcut_rows.items()
        if str(source["final_answer"]).strip()
        != str(final_rows[task_id].get("final_answer", "")).strip()
    ]
    if mismatches:
        raise ValueError(f"source-first answers differ from V7 final: {mismatches}")

    total_latency = sum(_usage(row, "latency_s") for row in final_rows.values())
    avoidable_latency = sum(
        _usage(final_rows[task_id], "latency_s") for task_id in shortcut_rows
    )
    total_input = int(sum(_usage(row, "input_tokens") for row in final_rows.values()))
    avoidable_input = int(
        sum(_usage(final_rows[task_id], "input_tokens") for task_id in shortcut_rows)
    )
    total_output = int(sum(_usage(row, "output_tokens") for row in final_rows.values()))
    avoidable_output = int(
        sum(_usage(final_rows[task_id], "output_tokens") for task_id in shortcut_rows)
    )

    result: dict[str, Any] = {
        "schema_version": SCHEMA,
        "rows": expected_rows,
        "source_shortcuts": len(shortcut_rows),
        "source_shortcut_rate": len(shortcut_rows) / expected_rows,
        "anchor_fallbacks": expected_rows - len(shortcut_rows),
        "answer_equivalent_rows": len(shortcut_rows),
        "conflicting_source_rows": conflicts,
        "recorded_anchor_usage": {
            "total_latency_s": total_latency,
            "avoidable_latency_s": avoidable_latency,
            "avoidable_latency_fraction": avoidable_latency / total_latency,
            "total_input_tokens": total_input,
            "avoidable_input_tokens": avoidable_input,
            "avoidable_input_fraction": avoidable_input / total_input,
            "total_output_tokens": total_output,
            "avoidable_output_tokens": avoidable_output,
            "avoidable_output_fraction": avoidable_output / total_output,
        },
        "claims": {
            "artifact_answer_equivalence_measured": True,
            "certificate_artifacts_replayed": True,
            "online_wall_clock_speedup_measured": False,
            "source_lookup_cost_included": False,
            "accuracy_or_gold_read": False,
            "task_id_policy_feature": False,
            "task_id_alignment_only": True,
        },
        "inputs": {
            "profile": {"path": str(profile_path), "sha256": _sha256(profile_path)},
            "main_manifest": {
                "path": str(main_manifest_path),
                "sha256": _sha256(main_manifest_path),
            },
            "main_candidate": {
                "path": str(main_candidate_path),
                "sha256": _sha256(main_candidate_path),
            },
            "main_certificates": {
                "path": str(main_certificate_path),
                "sha256": _sha256(main_certificate_path),
                "rows": len(main_certificates),
            },
            "history_manifest": {
                "path": str(history_manifest_path),
                "sha256": _sha256(history_manifest_path),
            },
            "history_candidate": {
                "path": str(history_candidate_path),
                "sha256": _sha256(history_candidate_path),
            },
            "history_certificates": {
                "path": str(history_certificate_path),
                "sha256": _sha256(history_certificate_path),
                "rows": len(history_certificates),
            },
            "final_solver": {
                "path": str(final_solver_path),
                "sha256": _sha256(final_solver_path),
            },
        },
    }
    _atomic_write(
        output_json,
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    _atomic_write(output_markdown, _markdown(result))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        type=Path,
        default=Path("configs/maxim_composite_source_pipeline_v7.json"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("reports/maxim_v7_source_first_speed_v1_20260808/analysis.json"),
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=Path("reports/maxim_v7_source_first_speed_v1_20260808/REPORT.md"),
    )
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    args = parser.parse_args()
    result = analyze(
        args.profile,
        args.output_json,
        args.output_markdown,
        args.repo_root,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
