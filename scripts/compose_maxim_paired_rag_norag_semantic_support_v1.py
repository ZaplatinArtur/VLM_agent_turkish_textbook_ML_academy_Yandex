#!/usr/bin/env python3
"""Compose a fail-closed full274 solver from semantic-support verdicts.

The RAG row is selected only when every preregistered gate passes.  Citation
quotes are checked as exact Python substrings of the exact queued chunk text;
any invalid citation, contradiction, unsupported decisive step, model error,
or missing verdict selects the frozen no-RAG row.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    import prepare_maxim_paired_rag_norag_semantic_support_v1 as preparation
    import run_maxim_paired_rag_norag_semantic_support_v1 as runner
except ModuleNotFoundError:  # Imported as scripts.*
    from scripts import prepare_maxim_paired_rag_norag_semantic_support_v1 as preparation
    from scripts import run_maxim_paired_rag_norag_semantic_support_v1 as runner


SCHEMA_VERSION = "maxim-paired-rag-norag-semantic-support-composition-v1"
CONDITION = "paired_rag_norag_semantic_support_on_pinned_structural_context_v1"
RULE_ID = "rag_only_if_exact_cited_support_else_no_rag_v1"


class CompositionError(RuntimeError):
    pass


def _index(rows: Sequence[Mapping[str, Any]], label: str) -> dict[str, Mapping[str, Any]]:
    _, result = preparation.index_unique(rows, label)
    return result


def _verify_source_hashes(
    manifest: Mapping[str, Any], paths: Mapping[str, Path]
) -> None:
    for label, path in paths.items():
        expected = str(manifest["sources"][label]["sha256"])
        actual = preparation.sha256_file(path)
        if actual != expected:
            raise CompositionError(
                f"{label} hash mismatch: expected {expected}, got {actual}"
            )


def exact_citation_audit(
    verdict: Mapping[str, Any], queue_row: Mapping[str, Any]
) -> dict[str, Any]:
    context_by_id = {
        str(item.get("chunk_id") or ""): str(item.get("text") or "")
        for item in queue_row.get("contexts") or []
        if isinstance(item, Mapping)
    }
    citations = verdict.get("citations")
    if not isinstance(citations, list):
        return {
            "all_valid": False,
            "valid_count": 0,
            "decisive_valid_count": 0,
            "rows": [],
            "error": "citations_not_list",
        }
    rows: list[dict[str, Any]] = []
    all_valid = True
    decisive_valid_count = 0
    for citation in citations:
        if not isinstance(citation, Mapping):
            rows.append({"valid": False, "reason": "citation_not_object"})
            all_valid = False
            continue
        chunk_id = str(citation.get("chunk_id") or "")
        quote = str(citation.get("exact_quote") or "")
        known_chunk = chunk_id in context_by_id
        exact_substring = bool(quote) and known_chunk and quote in context_by_id[chunk_id]
        valid = bool(known_chunk and exact_substring)
        decisive = citation.get("decisive") is True
        if valid and decisive:
            decisive_valid_count += 1
        all_valid = all_valid and valid
        rows.append(
            {
                "chunk_id": chunk_id,
                "quote_sha256": preparation.stable_sha256(quote),
                "quote_chars": len(quote),
                "known_chunk": known_chunk,
                "exact_substring": exact_substring,
                "decisive": decisive,
                "valid": valid,
            }
        )
    return {
        "all_valid": all_valid,
        "valid_count": sum(bool(item.get("valid")) for item in rows),
        "decisive_valid_count": decisive_valid_count,
        "rows": rows,
        "error": None,
    }


def gate_verdict(
    result_row: Mapping[str, Any],
    queue_row: Mapping[str, Any],
    profile: Mapping[str, Any],
) -> tuple[bool, list[str], dict[str, Any]]:
    reasons: list[str] = []
    if result_row.get("error"):
        reasons.append("runner_error")
    if result_row.get("request_sha256") != queue_row.get("request_sha256"):
        reasons.append("request_sha_mismatch")
    parsed = result_row.get("parsed")
    if not isinstance(parsed, Mapping):
        reasons.append("missing_valid_verdict")
        return False, reasons, {
            "all_valid": False,
            "valid_count": 0,
            "decisive_valid_count": 0,
            "rows": [],
            "error": "missing_verdict",
        }
    try:
        parsed = runner.validate_verdict(parsed)
    except Exception as exc:
        reasons.append(f"invalid_verdict_schema:{type(exc).__name__}")
        return False, reasons, {
            "all_valid": False,
            "valid_count": 0,
            "decisive_valid_count": 0,
            "rows": [],
            "error": "invalid_verdict_schema",
        }

    gate = profile["selection_gate"]
    citation_audit = exact_citation_audit(parsed, queue_row)
    if parsed.get("rag_answer_supported") is not True:
        reasons.append("rag_not_supported")
    if float(parsed.get("confidence") or 0.0) < float(gate["min_confidence"]):
        reasons.append("confidence_below_threshold")
    if parsed.get("contradiction_found") is not False:
        reasons.append("contradiction_found")
    unsupported = parsed.get("unsupported_decisive_steps")
    if not isinstance(unsupported, list) or unsupported:
        reasons.append("unsupported_decisive_steps")
    if parsed.get("answer_format_verified") is not True:
        reasons.append("answer_format_not_verified")
    if citation_audit["decisive_valid_count"] < int(gate["min_decisive_valid_citations"]):
        reasons.append("no_decisive_valid_citation")
    if gate.get("all_returned_citations_must_validate") and not citation_audit["all_valid"]:
        reasons.append("invalid_returned_citation")
    return not reasons, reasons, citation_audit


def _row_sha256(row: Mapping[str, Any]) -> str:
    return preparation.stable_sha256(dict(row))


def _compose_row(
    source: Mapping[str, Any],
    *,
    task_id: str,
    selected_source: str,
    route: str,
    gate_reasons: Sequence[str],
    citation_audit: Mapping[str, Any] | None,
    page_row: Mapping[str, Any],
    no_rag_row: Mapping[str, Any],
    verdict_row: Mapping[str, Any] | None,
) -> dict[str, Any]:
    output = copy.deepcopy(dict(source))
    original_generation = (
        copy.deepcopy(output.get("generation"))
        if isinstance(output.get("generation"), Mapping)
        else {}
    )
    output["condition"] = CONDITION
    output["prompt_version"] = CONDITION
    output["generation"] = {
        **original_generation,
        "gold_access": False,
        "semantic_support_composition": {
            "schema_version": SCHEMA_VERSION,
            "rule_id": RULE_ID,
            "route": route,
            "selected_source": selected_source,
            "gate_reasons": list(gate_reasons),
            "citation_audit": copy.deepcopy(citation_audit),
            "verdict_request_sha256": (
                verdict_row.get("request_sha256") if verdict_row else None
            ),
            "source_row_sha256": {
                "page_rag": _row_sha256(page_row),
                "no_rag": _row_sha256(no_rag_row),
            },
        },
    }
    output["error"] = None
    if not str(output.get("final_answer") or "").strip():
        raise CompositionError(f"{task_id}: selected source has empty final_answer")
    return output


def compose(
    benchmark_rows: Sequence[Mapping[str, Any]],
    page_rows: Sequence[Mapping[str, Any]],
    no_rag_rows: Sequence[Mapping[str, Any]],
    context_rows: Sequence[Mapping[str, Any]],
    queue_rows: Sequence[Mapping[str, Any]],
    verdict_rows: Sequence[Mapping[str, Any]],
    profile: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    order, benchmark = preparation.index_unique(benchmark_rows, "benchmark")
    page = _index(page_rows, "page_rag")
    no_rag = _index(no_rag_rows, "no_rag")
    contexts = _index(context_rows, "contexts")
    queue = _index(queue_rows, "queue")
    verdicts = _index(verdict_rows, "verdicts")
    if len(order) != preparation.EXPECTED_ROWS:
        raise CompositionError("benchmark row count mismatch")
    expected = set(order)
    for label, source in (("page", page), ("no_rag", no_rag), ("contexts", contexts)):
        if set(source) != expected:
            raise CompositionError(f"{label}: task-id set mismatch")
    if set(verdicts) != set(queue):
        raise CompositionError("verdict task-id set must exactly equal the model queue")

    output: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    rag_task_ids: list[str] = []
    for task_id in order:
        route, _ = preparation.classify_task(
            benchmark[task_id], page[task_id], no_rag[task_id], contexts[task_id]
        )
        if route == "semantic_support_verifier":
            if task_id not in queue:
                raise CompositionError(f"{task_id}: eligible task missing from queue")
            verdict = verdicts[task_id]
            passed, reasons, citation_audit = gate_verdict(
                verdict, queue[task_id], profile
            )
            selected = "page_rag" if passed else "no_rag"
            if passed:
                rag_task_ids.append(task_id)
            decision_key = (
                "semantic_gate_select_rag" if passed else "semantic_gate_default_no_rag"
            )
        else:
            if task_id in queue:
                raise CompositionError(f"{task_id}: skipped task unexpectedly present in queue")
            verdict = None
            reasons = [route]
            citation_audit = None
            selected = "no_rag"
            decision_key = route
        counts[decision_key] = counts.get(decision_key, 0) + 1
        source = page[task_id] if selected == "page_rag" else no_rag[task_id]
        output.append(
            _compose_row(
                source,
                task_id=task_id,
                selected_source=selected,
                route=route,
                gate_reasons=reasons,
                citation_audit=citation_audit,
                page_row=page[task_id],
                no_rag_row=no_rag[task_id],
                verdict_row=verdict,
            )
        )
    return output, {
        "rows": len(output),
        "decision_counts": dict(sorted(counts.items())),
        "rag_selected_task_ids": rag_task_ids,
        "rag_selected_rows": len(rag_task_ids),
        "no_rag_selected_rows": len(output) - len(rag_task_ids),
        "gold_access": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--page-rag", type=Path, required=True)
    parser.add_argument("--no-rag", type=Path, required=True)
    parser.add_argument("--contexts", type=Path, required=True)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--verdicts", type=Path, required=True)
    parser.add_argument("--preparation-manifest", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args(argv)

    paths = {
        "benchmark": args.benchmark.resolve(),
        "page_rag": args.page_rag.resolve(),
        "no_rag": args.no_rag.resolve(),
        "contexts": args.contexts.resolve(),
    }
    prep_manifest = preparation.load_json(args.preparation_manifest.resolve())
    profile = preparation.load_json(args.profile.resolve())
    if prep_manifest.get("schema_version") != preparation.SCHEMA_VERSION:
        raise CompositionError("preparation manifest schema mismatch")
    if profile.get("schema_version") != preparation.PROFILE_SCHEMA_VERSION:
        raise CompositionError("profile schema mismatch")
    _verify_source_hashes(prep_manifest, paths)
    if preparation.sha256_file(args.queue.resolve()) != prep_manifest["queue"]["sha256"]:
        raise CompositionError("queue hash mismatch")
    if preparation.sha256_file(args.profile.resolve()) != prep_manifest["profile"]["sha256"]:
        raise CompositionError("profile hash mismatch")

    output, stats = compose(
        preparation.load_jsonl(paths["benchmark"]),
        preparation.load_jsonl(paths["page_rag"]),
        preparation.load_jsonl(paths["no_rag"]),
        preparation.load_jsonl(paths["contexts"]),
        preparation.load_jsonl(args.queue.resolve()),
        preparation.load_jsonl(args.verdicts.resolve()),
        profile,
    )
    preparation.write_jsonl(args.output.resolve(), output)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "condition": CONDITION,
        "rule_id": RULE_ID,
        "gold_access": False,
        "sources": {
            **{
                label: {
                    "path": str(path),
                    "sha256": preparation.sha256_file(path),
                }
                for label, path in paths.items()
            },
            "queue": {
                "path": str(args.queue.resolve()),
                "sha256": preparation.sha256_file(args.queue.resolve()),
            },
            "verdicts": {
                "path": str(args.verdicts.resolve()),
                "sha256": preparation.sha256_file(args.verdicts.resolve()),
            },
            "profile": {
                "path": str(args.profile.resolve()),
                "sha256": preparation.sha256_file(args.profile.resolve()),
            },
            "preparation_manifest": {
                "path": str(args.preparation_manifest.resolve()),
                "sha256": preparation.sha256_file(args.preparation_manifest.resolve()),
            },
        },
        "code": {
            "composer": preparation.sha256_file(Path(__file__).resolve()),
            "prepare": prep_manifest["code"]["prepare"]["sha256"],
            "runner": prep_manifest["code"]["runner"]["sha256"],
        },
        "stats": stats,
        "output": {
            "path": str(args.output.resolve()),
            "rows": len(output),
            "sha256": preparation.sha256_file(args.output.resolve()),
        },
        "scoring_performed": False,
    }
    preparation.write_json(args.manifest.resolve(), manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
