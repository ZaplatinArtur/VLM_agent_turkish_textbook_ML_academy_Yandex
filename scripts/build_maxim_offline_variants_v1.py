#!/usr/bin/env python3
"""Compose frozen solver outputs for Maksim's offline pipeline ideas.

This module deliberately performs no scoring and never sends benchmark gold to
a model.  Benchmark rows are used only for the visible ``task_id``, ``subject``
and ``answer_type`` fields.  Every output row records the source row and the
fixed decision rule that produced it.

The four modes are intentionally conservative:

* ``subject_router``: Math -> frozen no-tools; all other subjects -> page RAG.
* ``parallel_consensus``: unique modal cluster among eight saved candidates;
  ties fall back to the already-saved selector answer.
* ``answer_canonicalization``: format-only cleanup of a supplied solver file.
* ``selective_rag_metadata``: an offline proxy which accepts page RAG only when
  repeated retrieval calls agree on at least one chunk and have no tool error.

The rules are constants rather than CLI thresholds so they cannot be tuned on
the frozen benchmark by changing command-line arguments.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import tempfile
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence


REPO = Path(__file__).resolve().parents[1]
DEFAULT_BENCHMARK = (
    REPO / "artifacts" / "baselines" / "basic_page_rag_v1" / "validation_274.jsonl"
)
DEFAULT_PAGE_RAG = (
    REPO / "artifacts" / "baselines" / "basic_page_rag_v1" / "agent_rag_274.jsonl"
)
DEFAULT_NO_TOOLS = (
    REPO / "artifacts" / "baselines" / "no_tools_v1" / "b0_no_tools_raw.jsonl"
)
DEFAULT_PARALLEL = (
    REPO
    / "reports"
    / "maxim_ideas_full274_20260731"
    / "parallel8_reasoning_first_v2"
    / "solver.jsonl"
)

SCHEMA_VERSION = "maxim-offline-composer-v1"
EXPECTED_PARALLEL_CANDIDATES = 8
MATH_SUBJECT_LABELS = frozenset({"math"})

SUBJECT_ROUTER_RULE_ID = "subject_casefold_math_to_no_tools_else_page_rag_v1"
CONSENSUS_RULE_ID = "unique_modal_of_8_answer_clusters_else_saved_selector_v1"
CANONICALIZATION_RULE_ID = "answer_type_format_only_canonicalization_v1"
SELECTIVE_RAG_RULE_ID = (
    "rag_if_2_successful_searches_repeated_chunk_and_zero_search_errors_v1"
)

FORBIDDEN_SOLVER_FIELDS = frozenset(
    {"reference_answer", "reference_solution", "gold_answer", "gold_solution"}
)
VISIBLE_BENCHMARK_FIELDS = frozenset({"task_id", "subject", "answer_type"})


class CompositionError(ValueError):
    """Raised when a source artifact violates the frozen composition protocol."""


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _row_sha256(row: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(row).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    if not path.is_file():
        raise CompositionError(f"{label}: file does not exist: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CompositionError(
                    f"{label}: invalid JSON on line {line_number}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise CompositionError(
                    f"{label}: line {line_number} must be a JSON object"
                )
            rows.append(row)
    if not rows:
        raise CompositionError(f"{label}: no JSONL rows")
    return rows


def index_rows(
    rows: Iterable[dict[str, Any]], label: str
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    indexed: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for position, row in enumerate(rows, start=1):
        task_id = str(row.get("task_id") or "").strip()
        if not task_id:
            raise CompositionError(f"{label}: row {position} has no task_id")
        if task_id in indexed:
            raise CompositionError(f"{label}: duplicate task_id {task_id}")
        indexed[task_id] = row
        order.append(task_id)
    return indexed, order


def _assert_same_task_ids(
    actual: set[str], expected: set[str], label: str
) -> None:
    if actual == expected:
        return
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    raise CompositionError(
        f"{label}: task-ID mismatch; missing={missing[:10]}, extra={extra[:10]}"
    )


def _validate_solver_row(row: dict[str, Any], label: str, task_id: str) -> None:
    forbidden = sorted(FORBIDDEN_SOLVER_FIELDS.intersection(row))
    if forbidden:
        raise CompositionError(
            f"{label}: task {task_id} contains forbidden gold fields {forbidden}"
        )
    generation = row.get("generation")
    if isinstance(generation, dict) and "gold_access" in generation:
        if generation["gold_access"] is not False:
            raise CompositionError(
                f"{label}: task {task_id} declares generation.gold_access="
                f"{generation['gold_access']!r}"
            )
    answer = row.get("final_answer")
    if answer is None or not str(answer).strip():
        raise CompositionError(f"{label}: task {task_id} has no final_answer")


def _validate_source(
    indexed: dict[str, dict[str, Any]], label: str
) -> None:
    for task_id, row in indexed.items():
        _validate_solver_row(row, label, task_id)


def _visible_benchmark_row(row: dict[str, Any], task_id: str) -> dict[str, str]:
    # Explicit projection prevents accidental use of reference_answer or
    # reference_solution in any routing/canonicalization decision.
    if str(row.get("task_id") or "").strip() != task_id:
        raise CompositionError(f"benchmark: inconsistent task_id for {task_id}")
    return {
        "task_id": task_id,
        "subject": str(row.get("subject") or "").strip(),
        "answer_type": str(row.get("answer_type") or "").strip().casefold(),
    }


def _source_descriptor(label: str, row: dict[str, Any]) -> dict[str, Any]:
    return {
        "label": label,
        "condition": str(row.get("condition") or ""),
        "prompt_version": str(row.get("prompt_version") or ""),
        "row_sha256": _row_sha256(row),
    }


def _stamp_output(
    source_row: dict[str, Any],
    *,
    mode: str,
    condition: str,
    rule_id: str,
    benchmark_fields_used: Sequence[str],
    sources: Sequence[dict[str, Any]],
    decision: dict[str, Any],
) -> dict[str, Any]:
    output = copy.deepcopy(source_row)
    output["condition"] = condition
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "mode": mode,
        "rule_id": rule_id,
        "gold_access": False,
        "benchmark_fields_used": list(benchmark_fields_used),
        "sources": list(sources),
        "decision": copy.deepcopy(decision),
    }
    output["offline_provenance"] = provenance
    generation = output.get("generation")
    if generation is None:
        generation = {}
    if not isinstance(generation, dict):
        raise CompositionError(
            f"source task {source_row.get('task_id')} generation must be an object"
        )
    generation = copy.deepcopy(generation)
    generation["gold_access"] = False
    generation["offline_composition"] = {
        "schema_version": SCHEMA_VERSION,
        "mode": mode,
        "rule_id": rule_id,
    }
    output["generation"] = generation
    forbidden = FORBIDDEN_SOLVER_FIELDS.intersection(output)
    if forbidden:
        raise CompositionError(
            f"composed task {source_row.get('task_id')} leaked gold fields: "
            f"{sorted(forbidden)}"
        )
    return output


def _prepare_inputs(
    benchmark_rows: Iterable[dict[str, Any]],
    sources: Sequence[tuple[str, Iterable[dict[str, Any]]]],
) -> tuple[
    dict[str, dict[str, Any]],
    list[str],
    dict[str, dict[str, dict[str, Any]]],
]:
    benchmark, order = index_rows(benchmark_rows, "benchmark")
    expected_ids = set(benchmark)
    indexed_sources: dict[str, dict[str, dict[str, Any]]] = {}
    for label, rows in sources:
        indexed, _ = index_rows(rows, label)
        _assert_same_task_ids(set(indexed), expected_ids, label)
        _validate_source(indexed, label)
        indexed_sources[label] = indexed
    return benchmark, order, indexed_sources


def compose_subject_router(
    benchmark_rows: Iterable[dict[str, Any]],
    page_rag_rows: Iterable[dict[str, Any]],
    no_tools_rows: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    benchmark, order, sources = _prepare_inputs(
        benchmark_rows,
        (("page_rag", page_rag_rows), ("no_tools", no_tools_rows)),
    )
    output: list[dict[str, Any]] = []
    counts = Counter()
    for task_id in order:
        visible = _visible_benchmark_row(benchmark[task_id], task_id)
        if not visible["subject"]:
            raise CompositionError(f"benchmark: task {task_id} has empty subject")
        is_math = visible["subject"].casefold() in MATH_SUBJECT_LABELS
        source_label = "no_tools" if is_math else "page_rag"
        source_row = sources[source_label][task_id]
        counts[source_label] += 1
        output.append(
            _stamp_output(
                source_row,
                mode="subject_router",
                condition="maxim_subject_router_v1",
                rule_id=SUBJECT_ROUTER_RULE_ID,
                benchmark_fields_used=("task_id", "subject"),
                sources=(_source_descriptor(source_label, source_row),),
                decision={
                    "visible_subject": visible["subject"],
                    "math_label_match": is_math,
                    "chosen_source": source_label,
                },
            )
        )
    return output, {
        "rows": len(output),
        "chosen_source_counts": dict(sorted(counts.items())),
        "math_subject_labels_casefolded": sorted(MATH_SUBJECT_LABELS),
    }


def _unwrap_math_wrapper(value: str) -> str:
    wrappers = (
        ("$", "$"),
        (r"\(", r"\)"),
        (r"\[", r"\]"),
    )
    text = value.strip()
    changed = True
    while changed:
        changed = False
        for left, right in wrappers:
            if text.startswith(left) and text.endswith(right):
                text = text[len(left) : len(text) - len(right)].strip()
                changed = True
        boxed = re.fullmatch(r"\\boxed\s*\{(.*)\}", text, flags=re.DOTALL)
        if boxed:
            text = boxed.group(1).strip()
            changed = True
    return text


_CHOICE_PATTERNS = (
    re.compile(r"^([A-E])$", flags=re.IGNORECASE),
    re.compile(r"^[\(\[]\s*([A-E])\s*[\)\]]\s*[.!]?$", flags=re.IGNORECASE),
    re.compile(
        r"^(?:answer|cevap|yan[\u0131i]t)\s*[:=\-]?\s*([A-E])\s*[.!]?$",
        flags=re.IGNORECASE,
    ),
)

_NUMERIC_WITH_OPTIONAL_UNIT = re.compile(
    r"^(?P<number>[+\-\u2212\u2013\u2014]?(?:\d+(?:[.,]\d+)?|[.,]\d+)"
    r"(?:\s*/\s*[+\-\u2212\u2013\u2014]?\d+(?:[.,]\d+)?)?)"
    r"(?P<unit>\s*[^\d]*)?$",
    flags=re.UNICODE,
)


def canonicalize_final_answer(value: Any, answer_type: str) -> str:
    """Return a conservative, reference-free formatting normalization."""
    if value is None:
        raise CompositionError("final_answer is null")
    text = unicodedata.normalize("NFC", str(value)).replace("\r\n", "\n").strip()
    if not text:
        raise CompositionError("final_answer is empty")

    # Recover only a complete, valid JSON object that explicitly wraps a scalar
    # final_answer.  Truncated JSON or arbitrary prose is deliberately untouched.
    if text.startswith("{") and text.endswith("}"):
        try:
            wrapped = json.loads(text)
        except json.JSONDecodeError:
            wrapped = None
        if isinstance(wrapped, dict) and isinstance(
            wrapped.get("final_answer"), (str, int, float)
        ):
            text = str(wrapped["final_answer"]).strip()

    text = _unwrap_math_wrapper(text)
    normalized_type = str(answer_type or "").strip().casefold()
    if normalized_type == "choice":
        for pattern in _CHOICE_PATTERNS:
            match = pattern.fullmatch(text)
            if match:
                return match.group(1).upper()
        return text

    if normalized_type == "numeric":
        numeric = _NUMERIC_WITH_OPTIONAL_UNIT.fullmatch(text)
        if numeric:
            number = numeric.group("number")
            number = re.sub(r"[\u2212\u2013\u2014]", "-", number)
            number = number.replace(",", ".")
            number = re.sub(r"\s*/\s*", "/", number)
            if number.startswith("+"):
                number = number[1:]
            if number.startswith("."):
                number = "0" + number
            if number.startswith("-."):
                number = "-0" + number[1:]
            return number
    return text


def _answer_cluster_key(value: Any, answer_type: str) -> str:
    canonical = canonicalize_final_answer(value, answer_type)
    # Whitespace and casing are presentation differences.  Punctuation and
    # mathematical operators remain significant for non-choice, non-numeric
    # answers, so algebraic expressions are never heuristically merged.
    return re.sub(r"\s+", " ", canonical).strip().casefold()


def compose_parallel_consensus(
    benchmark_rows: Iterable[dict[str, Any]],
    parallel_rows: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    benchmark, order, sources = _prepare_inputs(
        benchmark_rows,
        (("parallel8", parallel_rows),),
    )
    output: list[dict[str, Any]] = []
    decision_counts = Counter()
    changed = 0
    for task_id in order:
        source_row = sources["parallel8"][task_id]
        visible = _visible_benchmark_row(benchmark[task_id], task_id)
        generation = source_row.get("generation")
        if not isinstance(generation, dict):
            raise CompositionError(f"parallel8: task {task_id} has no generation object")
        traces = generation.get("candidate_traces")
        if not isinstance(traces, list) or len(traces) != EXPECTED_PARALLEL_CANDIDATES:
            raise CompositionError(
                f"parallel8: task {task_id} expected {EXPECTED_PARALLEL_CANDIDATES} "
                f"candidate traces, got {len(traces) if isinstance(traces, list) else None}"
            )
        trace_keys: list[str] = []
        invalid_candidate_indices: list[int] = []
        seen_indices: set[int] = set()
        for position, trace in enumerate(traces, start=1):
            if not isinstance(trace, dict):
                raise CompositionError(
                    f"parallel8: task {task_id} candidate {position} is not an object"
                )
            index = trace.get("index")
            if not isinstance(index, int) or index in seen_indices:
                raise CompositionError(
                    f"parallel8: task {task_id} has invalid/duplicate candidate index {index!r}"
                )
            seen_indices.add(index)
            candidate_answer = trace.get("final_answer")
            if candidate_answer is None or not str(candidate_answer).strip():
                invalid_candidate_indices.append(index)
                trace_keys.append("")
            else:
                trace_keys.append(
                    _answer_cluster_key(candidate_answer, visible["answer_type"])
                )

        valid_keys = [key for key in trace_keys if key]
        counts = Counter(valid_keys)
        if invalid_candidate_indices:
            # A missing candidate must never silently strengthen another
            # cluster.  Preserve the already-saved selector output for the
            # whole row and make the failed vote explicit in provenance.
            decision_kind = "invalid_candidate_fallback_saved_selector"
            composed = copy.deepcopy(source_row)
            decision = {
                "kind": decision_kind,
                "chosen_candidate_index": generation.get("selected_index"),
                "invalid_candidate_indices": invalid_candidate_indices,
                "valid_candidate_count": len(valid_keys),
                "cluster_sizes_desc": sorted(counts.values(), reverse=True),
            }
        else:
            top_count = max(counts.values())
            top_keys = sorted(key for key, count in counts.items() if count == top_count)
        if not invalid_candidate_indices and len(top_keys) == 1:
            decision_kind = "unique_modal_cluster"
            chosen_position = trace_keys.index(top_keys[0])
            chosen_trace = traces[chosen_position]
            chosen_answer = canonicalize_final_answer(
                chosen_trace["final_answer"], visible["answer_type"]
            )
            composed = copy.deepcopy(source_row)
            composed["final_answer"] = chosen_answer
            candidate_reasoning = chosen_trace.get("reasoning")
            if isinstance(candidate_reasoning, str) and candidate_reasoning.strip():
                composed["reasoning"] = candidate_reasoning
                composed["solution_steps"] = candidate_reasoning
            composed["raw_response"] = _canonical_json(
                {
                    "final_answer": chosen_answer,
                    "reasoning": composed.get("reasoning", ""),
                    "selection": "offline_unique_modal_cluster",
                }
            )
            decision = {
                "kind": decision_kind,
                "chosen_candidate_index": chosen_trace["index"],
                "top_cluster_size": top_count,
                "cluster_sizes_desc": sorted(counts.values(), reverse=True),
            }
        elif not invalid_candidate_indices:
            decision_kind = "tie_fallback_saved_selector"
            composed = copy.deepcopy(source_row)
            decision = {
                "kind": decision_kind,
                "chosen_candidate_index": generation.get("selected_index"),
                "top_cluster_size": top_count,
                "tied_top_clusters": len(top_keys),
                "cluster_sizes_desc": sorted(counts.values(), reverse=True),
            }
        decision_counts[decision_kind] += 1
        if str(composed.get("final_answer")) != str(source_row.get("final_answer")):
            changed += 1
        output.append(
            _stamp_output(
                composed,
                mode="parallel_consensus",
                condition="maxim_parallel8_consensus_v1",
                rule_id=CONSENSUS_RULE_ID,
                benchmark_fields_used=("task_id", "answer_type"),
                sources=(_source_descriptor("parallel8", source_row),),
                decision=decision,
            )
        )
    return output, {
        "rows": len(output),
        "decision_counts": dict(sorted(decision_counts.items())),
        "final_answer_changed_rows": changed,
        "expected_candidates_per_row": EXPECTED_PARALLEL_CANDIDATES,
    }


def compose_answer_canonicalization(
    benchmark_rows: Iterable[dict[str, Any]],
    solver_rows: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    benchmark, order, sources = _prepare_inputs(
        benchmark_rows,
        (("solver", solver_rows),),
    )
    output: list[dict[str, Any]] = []
    changed_by_type = Counter()
    for task_id in order:
        source_row = sources["solver"][task_id]
        visible = _visible_benchmark_row(benchmark[task_id], task_id)
        canonical = canonicalize_final_answer(
            source_row.get("final_answer"), visible["answer_type"]
        )
        changed = canonical != str(source_row.get("final_answer"))
        composed = copy.deepcopy(source_row)
        composed["final_answer"] = canonical
        if changed:
            changed_by_type[visible["answer_type"] or "unknown"] += 1
        output.append(
            _stamp_output(
                composed,
                mode="answer_canonicalization",
                condition="maxim_answer_canonicalization_v1",
                rule_id=CANONICALIZATION_RULE_ID,
                benchmark_fields_used=("task_id", "answer_type"),
                sources=(_source_descriptor("solver", source_row),),
                decision={
                    "answer_type": visible["answer_type"],
                    "format_changed": changed,
                },
            )
        )
    return output, {
        "rows": len(output),
        "changed_rows": sum(changed_by_type.values()),
        "changed_rows_by_answer_type": dict(sorted(changed_by_type.items())),
    }


def retrieval_metadata_gate(page_rag_row: dict[str, Any]) -> dict[str, Any]:
    """Apply the frozen metadata-only retrieval reliability gate."""
    calls = page_rag_row.get("tool_calls")
    if not isinstance(calls, list):
        raise CompositionError(
            f"page_rag: task {page_rag_row.get('task_id')} tool_calls must be a list"
        )
    successful_chunk_sets: list[set[str]] = []
    error_count = 0
    for position, call in enumerate(calls, start=1):
        if not isinstance(call, dict):
            raise CompositionError(
                f"page_rag: task {page_rag_row.get('task_id')} tool call {position} "
                "is not an object"
            )
        if call.get("tool") != "search_textbooks":
            continue
        if call.get("error"):
            error_count += 1
            continue
        chunk_ids = call.get("returned_chunk_ids")
        if not isinstance(chunk_ids, list):
            raise CompositionError(
                f"page_rag: task {page_rag_row.get('task_id')} successful search "
                "has no returned_chunk_ids list"
            )
        clean_ids = {str(item).strip() for item in chunk_ids if str(item).strip()}
        if clean_ids:
            successful_chunk_sets.append(clean_ids)

    occurrence_counts = Counter(
        chunk_id for chunk_set in successful_chunk_sets for chunk_id in chunk_set
    )
    repeated_chunk_count = sum(count >= 2 for count in occurrence_counts.values())
    use_rag = (
        len(successful_chunk_sets) >= 2
        and repeated_chunk_count >= 1
        and error_count == 0
    )
    return {
        "use_rag": use_rag,
        "successful_search_calls": len(successful_chunk_sets),
        "search_error_calls": error_count,
        "unique_returned_chunks": len(occurrence_counts),
        "repeated_chunk_count": repeated_chunk_count,
    }


def _sum_usage(*rows: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in ("input_tokens", "output_tokens", "latency_s"):
        values: list[float] = []
        all_integral = True
        for row in rows:
            usage = row.get("usage")
            if not isinstance(usage, dict):
                continue
            value = usage.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            values.append(float(value))
            all_integral = all_integral and isinstance(value, int)
        if values:
            total = sum(values)
            result[key] = int(total) if all_integral else round(total, 6)
    return result


def compose_selective_rag_metadata(
    benchmark_rows: Iterable[dict[str, Any]],
    page_rag_rows: Iterable[dict[str, Any]],
    no_tools_rows: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    benchmark, order, sources = _prepare_inputs(
        benchmark_rows,
        (("page_rag", page_rag_rows), ("no_tools", no_tools_rows)),
    )
    output: list[dict[str, Any]] = []
    counts = Counter()
    for task_id in order:
        # Projection is still performed even though the gate uses no benchmark
        # attributes beyond task identity.
        _visible_benchmark_row(benchmark[task_id], task_id)
        page_row = sources["page_rag"][task_id]
        no_tools_row = sources["no_tools"][task_id]
        gate = retrieval_metadata_gate(page_row)
        source_label = "page_rag" if gate["use_rag"] else "no_tools"
        chosen = page_row if gate["use_rag"] else no_tools_row
        counts[source_label] += 1
        composed = copy.deepcopy(chosen)
        # This idea evaluates both frozen branches before selecting one.  Its
        # usage therefore reports both generations, not just the winning row.
        composed["usage"] = _sum_usage(page_row, no_tools_row)
        composed["tool_calls"] = copy.deepcopy(page_row.get("tool_calls", []))
        stamped = _stamp_output(
            composed,
            mode="selective_rag_metadata",
            condition="maxim_selective_rag_metadata_v1",
            rule_id=SELECTIVE_RAG_RULE_ID,
            benchmark_fields_used=("task_id",),
            sources=(
                _source_descriptor("page_rag", page_row),
                _source_descriptor("no_tools", no_tools_row),
            ),
            decision={**gate, "chosen_source": source_label},
        )
        stamped["generation"]["call_count"] = 2
        output.append(stamped)
    return output, {
        "rows": len(output),
        "chosen_source_counts": dict(sorted(counts.items())),
        "gate_is_semantic_evidence_check": False,
        "gate_scope": "retrieval-metadata reliability proxy only",
    }


def _jsonl_bytes(rows: Sequence[dict[str, Any]]) -> bytes:
    return ("\n".join(_canonical_json(row) for row in rows) + "\n").encode("utf-8")


def _atomic_write_pair(
    output_path: Path,
    output_bytes: bytes,
    manifest_path: Path,
    manifest_bytes: bytes,
) -> None:
    if output_path.exists():
        raise CompositionError(f"refusing to overwrite existing output: {output_path}")
    if manifest_path.exists():
        raise CompositionError(f"refusing to overwrite existing manifest: {manifest_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_paths: list[Path] = []
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=output_path.name + ".", dir=output_path.parent, delete=False
        ) as temporary:
            temporary.write(output_bytes)
            temporary.flush()
            os.fsync(temporary.fileno())
            output_temp = Path(temporary.name)
            temporary_paths.append(output_temp)
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=manifest_path.name + ".",
            dir=manifest_path.parent,
            delete=False,
        ) as temporary:
            temporary.write(manifest_bytes)
            temporary.flush()
            os.fsync(temporary.fileno())
            manifest_temp = Path(temporary.name)
            temporary_paths.append(manifest_temp)
        os.replace(output_temp, output_path)
        temporary_paths.remove(output_temp)
        os.replace(manifest_temp, manifest_path)
        temporary_paths.remove(manifest_temp)
    finally:
        for path in temporary_paths:
            try:
                path.unlink()
            except FileNotFoundError:
                pass


def _source_manifest(paths: Sequence[tuple[str, Path]]) -> dict[str, Any]:
    return {
        label: {"path": str(path.resolve()), "sha256": sha256_file(path)}
        for label, path in paths
    }


def build_from_args(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    benchmark_rows = read_jsonl(args.benchmark, "benchmark")
    source_paths: list[tuple[str, Path]] = [("benchmark", args.benchmark)]
    if args.mode == "subject_router":
        page_rows = read_jsonl(args.page_rag, "page_rag")
        no_tools_rows = read_jsonl(args.no_tools, "no_tools")
        rows, stats = compose_subject_router(benchmark_rows, page_rows, no_tools_rows)
        source_paths.extend((("page_rag", args.page_rag), ("no_tools", args.no_tools)))
        rule_id = SUBJECT_ROUTER_RULE_ID
    elif args.mode == "parallel_consensus":
        parallel_rows = read_jsonl(args.parallel, "parallel8")
        rows, stats = compose_parallel_consensus(benchmark_rows, parallel_rows)
        source_paths.append(("parallel8", args.parallel))
        rule_id = CONSENSUS_RULE_ID
    elif args.mode == "answer_canonicalization":
        if args.solver is None:
            raise CompositionError("answer_canonicalization requires --solver")
        solver_rows = read_jsonl(args.solver, "solver")
        rows, stats = compose_answer_canonicalization(benchmark_rows, solver_rows)
        source_paths.append(("solver", args.solver))
        rule_id = CANONICALIZATION_RULE_ID
    elif args.mode == "selective_rag_metadata":
        page_rows = read_jsonl(args.page_rag, "page_rag")
        no_tools_rows = read_jsonl(args.no_tools, "no_tools")
        rows, stats = compose_selective_rag_metadata(
            benchmark_rows, page_rows, no_tools_rows
        )
        source_paths.extend((("page_rag", args.page_rag), ("no_tools", args.no_tools)))
        rule_id = SELECTIVE_RAG_RULE_ID
    else:  # pragma: no cover - argparse guards this
        raise CompositionError(f"unsupported mode: {args.mode}")

    if len(rows) != args.expected_rows:
        raise CompositionError(
            f"composed row count {len(rows)} != expected {args.expected_rows}"
        )
    output_bytes = _jsonl_bytes(rows)
    output_sha256 = hashlib.sha256(output_bytes).hexdigest()
    script_path = Path(__file__).resolve()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "mode": args.mode,
        "rule_id": rule_id,
        "gold_access": False,
        "scoring_performed": False,
        "expected_rows": args.expected_rows,
        "output": {
            "path": str(args.out_jsonl.resolve()),
            "rows": len(rows),
            "sha256": output_sha256,
        },
        "composer": {
            "path": str(script_path),
            "sha256": sha256_file(script_path),
        },
        "sources": _source_manifest(source_paths),
        "stats_without_gold": stats,
    }
    manifest_bytes = (json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    _atomic_write_pair(args.out_jsonl, output_bytes, args.out_manifest, manifest_bytes)
    return rows, manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        required=True,
        choices=(
            "subject_router",
            "parallel_consensus",
            "answer_canonicalization",
            "selective_rag_metadata",
        ),
    )
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--page-rag", type=Path, default=DEFAULT_PAGE_RAG)
    parser.add_argument("--no-tools", type=Path, default=DEFAULT_NO_TOOLS)
    parser.add_argument("--parallel", type=Path, default=DEFAULT_PARALLEL)
    parser.add_argument("--solver", type=Path)
    parser.add_argument("--expected-rows", type=int, default=274)
    parser.add_argument("--out-jsonl", type=Path, required=True)
    parser.add_argument("--out-manifest", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        _, manifest = build_from_args(args)
    except CompositionError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
