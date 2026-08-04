#!/usr/bin/env python3
"""Freeze the gold-blind queue for the paired RAG/no-RAG support verifier.

The historical page-RAG and no-tools answers are reused, but the evidence is
explicitly the separately pinned structural-context artifact.  Tasks whose
canonical final answers agree, or whose structural context is not safe, never
enter the model queue and deterministically default to the no-RAG row.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "maxim-paired-rag-norag-semantic-support-preparation-v1"
QUEUE_SCHEMA_VERSION = "maxim-paired-rag-norag-semantic-support-queue-v1"
PROFILE_SCHEMA_VERSION = "maxim-paired-rag-norag-semantic-support-profile-v1"
EXPECTED_ROWS = 274
MAX_CONTEXT_CHARS = 9_000
FROZEN_BENCHMARK_SHA256 = (
    "5a6a38ccae7835f0d015f6e5979834208347b8e6e7a8d6884e4af97605f51ed9"
)
FROZEN_PAGE_RAG_SHA256 = (
    "62bc952c3802308bc0fbf8d8dc1f82ec523a3ab1e3264bae87a5f8828021d75d"
)
FROZEN_NO_RAG_SHA256 = (
    "496236da966ed68aa81af3d33da1c40b85c5a11b342de253ada244f97320de8f"
)
FROZEN_STRUCTURAL_CONTEXT_SHA256 = (
    "293f15f9acb847436f1810ef95d8e54d429f9db57d267e60f2ed4ccfb8e18da7"
)

VISIBLE_TASK_FIELDS = (
    "task_id",
    "subject",
    "grade",
    "question",
    "question_images",
    "answer_type",
)
FORBIDDEN_KEYS = frozenset(
    {
        "reference_answer",
        "reference_solution",
        "gold_answer",
        "gold_solution",
        "acceptable_answers",
        "score",
        "scores",
        "accuracy",
        "correct",
        "strict_correct",
        "judge",
        "judge_verdict",
        "reward",
    }
)


class PreparationError(RuntimeError):
    pass


def stable_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def stable_sha256(value: Any) -> str:
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PreparationError(f"cannot load JSON {path}: {exc}") from exc


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8-sig") as source:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise PreparationError(
                        f"invalid JSONL {path}:{line_number}: {exc}"
                    ) from exc
                if not isinstance(value, dict):
                    raise PreparationError(
                        f"JSONL row is not an object: {path}:{line_number}"
                    )
                rows.append(value)
    except OSError as exc:
        raise PreparationError(f"cannot load JSONL {path}: {exc}") from exc
    return rows


def index_unique(
    rows: Iterable[Mapping[str, Any]], source_name: str
) -> tuple[list[str], dict[str, Mapping[str, Any]]]:
    order: list[str] = []
    indexed: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        task_id = str(row.get("task_id") or "")
        if not task_id:
            raise PreparationError(f"{source_name}: row without task_id")
        if task_id in indexed:
            raise PreparationError(f"{source_name}: duplicate task_id {task_id}")
        order.append(task_id)
        indexed[task_id] = row
    return order, indexed


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="\n") as sink:
        for row in rows:
            sink.write(stable_json(row) + "\n")
    os.replace(temporary, path)


def audit_gold_free(value: Any, path: str = "$") -> None:
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key).casefold()
            if key in FORBIDDEN_KEYS:
                raise PreparationError(f"forbidden key {raw_key!r} at {path}")
            audit_gold_free(child, f"{path}.{raw_key}")
    elif isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        for index, child in enumerate(value):
            audit_gold_free(child, f"{path}[{index}]")


def _compact_text(value: Any, limit: int) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    marker = " [truncated]"
    return text[: limit - len(marker)].rstrip() + marker


def canonical_answer(value: Any, answer_type: str) -> str:
    """Conservative, answer-type-aware equality used only for call skipping."""

    text = unicodedata.normalize("NFKC", str(value or "")).strip()
    text = re.sub(r"^\\boxed\{(.*)\}$", r"\1", text, flags=re.DOTALL)
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'"}:
        text = text[1:-1].strip()
    kind = str(answer_type or "").strip().casefold()
    if kind == "choice":
        match = re.fullmatch(r"\s*([A-Ea-e])(?:[\).:\-])?\s*", text)
        return match.group(1).upper() if match else text.casefold()
    if kind == "numeric":
        text = text.replace("−", "-").replace("–", "-")
        text = re.sub(r"\s+", "", text)
        if re.fullmatch(r"[+-]?\d+,\d+(?:[a-zA-Z%°]+)?", text):
            text = text.replace(",", ".", 1)
        return text.casefold()
    return re.sub(r"\s+", " ", text).casefold()


def visible_task(row: Mapping[str, Any]) -> dict[str, Any]:
    result = {key: row.get(key) for key in VISIBLE_TASK_FIELDS}
    if not str(result.get("task_id") or ""):
        raise PreparationError("benchmark row has no task_id")
    audit_gold_free(result)
    return result


def candidate_payload(row: Mapping[str, Any], label: str) -> dict[str, str]:
    answer = _compact_text(row.get("final_answer"), 160)
    if not answer:
        raise PreparationError(f"{label}: empty final_answer")
    return {
        "final_answer": answer,
        "reasoning": _compact_text(row.get("reasoning"), 1_800),
        "solution_steps": _compact_text(row.get("solution_steps"), 2_000),
    }


def pack_safe_context(context_row: Mapping[str, Any]) -> list[dict[str, Any]]:
    safety = context_row.get("safety")
    if context_row.get("route") != "structural_evidence":
        return []
    if not isinstance(safety, Mapping) or safety.get("safe") is not True:
        return []
    evidence = context_row.get("evidence")
    if not isinstance(evidence, list):
        return []

    packed: list[dict[str, Any]] = []
    remaining = MAX_CONTEXT_CHARS
    seen: set[str] = set()
    for item in evidence:
        if not isinstance(item, Mapping):
            continue
        chunk_id = str(item.get("chunk_id") or "").strip()
        text = str(item.get("text") or "")
        if not chunk_id or not text or chunk_id in seen or remaining <= 0:
            continue
        seen.add(chunk_id)
        selected = text[:remaining]
        if not selected:
            break
        packed.append(
            {
                "chunk_id": chunk_id,
                "document_id": str(item.get("document_id") or ""),
                "page_number": item.get("page_number"),
                "primary_type": str(item.get("primary_type") or ""),
                "relation": str(item.get("relation") or ""),
                "text": selected,
                "text_cut_by_queue": len(selected) < len(text),
            }
        )
        remaining -= len(selected)
    return packed


def classify_task(
    benchmark_row: Mapping[str, Any],
    page_row: Mapping[str, Any],
    no_rag_row: Mapping[str, Any],
    context_row: Mapping[str, Any],
) -> tuple[str, list[dict[str, Any]]]:
    answer_type = str(benchmark_row.get("answer_type") or "")
    if canonical_answer(page_row.get("final_answer"), answer_type) == canonical_answer(
        no_rag_row.get("final_answer"), answer_type
    ):
        return "same_answer_default_no_rag", []
    contexts = pack_safe_context(context_row)
    if not contexts:
        return "unsafe_or_missing_context_default_no_rag", []
    return "semantic_support_verifier", contexts


def build_queue(
    benchmark_rows: Sequence[Mapping[str, Any]],
    page_rows: Sequence[Mapping[str, Any]],
    no_rag_rows: Sequence[Mapping[str, Any]],
    context_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    order, benchmark = index_unique(benchmark_rows, "benchmark")
    _, page = index_unique(page_rows, "page_rag")
    _, no_rag = index_unique(no_rag_rows, "no_rag")
    _, contexts = index_unique(context_rows, "structural_contexts")
    if len(order) != EXPECTED_ROWS:
        raise PreparationError(f"benchmark rows must be {EXPECTED_ROWS}, got {len(order)}")
    expected = set(order)
    for label, source in (("page_rag", page), ("no_rag", no_rag), ("contexts", contexts)):
        if set(source) != expected:
            raise PreparationError(f"{label}: task-id set differs from benchmark")

    queue: list[dict[str, Any]] = []
    route_counts: dict[str, int] = {}
    route_task_ids: dict[str, list[str]] = {}
    for task_id in order:
        task = visible_task(benchmark[task_id])
        page_payload = candidate_payload(page[task_id], f"page_rag:{task_id}")
        no_rag_payload = candidate_payload(no_rag[task_id], f"no_rag:{task_id}")
        route, packed_contexts = classify_task(
            task, page[task_id], no_rag[task_id], contexts[task_id]
        )
        route_counts[route] = route_counts.get(route, 0) + 1
        route_task_ids.setdefault(route, []).append(task_id)
        if route != "semantic_support_verifier":
            continue
        payload: dict[str, Any] = {
            "schema_version": QUEUE_SCHEMA_VERSION,
            "queue_index": len(queue),
            **task,
            "rag_candidate": page_payload,
            "no_rag_candidate": no_rag_payload,
            "contexts": packed_contexts,
        }
        audit_gold_free(payload)
        payload["request_sha256"] = stable_sha256(payload)
        queue.append(payload)
    return queue, {
        "benchmark_rows": len(order),
        "model_queue_rows": len(queue),
        "route_counts": dict(sorted(route_counts.items())),
        "route_task_ids": {key: value for key, value in sorted(route_task_ids.items())},
        "same_answer_definition": "conservative answer-type-aware canonical equality",
        "context_character_cap_per_task": MAX_CONTEXT_CHARS,
    }


def source_descriptor(path: Path, expected_sha256: str) -> dict[str, Any]:
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise PreparationError(
            f"source hash mismatch for {path}: expected {expected_sha256}, got {actual}"
        )
    rows = len(load_jsonl(path))
    return {"path": str(path), "rows": rows, "sha256": actual, "bytes": path.stat().st_size}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--page-rag", type=Path, required=True)
    parser.add_argument("--no-rag", type=Path, required=True)
    parser.add_argument("--contexts", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args(argv)

    paths = {
        "benchmark": args.benchmark.resolve(),
        "page_rag": args.page_rag.resolve(),
        "no_rag": args.no_rag.resolve(),
        "contexts": args.contexts.resolve(),
        "profile": args.profile.resolve(),
    }
    expected = {
        "benchmark": FROZEN_BENCHMARK_SHA256,
        "page_rag": FROZEN_PAGE_RAG_SHA256,
        "no_rag": FROZEN_NO_RAG_SHA256,
        "contexts": FROZEN_STRUCTURAL_CONTEXT_SHA256,
    }
    sources = {
        label: source_descriptor(paths[label], digest)
        for label, digest in expected.items()
    }
    if any(value["rows"] != EXPECTED_ROWS for value in sources.values()):
        raise PreparationError("every frozen JSONL source must contain 274 rows")

    profile = load_json(paths["profile"])
    if not isinstance(profile, Mapping) or profile.get("schema_version") != PROFILE_SCHEMA_VERSION:
        raise PreparationError("profile schema mismatch")
    profile_sha256 = sha256_file(paths["profile"])

    queue, stats = build_queue(
        load_jsonl(paths["benchmark"]),
        load_jsonl(paths["page_rag"]),
        load_jsonl(paths["no_rag"]),
        load_jsonl(paths["contexts"]),
    )
    write_jsonl(args.queue.resolve(), queue)

    script_dir = Path(__file__).resolve().parent
    code_paths = {
        "prepare": Path(__file__).resolve(),
        "runner": script_dir / "run_maxim_paired_rag_norag_semantic_support_v1.py",
        "composer": script_dir / "compose_maxim_paired_rag_norag_semantic_support_v1.py",
        "tests": script_dir / "test_maxim_paired_rag_norag_semantic_support_v1.py",
    }
    missing_code = [str(path) for path in code_paths.values() if not path.is_file()]
    if missing_code:
        raise PreparationError(f"required code files are missing: {missing_code}")

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "frozen_before_generation_and_before_score",
        "generation_launched": False,
        "score_or_judge_inspected_by_this_stage": False,
        "gold_access": False,
        "sources": sources,
        "profile": {
            "path": str(paths["profile"]),
            "sha256": profile_sha256,
        },
        "code": {
            label: {"path": str(path), "sha256": sha256_file(path)}
            for label, path in code_paths.items()
        },
        "queue": {
            "path": str(args.queue.resolve()),
            "rows": len(queue),
            "sha256": sha256_file(args.queue.resolve()),
            "bytes": args.queue.resolve().stat().st_size,
        },
        "stats": stats,
        "freeze_rule": (
            "Any source, profile, queue, code, threshold, retry, prompt, schema, "
            "or composition change requires a new preregistration before generation."
        ),
    }
    write_json(args.manifest.resolve(), manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
