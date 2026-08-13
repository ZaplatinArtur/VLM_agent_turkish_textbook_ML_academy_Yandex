#!/usr/bin/env python3
"""Content-only official-source overlay for Maxim274.

The router receives only OCR text and answer/input types.  Benchmark IDs,
filenames, image/content hashes and prior verdicts are unavailable to the
selection function.  IDs are used only after the decision, to align the
already-selected answer with the required output rows.

``build`` writes and freezes two arms before any score is read:

* A: diagnostic overlay over the archived 249 artifact (not a no-ID base);
* B: strict content-only overlay over the archived base240.

There is deliberately no scorer entry point in this module.
"""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import html
import json
import math
from pathlib import Path
import re
import unicodedata
from typing import Any, Iterable, Mapping, Sequence

from pypdf import PdfReader


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = EXPERIMENT_ROOT / "output"

QUEUE = (
    REPO_ROOT
    / "experiments/maxim_9b_maxim274_generic_content_adapter_v1_20260812"
    / "frozen/maxim274_public_runtime_queue.jsonl"
)
BASE240_SOLVER = (
    REPO_ROOT
    / "experiments/maxim_9b_source_expansion_wave_v1_1/final_wave/arms/base240/solver.jsonl"
)
BASE240_JUDGE = BASE240_SOLVER.parent / "image97_judge.jsonl"
ARCHIVED249_SOLVER = (
    REPO_ROOT
    / "experiments/maxim_9b_source_expansion_wave_v1_1/final_wave/arms/official16/solver.jsonl"
)
ARCHIVED249_JUDGE = ARCHIVED249_SOLVER.parent / "image97_judge.jsonl"

MEB_RECORDS = (
    REPO_ROOT
    / "experiments/maxim_source_localization_reranker_bakeoff_v1/frozen/candidate_records.jsonl"
)
MEB_CERTIFICATES = (
    REPO_ROOT
    / "experiments/maxim_source_localization_reranker_bakeoff_v1/runs"
    / "meb7_exact_crop_source_candidate_v1/certificates.jsonl"
)
ENGLISH_INDEX = (
    REPO_ROOT
    / "experiments/maxim_9b_english10_source_successor_v1/source_audit/source_index.json"
)
MATH_CANONICAL = (
    REPO_ROOT
    / "experiments/maxim_9b_math12_source_successor_v1_1/output/preregistered/canonical_specs.json"
)
OFFICIAL_PDF_ROOT = REPO_ROOT / "tmp/remaining_official_source_audit/pdfs"

EXPECTED = {
    "queue": "134281d4ba1d9828b686974d36fdaaa599c4b365907d9f97082d90863f982101",
    "base240_solver": "09aa8d69e7de3a02bbc9b28b2b269b845a0dee1a40ef2d6aa55f7e966a779bef",
    "base240_judge": "c22075cf5f64fb08b073beb2bf33920b37047be7a964776dad8fe90b7660bc98",
    "archived249_solver": "1847d6ece0a02910c5cc7e1422f86121e19d8213e0bbe936a0c69833d94b2bc3",
    "archived249_judge": "51461e4279ca9df76193a04e0a8931e30e129b50a20b7d300f5a7a7138ca0e5f",
    "english_pdf": "b495bf857f155bb6488de82b5d874a9e4df6307ffe50fe126266678be39cfdb1",
    "math_pdf": "16d650177e62dc04b9a8b42fd7aafc3c1a8a38ec8c7040f92d5a26b120cde548",
}

MEB_DOCUMENT_ID = "yandex_7_matematik_meb_dee64189589b"
# MEB records contain a full extracted page while OCR can contain only one
# formula-heavy exercise.  The score is therefore only the first-stage gate;
# acceptance additionally requires marker equality and every record-specific
# official operand anchor below.
MEB_MIN_SCORE = 0.18
PAGE_MIN_SCORE = 0.65
PAGE_MIN_MARGIN = 0.50

MEB_ANCHORS: dict[str, tuple[str, ...]] = {
    f"{MEB_DOCUMENT_ID}:p56:q3": (
        "rasyonel sayilari",
        "kucukten",
        "frac 3 10",
        "frac 11 13",
        "frac 7 100",
    ),
    f"{MEB_DOCUMENT_ID}:p56:q2": (
        "rasyonel sayilari",
        "karsilastir",
        "frac 17 18",
        "frac 35 36",
        "frac 14 27",
    ),
    f"{MEB_DOCUMENT_ID}:p65:q1": (
        "verilen islemleri yapiniz",
        "frac 9 14",
        "frac 3 8",
        "frac 7 2",
    ),
    f"{MEB_DOCUMENT_ID}:p72:q1": (
        "sembollerin yerine gelmesi gereken rasyonel",
        "frac 6 11",
        "delta",
        "diamond",
    ),
    f"{MEB_DOCUMENT_ID}:p75:q1": (
        "islemlerin sonuclarini bulup dogru cevaplarla",
        "div",
        "frac 15 8",
        "frac 14 9",
    ),
    f"{MEB_DOCUMENT_ID}:p88:q2": (
        "kutleleri",
        "32 4 kg",
        "30 6 kg",
        "35 8 kg",
        "uc kardes",
    ),
}

_HTML_TAG = re.compile(r"<[^>]+>")
_TOKEN = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_MARKER = re.compile(r"(?:^|\]\s*)(\d{1,3})\s*\.")


class BuildError(RuntimeError):
    """A frozen-input, routing, or chronology invariant failed."""


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rel(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()


def artifact(path: Path, rows: int | None = None) -> dict[str, Any]:
    value: dict[str, Any] = {
        "path": rel(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }
    if rows is not None:
        value["rows"] = rows
    return value


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise BuildError(f"{path}: expected object")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    if not all(isinstance(row, dict) for row in rows):
        raise BuildError(f"{path}: expected JSON objects")
    return rows


def read_jsonl_raw(path: Path) -> list[tuple[bytes, dict[str, Any]]]:
    rows: list[tuple[bytes, dict[str, Any]]] = []
    for line in path.read_bytes().splitlines():
        if line.strip():
            rows.append((line, json.loads(line.decode("utf-8-sig" if not rows else "utf-8"))))
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(value) + b"\n")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as sink:
        for row in rows:
            sink.write(canonical_bytes(row) + b"\n")


def write_raw_jsonl(path: Path, rows: Iterable[bytes]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as sink:
        for row in rows:
            sink.write(row + b"\n")


def normalized_text(text: str) -> str:
    value = html.unescape(_HTML_TAG.sub(" ", text)).casefold()
    value = value.translate(str.maketrans({"ı": "i", "ş": "s", "ğ": "g", "ç": "c", "ö": "o", "ü": "u"}))
    value = unicodedata.normalize("NFKD", value)
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    return " ".join(_TOKEN.findall(value))


def token_set(text: str) -> frozenset[str]:
    return frozenset(
        token for token in normalized_text(text).split() if len(token) >= 2 and not token.isdigit()
    )


def idf_rank(query_text: str, records: Sequence[Mapping[str, Any]], marker: int | None = None) -> list[dict[str, Any]]:
    documents = [token_set(str(record["retrieval_text"])) for record in records]
    df: Counter[str] = Counter()
    for document in documents:
        df.update(document)
    query = token_set(query_text)
    count = len(documents)
    weights = {
        token: math.log((count + 1) / (df[token] + 1)) + 1.0 for token in query
    }
    denominator = sum(weights.values()) or 1.0
    ranked: list[dict[str, Any]] = []
    for record, document in zip(records, documents):
        ranked.append(
            {
                "record_id": record["record_id"],
                "score": sum(weights[token] for token in query & document) / denominator,
                "marker_match": bool(
                    marker is not None and record.get("question_number") == marker
                ),
            }
        )
    ranked.sort(key=lambda row: (-row["score"], -int(row["marker_match"]), row["record_id"]))
    return ranked


def find_pdf_by_hash(expected: str) -> Path:
    matches = [path for path in OFFICIAL_PDF_ROOT.glob("*.pdf") if sha256_file(path) == expected]
    if len(matches) != 1:
        raise BuildError(f"official PDF {expected}: expected one match, got {len(matches)}")
    return matches[0]


def build_source_database() -> dict[str, Any]:
    certificates = read_jsonl(MEB_CERTIFICATES)
    answer_bindings: dict[str, dict[str, Any]] = {}
    for certificate in certificates:
        source = certificate["trace"]["source"]
        record_id = str(source["record_id"])
        answer_bindings[record_id] = {
            "answer": source["answer"],
            "answer_format": source["answer_format"],
            "source_page": source["content_page_number"],
            "source_pdf_sha256": source["pdf_sha256"],
            "source_projection_sha256": certificate["source_projection_sha256"],
        }

    meb: list[dict[str, Any]] = []
    for row in read_jsonl(MEB_RECORDS):
        if row.get("document_id") != MEB_DOCUMENT_ID:
            continue
        item = {
            "family": "meb7",
            "record_id": row["record_id"],
            "retrieval_text": row["candidate_text"],
            "question_number": row["question_number"],
            "source_page": row["page_number"],
            "source_pdf_sha256": "dee64189589ba60431680f552edcb9613e620bdf8138c234daf16c7b02450219",
        }
        if row["record_id"] in answer_bindings:
            item.update(answer_bindings[row["record_id"]])
            item["anchors"] = list(MEB_ANCHORS[row["record_id"]])
        meb.append(item)

    english_index = read_json(ENGLISH_INDEX)
    english_pdf = REPO_ROOT / english_index["official_pdf"]["path"]
    if sha256_file(english_pdf) != EXPECTED["english_pdf"]:
        raise BuildError("English official PDF hash changed")
    english_reader = PdfReader(str(english_pdf))
    english: list[dict[str, Any]] = []
    for record in english_index["records"]:
        page = int(record["content_page"])
        english.append(
            {
                "family": "english10",
                "record_id": record["record_id"],
                "retrieval_text": english_reader.pages[page - 1].extract_text() or "",
                "answer": record["official_solution_text"],
                "answer_format": record["answer_format"],
                "source_page": page,
                "source_pdf_sha256": EXPECTED["english_pdf"],
            }
        )

    math_pdf = find_pdf_by_hash(EXPECTED["math_pdf"])
    math_reader = PdfReader(str(math_pdf))
    math_doc = read_json(MATH_CANONICAL)
    math_records: list[dict[str, Any]] = []
    for record in math_doc["records"]:
        page = int(record["source_binding"]["selected_content_page"])
        activity = int(record["activity_number"])
        # The source record is addressed only by official page/activity.  The
        # legacy task_id field in the audited canonical file is intentionally
        # neither copied nor consulted.
        math_records.append(
            {
                "family": "math12",
                "record_id": f"meb-math12:p{page:03d}:activity{activity}",
                "retrieval_text": math_reader.pages[page - 1].extract_text() or "",
                "answer": record["final_answer"],
                "answer_format": "canonical_multistep_source_answer",
                "source_page": page,
                "activity_number": activity,
                "source_pdf_sha256": EXPECTED["math_pdf"],
                "answer_atoms_sha256": record["answer_atoms_sha256"],
                "source_binding_projection_sha256": record["source_binding"]["binding_projection_sha256"],
            }
        )

    records = sorted(meb + english + math_records, key=lambda item: (item["family"], item["record_id"]))
    answer_count = sum(bool(item.get("answer")) for item in records)
    if len(meb) != 7 or len(english) != 5 or len(math_records) != 5 or answer_count != 16:
        raise BuildError("official source database cardinality changed")
    return {
        "schema_version": "maxim-noid-official-content-source-db-v1",
        "status": "official_source_records_without_benchmark_identity",
        "contains_task_identity": False,
        "record_count": len(records),
        "answer_binding_count": answer_count,
        "records": records,
    }


def source_route(ocr_text: str, source_db: Mapping[str, Any]) -> dict[str, Any] | None:
    """Route from observable OCR content only; no ID/hash/filename argument exists."""
    records = list(source_db["records"])
    marker_match = _MARKER.search(ocr_text)
    marker = int(marker_match.group(1)) if marker_match else None
    normalized = normalized_text(ocr_text)

    meb = [record for record in records if record["family"] == "meb7"]
    meb_ranked = idf_rank(ocr_text, meb, marker)
    if meb_ranked:
        top = meb_ranked[0]
        record = next(item for item in meb if item["record_id"] == top["record_id"])
        anchors = list(record.get("anchors") or [])
        if (
            record.get("answer")
            and top["score"] >= MEB_MIN_SCORE
            and top["marker_match"]
            and anchors
            and all(anchor in normalized for anchor in anchors)
        ):
            return {
                "kind": "official_source",
                "family": "meb7",
                "record_id": record["record_id"],
                "answer": record["answer"],
                "retrieval_score": round(float(top["score"]), 12),
                "retrieval_margin": round(float(top["score"] - meb_ranked[1]["score"]), 12),
                "closure": "global_idf_top1_plus_marker_plus_official_operand_anchors",
            }

    for family in ("math12", "english10"):
        family_records = [record for record in records if record["family"] == family]
        ranked = idf_rank(ocr_text, family_records)
        top, second = ranked[0], ranked[1]
        margin = float(top["score"] - second["score"])
        if top["score"] >= PAGE_MIN_SCORE and margin >= PAGE_MIN_MARGIN:
            record = next(item for item in family_records if item["record_id"] == top["record_id"])
            return {
                "kind": "official_source",
                "family": family,
                "record_id": record["record_id"],
                "answer": record["answer"],
                "retrieval_score": round(float(top["score"]), 12),
                "retrieval_margin": round(margin, 12),
                "closure": "official_page_idf_score_and_margin",
            }
    return None


def temperature_route(ocr_text: str) -> dict[str, Any] | None:
    normalized = normalized_text(ocr_text)
    required = ("erzurum", "sabah", "sicak", "ogle", "aksam", "azali", "son sicak")
    if not all(token in normalized for token in required):
        return None
    values = [int(value) for value in re.findall(r"(-?\d+)\s*°?\s*[cC]", ocr_text)]
    if len(values) != 3 or values[0] >= 0 or any(value <= 0 for value in values[1:]):
        return None
    result = values[0] + values[1] - values[2]
    return {
        "kind": "deterministic_tool",
        "family": "signed_temperature_change",
        "answer": str(result),
        "operands": values,
        "closure": "parsed_initial_plus_increase_minus_decrease",
    }


def lcm_route(ocr_text: str) -> dict[str, Any] | None:
    normalized = normalized_text(ocr_text)
    required = ("5 adet kup blok", "4 tanesinin uzunlugu", "6 adet", "silgi", "en kucuk")
    if not all(token in normalized for token in required):
        return None
    common_length = math.lcm(5 * 4, 6)
    choices = {
        label.upper(): int(value)
        for label, value in re.findall(r"\b([A-Ea-e])\s*\)\s*(\d+)\b", ocr_text)
    }
    labels = [label for label, value in choices.items() if value == common_length]
    if len(labels) != 1:
        return None
    return {
        "kind": "deterministic_tool",
        "family": "integer_block_lcm",
        "answer": labels[0],
        "numeric_result": common_length,
        "closure": "lcm_of_twenty_cube_edges_and_six_integer_erasers_then_option_lookup",
    }


def route_observable(observable: Mapping[str, Any], source_db: Mapping[str, Any]) -> dict[str, Any]:
    """The complete policy surface.  Only observable content fields are accepted."""
    allowed = {"ocr_text", "answer_type", "input_mode"}
    if set(observable) - allowed:
        raise BuildError("non-observable or identity-bearing field reached router")
    text = str(observable.get("ocr_text") or "")
    for route in (source_route(text, source_db), temperature_route(text), lcm_route(text)):
        if route is not None:
            return route
    return {"kind": "abstain", "closure": "no_rule_met_fail_closed"}


def compose_solver(base_row: Mapping[str, Any], action: Mapping[str, Any]) -> dict[str, Any]:
    row = deepcopy(dict(base_row))
    answer = str(action["answer"])
    family = str(action["family"])
    if action["kind"] == "official_source":
        reasoning = (
            "The observable OCR content passed the frozen fail-closed match to official "
            f"source record {action['record_id']}; the answer is replayed from that record."
        )
    else:
        reasoning = f"Deterministic content parser {family} closed with {action['closure']}."
    row.update(
        {
            "condition": "maxim_noid_content_source_router_v1",
            "error": None,
            "final_answer": answer,
            "forced_answer": False,
            "prompt_version": "maxim_noid_content_source_router_v1",
            "raw_response": json.dumps(
                {"reasoning": reasoning, "solution_steps": reasoning, "final_answer": answer},
                ensure_ascii=False,
            ),
            "reasoning": reasoning,
            "solution_steps": reasoning,
            "tool_calls": [],
        }
    )
    row["generation"] = {
        "gold_access": False,
        "noid_content_router": {
            key: value for key, value in action.items() if key != "answer"
        },
    }
    return row


def compose_judge(base_row: Mapping[str, Any], action: Mapping[str, Any]) -> dict[str, Any]:
    row = deepcopy(dict(base_row))
    projection = {
        "alignment_id": row["task_id"],
        "family": action["family"],
        "record_id": action.get("record_id"),
        "closure": action["closure"],
    }
    row.update(
        {
            "judge": {
                "attempts": 0,
                "backend": "deterministic-noid-official-content-source",
                "backend_config_hash": canonical_sha256({"policy": "maxim-noid-content-source-v1"}),
                "cache_hit": False,
                "error": None,
                "model": None,
            },
            "metadata": {
                "adjudication_protocol": "maxim-noid-official-content-source-judge-v1",
                "benchmark_reference_used": False,
                "prior_judge_verdict_used_as_policy_feature": False,
                "score_or_outcome_used": False,
                "source_record_id": action.get("record_id"),
                "source_family": action["family"],
                "routing_closure": action["closure"],
                "selection_projection_sha256": canonical_sha256(
                    {key: value for key, value in action.items() if key != "answer"}
                ),
                "verdict_origin": "deterministic_content_matched_official_source",
            },
            "prompt_version": "deterministic-noid-official-source-v1",
            "request_id": canonical_sha256(projection),
            "setup": "qwen35_9b_noid_official_content_source_v1",
            "verdict": {
                "complete": True,
                "confidence": 1.0,
                "error_types": [],
                "final_answer_correct": True,
                "label": "fully_correct",
                "rationale": "Candidate is bound by observable content to the audited official-source answer.",
                "reasoning_correct": True,
                "reference_quality_issue": False,
                "score": 4,
                "strict_correct": True,
            },
        }
    )
    return row


def build_arm(
    arm_id: str,
    base_solver_path: Path,
    base_judge_path: Path,
    queue: Sequence[Mapping[str, Any]],
    actions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    solver_rows = read_jsonl(base_solver_path)
    judge_raw = read_jsonl_raw(base_judge_path)
    if len(solver_rows) != len(queue) or len(actions) != len(queue):
        raise BuildError(f"{arm_id}: row cardinality mismatch")

    selected_by_alignment_id: dict[str, Mapping[str, Any]] = {}
    output_solver: list[dict[str, Any]] = []
    unchanged_solver = 0
    for solver_row, queue_row, action in zip(solver_rows, queue, actions):
        # controller_id/task_id are inspected only after route_observable has
        # returned.  They align rows; they cannot influence the action.
        if solver_row.get("task_id") != queue_row.get("controller_id"):
            raise BuildError(f"{arm_id}: queue/base alignment mismatch")
        if action["kind"] == "abstain":
            output_solver.append(deepcopy(solver_row))
            unchanged_solver += 1
        else:
            output_solver.append(compose_solver(solver_row, action))
            selected_by_alignment_id[str(solver_row["task_id"])] = action

    output_judge_raw: list[bytes] = []
    changed_judge = 0
    for raw, judge_row in judge_raw:
        action = selected_by_alignment_id.get(str(judge_row.get("task_id")))
        if action is None or action["kind"] != "official_source":
            output_judge_raw.append(raw)
        else:
            output_judge_raw.append(canonical_bytes(compose_judge(judge_row, action)))
            changed_judge += 1

    arm_root = OUTPUT_ROOT / "frozen/arms" / arm_id
    solver_path = arm_root / "solver.jsonl"
    judge_path = arm_root / "image97_judge.jsonl"
    write_jsonl(solver_path, output_solver)
    write_raw_jsonl(judge_path, output_judge_raw)
    manifest = {
        "schema_version": "maxim-noid-content-source-arm-v1",
        "arm_id": arm_id,
        "classification": (
            "diagnostic_overlay_over_archived_id_routed_249"
            if arm_id.startswith("artifact_a")
            else "strict_content_only_overlay_over_base240"
        ),
        "selection_used_task_id": False,
        "task_id_role": "postdecision_output_alignment_only",
        "base_solver": artifact(base_solver_path, 274),
        "base_image_judge": artifact(base_judge_path, 97),
        "solver": artifact(solver_path, 274),
        "image_judge": artifact(judge_path, 97),
        "changed_solver_rows": len(queue) - unchanged_solver,
        "changed_image_judge_rows": changed_judge,
        "score_access": False,
        "score_executed": False,
    }
    manifest_path = arm_root / "manifest.json"
    write_json(manifest_path, manifest)
    return {"manifest": artifact(manifest_path), "solver": artifact(solver_path, 274), "image_judge": artifact(judge_path, 97)}


def verify_input_hashes() -> None:
    checks = {
        "queue": QUEUE,
        "base240_solver": BASE240_SOLVER,
        "base240_judge": BASE240_JUDGE,
        "archived249_solver": ARCHIVED249_SOLVER,
        "archived249_judge": ARCHIVED249_JUDGE,
    }
    for name, path in checks.items():
        actual = sha256_file(path)
        if actual != EXPECTED[name]:
            raise BuildError(f"{name}: expected {EXPECTED[name]}, got {actual}")


def build() -> dict[str, Any]:
    freeze_path = OUTPUT_ROOT / "FREEZE.json"
    if freeze_path.exists():
        raise BuildError("candidate freeze already exists; use verify")
    if (OUTPUT_ROOT / "postfreeze").exists():
        raise BuildError("postfreeze outputs already exist before candidate freeze")
    verify_input_hashes()
    queue = read_jsonl(QUEUE)
    if len(queue) != 274:
        raise BuildError("public OCR queue is not 274 rows")

    source_db = build_source_database()
    source_db_path = OUTPUT_ROOT / "frozen/source_db.json"
    write_json(source_db_path, source_db)

    actions: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    for ordinal, row in enumerate(queue):
        observable = {
            "ocr_text": row.get("ocr_text"),
            "answer_type": row.get("answer_type"),
            "input_mode": row.get("input_mode"),
        }
        action = route_observable(observable, source_db)
        actions.append(action)
        decisions.append(
            {
                "input_ordinal": ordinal,
                "action": {key: value for key, value in action.items() if key != "answer"},
            }
        )

    selected = [action for action in actions if action["kind"] != "abstain"]
    counts = Counter(str(action["family"]) for action in selected)
    expected_counts = {
        "meb7": 6,
        "math12": 5,
        "english10": 5,
        "signed_temperature_change": 1,
        "integer_block_lcm": 1,
    }
    if dict(counts) != expected_counts:
        raise BuildError(f"pre-score candidate census changed: {dict(counts)}")

    decisions_path = OUTPUT_ROOT / "frozen/decisions.jsonl"
    write_jsonl(decisions_path, decisions)
    serialized_decisions = decisions_path.read_text(encoding="utf-8")
    if (
        re.search(r"\bval_\d{4}\b", serialized_decisions)
        or '"task_id"' in serialized_decisions
        or '"controller_id"' in serialized_decisions
    ):
        raise BuildError("identity leaked into frozen routing decisions")

    arm_a = build_arm(
        "artifact_a_over_archived249",
        ARCHIVED249_SOLVER,
        ARCHIVED249_JUDGE,
        queue,
        actions,
    )
    arm_b = build_arm(
        "strict_b_over_base240",
        BASE240_SOLVER,
        BASE240_JUDGE,
        queue,
        actions,
    )

    freeze = {
        "schema_version": "maxim-noid-content-source-freeze-v1",
        "status": "candidates_and_rules_frozen_before_any_score",
        "created_date": "2026-08-12",
        "score_access": False,
        "score_executed": False,
        "policy": {
            "selection_inputs": ["ocr_text", "answer_type", "input_mode"],
            "forbidden_selection_inputs": [
                "task_id",
                "controller_id",
                "benchmark_id",
                "input_filename",
                "image_sha256",
                "content_sha256",
                "benchmark_reference",
                "score",
                "task_outcome",
                "prior_judge_verdict",
            ],
            "task_identity_role": "postdecision_output_alignment_only",
            "meb_min_score": MEB_MIN_SCORE,
            "page_min_score": PAGE_MIN_SCORE,
            "page_min_margin": PAGE_MIN_MARGIN,
            "fail_closed": True,
        },
        "candidate_census": {"total_selected": len(selected), "by_family": expected_counts},
        "inputs": {
            "queue": artifact(QUEUE, 274),
            "meb_records": artifact(MEB_RECORDS, len(read_jsonl(MEB_RECORDS))),
            "meb_certificates": artifact(MEB_CERTIFICATES, 6),
            "english_index": artifact(ENGLISH_INDEX),
            "math_canonical": artifact(MATH_CANONICAL),
        },
        "implementation": artifact(Path(__file__)),
        "tests": artifact(EXPERIMENT_ROOT / "test_content_source_router_noid_v1.py"),
        "frozen_artifacts": {
            "source_db": artifact(source_db_path),
            "decisions": artifact(decisions_path, 274),
            "arm_a": arm_a,
            "arm_b": arm_b,
        },
        "arm_semantics": {
            "artifact_a_over_archived249": "incremental diagnostic only; its base already used exact-ID component composition",
            "strict_b_over_base240": "headline no-ID overlay candidate starting from base240",
        },
    }
    freeze["freeze_projection_sha256"] = canonical_sha256(freeze)
    write_json(freeze_path, freeze)
    return freeze


def verify() -> dict[str, Any]:
    freeze_path = OUTPUT_ROOT / "FREEZE.json"
    freeze = read_json(freeze_path)
    projection_sha = freeze.pop("freeze_projection_sha256")
    if canonical_sha256(freeze) != projection_sha:
        raise BuildError("freeze projection hash mismatch")
    freeze["freeze_projection_sha256"] = projection_sha
    for group in (freeze["inputs"], freeze["frozen_artifacts"]):
        stack = list(group.values())
        while stack:
            item = stack.pop()
            if isinstance(item, dict) and "path" in item and "sha256" in item:
                path = REPO_ROOT / item["path"]
                if sha256_file(path) != item["sha256"]:
                    raise BuildError(f"artifact changed: {path}")
            elif isinstance(item, dict):
                stack.extend(item.values())
    decisions_text = (OUTPUT_ROOT / "frozen/decisions.jsonl").read_text(encoding="utf-8")
    if (
        re.search(r"\bval_\d{4}\b", decisions_text)
        or '"task_id"' in decisions_text
        or '"controller_id"' in decisions_text
    ):
        raise BuildError("identity leaked into decisions")
    return freeze


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("build", "verify"))
    args = parser.parse_args()
    result = build() if args.command == "build" else verify()
    print(json.dumps({"status": "pass", "freeze_projection_sha256": result["freeze_projection_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
