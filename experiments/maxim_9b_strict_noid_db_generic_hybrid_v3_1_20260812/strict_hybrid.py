"""Strict no-ID certified-router + generic Qwen3.5-9B hybrid adapter.

The only values passed to the branch selector are OCR text, public answer type,
and public input mode. Runtime IDs are read by the outer adapter only after the
selector has returned an action, solely to preserve evaluation row alignment.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
NOID_DIR = ROOT / "experiments/maxim_9b_content_source_router_noid_v1_20260812"
NOID_IMPLEMENTATION = NOID_DIR / "content_source_router_noid_v1.py"
NOID_FREEZE = NOID_DIR / "output/FREEZE.json"
NOID_SOURCE_DB = NOID_DIR / "output/frozen/source_db.json"
NOID_AUDIT = NOID_DIR / "INDEPENDENT_AUDIT.json"
BASE240 = ROOT / "experiments/maxim_9b_source_expansion_wave_v1_1/final_wave/arms/base240/solver.jsonl"
FREEZE = HERE / "HYBRID_RULE_FREEZE.json"
SIDECAR = HERE / "HYBRID_RULE_FREEZE_SHA256.txt"


class HybridError(RuntimeError):
    pass


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if type(value) is not dict:
            raise HybridError(f"non-object JSONL row {path}:{number}")
        rows.append(value)
    return rows


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(canonical_bytes(dict(row)) for row in rows))


def load_noid_module() -> Any:
    spec = importlib.util.spec_from_file_location("certified_noid_router_frozen_v1", NOID_IMPLEMENTATION)
    if spec is None or spec.loader is None:
        raise HybridError("cannot load certified no-ID router")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_freeze(expected: str) -> tuple[dict[str, Any], dict[str, Any], Any]:
    if sha256(FREEZE) != expected or SIDECAR.read_text(encoding="ascii").strip() != expected:
        raise HybridError("hybrid freeze pin mismatch")
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    if freeze.get("state") != "frozen_unexecuted_unscored":
        raise HybridError("hybrid freeze state mismatch")
    for key, path in {
        "noid_freeze": NOID_FREEZE,
        "noid_source_db": NOID_SOURCE_DB,
        "noid_implementation": NOID_IMPLEMENTATION,
        "noid_independent_audit": NOID_AUDIT,
        "strict_maxim_anchor_base240": BASE240,
        "generic_output_contract": HERE / "GENERIC_OUTPUT_CONTRACT.json",
    }.items():
        descriptor = freeze["artifacts"][key]
        if sha256(path) != descriptor["sha256"] or path.stat().st_size != descriptor["size"]:
            raise HybridError(f"frozen closure mismatch: {key}")
    noid_freeze = json.loads(NOID_FREEZE.read_text(encoding="utf-8"))
    noid_audit = json.loads(NOID_AUDIT.read_text(encoding="utf-8"))
    if (
        noid_audit.get("status") != "PASS"
        or noid_audit.get("freeze_sha256") != sha256(NOID_FREEZE)
        or noid_freeze.get("freeze_projection_sha256") != freeze["certified_branch"]["noid_projection_sha256"]
    ):
        raise HybridError("certified no-ID audit/freeze mismatch")
    source_db = json.loads(NOID_SOURCE_DB.read_text(encoding="utf-8"))
    if source_db.get("contains_task_identity") is not False:
        raise HybridError("source DB identity guard failed")
    return freeze, source_db, load_noid_module()


def selector_observable(row: Mapping[str, Any], eval_set: str) -> dict[str, Any]:
    if eval_set == "maxim274":
        observable = {
            "ocr_text": row.get("ocr_text"),
            "answer_type": row.get("answer_type"),
            "input_mode": row.get("input_mode"),
        }
    elif eval_set == "ykslop_dev185":
        question, choices = row.get("question"), row.get("choices")
        if type(question) is not str or type(choices) is not dict or set(choices) != set("ABCDE"):
            raise HybridError("invalid YKSLOP public content row")
        visible = question + "\n" + "\n".join(f"{label}) {choices[label]}" for label in "ABCDE")
        observable = {"ocr_text": visible, "answer_type": "choice", "input_mode": "text_only"}
    else:
        raise HybridError(f"unsupported eval set: {eval_set}")
    if set(observable) != {"ocr_text", "answer_type", "input_mode"} or type(observable["ocr_text"]) is not str:
        raise HybridError("selector observable schema mismatch")
    return observable


def alignment_id(row: Mapping[str, Any], eval_set: str) -> str:
    value = row.get("controller_id") if eval_set == "maxim274" else row.get("content_sha256")
    if type(value) is not str or not value:
        raise HybridError("missing outer alignment ID")
    return value


def route(eval_set: str, public_path: Path, output_dir: Path, expected_freeze: str) -> dict[str, Any]:
    _freeze, source_db, noid = verify_freeze(expected_freeze)
    public = read_jsonl(public_path)
    decisions: list[dict[str, Any]] = []
    generic_queue: list[dict[str, Any]] = []
    family_counts: dict[str, int] = {}
    accepted_ids: list[str] = []
    for ordinal, row in enumerate(public):
        observable = selector_observable(row, eval_set)
        action = noid.route_observable(observable, source_db)
        identifier = alignment_id(row, eval_set)  # after selector return; alignment only
        accepted = action.get("kind") != "abstain"
        family = str(action.get("family") or "abstain")
        family_counts[family] = family_counts.get(family, 0) + 1
        decisions.append({
            "schema_version": "strict-noid-db-generic-route-decision-v1",
            "input_ordinal": ordinal,
            "runtime_alignment_id": identifier,
            "runtime_alignment_id_used_by_selector": False,
            "selector_projection_fields": ["ocr_text", "answer_type", "input_mode"],
            "branch": "certified_noid" if accepted else "generic_qwen35_9b",
            "action": action,
        })
        if accepted:
            accepted_ids.append(identifier)
        else:
            generic_queue.append(row)
    output_dir.mkdir(parents=True, exist_ok=False)
    write_jsonl(output_dir / "route_decisions.jsonl", decisions)
    write_jsonl(output_dir / "generic_queue.jsonl", generic_queue)
    summary = {
        "schema_version": "strict-noid-db-generic-coverage-v1",
        "eval_set": eval_set,
        "freeze_sha256": expected_freeze,
        "rows": len(public),
        "certified_accepted": len(accepted_ids),
        "certified_coverage": len(accepted_ids) / len(public),
        "abstained_to_generic": len(generic_queue),
        "abstention_rate": len(generic_queue) / len(public),
        "family_counts": family_counts,
        "accepted_runtime_alignment_ids": accepted_ids,
        "score_gold_or_outcomes_used": False,
    }
    (output_dir / "coverage.json").write_bytes(canonical_bytes(summary))
    return summary


def prediction_id(row: Mapping[str, Any]) -> str:
    for key in ("runtime_alignment_id", "controller_id", "task_id", "content_sha256"):
        if type(row.get(key)) is str and row[key]:
            return str(row[key])
    raise HybridError("prediction lacks alignment ID")


def prediction_answer(row: Mapping[str, Any]) -> str:
    for key in ("final_answer", "answer"):
        if type(row.get(key)) is str and row[key].strip():
            return str(row[key]).strip()
    raise HybridError("prediction lacks answer")


def verify_generic_candidate(candidate: Path, candidate_sha: str, audit: Path, audit_sha: str) -> None:
    if sha256(candidate) != candidate_sha or sha256(audit) != audit_sha:
        raise HybridError("generic candidate or audit external pin mismatch")
    candidate_value = json.loads(candidate.read_text(encoding="utf-8"))
    audit_value = json.loads(audit.read_text(encoding="utf-8"))
    serialized = json.dumps(candidate_value, ensure_ascii=False).casefold()
    if "qwen/qwen3.5-9b" not in serialized or audit_value.get("status") != "PASS":
        raise HybridError("generic candidate exact-model/PASS guard failed")
    pinned = {value for key, value in audit_value.items() if key.endswith("freeze_sha256") and type(value) is str}
    if candidate_sha not in pinned:
        raise HybridError("generic audit does not pin candidate freeze")


def compose(
    decisions_path: Path,
    predictions_path: Path,
    output_path: Path,
    expected_freeze: str,
    generic_mode: str,
    candidate: Path | None = None,
    candidate_sha: str | None = None,
    audit: Path | None = None,
    audit_sha: str | None = None,
) -> dict[str, Any]:
    freeze, _db, _noid = verify_freeze(expected_freeze)
    decisions, predictions = read_jsonl(decisions_path), read_jsonl(predictions_path)
    candidate_pin: str | None = None
    if generic_mode == "maxim_base240_control":
        if sha256(predictions_path) != freeze["artifacts"]["strict_maxim_anchor_base240"]["sha256"]:
            raise HybridError("base240 control pin mismatch; archived base249 is forbidden")
    elif generic_mode == "qwen35_9b_frozen_candidate":
        if None in (candidate, candidate_sha, audit, audit_sha):
            raise HybridError("generic candidate freeze and independent audit pins are required")
        verify_generic_candidate(candidate, str(candidate_sha), audit, str(audit_sha))  # type: ignore[arg-type]
        candidate_pin = str(candidate_sha)
    else:
        raise HybridError("unsupported generic mode")
    by_id: dict[str, dict[str, Any]] = {}
    for row in predictions:
        identifier = prediction_id(row)
        if identifier in by_id:
            raise HybridError("duplicate prediction alignment ID")
        by_id[identifier] = row
    required = {row["runtime_alignment_id"] for row in decisions if row["branch"] == "generic_qwen35_9b"}
    if generic_mode == "qwen35_9b_frozen_candidate" and set(by_id) != required:
        raise HybridError("prediction IDs must exactly equal generic queue IDs")
    output: list[dict[str, Any]] = []
    for row in decisions:
        identifier, action = row["runtime_alignment_id"], row["action"]
        if row["branch"] == "certified_noid":
            answer, selected = str(action["answer"]), "certified_noid"
        else:
            if identifier not in by_id:
                raise HybridError(f"missing fallback prediction: {identifier}")
            answer, selected = prediction_answer(by_id[identifier]), generic_mode
        output.append({
            "task_id": identifier,
            "final_answer": answer,
            "condition": "strict_noid_db_generic_hybrid_v3",
            "generation": {
                "selected_branch": selected,
                "identity_used_by_branch_selector": False,
                "selector_projection_fields": ["ocr_text", "answer_type", "input_mode"],
                "hybrid_freeze_sha256": expected_freeze,
                "generic_candidate_freeze_sha256": candidate_pin,
                "generic_transport_used_by_branch_selector": False,
            },
        })
    write_jsonl(output_path, output)
    return {"rows": len(output), "output_sha256": sha256(output_path)}
