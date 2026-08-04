#!/usr/bin/env python3
"""Compose preregistered, outcome-blind consensus repairs over Meta-V2.1.

The default row is always the exact Meta-V2.1 row.  A policy may copy an
exact row from V3.1, Active-Crop, or no-tools only when the frozen normalized
answer-agreement rule fires.  The profile is shared by the complete policy
sweep and must be frozen before source row values or target scores are read.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import unicodedata
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "maxim-consensus-repair-composition-v1"
PROFILE_SCHEMA_VERSION = "maxim-consensus-repair-sweep-preregistration-v1"
POLICIES = (
    "v31_active_pair",
    "v31_no_tools_pair",
    "active_no_tools_pair",
    "v31_active_no_tools_triple",
    "two_of_three_majority",
)
SOURCE_PRIORITY = ("v31", "active", "no_tools")
FORBIDDEN_TOP_LEVEL = {
    "reference_answer",
    "acceptable_answers",
    "gold",
    "gold_answer",
    "correct",
    "new_correct",
    "verdict",
    "judge",
    "score",
}
PREFIX = re.compile(
    r"^(?:final\s+answer|answer|cevap|yan[ıi]t|ответ)\s*[:=\-]\s*",
    re.IGNORECASE,
)


class ConsensusRepairError(ValueError):
    """Raised when a frozen profile or source binding fails."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def normalize_answer(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).strip().casefold()
    text = PREFIX.sub("", text)
    text = text.replace("−", "-").replace("–", "-").replace("—", "-")
    # Agreement only: retain semantic alphanumerics and common numeric symbols,
    # while making whitespace and presentation punctuation irrelevant.
    return "".join(
        character
        for character in text
        if character.isalnum() or character in {".", ",", "/", "%", "+", "-"}
    )


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConsensusRepairError(f"{label}: invalid JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ConsensusRepairError(f"{label}: expected object")
    return value


def _read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except OSError as exc:
        raise ConsensusRepairError(f"{label}: cannot read {path}: {exc}") from exc
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ConsensusRepairError(f"{label}:{line_number}: invalid JSON") from exc
        if not isinstance(row, dict):
            raise ConsensusRepairError(f"{label}:{line_number}: row is not an object")
        forbidden = FORBIDDEN_TOP_LEVEL.intersection(row)
        if forbidden:
            raise ConsensusRepairError(
                f"{label}:{line_number}: forbidden top-level fields: {sorted(forbidden)}"
            )
        task_id = str(row.get("task_id") or "")
        if not task_id or task_id in seen:
            raise ConsensusRepairError(f"{label}:{line_number}: missing or duplicate task_id")
        seen.add(task_id)
        rows.append(row)
    return rows


def _clean_vote(row: Mapping[str, Any]) -> tuple[bool, str]:
    answer = str(row.get("final_answer") or "").strip()
    normalized = normalize_answer(answer)
    clean = (
        not bool(row.get("error"))
        and row.get("forced_answer") is not True
        and bool(normalized)
        and len(answer) <= 256
        and len(normalized) <= 192
    )
    return clean, normalized


def _winner(
    policy: str, votes: Mapping[str, tuple[bool, str]]
) -> tuple[str | None, str]:
    def pair(left: str, right: str) -> tuple[str | None, str]:
        left_clean, left_value = votes[left]
        right_clean, right_value = votes[right]
        if left_clean and right_clean and left_value == right_value:
            return left, f"agreement_{left}_{right}"
        return None, f"no_agreement_{left}_{right}"

    if policy == "v31_active_pair":
        return pair("v31", "active")
    if policy == "v31_no_tools_pair":
        return pair("v31", "no_tools")
    if policy == "active_no_tools_pair":
        return pair("active", "no_tools")
    if policy == "v31_active_no_tools_triple":
        clean = all(votes[name][0] for name in SOURCE_PRIORITY)
        values = {votes[name][1] for name in SOURCE_PRIORITY}
        return (
            ("v31", "agreement_v31_active_no_tools")
            if clean and len(values) == 1
            else (None, "no_three_way_agreement")
        )
    if policy == "two_of_three_majority":
        groups: dict[str, list[str]] = {}
        for name in SOURCE_PRIORITY:
            clean, value = votes[name]
            if clean:
                groups.setdefault(value, []).append(name)
        eligible = [(value, names) for value, names in groups.items() if len(names) >= 2]
        if len(eligible) != 1:
            return None, "no_unique_two_of_three_majority"
        _value, names = eligible[0]
        selected = next(name for name in SOURCE_PRIORITY if name in names)
        return selected, "unique_two_of_three_majority"
    raise ConsensusRepairError(f"unknown policy: {policy}")


def compose(*, profile_path: Path, policy: str, output_dir: Path) -> dict[str, Any]:
    profile_path = profile_path.resolve()
    output_dir = output_dir.resolve()
    if policy not in POLICIES:
        raise ConsensusRepairError(f"policy must be one of {POLICIES}")
    if output_dir.exists():
        raise ConsensusRepairError(f"output directory already exists: {output_dir}")
    profile = _read_json(profile_path, "profile")
    if profile.get("schema_version") != PROFILE_SCHEMA_VERSION:
        raise ConsensusRepairError("profile schema mismatch")
    if profile.get("status") != "frozen_before_source_row_values_and_target_scoring":
        raise ConsensusRepairError("profile status mismatch")
    if profile.get("policies") != list(POLICIES):
        raise ConsensusRepairError("profile policy sweep mismatch")
    if profile.get("source_outcomes_used_for_policy_design") is not False:
        raise ConsensusRepairError("profile permits outcome-conditioned policy design")
    if profile.get("composer_sha256") != sha256_file(Path(__file__).resolve()):
        raise ConsensusRepairError("composer SHA256 mismatch")
    conditions = profile.get("conditions")
    if not isinstance(conditions, Mapping) or conditions.get(policy) != f"maxim_consensus_repair_{policy}_v1":
        raise ConsensusRepairError("condition binding mismatch")

    source_records = profile.get("sources")
    if not isinstance(source_records, Mapping) or list(source_records) != [
        "default_v21",
        "v31",
        "active",
        "no_tools",
    ]:
        raise ConsensusRepairError("source order mismatch")
    loaded: dict[str, list[dict[str, Any]]] = {}
    for name, record in source_records.items():
        if not isinstance(record, Mapping):
            raise ConsensusRepairError(f"source binding is not an object: {name}")
        path = Path(str(record.get("path") or "")).resolve()
        if sha256_file(path) != record.get("sha256"):
            raise ConsensusRepairError(f"source SHA256 mismatch: {name}")
        rows = _read_jsonl(path, name)
        if len(rows) != profile.get("rows") or record.get("rows") != len(rows):
            raise ConsensusRepairError(f"source row count mismatch: {name}")
        loaded[name] = rows
    task_order = [str(row["task_id"]) for row in loaded["default_v21"]]
    for name, rows in loaded.items():
        if [str(row["task_id"]) for row in rows] != task_order:
            raise ConsensusRepairError(f"task order mismatch: {name}")

    output_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    counts: dict[str, int] = {"default": 0, "v31": 0, "active": 0, "no_tools": 0}
    condition = str(conditions[policy])
    for index, task_id in enumerate(task_order):
        default = loaded["default_v21"][index]
        default_norm = normalize_answer(default.get("final_answer"))
        votes = {
            "v31": _clean_vote(loaded["v31"][index]),
            "active": _clean_vote(loaded["active"][index]),
            "no_tools": _clean_vote(loaded["no_tools"][index]),
        }
        selected, decision = _winner(policy, votes)
        if selected is not None and votes[selected][1] == default_norm:
            selected = None
            decision = "agreement_equals_default"
        source_name = selected or "default"
        source_key = "default_v21" if source_name == "default" else source_name
        output_row = copy.deepcopy(loaded[source_key][index])
        output_row["condition"] = condition
        output_rows.append(output_row)
        counts[source_name] += 1
        audit_rows.append(
            {
                "schema_version": "maxim-consensus-repair-selection-audit-v1",
                "queue_index": index,
                "task_id": task_id,
                "policy": policy,
                "selected_source": source_name,
                "decision": decision,
                "default_answer_hash": canonical_sha256(default_norm),
                "voter_answer_hashes": {
                    name: canonical_sha256(value) if clean else None
                    for name, (clean, value) in votes.items()
                },
                "gold_access": False,
                "score_or_judge_access": False,
            }
        )

    output_dir.mkdir(parents=True)
    solver_path = output_dir / "solver.jsonl"
    audit_path = output_dir / "selection_audit.jsonl"
    manifest_path = output_dir / "composition_manifest.json"
    solver_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in output_rows),
        encoding="utf-8",
    )
    audit_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in audit_rows),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "COMPLETE_OUTCOME_BLIND_COMPOSITION",
        "condition": condition,
        "policy": policy,
        "rows": len(output_rows),
        "source_counts": counts,
        "override_rows": len(output_rows) - counts["default"],
        "model_calls": 0,
        "gold_access": False,
        "score_or_judge_access": False,
        "profile_sha256": sha256_file(profile_path),
        "composer_sha256": sha256_file(Path(__file__).resolve()),
        "solver_sha256": sha256_file(solver_path),
        "selection_audit_sha256": sha256_file(audit_path),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--policy", choices=POLICIES, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = compose(profile_path=args.profile, policy=args.policy, output_dir=args.output_dir)
    except (ConsensusRepairError, OSError, json.JSONDecodeError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
