#!/usr/bin/env python3
"""Compare already-resolved dev certificates with five source addresses.

Expected activity numbers are consumed only here, after generic visual resolve.
They are never passed to the resolver and never alter a certificate.  This is
an alignment audit, not an accuracy or answer-correctness evaluation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from evidence_os.math12_activity_source import (  # noqa: E402
    load_math12_inventory,
    load_math12_render_manifest,
    load_math12_source_certificate,
    verify_math12_source_certificate,
    write_canonical_json,
)
from evidence_os.official_ogm import canonical_json_sha256  # noqa: E402


AUDIT_SCHEMA = "math12-dev5-source-binding-audit-v1"


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"certificate is not an object: {path}")
    return value


def _parse_case(value: str) -> tuple[str, int, Path]:
    label, expected, path = value.split("=", 2)
    if not label or not expected.isdigit():
        raise argparse.ArgumentTypeError("case must be LABEL=EXPECTED_ACTIVITY=CERTIFICATE")
    return label, int(expected), Path(path)


def _parse_solution(value: str) -> tuple[str, Path]:
    label, path = value.split("=", 1)
    if not label:
        raise argparse.ArgumentTypeError("solution must be LABEL=OFFICIAL_SOLUTION")
    return label, Path(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--render-manifest", type=Path, required=True)
    parser.add_argument("--page-root", type=Path)
    parser.add_argument("--case", action="append", type=_parse_case, required=True)
    parser.add_argument("--solution", action="append", type=_parse_solution, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.case) != 5 or len({item[0] for item in args.case}) != 5:
        parser.error("the dev audit requires exactly five unique labels")
    solutions = dict(args.solution)
    if len(args.solution) != 5 or set(solutions) != {item[0] for item in args.case}:
        parser.error("the dev audit requires one unique solution for every case")
    inventory = load_math12_inventory(args.inventory)
    manifest = load_math12_render_manifest(
        args.render_manifest, inventory, page_root=args.page_root
    )
    rows: list[dict[str, Any]] = []
    identities: set[tuple[str, str, str]] = set()
    for label, expected, path in args.case:
        certificate = load_math12_source_certificate(path)
        verify_math12_source_certificate(inventory, manifest, certificate)
        value = certificate.to_mapping()
        decision = value["decision"]
        selected = decision.get("selected_activity_number")
        solution = _load(solutions[label])
        solution_text = str(solution.get("official_solution_text") or "")
        next_header = f"Etkinlik No.: {expected + 1}"
        no_next_header = next_header not in solution_text
        if not no_next_header:
            raise ValueError(f"{label}: official solution leaks next activity header")
        if (
            solution.get("activity_number") != expected
            or solution.get("source_certificate_projection_sha256")
            != certificate.certificate_projection_sha256
        ):
            raise ValueError(f"{label}: solution record is not bound to its certificate")
        identities.add(
            (
                value["document_id"],
                value["pdf_sha256"],
                value["inventory_projection_sha256"],
            )
        )
        rows.append(
            {
                "label": label,
                "certificate_path": str(path),
                "task_image_sha256": value["task_image_sha256"],
                "expected_activity_for_alignment_audit_only": expected,
                "resolver_accepted": bool(decision.get("accepted")),
                "resolved_content_page": decision.get("selected_content_page"),
                "resolved_activity_number": selected,
                "resolved_key_page_start": decision.get("key_page_start"),
                "resolved_key_page_end": decision.get("key_page_end"),
                "alignment_matches_expected": bool(decision.get("accepted"))
                and selected == expected,
                "next_activity_header": next_header,
                "next_activity_header_absent_from_solution_text": no_next_header,
                "certificate_projection_sha256": value[
                    "certificate_projection_sha256"
                ],
            }
        )
    if len(identities) != 1:
        raise ValueError("dev certificates do not share one pinned source identity")
    document_id, pdf_sha, inventory_sha = next(iter(identities))
    projection = {
        "schema_version": AUDIT_SCHEMA,
        "scope": "source_binding_alignment_only_no_answers_no_correctness_no_score",
        "selection_disclosure": (
            "Math12 source family was selected post-hoc after inspecting the five dev inputs; "
            "this audit cannot establish transfer or a new accuracy result."
        ),
        "resolver_is_blind_to_expected_activity": True,
        "document_id": document_id,
        "pdf_sha256": pdf_sha,
        "inventory_projection_sha256": inventory_sha,
        "rows": rows,
        "summary": {
            "cases": len(rows),
            "resolver_accepted": sum(item["resolver_accepted"] for item in rows),
            "alignment_matches": sum(item["alignment_matches_expected"] for item in rows),
            "solutions_without_next_activity_header": sum(
                item["next_activity_header_absent_from_solution_text"] for item in rows
            ),
        },
    }
    projection["audit_projection_sha256"] = canonical_json_sha256(projection)
    write_canonical_json(args.output, projection)
    print(json.dumps(projection["summary"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
