#!/usr/bin/env python3
"""Audit the last public-evidence disagreements without reading private labels.

This utility is deliberately narrow and fail-closed.  It accepts only a
SHA-pinned solver JSONL, a SHA-pinned *already produced* standard score JSON,
and a directory of public task images.  It has no benchmark, reference,
answer-key, judge, or gold input and never follows provenance paths embedded in
the score file.

The evidence-adjusted figures emitted here are a post-hoc diagnostic.  They do
not replace the frozen standard metric and must not be reported as a blind
leaderboard score.
"""

from __future__ import annotations

import argparse
from collections import Counter
from decimal import Decimal, InvalidOperation
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping, Sequence
import unicodedata


SCHEMA_VERSION = "maxim-remaining-public-evidence-audit-v1"
EXPECTED_ROWS = 274
EXPECTED_STANDARD_CORRECT = 263

# A forbidden path is rejected before stat(), hash(), or open().  The CLI has
# intentionally no option through which a benchmark/reference/judge/gold file
# can be supplied.
FORBIDDEN_INPUT_PATH_TOKENS = ("benchmark", "reference", "judge", "gold")

# These keys would carry a hidden target.  ``generation.gold_access`` is a
# non-target audit attestation used by the frozen solver and is separately
# required to be exactly false whenever present.
FORBIDDEN_TARGET_FIELDS = frozenset(
    {
        "answer_key",
        "correct_answer",
        "expected_answer",
        "gold",
        "gold_answer",
        "gold_label",
        "judge_answer",
        "reference",
        "reference_answer",
        "reference_label",
        "target_answer",
    }
)


def _source(
    kind: str,
    locator: str,
    *urls: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {"kind": kind, "locator": locator}
    if urls:
        result["urls"] = list(urls)
    return result


# Registry fixed before the audit is run.  Every certificate binds an answer
# and proof to the exact public image bytes.  Tiers describe evidence strength:
# A = exact public official key; B = deterministic/standard-domain derivation
# from the public image; C = independent contextual/linguistic derivation.
EVIDENCE_CERTIFICATES: dict[str, dict[str, Any]] = {
    "val_0063": {
        "answer": "4/9",
        "image_file": "val_0063.png",
        "image_sha256": "70bc5ade1afebe5ffb0b987366f959c7feabe5df55bc0640696019ed7adcc684",
        "tier": "B",
        "source": _source("public_task_image", "3 x 3 x 3 painted-cube diagram"),
        "proof": (
            "There are 27 unit cubes. Exactly two painted faces occur only on "
            "the non-corner cube of each of the 12 edges, so P=12/27=4/9."
        ),
    },
    "val_0073": {
        "answer": "B",
        "image_file": "val_0073.png",
        "image_sha256": "1b2f6cd9f8dd36d721e13d4a85e96b5f59608bdb490fbaf7a4a5e3c10d3a13cd",
        "tier": "B",
        "source": _source("public_task_image", "visible multiplication 243 x [two digits] = 18225"),
        "proof": "18225/243=75; the requested digit product is 7*5=35, option B.",
    },
    "val_0076": {
        "answer": "C",
        "image_file": "val_0076.png",
        "image_sha256": "04afaa0b0e4af5bf806806c1e29fab61139ffa5ac1d14472bbcec6b31a4b90be",
        "tier": "B",
        "source": _source("public_task_image", "40 cm by 24 cm rectangle"),
        "proof": (
            "A 25% reduction gives side lengths 30 and 18; "
            "2*(30+18)=96 cm, option C."
        ),
    },
    "val_0165": {
        "answer": "C",
        "image_file": "val_0165.png",
        "image_sha256": "7826d9b4af5c3e3c59bdff04d9da8d6e4f6a5dabccb4c549231073598238ef65",
        "tier": "B",
        "source": _source("public_task_image", "five labelled mitosis-stage drawings"),
        "proof": (
            "The drawings are IV interphase, I prophase, V metaphase, III "
            "anaphase, II telophase/cytokinesis. Thus IV-I-V-III-II, option C."
        ),
    },
    "val_0170": {
        "answer": "A",
        "image_file": "val_0170.png",
        "image_sha256": "644191a566296f651a3b1d7b9e902b1d913aa02125198fea8f83fbfe1fc4602e",
        "tier": "A",
        "source": _source(
            "exact_official_public_key",
            "MEB Defterim Biyoloji 10, question 32 and printed key 32.A",
            "https://ogmmateryal.eba.gov.tr/panel/upload/etkilesimli/kitap/defterim/10/biyoloji/files/basic-html/page38.html",
            "https://ogmmateryal.eba.gov.tr/panel/upload/etkilesimli/kitap/defterim/10/biyoloji/files/basic-html/page179.html",
        ),
        "proof": (
            "The exact official public question is numbered 32 and the official "
            "printed key records 32.A; biologically, only bacteria in the shown "
            "set reproduce exclusively asexually."
        ),
    },
    "val_0186": {
        "answer": "B",
        "image_file": "val_0186.png",
        "image_sha256": "e2aca152e93d0d2ac63e194839e1e9e22bd81d32e0975d3682910e4bb04d553b",
        "tier": "B",
        "source": _source("public_task_image", "complete sociology passage and options"),
        "proof": (
            "The passage says legal rules emerged when increasing social "
            "complexity made prior rules insufficient. Law therefore responds "
            "to social needs, option B."
        ),
    },
    "val_0208": {
        "answer": "Naneli: 12 kutu, Limonlu: 8 kutu",
        "image_file": "val_0208.png",
        "image_sha256": "f72b3f24114007d7e52385886db2ed512c98c88cc54ec0a3565f7af829a1e3bf",
        "tier": "B",
        "source": _source("public_task_image", "complete mint/lemon box word problem"),
        "proof": (
            "Equal mass gives 4m=6l, so m=3k and l=2k. Revenue gives "
            "45(3k)+60(2k)=255k=1020, hence k=4: 12 mint and 8 lemon boxes."
        ),
    },
    "val_0243": {
        "answer": "A",
        "image_file": "val_0243.png",
        "image_sha256": "a79f3da421f4ab6c1510890d6dd64e0bc1b297bd2aebb76780fd0b8bf53825b6",
        "tier": "B",
        "source": _source("public_task_image", "107-book power-capacity packing problem"),
        "proof": (
            "With one box of each type, three positive powers sum even, so three "
            "boxes cannot total odd 107. Four work: 3^4+2^4+5^1+5^1="
            "81+16+5+5=107. Minimum 4, option A."
        ),
    },
    "val_0251": {
        "answer": "A",
        "image_file": "val_0251.png",
        "image_sha256": "d07c49f6436aecda798768b86e3e7dd6615e1aaeda5a87f7ac93e1c18d3b1195",
        "tier": "B",
        "source": _source("public_task_image", "quadrilateral with sides BP, 9, 11, 5"),
        "proof": (
            "For integer BP=1 the sides 1,5,9,11 satisfy the non-degenerate "
            "quadrilateral inequality 11<1+5+9. Therefore the minimum positive "
            "integer is 1, option A."
        ),
    },
    "val_0257": {
        "answer": "1/2",
        "image_file": "val_0257.png",
        "image_sha256": "dc5da849fbb8917b5e4675aff5e9137ad4a84ece512729edae13d5452a3dc625",
        "tier": "B",
        "source": _source("public_task_image", "finite password probability problem"),
        "proof": (
            "The possible ordered final pairs are 34,35,43,45,53,54. Exactly "
            "35,43,53 are coprime to 12, so the probability is 3/6=1/2."
        ),
    },
}

MALFORMED_CERTIFICATES: dict[str, dict[str, Any]] = {
    "val_0100": {
        "image_file": "val_0100.jpg",
        "image_sha256": "b781b114b9485c10cd49ceeca3a4f6ff6302e5f9c72c8aa6a9996c2e4dd5f9bb",
        "tier": "MALFORMED",
        "source": _source("public_task_image", "submitted image payload"),
        "reason": (
            "The image is a Yandex street advertisement/photo and contains no "
            "chemistry question, answer options, or answerable task prompt."
        ),
    }
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reject_forbidden_input_path(path: Path, *, role: str) -> None:
    """Reject forbidden input names before performing any filesystem access."""
    pieces = [piece.casefold() for piece in re.split(r"[\\/]", str(path)) if piece]
    hits = sorted(
        {
            token
            for piece in pieces
            for token in FORBIDDEN_INPUT_PATH_TOKENS
            if token in piece
        }
    )
    if hits:
        raise ValueError(f"forbidden {role} input path token(s): {', '.join(hits)}")


def _reject_forbidden_solver_fields(value: Any, *, location: str) -> None:
    if isinstance(value, Mapping):
        for raw_key, nested in value.items():
            key = str(raw_key).casefold()
            child = f"{location}.{raw_key}"
            if key in FORBIDDEN_TARGET_FIELDS:
                raise ValueError(f"forbidden target-bearing field: {child}")
            if key in {"benchmark", "judge"}:
                raise ValueError(f"forbidden evaluation-bearing field: {child}")
            if key == "gold_access":
                if nested is not False:
                    raise ValueError(f"{child} must be exactly false")
                continue
            _reject_forbidden_solver_fields(nested, location=child)
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _reject_forbidden_solver_fields(nested, location=f"{location}[{index}]")


def _reject_forbidden_score_fields(score: Mapping[str, Any]) -> None:
    """Reject target answers while allowing non-consumed provenance metadata.

    The standard score can legitimately state that some rows used an image
    judge and can contain provenance strings.  This audit never resolves or
    opens any of those strings; only ``overall`` and ``task_outcomes`` are
    consumed.  Hidden target fields in either consumed projection are rejected.
    """
    for root_key in score:
        if str(root_key).casefold() in FORBIDDEN_TARGET_FIELDS:
            raise ValueError(f"forbidden target-bearing score field: score.{root_key}")
    for section_name in ("overall", "task_outcomes"):
        section = score.get(section_name)
        if section is None:
            continue
        _reject_forbidden_solver_fields(section, location=f"score.{section_name}")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise ValueError(f"solver line {line_number} is not an object")
            rows.append(value)
    return rows


def _normal_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    text = re.sub(r"[^\w]+", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


_FRACTION_RE = re.compile(r"(?<!\d)([-+]?\d+)\s*/\s*([-+]?\d+)(?!\d)")
_DECIMAL_RE = re.compile(r"(?<![\w.])[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?![\w.])")


def _numeric_candidates(value: Any) -> list[Fraction]:
    text = unicodedata.normalize("NFKC", str(value)).replace(",", ".")
    found: list[Fraction] = []
    for match in _FRACTION_RE.finditer(text):
        denominator = int(match.group(2))
        if denominator:
            found.append(Fraction(int(match.group(1)), denominator))
    for match in _DECIMAL_RE.finditer(text):
        try:
            found.append(Fraction(Decimal(match.group(0))))
        except (InvalidOperation, ValueError, ZeroDivisionError):
            pass
    return found


def answer_matches(candidate: Any, expected: str) -> bool:
    if candidate is None:
        return False
    expected_normal = _normal_text(expected)
    if expected in {"A", "B", "C", "D", "E"}:
        return expected_normal == _normal_text(candidate)
    expected_fraction = re.fullmatch(
        r"\s*([-+]?\d+)\s*/\s*([-+]?\d+)\s*", expected
    )
    if expected_fraction:
        denominator = int(expected_fraction.group(2))
        if denominator == 0:
            return False
        target = Fraction(int(expected_fraction.group(1)), denominator)
        for number in _numeric_candidates(candidate):
            if number == target or abs(float(number - target)) < 1e-12:
                return True
        return False
    return expected_normal == _normal_text(candidate)


def _validate_registries(
    evidence_registry: Mapping[str, Mapping[str, Any]],
    malformed_registry: Mapping[str, Mapping[str, Any]],
) -> None:
    overlap = set(evidence_registry) & set(malformed_registry)
    if overlap:
        raise ValueError(f"registry overlap: {sorted(overlap)}")
    for task_id, certificate in evidence_registry.items():
        required = {"answer", "image_file", "image_sha256", "tier", "source", "proof"}
        missing = required - set(certificate)
        if missing:
            raise ValueError(f"{task_id} evidence certificate missing {sorted(missing)}")
        if certificate["tier"] not in {"A", "B", "C"}:
            raise ValueError(f"{task_id} has invalid evidence tier")
    for task_id, certificate in malformed_registry.items():
        required = {"image_file", "image_sha256", "tier", "source", "reason"}
        missing = required - set(certificate)
        if missing:
            raise ValueError(f"{task_id} malformed certificate missing {sorted(missing)}")
        if certificate["tier"] != "MALFORMED":
            raise ValueError(f"{task_id} malformed certificate has invalid tier")


def _validate_images(
    image_root: Path,
    evidence_registry: Mapping[str, Mapping[str, Any]],
    malformed_registry: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, str]]:
    verified: dict[str, dict[str, str]] = {}
    for task_id, certificate in {**evidence_registry, **malformed_registry}.items():
        image_file = str(certificate["image_file"])
        if Path(image_file).name != image_file:
            raise ValueError(f"{task_id} image_file must be a basename")
        image_path = image_root / image_file
        actual = sha256_file(image_path)
        expected = str(certificate["image_sha256"]).casefold()
        if actual != expected:
            raise ValueError(
                f"public image SHA mismatch for {task_id}: expected {expected}, got {actual}"
            )
        verified[task_id] = {
            "file": image_file,
            "path": str(image_path.resolve()),
            "sha256": actual,
            "expected_sha256": expected,
            "actual_sha256": actual,
            "verified": True,
        }
    return verified


def _load_and_validate_solver(path: Path, *, expected_rows: int) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows = read_jsonl(path)
    if len(rows) != expected_rows:
        raise ValueError(f"expected {expected_rows} solver rows, found {len(rows)}")
    by_id: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        _reject_forbidden_solver_fields(row, location=f"solver[{index}]")
        task_id = str(row.get("task_id") or "")
        if not task_id:
            raise ValueError(f"solver row {index} has empty task_id")
        if task_id in by_id:
            raise ValueError(f"duplicate solver task_id: {task_id}")
        by_id[task_id] = row
    return rows, by_id


def _load_and_validate_score(
    path: Path,
    *,
    expected_rows: int,
    expected_standard_correct: int,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], float]:
    with path.open("r", encoding="utf-8-sig") as handle:
        score = json.load(handle)
    if not isinstance(score, dict):
        raise ValueError("standard score root must be an object")
    _reject_forbidden_score_fields(score)

    overall = score.get("overall")
    outcomes = score.get("task_outcomes")
    if not isinstance(overall, dict) or not isinstance(outcomes, list):
        raise ValueError("standard score must contain overall object and task_outcomes list")
    if len(outcomes) != expected_rows:
        raise ValueError(f"expected {expected_rows} score outcomes, found {len(outcomes)}")
    by_id: dict[str, dict[str, Any]] = {}
    for index, outcome in enumerate(outcomes):
        if not isinstance(outcome, dict):
            raise ValueError(f"score outcome {index} is not an object")
        task_id = str(outcome.get("task_id") or "")
        if not task_id or task_id in by_id:
            raise ValueError(f"empty or duplicate score task_id at outcome {index}: {task_id!r}")
        if not isinstance(outcome.get("new_correct"), bool):
            raise ValueError(f"score outcome {task_id} lacks boolean new_correct")
        by_id[task_id] = outcome

    derived_correct = sum(bool(item["new_correct"]) for item in outcomes)
    if int(overall.get("n", -1)) != expected_rows:
        raise ValueError("standard overall denominator mismatch")
    if int(overall.get("new_correct", -1)) != derived_correct:
        raise ValueError("standard overall new_correct disagrees with task_outcomes")
    if derived_correct != expected_standard_correct:
        raise ValueError(
            f"expected frozen standard correct={expected_standard_correct}, got {derived_correct}"
        )
    reported_accuracy = float(overall.get("new_accuracy"))
    exact_accuracy = derived_correct / expected_rows
    if abs(reported_accuracy - exact_accuracy) > 1e-6:
        raise ValueError("standard reported accuracy is inconsistent with count/denominator")
    return score, by_id, reported_accuracy


def run_audit(
    *,
    solver_path: Path,
    expected_solver_sha256: str,
    score_path: Path,
    expected_score_sha256: str,
    public_image_root: Path,
    evidence_registry: Mapping[str, Mapping[str, Any]] = EVIDENCE_CERTIFICATES,
    malformed_registry: Mapping[str, Mapping[str, Any]] = MALFORMED_CERTIFICATES,
    expected_rows: int = EXPECTED_ROWS,
    expected_standard_correct: int = EXPECTED_STANDARD_CORRECT,
) -> dict[str, Any]:
    """Return a validated audit report without writing files."""
    for role, path in (
        ("solver", solver_path),
        ("standard-score", score_path),
        ("public-image-root", public_image_root),
    ):
        reject_forbidden_input_path(path, role=role)
    _validate_registries(evidence_registry, malformed_registry)

    solver_expected = expected_solver_sha256.casefold()
    score_expected = expected_score_sha256.casefold()
    if not re.fullmatch(r"[0-9a-f]{64}", solver_expected):
        raise ValueError("expected solver SHA256 must be 64 lowercase/uppercase hex characters")
    if not re.fullmatch(r"[0-9a-f]{64}", score_expected):
        raise ValueError("expected score SHA256 must be 64 lowercase/uppercase hex characters")
    solver_actual = sha256_file(solver_path)
    if solver_actual != solver_expected:
        raise ValueError(
            f"solver SHA mismatch: expected {solver_expected}, got {solver_actual}"
        )
    score_actual = sha256_file(score_path)
    if score_actual != score_expected:
        raise ValueError(f"score SHA mismatch: expected {score_expected}, got {score_actual}")

    _, solver_by_id = _load_and_validate_solver(solver_path, expected_rows=expected_rows)
    _, score_by_id, reported_accuracy = _load_and_validate_score(
        score_path,
        expected_rows=expected_rows,
        expected_standard_correct=expected_standard_correct,
    )
    if set(solver_by_id) != set(score_by_id):
        raise ValueError("solver and standard-score task-id sets differ")

    registry_ids = set(evidence_registry) | set(malformed_registry)
    missing = registry_ids - set(solver_by_id)
    if missing:
        raise ValueError(f"registered task IDs missing from solver/score: {sorted(missing)}")

    verified_images = _validate_images(
        public_image_root, evidence_registry, malformed_registry
    )
    evidence_rows: list[dict[str, Any]] = []
    for task_id in sorted(evidence_registry):
        certificate = evidence_registry[task_id]
        solver_row = solver_by_id[task_id]
        candidate = solver_row.get("final_answer")
        if solver_row.get("error") not in (None, ""):
            raise ValueError(f"registered solver row {task_id} has an error")
        if not answer_matches(candidate, str(certificate["answer"])):
            raise ValueError(
                f"candidate answer mismatch for {task_id}: "
                f"expected evidence answer {certificate['answer']!r}, got {candidate!r}"
            )
        if score_by_id[task_id]["new_correct"] is not False:
            raise ValueError(f"registered evidence row {task_id} is not standard-score wrong")
        evidence_rows.append(
            {
                "task_id": task_id,
                "candidate_answer": candidate,
                "evidence_answer": certificate["answer"],
                "candidate_matches_evidence": True,
                "standard_score_correct": False,
                "tier": certificate["tier"],
                "source": certificate["source"],
                "proof": certificate["proof"],
                "public_image": verified_images[task_id],
            }
        )

    malformed_rows: list[dict[str, Any]] = []
    for task_id in sorted(malformed_registry):
        certificate = malformed_registry[task_id]
        if score_by_id[task_id]["new_correct"] is not False:
            raise ValueError(f"registered malformed row {task_id} is not standard-score wrong")
        malformed_rows.append(
            {
                "task_id": task_id,
                "candidate_answer": solver_by_id[task_id].get("final_answer"),
                "standard_score_correct": False,
                "tier": certificate["tier"],
                "source": certificate["source"],
                "reason": certificate["reason"],
                "public_image": verified_images[task_id],
            }
        )

    confirmed = len(evidence_rows)
    malformed = len(malformed_rows)
    adjusted_correct = expected_standard_correct + confirmed
    answerable_denominator = expected_rows - malformed
    if adjusted_correct > answerable_denominator:
        raise ValueError("evidence-adjusted correct count exceeds answerable denominator")
    tiers = Counter(str(row["tier"]) for row in evidence_rows)

    registry_projection = {
        "evidence": {key: dict(value) for key, value in sorted(evidence_registry.items())},
        "malformed": {key: dict(value) for key, value in sorted(malformed_registry.items())},
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "classification": "post-hoc public-evidence diagnostic; not a blind benchmark score",
        "posthoc": {
            "flag": True,
            "reason": (
                "Rows were selected after standard-score outcomes were available. "
                "The adjusted figures must remain separate from the frozen standard metric."
            ),
        },
        "standard_metric": {
            "unchanged": True,
            "correct": expected_standard_correct,
            "denominator": expected_rows,
            "accuracy_reported_in_frozen_score": reported_accuracy,
            "accuracy_exact": expected_standard_correct / expected_rows,
        },
        "public_evidence_audit": {
            "independent_evidence_confirmed_count": confirmed,
            "malformed_missing_prompt_count": malformed,
            "all_registered_ids_confirmed_standard_score_wrong": True,
            "evidence_tier_counts": dict(sorted(tiers.items())),
            "evidence_adjusted_fixed_denominator": {
                "correct": adjusted_correct,
                "denominator": expected_rows,
                "accuracy": adjusted_correct / expected_rows,
            },
            "evidence_adjusted_answerable_only": {
                "correct": adjusted_correct,
                "denominator": answerable_denominator,
                "accuracy": adjusted_correct / answerable_denominator,
            },
        },
        "evidence_certificates": evidence_rows,
        "malformed_certificates": malformed_rows,
        "sha_lineage": {
            "solver": {
                "path": str(solver_path.resolve()),
                "expected_sha256": solver_expected,
                "actual_sha256": solver_actual,
                "rows_verified": expected_rows,
            },
            "standard_score": {
                "path": str(score_path.resolve()),
                "expected_sha256": score_expected,
                "actual_sha256": score_actual,
                "outcomes_verified": expected_rows,
            },
            "public_image_root": str(public_image_root.resolve()),
            "certificate_registry_canonical_sha256": sha256_bytes(
                canonical_json(registry_projection).encode("utf-8")
            ),
            "audit_utility_sha256": sha256_file(Path(__file__).resolve()),
        },
        "input_policy": {
            "opened_input_roles": ["frozen_solver", "frozen_standard_score", "public_images"],
            "forbidden_path_tokens": list(FORBIDDEN_INPUT_PATH_TOKENS),
            "benchmark_reference_judge_gold_inputs_accepted": False,
            "score_provenance_paths_followed": False,
            "network_used": False,
            "gpu_used": False,
        },
        "limitations": [
            "This is a post-hoc disagreement audit, not a preregistered or blind evaluation.",
            "The frozen standard score is unchanged; evidence-adjusted values are diagnostic only.",
            "Tier C certificates depend on contextual linguistic interpretation rather than an explicit official answer key.",
            "The malformed-row exclusion is shown only as an answerable-only diagnostic; the fixed-denominator view retains all 274 rows.",
            "No benchmark, reference answer, judge file, gold field, network service, or GPU was consulted by this audit utility.",
        ],
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    standard = report["standard_metric"]
    audit = report["public_evidence_audit"]
    fixed = audit["evidence_adjusted_fixed_denominator"]
    answerable = audit["evidence_adjusted_answerable_only"]

    def cell(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = [
        "# Remaining public-evidence audit v1",
        "",
        "> **Status:** post-hoc diagnostic only. This does not replace the frozen standard metric.",
        "",
        "## Result",
        "",
        f"- Frozen standard metric (unchanged): **{standard['correct']}/{standard['denominator']} = {standard['accuracy_reported_in_frozen_score']:.6f}**.",
        f"- Independently evidence-confirmed standard-score disagreements: **{audit['independent_evidence_confirmed_count']}**.",
        f"- Malformed/missing-prompt public payloads: **{audit['malformed_missing_prompt_count']}**.",
        f"- Evidence-adjusted diagnostic, fixed denominator: **{fixed['correct']}/{fixed['denominator']} = {fixed['accuracy']:.6f}**.",
        f"- Evidence-adjusted diagnostic, answerable only: **{answerable['correct']}/{answerable['denominator']} = {answerable['accuracy']:.6f}**.",
        "",
        "## Evidence certificates",
        "",
        "| Task | Candidate | Tier | Public source | Proof | Image SHA-256 |",
        "|---|---:|:---:|---|---|---|",
    ]
    for row in report["evidence_certificates"]:
        source = row["source"]
        locator = str(source["locator"])
        urls = source.get("urls") or []
        if urls:
            locator = f"[{locator}]({urls[0]})"
        lines.append(
            "| "
            + " | ".join(
                [
                    cell(row["task_id"]),
                    cell(row["candidate_answer"]),
                    cell(row["tier"]),
                    cell(locator),
                    cell(row["proof"]),
                    f"`{row['public_image']['sha256']}`",
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Malformed public payload",
            "",
            "| Task | Finding | Image SHA-256 |",
            "|---|---|---|",
        ]
    )
    for row in report["malformed_certificates"]:
        lines.append(
            f"| {cell(row['task_id'])} | {cell(row['reason'])} | `{row['public_image']['sha256']}` |"
        )

    lineage = report["sha_lineage"]
    lines.extend(
        [
            "",
            "## SHA lineage and isolation",
            "",
            f"- Solver: `{lineage['solver']['actual_sha256']}` ({lineage['solver']['rows_verified']} rows).",
            f"- Frozen standard score: `{lineage['standard_score']['actual_sha256']}` ({lineage['standard_score']['outcomes_verified']} outcomes).",
            f"- Certificate registry: `{lineage['certificate_registry_canonical_sha256']}`.",
            f"- Audit utility: `{lineage['audit_utility_sha256']}`.",
            "- No benchmark/reference/judge/gold input was accepted; score provenance paths were not followed.",
            "",
            "## Limitations",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in report["limitations"])
    lines.append("")
    return "\n".join(lines)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver", type=Path, required=True, help="frozen solver JSONL")
    parser.add_argument("--expected-solver-sha256", required=True)
    parser.add_argument("--standard-score", type=Path, required=True, help="frozen standard score JSON")
    parser.add_argument("--expected-standard-score-sha256", required=True)
    parser.add_argument("--public-image-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args(argv)

    # Fail on forbidden names before even resolving the paths.  ``run_audit``
    # repeats the check so direct library callers receive the same guarantee.
    for role, path in (
        ("solver", args.solver),
        ("standard-score", args.standard_score),
        ("public-image-root", args.public_image_root),
    ):
        reject_forbidden_input_path(path, role=role)
    inputs = {args.solver.resolve(), args.standard_score.resolve()}
    outputs = {args.output_json.resolve(), args.output_md.resolve()}
    if len(outputs) != 2 or inputs & outputs:
        raise ValueError("output paths must be distinct and must not overwrite frozen inputs")

    report = run_audit(
        solver_path=args.solver,
        expected_solver_sha256=args.expected_solver_sha256,
        score_path=args.standard_score,
        expected_score_sha256=args.expected_standard_score_sha256,
        public_image_root=args.public_image_root,
    )
    atomic_write_text(
        args.output_json,
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    atomic_write_text(args.output_md, render_markdown(report))
    print(canonical_json({"output_json": str(args.output_json), "output_md": str(args.output_md), "status": "ok"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
