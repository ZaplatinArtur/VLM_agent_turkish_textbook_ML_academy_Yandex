from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
from typing import Any

import pytest

from evidence_os.fill_blank_page_activity import VERIFIER
from evidence_os.official_ogm import canonical_json_bytes, sha256_file
from scripts.build_maxim_fill_blank_page_activity_image_judge_v1 import (
    BASE_COMPOSITION_SCHEMA,
    BASE_IMAGE_JUDGE_SCHEMA,
    EXPECTED_IMAGE_ROWS,
    EXPECTED_ROWS,
    HistoryImageJudgeError,
    build,
)
from scripts.compose_maxim_fill_blank_page_activity_v1 import (
    COMPOSITION_SCHEMA,
    PROFILE_SCHEMA,
    RUN_SCHEMA,
)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(canonical_json_bytes(row) + b"\n" for row in rows))


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


@dataclass
class SyntheticChain:
    root: Path
    profile: Path
    resolver: Path
    composition: Path
    base_solver: Path
    base_judge: Path
    base_judge_manifest: Path
    base_composition: Path
    frozen_solver: Path
    frozen_decisions: Path
    trusted_solver: Path
    trusted_decisions: Path
    certificates: Path
    source_ids: tuple[str, ...]

    def replay(self, _profile: Path, _resolver: Path, output_dir: Path) -> dict[str, Any]:
        output_dir.mkdir(parents=True, exist_ok=True)
        solver = output_dir / "solver.jsonl"
        decisions = output_dir / "decisions.jsonl"
        shutil.copyfile(self.trusted_solver, solver)
        shutil.copyfile(self.trusted_decisions, decisions)
        return {
            "artifacts": {
                "solver": {"path": str(solver), "sha256": sha256_file(solver)},
                "decisions": {
                    "path": str(decisions),
                    "sha256": sha256_file(decisions),
                },
            }
        }

    def refresh_composition(self) -> None:
        value = _load_json(self.composition)
        value["resolver_manifest"]["sha256"] = sha256_file(self.resolver)
        value["source_overrides"] = len(self.source_ids)
        value["opaque_anchor_rows_copied"] = EXPECTED_ROWS - len(self.source_ids)
        value["artifacts"]["solver"]["sha256"] = sha256_file(self.frozen_solver)
        value["artifacts"]["decisions"]["sha256"] = sha256_file(
            self.frozen_decisions
        )
        _write_json(self.composition, value)

    def call(self, *, output_name: str = "judge.jsonl", **overrides: Any):
        values: dict[str, Any] = {
            "profile_path": self.profile,
            "expected_profile_sha256": sha256_file(self.profile),
            "resolver_manifest_path": self.resolver,
            "expected_resolver_manifest_sha256": sha256_file(self.resolver),
            "composition_manifest_path": self.composition,
            "expected_composition_manifest_sha256": sha256_file(self.composition),
            "base_main_solver_path": self.base_solver,
            "expected_base_main_solver_sha256": sha256_file(self.base_solver),
            "base_main_image_judge_path": self.base_judge,
            "expected_base_main_image_judge_sha256": sha256_file(self.base_judge),
            "base_main_image_judge_manifest_path": self.base_judge_manifest,
            "expected_base_main_image_judge_manifest_sha256": sha256_file(
                self.base_judge_manifest
            ),
            "output_path": self.root / output_name,
            "manifest_path": self.root / f"{output_name}.manifest.json",
            "compose_fn": self.replay,
        }
        values.update(overrides)
        return build(**values)


def _chain(
    tmp_path: Path,
    *,
    source_indexes: tuple[int, ...] = (96,),
    source_answer_equals_base: bool = True,
) -> SyntheticChain:
    source_ids = tuple(f"task-{index:03d}" for index in source_indexes)
    base_solver = tmp_path / "base_solver.jsonl"
    frozen_solver = tmp_path / "frozen" / "solver.jsonl"
    frozen_decisions = tmp_path / "frozen" / "decisions.jsonl"
    trusted_solver = tmp_path / "trusted" / "solver.jsonl"
    trusted_decisions = tmp_path / "trusted" / "decisions.jsonl"
    base_rows: list[dict[str, Any]] = []
    final_rows: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    fingerprints = {
        task_id: f"{index + 1:064x}" for index, task_id in enumerate(source_ids)
    }
    for index in range(EXPECTED_ROWS):
        task_id = f"task-{index:03d}"
        base_answer = "source-answer" if task_id in source_ids else f"answer-{index}"
        base = {
            "task_id": task_id,
            "final_answer": base_answer,
            "generation": {"gold_access": False, "opaque": index},
        }
        final = dict(base)
        if task_id in source_ids:
            final["final_answer"] = (
                base_answer if source_answer_equals_base else f"certified-{index}"
            )
            generation = dict(base["generation"])
            generation["fill_blank_page_activity_override"] = {
                "verifier": VERIFIER,
                "trace_fingerprint": fingerprints[task_id],
                "anchor_answer_compared": False,
            }
            final["generation"] = generation
            decisions.append(
                {
                    "task_id": task_id,
                    "source_override": True,
                    "anchor_bytes_copied": False,
                    "certificate_trace_fingerprint": fingerprints[task_id],
                }
            )
        else:
            decisions.append(
                {
                    "task_id": task_id,
                    "source_override": False,
                    "anchor_bytes_copied": True,
                }
            )
        base_rows.append(base)
        final_rows.append(final)
    _write_jsonl(base_solver, base_rows)
    _write_jsonl(frozen_solver, final_rows)
    _write_jsonl(frozen_decisions, decisions)
    _write_jsonl(trusted_solver, final_rows)
    _write_jsonl(trusted_decisions, decisions)

    base_judge = tmp_path / "base_judge.jsonl"
    _write_jsonl(
        base_judge,
        [
            {
                "task_id": f"task-{index:03d}",
                "opaque_metadata": index,
                "verdict": {
                    "strict_correct": False if index == source_indexes[0] else index % 2 == 0
                },
            }
            for index in range(EXPECTED_IMAGE_ROWS)
        ],
    )
    base_composition = tmp_path / "base_composition" / "manifest.json"
    _write_json(
        base_composition,
        {
            "schema_version": BASE_COMPOSITION_SCHEMA,
            "rows": EXPECTED_ROWS,
            "gold_access": False,
            "score_or_outcome_access": False,
            "output": {
                "solver": {
                    "path": str(base_solver),
                    "sha256": sha256_file(base_solver),
                },
                "decisions": {
                    "path": str(tmp_path / "base_composition" / "decisions.jsonl"),
                    "sha256": "d" * 64,
                },
            },
        },
    )
    base_judge_manifest = tmp_path / "base_judge.manifest.json"
    _write_json(
        base_judge_manifest,
        {
            "schema_version": BASE_IMAGE_JUDGE_SCHEMA,
            "solver_and_source_certificates_hashed_before_adjudication": True,
            "benchmark_reference_answers_opened": False,
            "base_image_judge_outcomes_read_and_copied_for_unchanged_rows": True,
            "base_image_judge_outcomes_used_for_changed_rows": False,
            "composition_manifest": {
                "path": str(base_composition),
                "sha256": sha256_file(base_composition),
            },
            "output": {
                "path": str(base_judge),
                "sha256": sha256_file(base_judge),
                "rows": EXPECTED_IMAGE_ROWS,
            },
        },
    )

    profile = tmp_path / "profile.json"
    _write_json(
        profile,
        {
            "schema_version": PROFILE_SCHEMA,
            "profile_name": "synthetic-history-source-only",
            "expected_rows": EXPECTED_ROWS,
            "anchor": {
                "path": str(base_solver),
                "sha256": sha256_file(base_solver),
                "role": "opaque_fail_closed_anchor_only",
            },
            "inputs": {},
            "runtime": {},
            "policy": {
                "task_id_is_policy_feature": False,
                "benchmark_candidate_or_outcome_access": False,
            },
        },
    )
    candidate = tmp_path / "resolver" / "candidate.jsonl"
    certificates = tmp_path / "resolver" / "certificates.jsonl"
    audit = tmp_path / "resolver" / "audit.jsonl"
    _write_jsonl(
        candidate,
        [{"task_id": task_id, "final_answer": "source-answer"} for task_id in source_ids],
    )
    _write_jsonl(
        certificates,
        [
            {
                "task_id": task_id,
                "schema_version": "maxim-fill-blank-page-activity-certificate-v1",
                "verifier": VERIFIER,
                "kind": "source_entailment",
                "strength": "strong",
                "status": "pass",
                "trace_fingerprint": fingerprints[task_id],
                "trace": {
                    "source": {
                        "document_id": "synthetic-history-document",
                        "record_id": "synthetic-history:p9:fill_blank",
                        "pdf_sha256": "a" * 64,
                        "answer_format": "short_text",
                    },
                    "provenance": {"profile_sha256": sha256_file(profile)},
                },
            }
            for task_id in source_ids
        ],
    )
    _write_jsonl(audit, [{"task_id": f"task-{index:03d}"} for index in range(EXPECTED_ROWS)])
    resolver = tmp_path / "resolver" / "manifest.json"
    _write_json(
        resolver,
        {
            "schema_version": RUN_SCHEMA,
            "gold_access": False,
            "benchmark_candidate_or_outcome_access": False,
            "task_id_used_for_alignment_only": True,
            "rows": EXPECTED_ROWS,
            "accepted_certificates": len(source_ids),
            "abstentions": EXPECTED_ROWS - len(source_ids),
            "profile": {"path": str(profile), "sha256": sha256_file(profile)},
            "artifacts": {
                "candidate": {"path": str(candidate), "sha256": sha256_file(candidate)},
                "certificates": {
                    "path": str(certificates),
                    "sha256": sha256_file(certificates),
                },
                "audit": {"path": str(audit), "sha256": sha256_file(audit)},
            },
        },
    )
    composition = tmp_path / "frozen" / "manifest.json"
    _write_json(
        composition,
        {
            "schema_version": COMPOSITION_SCHEMA,
            "gold_access": False,
            "benchmark_candidate_or_outcome_access": False,
            "task_id_used_for_alignment_only": True,
            "anchor_answer_used_as_policy_feature": False,
            "anchor_answer_compared": False,
            "profile": {"path": str(profile), "sha256": sha256_file(profile)},
            "resolver_manifest": {
                "path": str(resolver),
                "sha256": sha256_file(resolver),
            },
            "anchor": {"path": str(base_solver), "sha256": sha256_file(base_solver)},
            "rows": EXPECTED_ROWS,
            "source_overrides": len(source_ids),
            "opaque_anchor_rows_copied": EXPECTED_ROWS - len(source_ids),
            "artifacts": {
                "solver": {
                    "path": str(frozen_solver),
                    "sha256": sha256_file(frozen_solver),
                },
                "decisions": {
                    "path": str(frozen_decisions),
                    "sha256": sha256_file(frozen_decisions),
                },
            },
        },
    )
    return SyntheticChain(
        root=tmp_path,
        profile=profile,
        resolver=resolver,
        composition=composition,
        base_solver=base_solver,
        base_judge=base_judge,
        base_judge_manifest=base_judge_manifest,
        base_composition=base_composition,
        frozen_solver=frozen_solver,
        frozen_decisions=frozen_decisions,
        trusted_solver=trusted_solver,
        trusted_decisions=trusted_decisions,
        certificates=certificates,
        source_ids=source_ids,
    )


def test_success_is_deterministic_when_source_answer_equals_base_and_copies_96_bytes(
    tmp_path: Path,
) -> None:
    chain = _chain(tmp_path, source_answer_equals_base=True)
    result = chain.call()
    source_id = chain.source_ids[0]
    base_by_id = {
        json.loads(line)["task_id"]: line
        for line in chain.base_judge.read_bytes().splitlines()
    }
    output_by_id = {
        json.loads(line)["task_id"]: line
        for line in (tmp_path / "judge.jsonl").read_bytes().splitlines()
    }
    assert len(output_by_id) == EXPECTED_IMAGE_ROWS
    assert result["copied_unchanged_rows"] == EXPECTED_IMAGE_ROWS - 1
    assert all(
        output_by_id[task_id] == raw
        for task_id, raw in base_by_id.items()
        if task_id != source_id
    )
    source = json.loads(output_by_id[source_id])
    assert source["verdict"]["strict_correct"] is True
    assert source["verdict"]["label"] == "fully_correct"
    assert source["judge"]["backend"] == (
        "deterministic-pinned-pdf-fill-blank-certificate"
    )
    assert result["base_answer_compared_for_adjudication"] is False
    assert result["base_image_judge_outcome_fields_parsed"] is False


def test_source_verdict_does_not_depend_on_base_judge_source_outcome(
    tmp_path: Path,
) -> None:
    chain = _chain(tmp_path)
    chain.call(output_name="first.jsonl")
    first = (tmp_path / "first.jsonl").read_bytes()
    rows = _load_jsonl(chain.base_judge)
    source = next(row for row in rows if row["task_id"] == chain.source_ids[0])
    source["verdict"] = {"strict_correct": True, "forged_outcome_detail": "ignored"}
    _write_jsonl(chain.base_judge, rows)
    base_manifest = _load_json(chain.base_judge_manifest)
    base_manifest["output"]["sha256"] = sha256_file(chain.base_judge)
    _write_json(chain.base_judge_manifest, base_manifest)
    chain.call(output_name="second.jsonl")
    assert (tmp_path / "second.jsonl").read_bytes() == first


@pytest.mark.parametrize(
    "argument",
    [
        "expected_profile_sha256",
        "expected_resolver_manifest_sha256",
        "expected_composition_manifest_sha256",
        "expected_base_main_solver_sha256",
        "expected_base_main_image_judge_sha256",
        "expected_base_main_image_judge_manifest_sha256",
    ],
)
def test_wrong_top_level_hashes_fail_closed(tmp_path: Path, argument: str) -> None:
    chain = _chain(tmp_path)
    with pytest.raises(HistoryImageJudgeError, match="SHA-256 mismatch"):
        chain.call(**{argument: "0" * 64})


def test_non_image_source_override_fails_closed(tmp_path: Path) -> None:
    chain = _chain(tmp_path, source_indexes=(150,))
    with pytest.raises(HistoryImageJudgeError, match="97-row image partition"):
        chain.call()


def test_base_judge_manifest_output_binding_fails_closed(tmp_path: Path) -> None:
    chain = _chain(tmp_path)
    manifest = _load_json(chain.base_judge_manifest)
    manifest["output"]["sha256"] = "e" * 64
    _write_json(chain.base_judge_manifest, manifest)
    with pytest.raises(HistoryImageJudgeError, match="output does not bind"):
        chain.call()


def test_base_judge_composition_solver_binding_fails_closed(tmp_path: Path) -> None:
    chain = _chain(tmp_path)
    composition = _load_json(chain.base_composition)
    composition["output"]["solver"]["sha256"] = "e" * 64
    _write_json(chain.base_composition, composition)
    manifest = _load_json(chain.base_judge_manifest)
    manifest["composition_manifest"]["sha256"] = sha256_file(
        chain.base_composition
    )
    _write_json(chain.base_judge_manifest, manifest)
    with pytest.raises(HistoryImageJudgeError, match="does not bind the supplied base solver"):
        chain.call()


def test_base_judge_pinned_composition_hash_fails_closed(tmp_path: Path) -> None:
    chain = _chain(tmp_path)
    manifest = _load_json(chain.base_judge_manifest)
    manifest["composition_manifest"]["sha256"] = "e" * 64
    _write_json(chain.base_judge_manifest, manifest)
    with pytest.raises(HistoryImageJudgeError, match="composition manifest SHA-256"):
        chain.call()


@pytest.mark.parametrize(
    ("sink_argument", "input_attribute"),
    [
        ("output_path", "frozen_solver"),
        ("manifest_path", "certificates"),
        ("output_path", "base_composition"),
        ("manifest_path", "base_judge_manifest"),
    ],
)
def test_output_paths_cannot_collide_with_discovered_inputs(
    tmp_path: Path,
    sink_argument: str,
    input_attribute: str,
) -> None:
    chain = _chain(tmp_path)
    collision = getattr(chain, input_attribute)
    with pytest.raises(HistoryImageJudgeError, match="collide with a pinned"):
        chain.call(**{sink_argument: collision})


def test_output_and_manifest_paths_must_differ(tmp_path: Path) -> None:
    chain = _chain(tmp_path)
    collision = tmp_path / "same-output-path.jsonl"
    with pytest.raises(HistoryImageJudgeError, match="must differ"):
        chain.call(output_path=collision, manifest_path=collision)


def test_extra_source_override_fails_closed(tmp_path: Path) -> None:
    chain = _chain(tmp_path, source_indexes=(95, 96))
    with pytest.raises(HistoryImageJudgeError, match="resolver is not bound"):
        chain.call()


def test_forged_frozen_solver_is_rejected_by_byte_exact_fresh_replay(
    tmp_path: Path,
) -> None:
    chain = _chain(tmp_path)
    rows = _load_jsonl(chain.frozen_solver)
    rows[0]["forged"] = True
    _write_jsonl(chain.frozen_solver, rows)
    chain.refresh_composition()
    with pytest.raises(HistoryImageJudgeError, match="differs byte-for-byte"):
        chain.call()


def test_forged_generation_is_rejected_even_if_fresh_artifacts_match(
    tmp_path: Path,
) -> None:
    chain = _chain(tmp_path)
    for path in (chain.frozen_solver, chain.trusted_solver):
        rows = _load_jsonl(path)
        source = next(row for row in rows if row["task_id"] == chain.source_ids[0])
        source["generation"]["fill_blank_page_activity_override"][
            "anchor_answer_compared"
        ] = True
        _write_jsonl(path, rows)
    chain.refresh_composition()
    with pytest.raises(HistoryImageJudgeError, match="certificate override"):
        chain.call()


def test_forged_decision_is_rejected_even_if_fresh_artifacts_match(
    tmp_path: Path,
) -> None:
    chain = _chain(tmp_path)
    for path in (chain.frozen_decisions, chain.trusted_decisions):
        rows = _load_jsonl(path)
        source = next(row for row in rows if row["task_id"] == chain.source_ids[0])
        source["unexpected_policy_field"] = True
        _write_jsonl(path, rows)
    chain.refresh_composition()
    with pytest.raises(HistoryImageJudgeError, match="decision .* malformed"):
        chain.call()


def test_forged_certificate_fingerprint_is_rejected_after_rehashing_chain(
    tmp_path: Path,
) -> None:
    chain = _chain(tmp_path)
    rows = _load_jsonl(chain.certificates)
    rows[0]["trace_fingerprint"] = "f" * 64
    _write_jsonl(chain.certificates, rows)
    resolver = _load_json(chain.resolver)
    resolver["artifacts"]["certificates"]["sha256"] = sha256_file(chain.certificates)
    _write_json(chain.resolver, resolver)
    chain.refresh_composition()
    with pytest.raises(HistoryImageJudgeError, match="fingerprint or contract"):
        chain.call()
