from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from evidence_os.math12_opaque_batch import (
    Math12OpaqueBatchError,
    execute_opaque_batch,
    load_opaque_inputs,
)
from evidence_os.official_ogm import canonical_json_bytes, canonical_json_sha256


@dataclass(frozen=True)
class _FakeCertificate:
    task_image_sha256: str
    decision: Any
    certificate_projection_sha256: str

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": "fake-source-certificate-v1",
            "task_image_sha256": self.task_image_sha256,
            "decision": {
                "accepted": self.decision.accepted,
                "reason": self.decision.reason,
                "checks": list(self.decision.checks),
                "selected_content_page": self.decision.selected_content_page,
                "selected_activity_number": self.decision.selected_activity_number,
            },
            "certificate_projection_sha256": self.certificate_projection_sha256,
        }


@dataclass(frozen=True)
class _FakeSolution:
    task_image_sha256: str
    activity_number: int
    answer_bound_certificate_projection_sha256: str

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": "fake-official-source-record-v1",
            "task_image_sha256": self.task_image_sha256,
            "activity_number": self.activity_number,
            "official_solution_text": f"official source for {self.activity_number}",
            "answer_bound_certificate_projection_sha256": (
                self.answer_bound_certificate_projection_sha256
            ),
        }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_bytes(b"".join(canonical_json_bytes(row) + b"\n" for row in rows))


def _row(input_id: str, assets: list[tuple[str, bytes]]) -> dict[str, Any]:
    return {
        "schema_version": "holdout80-opaque-resolver-input-v1",
        "input_id": input_id,
        "language": "tr",
        "prompt": "opaque",
        "expected_response_format": "numbered_multi_part_solution",
        "images": [
            {
                "path": path,
                "sha256": hashlib.sha256(content).hexdigest(),
            }
            for path, content in assets
        ],
    }


def _materialize_inputs(
    tmp_path: Path,
    rows_and_bytes: list[tuple[dict[str, Any], list[tuple[str, bytes]]]],
) -> tuple[Path, Path]:
    root = tmp_path / "asset-root"
    root.mkdir()
    rows: list[dict[str, Any]] = []
    for row, assets in rows_and_bytes:
        rows.append(row)
        for relative_path, content in assets:
            path = root / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
    jsonl = tmp_path / "opaque.jsonl"
    _write_jsonl(jsonl, rows)
    return jsonl, root


def _runtime(activity_by_bytes: dict[bytes, int | None]):
    def resolve(image_bytes: bytes, _inventory: Any, _render_manifest: Any):
        activity = activity_by_bytes[image_bytes]
        image_sha = hashlib.sha256(image_bytes).hexdigest()
        accepted = activity is not None
        projection = {
            "image_sha256": image_sha,
            "activity": activity,
            "accepted": accepted,
        }
        decision = SimpleNamespace(
            accepted=accepted,
            reason="accepted" if accepted else "insufficient_geometry",
            checks=(("strong_geometry", accepted),),
            selected_content_page=activity + 3 if activity is not None else None,
            selected_activity_number=activity,
        )
        return _FakeCertificate(
            task_image_sha256=image_sha,
            decision=decision,
            certificate_projection_sha256=canonical_json_sha256(projection),
        )

    def verify(_inventory: Any, _render_manifest: Any, certificate: _FakeCertificate):
        return certificate.decision

    def extract(
        _pdf_path: Path,
        _inventory: Any,
        _render_manifest: Any,
        certificate: _FakeCertificate,
    ):
        activity = certificate.decision.selected_activity_number
        assert isinstance(activity, int)
        projection = {
            "certificate": certificate.certificate_projection_sha256,
            "activity": activity,
        }
        return _FakeSolution(
            task_image_sha256=certificate.task_image_sha256,
            activity_number=activity,
            answer_bound_certificate_projection_sha256=canonical_json_sha256(
                projection
            ),
        )

    return resolve, verify, extract


def _execute(
    tmp_path: Path,
    *,
    contents: tuple[bytes, ...],
    activities: tuple[int | None, ...],
    output_name: str = "output",
) -> tuple[dict[str, Any], Path]:
    assets = [(f"opaque/assets/page-{index}.jpg", content) for index, content in enumerate(contents, 1)]
    row = _row("input-safe", assets)
    jsonl, asset_root = _materialize_inputs(tmp_path, [(row, assets)])
    opaque_inputs = load_opaque_inputs(jsonl, asset_root)
    resolve, verify, extract = _runtime(dict(zip(contents, activities, strict=True)))
    inventory = SimpleNamespace(
        inventory_projection_sha256="a" * 64,
        pdf_sha256="b" * 64,
    )
    render_manifest = SimpleNamespace(render_manifest_projection_sha256="c" * 64)
    output = tmp_path / output_name
    manifest = execute_opaque_batch(
        opaque_inputs=opaque_inputs,
        input_jsonl_sha256=hashlib.sha256(jsonl.read_bytes()).hexdigest(),
        output_dir=output,
        inventory=inventory,
        render_manifest=render_manifest,
        pdf_path=tmp_path / "unused-pinned.pdf",
        resolve_image=resolve,
        verify_certificate=verify,
        extract_solution=extract,
    )
    return manifest, output


def _read_only_result(output: Path) -> dict[str, Any]:
    lines = (output / "results.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    return json.loads(lines[0])


def test_multi_page_accepted_certificates_must_agree(tmp_path: Path) -> None:
    manifest, output = _execute(
        tmp_path,
        contents=(b"page-a", b"page-b"),
        activities=(17, 17),
    )
    result = _read_only_result(output)
    assert result["aggregate"] == {
        "accepted": True,
        "reason": "accepted_activity_agreement",
        "selected_activity_number": 17,
    }
    assert result["accepted_certificate_count"] == 2
    assert manifest["accepted_input_count"] == 1
    assert len(list((output / "certificates").rglob("*.json"))) == 2
    assert len(list((output / "solution_records").rglob("*.json"))) == 2


def test_multi_page_conflicting_accepted_certificates_abstain(tmp_path: Path) -> None:
    manifest, output = _execute(
        tmp_path,
        contents=(b"page-a", b"page-b"),
        activities=(17, 18),
    )
    result = _read_only_result(output)
    assert result["aggregate"] == {
        "accepted": False,
        "reason": "abstain_conflicting_accepted_activities",
        "selected_activity_number": None,
    }
    assert manifest["abstained_input_count"] == 1
    # Per-image source records remain auditable, but no input-level choice is made.
    assert len(list((output / "solution_records").rglob("*.json"))) == 2


def test_zero_accepted_certificates_abstains(tmp_path: Path) -> None:
    _, output = _execute(
        tmp_path,
        contents=(b"page-a", b"page-b"),
        activities=(None, None),
    )
    result = _read_only_result(output)
    assert result["aggregate"] == {
        "accepted": False,
        "reason": "abstain_no_accepted_certificate",
        "selected_activity_number": None,
    }
    assert result["accepted_certificate_count"] == 0
    assert not (output / "solution_records").exists()


def test_one_resolver_exception_makes_the_multi_page_input_incomplete(
    tmp_path: Path,
) -> None:
    contents = (b"page-a", b"page-b")
    assets = [(f"assets/{index}.jpg", content) for index, content in enumerate(contents, 1)]
    row = _row("input-safe", assets)
    jsonl, asset_root = _materialize_inputs(tmp_path, [(row, assets)])
    opaque_inputs = load_opaque_inputs(jsonl, asset_root)
    normal_resolve, verify, extract = _runtime({b"page-a": 17})
    calls: list[bytes] = []

    def resolve(image_bytes: bytes, inventory: Any, render_manifest: Any):
        calls.append(image_bytes)
        if image_bytes == b"page-b":
            raise RuntimeError("local implementation detail must not leak")
        return normal_resolve(image_bytes, inventory, render_manifest)

    output = tmp_path / "output"
    execute_opaque_batch(
        opaque_inputs=opaque_inputs,
        input_jsonl_sha256=hashlib.sha256(jsonl.read_bytes()).hexdigest(),
        output_dir=output,
        inventory=SimpleNamespace(
            inventory_projection_sha256="a" * 64,
            pdf_sha256="b" * 64,
        ),
        render_manifest=SimpleNamespace(render_manifest_projection_sha256="c" * 64),
        pdf_path=tmp_path / "unused-pinned.pdf",
        resolve_image=resolve,
        verify_certificate=verify,
        extract_solution=extract,
    )
    result = _read_only_result(output)
    assert calls == [b"page-a", b"page-b"]
    assert result["aggregate"] == {
        "accepted": False,
        "reason": "abstain_incomplete_image_processing",
        "selected_activity_number": None,
    }
    assert result["images"][1]["certificate_reason"] == "resolver_error:RuntimeError"
    assert "implementation detail" not in (output / "results.jsonl").read_text("utf-8")


def test_strict_verifier_failure_preserves_certificate_but_forces_abstain(
    tmp_path: Path,
) -> None:
    assets = [("assets/a.jpg", b"page-a")]
    row = _row("input-safe", assets)
    jsonl, asset_root = _materialize_inputs(tmp_path, [(row, assets)])
    opaque_inputs = load_opaque_inputs(jsonl, asset_root)
    resolve, _verify, extract = _runtime({b"page-a": 17})

    def reject_verification(_inventory: Any, _render: Any, _certificate: Any):
        raise ValueError("private verifier detail must not leak")

    output = tmp_path / "output"
    execute_opaque_batch(
        opaque_inputs=opaque_inputs,
        input_jsonl_sha256=hashlib.sha256(jsonl.read_bytes()).hexdigest(),
        output_dir=output,
        inventory=SimpleNamespace(
            inventory_projection_sha256="a" * 64,
            pdf_sha256="b" * 64,
        ),
        render_manifest=SimpleNamespace(render_manifest_projection_sha256="c" * 64),
        pdf_path=tmp_path / "unused-pinned.pdf",
        resolve_image=resolve,
        verify_certificate=reject_verification,
        extract_solution=extract,
    )
    result = _read_only_result(output)
    image = result["images"][0]
    assert result["aggregate"]["accepted"] is False
    assert result["aggregate"]["reason"] == "abstain_incomplete_image_processing"
    assert image["processing_status"] == "certificate_verification_error"
    assert image["certificate_reason"] == "verification_error:ValueError"
    assert image["certificate_path"] is not None
    assert (output / image["certificate_path"]).is_file()
    assert "private verifier detail" not in (output / "results.jsonl").read_text("utf-8")


def test_path_traversal_is_rejected_before_asset_read(tmp_path: Path) -> None:
    asset_root = tmp_path / "asset-root"
    asset_root.mkdir()
    outside = tmp_path / "outside.jpg"
    outside.write_bytes(b"must-not-be-opened")
    jsonl = tmp_path / "opaque.jsonl"
    _write_jsonl(jsonl, [_row("input-safe", [("../outside.jpg", outside.read_bytes())])])
    with pytest.raises(Math12OpaqueBatchError, match="escapes"):
        load_opaque_inputs(jsonl, asset_root)


def test_duplicate_input_ids_are_rejected(tmp_path: Path) -> None:
    assets_a = [("assets/a.jpg", b"a")]
    assets_b = [("assets/b.jpg", b"b")]
    rows = [
        (_row("same-id", assets_a), assets_a),
        (_row("same-id", assets_b), assets_b),
    ]
    jsonl, asset_root = _materialize_inputs(tmp_path, rows)
    with pytest.raises(Math12OpaqueBatchError, match="duplicate input_id"):
        load_opaque_inputs(jsonl, asset_root)


@pytest.mark.parametrize("same_path", [True, False])
def test_duplicate_asset_path_or_bytes_are_rejected(
    tmp_path: Path, same_path: bool
) -> None:
    first = ("assets/a.jpg", b"same")
    second = ("assets/a.jpg" if same_path else "assets/b.jpg", b"same")
    row = _row("input-safe", [first, second])
    # Do not materialize the same path twice through the generic helper.
    jsonl, asset_root = _materialize_inputs(tmp_path, [(row, list({first, second}))])
    expected = "duplicate opaque asset path" if same_path else "duplicate opaque asset bytes"
    with pytest.raises(Math12OpaqueBatchError, match=expected):
        load_opaque_inputs(jsonl, asset_root)


def test_asset_hash_mismatch_is_rejected(tmp_path: Path) -> None:
    assets = [("assets/a.jpg", b"actual")]
    row = _row("input-safe", assets)
    row["images"][0]["sha256"] = "0" * 64
    jsonl, asset_root = _materialize_inputs(tmp_path, [(row, assets)])
    with pytest.raises(Math12OpaqueBatchError, match="SHA-256 mismatch"):
        load_opaque_inputs(jsonl, asset_root)


def test_output_is_byte_deterministic(tmp_path: Path) -> None:
    contents = (b"page-a", b"page-b")
    assets = [(f"assets/{index}.jpg", content) for index, content in enumerate(contents, 1)]
    row = _row("input-safe", assets)
    jsonl, asset_root = _materialize_inputs(tmp_path, [(row, assets)])
    opaque_inputs = load_opaque_inputs(jsonl, asset_root)
    resolve, verify, extract = _runtime({b"page-a": 17, b"page-b": None})
    inventory = SimpleNamespace(
        inventory_projection_sha256="a" * 64,
        pdf_sha256="b" * 64,
    )
    render_manifest = SimpleNamespace(render_manifest_projection_sha256="c" * 64)
    kwargs = {
        "opaque_inputs": opaque_inputs,
        "input_jsonl_sha256": hashlib.sha256(jsonl.read_bytes()).hexdigest(),
        "inventory": inventory,
        "render_manifest": render_manifest,
        "pdf_path": tmp_path / "unused-pinned.pdf",
        "resolve_image": resolve,
        "verify_certificate": verify,
        "extract_solution": extract,
    }
    first_manifest = execute_opaque_batch(output_dir=tmp_path / "out-1", **kwargs)
    second_manifest = execute_opaque_batch(output_dir=tmp_path / "out-2", **kwargs)
    assert first_manifest == second_manifest
    assert _read_only_result(tmp_path / "out-1")["aggregate"] == {
        "accepted": True,
        "reason": "accepted_activity_agreement",
        "selected_activity_number": 17,
    }

    def snapshot(root: Path) -> dict[str, bytes]:
        return {
            path.relative_to(root).as_posix(): path.read_bytes()
            for path in sorted(root.rglob("*"))
            if path.is_file()
        }

    assert snapshot(tmp_path / "out-1") == snapshot(tmp_path / "out-2")
