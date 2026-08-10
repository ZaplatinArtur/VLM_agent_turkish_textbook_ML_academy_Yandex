from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from evidence_os.mcq_fullpage_source import (
    EXPECTED_FROZEN_BUNDLE_MANIFEST_PROJECTION_SHA256,
    EXPECTED_FROZEN_BUNDLE_MANIFEST_SHA256,
    EXPECTED_PAGE_PAYLOADS_PROJECTION_SHA256,
    McqSourceError,
    assert_frozen_mcq_bundle,
    assert_frozen_mcq_objects,
    load_mcq_inventory,
    load_mcq_key_index,
    load_mcq_render_manifest,
    write_canonical_json,
)
from evidence_os.mcq_opaque_batch import (
    EXPECTED_V11_RUNTIME_CODE_PATHS,
    McqOpaqueBatchError,
    assert_mcq_v11_code_freeze,
    execute_mcq_opaque_batch,
    load_mcq_opaque_inputs,
    run_mcq_opaque_batch,
)
from evidence_os.official_ogm import canonical_json_sha256, sha256_file


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "reports" / "maxim_mcq_fullpage_source_v1_20260808"


def _prompt(question_number: int = 7) -> str:
    return (
        f"Sayfadaki {question_number}. \u00e7oktan se\u00e7meli soruyu "
        "\u00e7\u00f6z\u00fcn\u00fcz. Yaln\u0131zca A, B, C, D veya E "
        "yaz\u0131n\u0131z."
    )


def _bundle():
    return assert_frozen_mcq_bundle(
        freeze_manifest_path=SOURCE_ROOT / "freeze_manifest.json",
        inventory_path=SOURCE_ROOT / "inventory.json",
        key_index_path=SOURCE_ROOT / "official_key_index.json",
        render_manifest_path=SOURCE_ROOT / "render_manifest.json",
        page_root=SOURCE_ROOT / "renders",
    )


def _raw_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _write_one_input(tmp_path: Path) -> tuple[Path, Path, bytes]:
    asset_root = tmp_path / "assets"
    asset_root.mkdir()
    image_bytes = b"deliberately foreign non-PNG image bytes"
    (asset_root / "page.bin").write_bytes(image_bytes)
    row = {
        "schema_version": "holdout80-opaque-resolver-input-v1",
        "input_id": "row-safe",
        "prompt": _prompt(),
        "language": "tr",
        "expected_response_format": "single_choice_ABCDE",
        "images": [
            {
                "path": "page.bin",
                "sha256": hashlib.sha256(image_bytes).hexdigest(),
            }
        ],
    }
    raw = (
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    input_path = tmp_path / "inputs.jsonl"
    input_path.write_bytes(raw)
    return input_path, asset_root, raw


def _write_current_code_freeze(
    tmp_path: Path,
) -> tuple[Path, str, str]:
    code_files = [
        {
            "path": relative,
            "sha256": sha256_file(REPO_ROOT / relative),
            "size_bytes": (REPO_ROOT / relative).stat().st_size,
        }
        for relative in sorted(EXPECTED_V11_RUNTIME_CODE_PATHS)
    ]
    projection: dict[str, object] = {
        "schema_version": "mcq-fullpage-source-adapter-freeze-v1.1",
        "status": "ready_for_commit_no_opaque_read_or_run",
        "accuracy_claim": None,
        "code": {
            "files": code_files,
            "combined_code_projection_sha256": canonical_json_sha256(code_files),
        },
    }
    freeze_projection = canonical_json_sha256(projection)
    projection["manifest_projection_sha256"] = freeze_projection
    path = tmp_path / "current-code-freeze.json"
    write_canonical_json(path, projection)
    return path, sha256_file(path), freeze_projection


def test_exact_published_bundle_is_the_only_runtime_trust_anchor() -> None:
    bundle = _bundle()
    assert bundle.freeze_manifest_sha256 == EXPECTED_FROZEN_BUNDLE_MANIFEST_SHA256
    assert (
        bundle.freeze_manifest_projection_sha256
        == EXPECTED_FROZEN_BUNDLE_MANIFEST_PROJECTION_SHA256
    )
    assert (
        bundle.page_payloads_projection_sha256
        == EXPECTED_PAGE_PAYLOADS_PROJECTION_SHA256
    )


def test_external_v11_freeze_pin_attests_all_runtime_code(tmp_path: Path) -> None:
    path, file_sha, projection_sha = _write_current_code_freeze(tmp_path)
    attestation = assert_mcq_v11_code_freeze(
        freeze_manifest_path=path,
        expected_freeze_sha256=file_sha,
        expected_freeze_projection_sha256=projection_sha,
    )
    assert attestation.code_file_count == len(EXPECTED_V11_RUNTIME_CODE_PATHS)
    assert attestation.code_projection_sha256

    forged = _raw_json(path)
    code = forged["code"]
    assert isinstance(code, dict)
    files = code["files"]
    assert isinstance(files, list) and isinstance(files[0], dict)
    files[0]["sha256"] = hashlib.sha256(b"forged-runtime-code").hexdigest()
    code["combined_code_projection_sha256"] = canonical_json_sha256(files)
    forged_projection = dict(forged)
    forged_projection.pop("manifest_projection_sha256")
    forged["manifest_projection_sha256"] = canonical_json_sha256(
        forged_projection
    )
    forged_path = tmp_path / "forged-code-freeze.json"
    write_canonical_json(forged_path, forged)
    with pytest.raises(McqOpaqueBatchError, match="code bytes changed"):
        assert_mcq_v11_code_freeze(
            freeze_manifest_path=forged_path,
            expected_freeze_sha256=sha256_file(forged_path),
            expected_freeze_projection_sha256=str(
                forged["manifest_projection_sha256"]
            ),
        )


def test_adapter_bad_v11_pin_exits_two_without_traceback(tmp_path: Path) -> None:
    path, _, projection_sha = _write_current_code_freeze(tmp_path)
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "mcq_fullpage_source_adapter.py"),
            "resolve",
            "--v11-freeze-manifest",
            str(path),
            "--expected-v11-freeze-sha256",
            "0" * 64,
            "--expected-v11-freeze-projection-sha256",
            projection_sha,
            "--freeze-manifest",
            str(SOURCE_ROOT / "freeze_manifest.json"),
            "--inventory",
            str(SOURCE_ROOT / "inventory.json"),
            "--key-index",
            str(SOURCE_ROOT / "official_key_index.json"),
            "--render-manifest",
            str(SOURCE_ROOT / "render_manifest.json"),
            "--page-root",
            str(SOURCE_ROOT / "renders"),
            "--prompt",
            _prompt(),
            "--image",
            str(tmp_path / "must-not-be-read.png"),
            "--output",
            str(tmp_path / "must-not-be-written.json"),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert result.returncode == 2
    assert "MCQ source adapter failed" in result.stderr
    assert "Traceback" not in result.stderr
    assert not (tmp_path / "must-not-be-written.json").exists()


def test_self_consistent_alternate_key_index_is_rejected(tmp_path: Path) -> None:
    inventory = load_mcq_inventory(SOURCE_ROOT / "inventory.json")
    raw = _raw_json(SOURCE_ROOT / "official_key_index.json")
    cells = raw["cells"]
    assert isinstance(cells, list) and isinstance(cells[0], dict)
    cell = cells[0]
    original = str(cell["answer"])
    answer = next(choice for choice in "ABCDE" if choice != original)
    cell["answer"] = answer
    cell["key_text"] = f"{cell['question_number']} {answer}"
    cell["key_text_sha256"] = hashlib.sha256(
        str(cell["key_text"]).encode("utf-8")
    ).hexdigest()
    cell["key_projection_sha256"] = hashlib.sha256(
        b"self-consistent-but-forged-key-cell"
    ).hexdigest()
    projection = dict(raw)
    projection.pop("key_index_projection_sha256")
    raw["key_index_projection_sha256"] = canonical_json_sha256(projection)
    forged_path = tmp_path / "official_key_index.json"
    write_canonical_json(forged_path, raw)

    forged_key = load_mcq_key_index(forged_path, inventory)
    render = load_mcq_render_manifest(
        SOURCE_ROOT / "render_manifest.json",
        inventory,
        page_root=SOURCE_ROOT / "renders",
    )
    with pytest.raises(McqSourceError, match="exact frozen official key"):
        assert_frozen_mcq_objects(inventory, forged_key, render)
    with pytest.raises(McqSourceError, match="artifact bytes changed"):
        assert_frozen_mcq_bundle(
            freeze_manifest_path=SOURCE_ROOT / "freeze_manifest.json",
            inventory_path=SOURCE_ROOT / "inventory.json",
            key_index_path=forged_path,
            render_manifest_path=SOURCE_ROOT / "render_manifest.json",
            page_root=SOURCE_ROOT / "renders",
        )


@pytest.mark.parametrize("artifact", ["inventory", "render"])
def test_self_consistent_altered_artifact_projection_is_rejected(
    tmp_path: Path, artifact: str
) -> None:
    inventory_path = SOURCE_ROOT / "inventory.json"
    render_path = SOURCE_ROOT / "render_manifest.json"
    if artifact == "inventory":
        raw = _raw_json(inventory_path)
        documents = raw["documents"]
        assert isinstance(documents, list) and isinstance(documents[0], dict)
        questions = documents[0]["questions"]
        assert isinstance(questions, list) and isinstance(questions[0], dict)
        questions[0]["content_marker_projection_sha256"] = hashlib.sha256(
            b"alternate-marker-projection"
        ).hexdigest()
        projection = dict(raw)
        projection.pop("inventory_projection_sha256")
        raw["inventory_projection_sha256"] = canonical_json_sha256(projection)
        forged = tmp_path / "inventory.json"
        write_canonical_json(forged, raw)
        alternate_inventory = load_mcq_inventory(forged)
        original_inventory = load_mcq_inventory(inventory_path)
        key = load_mcq_key_index(
            SOURCE_ROOT / "official_key_index.json", original_inventory
        )
        render = load_mcq_render_manifest(
            render_path,
            original_inventory,
            page_root=SOURCE_ROOT / "renders",
        )
        with pytest.raises(McqSourceError, match="exact frozen source census"):
            assert_frozen_mcq_objects(alternate_inventory, key, render)
    else:
        inventory = load_mcq_inventory(inventory_path)
        key = load_mcq_key_index(SOURCE_ROOT / "official_key_index.json", inventory)
        raw = _raw_json(render_path)
        pages = raw["pages"]
        assert isinstance(pages, list) and isinstance(pages[0], dict)
        pages[0]["sha256"] = hashlib.sha256(b"alternate-render").hexdigest()
        projection = dict(raw)
        projection.pop("render_manifest_projection_sha256")
        raw["render_manifest_projection_sha256"] = canonical_json_sha256(projection)
        forged = tmp_path / "render_manifest.json"
        write_canonical_json(forged, raw)
        alternate_render = load_mcq_render_manifest(forged, inventory)
        with pytest.raises(McqSourceError, match="exact frozen page set"):
            assert_frozen_mcq_objects(inventory, key, alternate_render)


def test_runner_attests_bundle_before_trying_to_read_opaque_input(
    tmp_path: Path,
) -> None:
    v11_path, v11_sha, v11_projection = _write_current_code_freeze(tmp_path)
    forged_freeze = tmp_path / "freeze.json"
    forged_freeze.write_text("{}", encoding="utf-8")
    with pytest.raises(McqSourceError, match="trust anchor"):
        run_mcq_opaque_batch(
            input_jsonl=tmp_path / "opaque-input-must-not-be-read.jsonl",
            asset_root=tmp_path / "opaque-assets-must-not-be-read",
            v11_freeze_manifest_path=v11_path,
            expected_v11_freeze_sha256=v11_sha,
            expected_v11_freeze_projection_sha256=v11_projection,
            freeze_manifest_path=forged_freeze,
            inventory_path=SOURCE_ROOT / "inventory.json",
            key_index_path=SOURCE_ROOT / "official_key_index.json",
            render_manifest_path=SOURCE_ROOT / "render_manifest.json",
            page_root=SOURCE_ROOT / "renders",
            output_dir=tmp_path / "output",
        )


def test_windows_reserved_output_stems_are_rejected(tmp_path: Path) -> None:
    input_path, asset_root, raw = _write_one_input(tmp_path)
    original = json.loads(raw.decode("utf-8"))
    for index, reserved in enumerate(("CON", "nul.json", "LPT1")):
        row = {**original, "input_id": reserved}
        candidate = tmp_path / f"reserved-{index}.jsonl"
        candidate.write_bytes(
            (
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            ).encode("utf-8")
        )
        with pytest.raises(McqOpaqueBatchError, match="unsafe or duplicate"):
            load_mcq_opaque_inputs(candidate, asset_root)


@pytest.mark.parametrize("mutation", ["traversal", "duplicate", "relabel", "foreign_image"])
def test_direct_execute_reparses_and_rejects_mutated_objects(
    tmp_path: Path, mutation: str
) -> None:
    bundle = _bundle()
    v11_path, v11_sha, v11_projection = _write_current_code_freeze(tmp_path)
    input_path, asset_root, raw = _write_one_input(tmp_path)
    inputs = load_mcq_opaque_inputs(input_path, asset_root)
    item = inputs[0]
    if mutation == "traversal":
        supplied = (replace(item, input_id="../escape"),)
    elif mutation == "duplicate":
        supplied = (item, item)
    elif mutation == "relabel":
        supplied = (replace(item, prompt=_prompt(8)),)
    else:
        supplied = (
            replace(
                item,
                image=replace(
                    item.image,
                    image_bytes=b"foreign replacement",
                    sha256=hashlib.sha256(b"foreign replacement").hexdigest(),
                ),
            ),
        )
    with pytest.raises(McqOpaqueBatchError, match="differs|count"):
        execute_mcq_opaque_batch(
            supplied,
            bundle.inventory,
            bundle.render_manifest,
            bundle.key_index,
            tmp_path / "output",
            v11_freeze_manifest_path=v11_path,
            expected_v11_freeze_sha256=v11_sha,
            expected_v11_freeze_projection_sha256=v11_projection,
            input_jsonl_bytes=raw,
            asset_root=asset_root,
        )
    assert not (tmp_path / "output").exists()


def test_run_manifest_pins_raw_input_bytes_size_and_ordered_projection(
    tmp_path: Path,
) -> None:
    bundle = _bundle()
    v11_path, v11_sha, v11_projection = _write_current_code_freeze(tmp_path)
    input_path, asset_root, raw = _write_one_input(tmp_path)
    inputs = load_mcq_opaque_inputs(input_path, asset_root)
    output = tmp_path / "output"
    manifest = execute_mcq_opaque_batch(
        inputs,
        bundle.inventory,
        bundle.render_manifest,
        bundle.key_index,
        output,
        v11_freeze_manifest_path=v11_path,
        expected_v11_freeze_sha256=v11_sha,
        expected_v11_freeze_projection_sha256=v11_projection,
        input_jsonl_bytes=raw,
        asset_root=asset_root,
    )
    assert manifest["input_jsonl_sha256"] == hashlib.sha256(raw).hexdigest()
    assert manifest["input_jsonl_size_bytes"] == len(raw)
    assert len(manifest["ordered_input_projection_sha256"]) == 64
    assert (
        manifest["source_bundle"][
            "exact_source_objects_attested_before_input_parse"
        ]
        is True
    )
    assert manifest["source_bundle"]["freeze_manifest_sha256"] == (
        EXPECTED_FROZEN_BUNDLE_MANIFEST_SHA256
    )
    assert manifest["v11_code_freeze"]["freeze_file_sha256"] == v11_sha
    assert manifest["v11_code_freeze"]["freeze_projection_sha256"] == (
        v11_projection
    )
