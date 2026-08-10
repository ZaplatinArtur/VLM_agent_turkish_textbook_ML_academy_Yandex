from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import pytest

from evidence_os.mcq_fullpage_source import (
    load_mcq_inventory,
    load_mcq_key_index,
    load_mcq_render_manifest,
)
from evidence_os.mcq_opaque_batch import (
    McqOpaqueBatchError,
    load_mcq_opaque_inputs,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
PUBLIC_SOURCE_ROOT = (
    REPO_ROOT / "reports" / "maxim_mcq_fullpage_source_v1_20260808"
)


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _prompt(question_number: int) -> str:
    return (
        f"Sayfadaki {question_number}. çoktan seçmeli soruyu çözünüz. "
        "Yalnızca A, B, C, D veya E yazınız."
    )


@pytest.fixture(scope="module")
def synthetic_source():
    inventory_path = PUBLIC_SOURCE_ROOT / "inventory.json"
    key_index_path = PUBLIC_SOURCE_ROOT / "official_key_index.json"
    assert inventory_path.is_file(), "the public source census must be generated first"
    assert key_index_path.is_file(), "the public official-key index must be generated first"
    inventory = load_mcq_inventory(inventory_path)
    key_index = load_mcq_key_index(key_index_path, inventory)
    manifest = load_mcq_render_manifest(
        PUBLIC_SOURCE_ROOT / "render_manifest.json",
        inventory,
        page_root=PUBLIC_SOURCE_ROOT / "renders",
    )
    return inventory, key_index, manifest


def _source_page_with_two_choices(inventory):
    grouped: dict[tuple[str, int], list[object]] = {}
    for record in inventory.questions:
        if record.source_response_kind == "choice_A-E":
            grouped.setdefault(
                (record.document_id, record.content_page_number), []
            ).append(record)
    return next(records[:2] for records in grouped.values() if len(records) >= 2)


def _opaque_row(
    *,
    input_id: str,
    prompt: str,
    image_path: str,
    image_sha256: str,
) -> dict[str, object]:
    return {
        "schema_version": "holdout80-opaque-resolver-input-v1",
        "input_id": input_id,
        "prompt": prompt,
        "language": "tr",
        "expected_response_format": "single_choice_ABCDE",
        "images": [{"path": image_path, "sha256": image_sha256}],
    }


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def test_repeated_image_sha_is_allowed_for_distinct_observable_prompts(
    tmp_path: Path,
    synthetic_source,
) -> None:
    inventory, _, _ = synthetic_source
    first, second = _source_page_with_two_choices(inventory)
    asset_root = tmp_path / "assets"
    asset_root.mkdir()
    image_bytes = b"same full textbook page for two printed questions"
    (asset_root / "page.bin").write_bytes(image_bytes)
    image_sha = _sha_bytes(image_bytes)
    input_path = tmp_path / "inputs.jsonl"
    _write_jsonl(
        input_path,
        [
            _opaque_row(
                input_id="row-a",
                prompt=_prompt(first.question_number),
                image_path="page.bin",
                image_sha256=image_sha,
            ),
            _opaque_row(
                input_id="row-b",
                prompt=_prompt(second.question_number),
                image_path="page.bin",
                image_sha256=image_sha,
            ),
        ],
    )

    loaded = load_mcq_opaque_inputs(input_path, asset_root)

    assert len(loaded) == 2
    assert loaded[0].image.sha256 == loaded[1].image.sha256 == image_sha
    assert loaded[0].prompt_sha256 != loaded[1].prompt_sha256


def test_duplicate_observable_prompt_image_pair_is_rejected_despite_new_id(
    tmp_path: Path,
) -> None:
    asset_root = tmp_path / "assets"
    asset_root.mkdir()
    image_bytes = b"one official page"
    (asset_root / "page.bin").write_bytes(image_bytes)
    image_sha = _sha_bytes(image_bytes)
    row = _opaque_row(
        input_id="row-a",
        prompt=_prompt(7),
        image_path="page.bin",
        image_sha256=image_sha,
    )
    duplicate = {**row, "input_id": "row-b"}
    input_path = tmp_path / "inputs.jsonl"
    _write_jsonl(input_path, [row, duplicate])

    with pytest.raises(McqOpaqueBatchError, match="duplicate observable"):
        load_mcq_opaque_inputs(input_path, asset_root)


@pytest.mark.parametrize("forbidden_key", ["task_id", "gold", "page", "source"])
def test_nested_task_gold_page_or_source_metadata_is_never_admitted(
    tmp_path: Path,
    forbidden_key: str,
) -> None:
    asset_root = tmp_path / "assets"
    asset_root.mkdir()
    image_bytes = b"opaque page bytes"
    (asset_root / "page.bin").write_bytes(image_bytes)
    row = _opaque_row(
        input_id="row-a",
        prompt=_prompt(7),
        image_path="page.bin",
        image_sha256=_sha_bytes(image_bytes),
    )
    image = row["images"][0]
    image["metadata"] = {"deep": [{forbidden_key: "must-not-enter-policy"}]}
    input_path = tmp_path / "inputs.jsonl"
    _write_jsonl(input_path, [row])

    with pytest.raises(McqOpaqueBatchError):
        load_mcq_opaque_inputs(input_path, asset_root)


@pytest.mark.parametrize(
    "raw_key",
    ["TASK-ID", "Gold", "page_number", "Source PDF"],
)
def test_forbidden_metadata_key_normalization_cannot_bypass_rejection(
    tmp_path: Path,
    raw_key: str,
) -> None:
    asset_root = tmp_path / "assets"
    asset_root.mkdir()
    image_bytes = b"opaque page bytes"
    (asset_root / "page.bin").write_bytes(image_bytes)
    row = _opaque_row(
        input_id="row-a",
        prompt=_prompt(7),
        image_path="page.bin",
        image_sha256=_sha_bytes(image_bytes),
    )
    image = row["images"][0]
    image["metadata"] = {raw_key: "must-not-enter-policy"}
    input_path = tmp_path / "inputs.jsonl"
    _write_jsonl(input_path, [row])

    with pytest.raises(McqOpaqueBatchError):
        load_mcq_opaque_inputs(input_path, asset_root)


def test_input_id_renaming_cannot_change_observable_resolver_inputs(
    tmp_path: Path,
    synthetic_source,
) -> None:
    inventory, _, _ = synthetic_source
    first, _ = _source_page_with_two_choices(inventory)
    asset_root = tmp_path / "assets"
    asset_root.mkdir()
    image_bytes = b"same observable page bytes under renamed alignment id"
    (asset_root / "page.bin").write_bytes(image_bytes)
    image_sha = _sha_bytes(image_bytes)
    prompt = _prompt(first.question_number)

    loaded_runs: list[tuple[tuple[object, ...], bytes]] = []
    for input_id in ("alignment-original", "alignment-renamed"):
        input_path = tmp_path / f"{input_id}.jsonl"
        _write_jsonl(
            input_path,
            [
                _opaque_row(
                    input_id=input_id,
                    prompt=prompt,
                    image_path="page.bin",
                    image_sha256=image_sha,
                )
            ],
        )
        loaded_runs.append(
            (load_mcq_opaque_inputs(input_path, asset_root), input_path.read_bytes())
        )

    original = loaded_runs[0][0][0]
    renamed = loaded_runs[1][0][0]
    original_observation = asdict(original)
    renamed_observation = asdict(renamed)
    original_observation.pop("input_id")
    renamed_observation.pop("input_id")
    assert original_observation == renamed_observation
