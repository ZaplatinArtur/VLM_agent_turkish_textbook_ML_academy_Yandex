import hashlib
import json
import os
from pathlib import Path


REPORT = Path(os.environ.get("VLM_HOLDOUT_REPORT_DIR", Path(__file__).resolve().parents[1])).resolve()


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path: Path):
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def test_frozen_manifest_is_intact_and_gold_free():
    freeze = json.loads((REPORT / "freeze.json").read_text(encoding="utf-8"))
    manifest_path = REPORT / "selection_manifest.jsonl"
    manifest = rows(manifest_path)
    assert digest(manifest_path) == freeze["manifest_sha256"]
    assert len(manifest) == 80
    assert len({row["task_id"] for row in manifest}) == 80
    forbidden = {"answer", "gold", "official_answer", "reference_solution", "official_reference_solution"}
    assert all(not (forbidden & set(row)) for row in manifest)


def test_declared_split_and_math_exclusions():
    manifest = rows(REPORT / "selection_manifest.jsonl")
    families = [row["source_family"] for row in manifest]
    assert families.count("math12_beceri") == 20
    assert families.count("biology9_textbook") == 30
    assert families.count("physics12_textbook") == 30
    selected = {row["activity_id"] for row in manifest if row["source_family"] == "math12_beceri"}
    assert not selected.intersection({3, 17, 31, 43, 88})
    assert all(row["benchmark_dedup"]["passed"] for row in manifest)


def test_question_asset_hashes():
    workspace = REPORT.parents[1]
    for row in rows(REPORT / "selection_manifest.jsonl"):
        for path_text, expected in zip(row["question_assets"], row["question_asset_sha256"]):
            path = (workspace / path_text).resolve()
            if not path.exists():
                # Assets are intentionally gitignored in the public bundle.
                continue
            assert digest(path) == expected


def test_sealed_gold_belongs_to_frozen_manifest():
    freeze = json.loads((REPORT / "freeze.json").read_text(encoding="utf-8"))
    seal_path = REPORT / "sealed" / "gold_seal.json"
    if not seal_path.exists():
        return
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    gold_path = REPORT / "sealed" / "sealed_gold.jsonl"
    if not gold_path.exists():
        # The public integration bundle ships only the hash/count seal.
        return
    gold = rows(gold_path)
    assert seal["frozen_manifest_sha256"] == freeze["manifest_sha256"]
    assert digest(gold_path) == seal["sealed_gold_sha256"]
    assert len(gold) == 80
    assert sum(row["scoring_type"] == "exact_choice" for row in gold) == 60
    assert sum(row["scoring_type"].startswith("manual") for row in gold) == 20


def test_opaque_resolver_inputs_do_not_leak_source_or_task_ids():
    forbidden = {"task_id", "source_family", "source_pdf", "activity_id", "unit", "question_pages", "official_answer"}
    for partition, expected_count in (("math12", 20), ("mcq", 60)):
        input_path = REPORT / "resolver_inputs" / f"{partition}.jsonl"
        if not input_path.exists():
            continue
        seal = json.loads((REPORT / "resolver_inputs" / f"{partition}.seal.json").read_text(encoding="utf-8"))
        inputs = rows(input_path)
        assert len(inputs) == expected_count
        assert len({row["input_id"] for row in inputs}) == expected_count
        assert digest(input_path) == seal["public_inputs_sha256"]
        assert all(not (forbidden & set(row)) for row in inputs)
