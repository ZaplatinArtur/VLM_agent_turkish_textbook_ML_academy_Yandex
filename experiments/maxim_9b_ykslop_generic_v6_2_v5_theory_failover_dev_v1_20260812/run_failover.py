"""Materialize the frozen V6.2 -> V5-theory failover candidate."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from failover_rule import apply_failover, selection_counts


HERE = Path(__file__).resolve().parent
FREEZE = HERE / "DEV_FAILOVER_FREEZE.json"
FREEZE_SHA = HERE / "DEV_FAILOVER_FREEZE_SHA256.txt"
FALLBACK = HERE / "frozen" / "v5_theory_fallback_content_only.jsonl"
V6_DIR = HERE.parent / "maxim_9b_ykslop_generic_reasoning_nonstream_alt_v6_2_siliconflow_dev_20260812"
V6_FREEZE = V6_DIR / "DEV_EXECUTION_FREEZE.json"
V6_COMPLETION = V6_DIR / "DEV_WAVE_COMPLETION.json"
OUTPUT = HERE / "runs" / "predictions_content_only.jsonl"
COMPLETION = HERE / "DEV_WAVE_COMPLETION.json"


class RunError(RuntimeError):
    pass


def stable_bytes(path: Path) -> bytes:
    with path.open("rb") as handle:
        return handle.read()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(stable_bytes(path))


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(stable_bytes(path).decode("utf-8"))
    if type(value) is not dict:
        raise RunError(f"JSON object required: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for number, raw in enumerate(stable_bytes(path).splitlines(), 1):
        if not raw.strip():
            continue
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RunError(f"invalid JSONL row {number}: {path}") from exc
        if type(value) is not dict:
            raise RunError(f"non-object JSONL row {number}: {path}")
        rows.append(value)
    return rows


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def exclusive_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise


def artifact(path: Path, *, rows: int | None = None) -> dict[str, Any]:
    value: dict[str, Any] = {
        "path": path.resolve().relative_to(HERE.resolve()).as_posix(),
        "sha256": sha256_file(path),
        "size": path.stat().st_size,
    }
    if rows is not None:
        value["rows"] = rows
    return value


def verify_descriptor(descriptor: Any, *, base: Path) -> Path:
    if type(descriptor) is not dict or set(descriptor) != {"path", "rows", "sha256", "size"}:
        raise RunError("V6 prediction descriptor mismatch")
    path = (base / descriptor["path"]).resolve()
    data = stable_bytes(path)
    if len(data) != descriptor["size"] or sha256_bytes(data) != descriptor["sha256"]:
        raise RunError("V6 prediction descriptor closure mismatch")
    if len([line for line in data.splitlines() if line.strip()]) != descriptor["rows"]:
        raise RunError("V6 prediction row descriptor mismatch")
    return path


def execute(args: argparse.Namespace) -> dict[str, Any]:
    if OUTPUT.exists() or COMPLETION.exists():
        raise RunError("failover output already exists")
    if sha256_file(FREEZE) != args.expected_freeze_sha256:
        raise RunError("failover freeze external pin mismatch")
    if FREEZE_SHA.read_text(encoding="ascii").strip() != args.expected_freeze_sha256:
        raise RunError("failover freeze sidecar mismatch")
    freeze = read_json(FREEZE)
    if (
        freeze.get("schema_version") != "generic-v6-v5-theory-failover-dev-freeze-v1"
        or freeze.get("state") != "pre_outcome_frozen_unexecuted"
        or freeze.get("v6_execution_freeze_sha256") != sha256_file(V6_FREEZE)
        or freeze.get("row_count") != 185
    ):
        raise RunError("failover freeze closure mismatch")
    fallback_descriptor = freeze.get("artifacts", {}).get("v5_theory_fallback")
    if type(fallback_descriptor) is not dict or fallback_descriptor.get("sha256") != sha256_file(FALLBACK):
        raise RunError("fallback artifact mismatch")
    if sha256_file(V6_COMPLETION) != args.expected_v6_completion_sha256:
        raise RunError("V6 completion external pin mismatch")
    v6_completion = read_json(V6_COMPLETION)
    if (
        v6_completion.get("schema_version") != "generic-medium-nonstream-dev-completion-v6"
        or v6_completion.get("freeze_sha256") != freeze["v6_execution_freeze_sha256"]
        or v6_completion.get("rows") != 185
        or v6_completion.get("gold_accessed") is not False
        or v6_completion.get("final_accessed") is not False
        or v6_completion.get("runtime_opaque_ids") is not False
    ):
        raise RunError("V6 completion guard mismatch")
    v6_predictions_path = verify_descriptor(v6_completion.get("predictions"), base=V6_DIR)
    v6_rows = read_jsonl(v6_predictions_path)
    fallback_rows = read_jsonl(FALLBACK)
    outputs = apply_failover(v6_rows, fallback_rows)
    payload = b"".join(canonical_bytes(row) for row in outputs)
    exclusive_write(OUTPUT, payload)
    completion = {
        "schema_version": "generic-v6-v5-theory-failover-dev-completion-v1",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        "freeze_sha256": args.expected_freeze_sha256,
        "v6_completion_sha256": args.expected_v6_completion_sha256,
        "rows": 185,
        "predictions": artifact(OUTPUT, rows=185),
        "selection_counts": selection_counts(outputs),
        "gold_accessed": False,
        "final_accessed": False,
        "opaque_identifier_access": False,
    }
    exclusive_write(COMPLETION, canonical_bytes(completion))
    return completion


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-freeze-sha256", required=True)
    parser.add_argument("--expected-v6-completion-sha256", required=True)
    result = execute(parser.parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
