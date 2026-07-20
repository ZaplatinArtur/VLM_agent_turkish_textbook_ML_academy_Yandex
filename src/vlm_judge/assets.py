from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


YANDEX_RESOURCE_API = "https://cloud-api.yandex.net/v1/disk/public/resources"


def resolve_yandex_asset(public_url: str, *, timeout: float = 15.0) -> dict[str, Any]:
    query = urllib.parse.urlencode({"public_key": public_url})
    request = urllib.request.Request(
        f"{YANDEX_RESOURCE_API}?{query}",
        headers={"User-Agent": "vlm-judge/0.1"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.load(response)
    return {
        "name": payload.get("name"),
        "size": payload.get("size"),
        "mime_type": payload.get("mime_type"),
        "md5": payload.get("md5"),
        "sha256": payload.get("sha256"),
    }


def _verify_one(entry: dict[str, Any], retries: int) -> dict[str, Any]:
    error: str | None = None
    for attempt in range(1, retries + 2):
        try:
            resolved = resolve_yandex_asset(entry["public_url"])
            return {**entry, "ok": True, "attempts": attempt, **resolved, "error": None}
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            if attempt <= retries:
                time.sleep(min(0.25 * attempt, 1.0))
    return {**entry, "ok": False, "attempts": retries + 1, "error": error}


def verify_benchmark_assets(
    benchmark_path: Path,
    manifest_path: Path,
    summary_path: Path,
    *,
    workers: int = 12,
    retries: int = 1,
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    with benchmark_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            task = json.loads(line)
            for role, key in (
                ("question", "question_image_url"),
                ("reference", "reference_image_url"),
            ):
                public_url = task.get(key)
                if public_url:
                    entries.append(
                        {
                            "task_id": task.get("task_id"),
                            "role": role,
                            "public_url": public_url,
                        }
                    )

    cached_ok: dict[tuple[str, str, str], dict[str, Any]] = {}
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("ok"):
                    key = (
                        str(record.get("task_id")),
                        str(record.get("role")),
                        str(record.get("public_url")),
                    )
                    cached_ok[key] = record

    results: list[dict[str, Any] | None] = [None] * len(entries)
    pending: list[tuple[int, dict[str, Any]]] = []
    for index, entry in enumerate(entries):
        key = (str(entry["task_id"]), str(entry["role"]), str(entry["public_url"]))
        if key in cached_ok:
            results[index] = cached_ok[key]
        else:
            pending.append((index, entry))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_index = {
            executor.submit(_verify_one, entry, retries): index
            for index, entry in pending
        }
        for future in as_completed(future_to_index):
            results[future_to_index[future]] = future.result()

    finalized = [result for result in results if result is not None]
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="\n") as handle:
        for result in finalized:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
    ok = [result for result in finalized if result.get("ok")]
    summary = {
        "assets": len(finalized),
        "available": len(ok),
        "unavailable": len(finalized) - len(ok),
        "question_assets": sum(result.get("role") == "question" for result in finalized),
        "reference_assets": sum(result.get("role") == "reference" for result in finalized),
        "total_bytes": sum(int(result.get("size") or 0) for result in ok),
        "reused_from_manifest": len(cached_ok),
        "checked_this_run": len(pending),
        "manifest": str(manifest_path),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return summary
