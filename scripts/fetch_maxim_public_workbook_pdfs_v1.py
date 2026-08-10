#!/usr/bin/env python3
"""Fetch SHA-pinned public workbook PDFs through the documented Yandex API."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, urlopen


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evidence_os.official_ogm import canonical_json_bytes, sha256_file  # noqa: E402
from evidence_os.official_workbook import parse_workbook_index  # noqa: E402


API_URL = "https://cloud-api.yandex.net/v1/disk/public/resources/download"


class FetchError(RuntimeError):
    pass


def _public_key_and_path(locator: str) -> tuple[str, str | None]:
    prefix = "ya-disk-public://"
    if not locator.startswith(prefix):
        raise FetchError("source index contains a non-Yandex public locator")
    body = locator[len(prefix) :]
    separator = body.find(":/")
    if separator < 0:
        public_key, public_path = body, None
    else:
        public_key = body[:separator]
        public_path = body[separator + 1 :]
    if not public_key or (public_path is not None and not public_path.startswith("/")):
        raise FetchError("Yandex public locator is malformed")
    return public_key, public_path


def _json_get(url: str) -> dict[str, Any]:
    request = Request(url, headers={"User-Agent": "VLM-public-source-fetcher/1"})
    with urlopen(request, timeout=60) as response:
        value = json.load(response)
    if not isinstance(value, dict):
        raise FetchError("Yandex download API returned a non-object")
    return value


def _download(url: str, destination: Path, expected_sha256: str) -> str:
    parsed = urlsplit(url)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise FetchError("Yandex API returned an unsafe download URL")
    request = Request(url, headers={"User-Agent": "VLM-public-source-fetcher/1"})
    digest = hashlib.sha256()
    temporary = destination.with_suffix(destination.suffix + ".part")
    with urlopen(request, timeout=300) as response, temporary.open("wb") as sink:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            sink.write(chunk)
    actual_sha256 = digest.hexdigest()
    if actual_sha256 != expected_sha256:
        temporary.unlink(missing_ok=True)
        raise FetchError(
            f"downloaded SHA-256 mismatch for {destination.stem}: {actual_sha256}"
        )
    temporary.replace(destination)
    return actual_sha256


def fetch(index_path: Path, output_dir: Path, manifest_path: Path) -> dict[str, Any]:
    raw_index = json.loads(index_path.read_text(encoding="utf-8-sig"))
    if not isinstance(raw_index, dict):
        raise FetchError("source index must be an object")
    source_index = parse_workbook_index(raw_index)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, dict[str, Any]] = {}
    for document in source_index.documents:
        destination = output_dir / f"{document.document_id}.pdf"
        if destination.exists() and sha256_file(destination) == document.pdf_sha256:
            actual_sha = document.pdf_sha256
            reused = True
        else:
            public_key, public_path = _public_key_and_path(
                document.identity.public_locator
            )
            query: dict[str, str] = {"public_key": public_key}
            if public_path is not None:
                query["path"] = public_path
            response = _json_get(API_URL + "?" + urlencode(query))
            href = str(response.get("href") or "")
            if not href:
                raise FetchError(f"Yandex API returned no href for {document.document_id}")
            actual_sha = _download(href, destination, document.pdf_sha256)
            reused = False
        if actual_sha != document.pdf_sha256:
            raise FetchError(
                f"downloaded SHA-256 mismatch for {document.document_id}: {actual_sha}"
            )
        artifacts[document.document_id] = {
            "path": str(destination),
            "sha256": actual_sha,
            "bytes": destination.stat().st_size,
            "reused": reused,
        }
    manifest = {
        "schema_version": "maxim-public-workbook-fetch-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "GET Yandex public resources download API",
        "source_index": {"path": str(index_path), "sha256": sha256_file(index_path)},
        "documents": artifacts,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_bytes(canonical_json_bytes(manifest) + b"\n")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-index", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = fetch(
            args.source_index.resolve(),
            args.output_dir.resolve(),
            args.manifest.resolve(),
        )
    except (FetchError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
