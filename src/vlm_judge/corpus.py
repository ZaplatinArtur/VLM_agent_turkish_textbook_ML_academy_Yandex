from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


BOILERPLATE_MARKERS = (
    "bu sayfada henüz çözüm bulunmamaktadır",
    "bu sayfada henuz cozum bulunmamaktadir",
)


def normalize_page_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[\t ]+", " ", line).strip() for line in text.split("\n")]
    normalized: list[str] = []
    previous_blank = False
    for line in lines:
        blank = not line
        if blank and previous_blank:
            continue
        normalized.append(line)
        previous_blank = blank
    return "\n".join(normalized).strip()


def is_boilerplate(text: str) -> bool:
    folded = unicodedata.normalize("NFKC", text).casefold()
    return any(marker in folded for marker in BOILERPLATE_MARKERS)


def split_text(text: str, *, max_chars: int = 1600, overlap_chars: int = 200) -> list[str]:
    if max_chars < 200:
        raise ValueError("max_chars must be at least 200")
    if overlap_chars < 0 or overlap_chars >= max_chars:
        raise ValueError("overlap_chars must be non-negative and smaller than max_chars")
    text = text.strip()
    if not text:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(text):
        hard_end = min(len(text), start + max_chars)
        end = hard_end
        if hard_end < len(text):
            minimum = start + int(max_chars * 0.6)
            candidates = [
                text.rfind("\n\n", minimum, hard_end),
                text.rfind("\n", minimum, hard_end),
                text.rfind(". ", minimum, hard_end),
                text.rfind(" ", minimum, hard_end),
            ]
            boundary = max(candidates)
            if boundary >= minimum:
                end = boundary + (2 if text[boundary : boundary + 2] in {"\n\n", ". "} else 1)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(start + 1, end - overlap_chars)
    return chunks


def _digest(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _quality(item: dict[str, Any]) -> tuple[int, int, int, str]:
    text = normalize_page_text(item.get("content"))
    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    images = metadata.get("image_urls") if isinstance(metadata.get("image_urls"), list) else []
    return (
        0 if is_boilerplate(text) else 1,
        len(text),
        len(images),
        str(metadata.get("scraped_at") or ""),
    )


def _chunk_id(page_hash: str, kind: str, index: int, payload: str) -> str:
    digest = hashlib.sha256(f"{page_hash}:{kind}:{index}:{payload}".encode("utf-8")).hexdigest()
    return f"chunk_{digest[:24]}"


def prepare_corpus(
    input_path: Path,
    output_dir: Path,
    *,
    max_chars: int = 1600,
    overlap_chars: int = 200,
) -> dict[str, Any]:
    canonical: dict[str, dict[str, Any]] = {}
    canonical_digests: dict[str, str] = {}
    variant_counts: Counter[str] = Counter()
    conflicting_variants: dict[str, list[dict[str, Any]]] = defaultdict(list)
    parse_errors: list[dict[str, Any]] = []
    rows = 0

    with input_path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            rows += 1
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                parse_errors.append({"line": line_number, "error": str(exc)})
                continue
            if not isinstance(item, dict):
                parse_errors.append({"line": line_number, "error": "record is not an object"})
                continue
            page_id = str(item.get("id") or "").strip()
            if not page_id:
                parse_errors.append({"line": line_number, "error": "missing id"})
                continue
            digest = _digest(item)
            variant_counts[page_id] += 1
            if page_id not in canonical:
                canonical[page_id] = item
                canonical_digests[page_id] = digest
                continue
            if digest != canonical_digests[page_id]:
                if not conflicting_variants[page_id]:
                    conflicting_variants[page_id].append(canonical[page_id])
                conflicting_variants[page_id].append(item)
                if _quality(item) > _quality(canonical[page_id]):
                    canonical[page_id] = item
                    canonical_digests[page_id] = digest

    output_dir.mkdir(parents=True, exist_ok=True)
    pages_path = output_dir / "pages.jsonl"
    chunks_path = output_dir / "chunks.jsonl"
    conflicts_path = output_dir / "conflicts.jsonl"
    errors_path = output_dir / "parse_errors.jsonl"
    report_path = output_dir / "report.json"

    content_hash_counts: Counter[str] = Counter()
    text_chunks = 0
    image_chunks = 0
    low_information_pages = 0
    boilerplate_pages = 0

    with pages_path.open("w", encoding="utf-8", newline="\n") as pages_handle, chunks_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as chunks_handle:
        for page_id in sorted(canonical):
            source = canonical[page_id]
            text = normalize_page_text(source.get("content"))
            metadata = source.get("metadata") if isinstance(source.get("metadata"), dict) else {}
            image_urls = [str(value) for value in metadata.get("image_urls", []) if value]
            content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            page_hash = _digest({"page_id": page_id, "content": text, "metadata": metadata})
            content_hash_counts[content_hash] += 1
            boilerplate = is_boilerplate(text)
            low_information = len(text) < 100 or boilerplate
            low_information_pages += int(low_information)
            boilerplate_pages += int(boilerplate)
            page_record = {
                "id": page_id,
                "content": text,
                "metadata": metadata,
                "provenance": {
                    "source_file": input_path.name,
                    "page_hash": page_hash,
                    "content_hash": content_hash,
                    "source_variants": variant_counts[page_id],
                    "conflicting_id": page_id in conflicting_variants,
                    "low_information": low_information,
                    "boilerplate": boilerplate,
                },
            }
            pages_handle.write(json.dumps(page_record, ensure_ascii=False) + "\n")

            text_parts = [] if boilerplate else split_text(
                text,
                max_chars=max_chars,
                overlap_chars=overlap_chars,
            )
            common_metadata = {
                "subject": metadata.get("ders"),
                "grade": metadata.get("sinif"),
                "book_id": metadata.get("kitap_id"),
                "book_title": metadata.get("kitap_title"),
                "page_number": metadata.get("sayfa_no"),
                "source_url": metadata.get("url"),
                "parent_page_hash": page_hash,
                "low_information": low_information,
            }
            for index, part in enumerate(text_parts):
                chunk = {
                    "chunk_id": _chunk_id(page_hash, "text", index, part),
                    "page_id": page_id,
                    "kind": "text",
                    "text": part,
                    "image_indices": list(range(len(image_urls))),
                    "metadata": {
                        **common_metadata,
                        "chunk_index": index,
                        "chunk_count": len(text_parts),
                        "index_policy": "downweight" if low_information else "normal",
                    },
                }
                chunks_handle.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                text_chunks += 1
            context = text[:500]
            for index, image_url in enumerate(image_urls):
                chunk = {
                    "chunk_id": _chunk_id(page_hash, "image", index, image_url),
                    "page_id": page_id,
                    "kind": "image",
                    "image_url": image_url,
                    "text_context": context,
                    "metadata": {
                        **common_metadata,
                        "image_index": index,
                        "index_policy": "normal",
                    },
                }
                chunks_handle.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                image_chunks += 1

    with conflicts_path.open("w", encoding="utf-8", newline="\n") as handle:
        for page_id in sorted(conflicting_variants):
            handle.write(
                json.dumps(
                    {"id": page_id, "variants": conflicting_variants[page_id]},
                    ensure_ascii=False,
                )
                + "\n"
            )
    with errors_path.open("w", encoding="utf-8", newline="\n") as handle:
        for error in parse_errors:
            handle.write(json.dumps(error, ensure_ascii=False) + "\n")

    report = {
        "source_rows": rows,
        "canonical_pages": len(canonical),
        "duplicate_rows_removed": rows - len(canonical) - len(parse_errors),
        "conflicting_ids": len(conflicting_variants),
        "parse_errors": len(parse_errors),
        "low_information_pages": low_information_pages,
        "boilerplate_pages_text_suppressed": boilerplate_pages,
        "text_chunks": text_chunks,
        "image_chunks": image_chunks,
        "total_chunks": text_chunks + image_chunks,
        "exact_content_duplicate_groups_across_ids": sum(count > 1 for count in content_hash_counts.values()),
        "max_chars": max_chars,
        "overlap_chars": overlap_chars,
        "outputs": {
            "pages": str(pages_path),
            "chunks": str(chunks_path),
            "conflicts": str(conflicts_path),
            "parse_errors": str(errors_path),
        },
    }
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return report
