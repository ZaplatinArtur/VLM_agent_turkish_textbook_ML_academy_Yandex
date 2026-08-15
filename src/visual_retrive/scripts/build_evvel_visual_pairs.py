"""Download validated Evvel page-image segments and emit SigLIP training pairs."""
from __future__ import annotations

import argparse
import concurrent.futures
import io
import json
import re
import threading
from collections import defaultdict
from pathlib import Path
from typing import Any

import requests
from PIL import Image


SUBJECTS = (
    (("matematik",), "math"),
    (("fizik",), "physics"),
    (("kimya",), "chemistry"),
    (("biyoloji",), "biology"),
    (("cografya", "coğrafya"), "geography"),
    (("tarih", "inkilap"), "history"),
    (("turk-dili", "türk-dili", "edebiyat"), "turkish language and literature"),
    (("ingilizce",), "english"),
    (("almanca",), "german"),
    (("din-kulturu", "din-kültürü", "kuran", "kur'an"), "religious culture and ethics"),
    (("felsefe",), "philosophy"),
    (("psikoloji",), "psychology"),
    (("sosyoloji",), "sociology"),
    (("mantik", "mantık"), "logic"),
)
lock = threading.Lock()


def subject_for(row: dict[str, Any]) -> str:
    text = " ".join(str(row.get(key) or "") for key in ("book_slug", "title", "page_url")).casefold()
    for needles, subject in SUBJECTS:
        if any(needle in text for needle in needles):
            return subject
    return "high school other"


def safe_slug(row: dict[str, Any]) -> str:
    value = f"evvel-{int(row['grade'])}-sinif-{row['book_slug']}".casefold()
    return re.sub(r"[^a-z0-9_-]+", "-", value).strip("-")


def download(item, root: Path):
    row, segment, url = item
    slug = safe_slug(row)
    encoded_page = int(row["page_number"]) * 100 + segment
    relative = Path("books") / slug / "pages" / f"{encoded_page:04d}.jpg"
    target = root / relative
    if target.is_file():
        try:
            with Image.open(target) as image:
                image.verify()
            return row, segment, str(relative).replace("\\", "/"), encoded_page
        except Exception:
            target.unlink(missing_ok=True)
    response = requests.get(url, timeout=(10, 60), headers={"User-Agent": "TurkishTextbookResearchBot/1.0"})
    response.raise_for_status()
    if len(response.content) > 25_000_000:
        raise ValueError("image exceeds 25 MB")
    with Image.open(io.BytesIO(response.content)) as image:
        image.load()
        if image.width < 400 or image.height < 500:
            raise ValueError(f"image too small: {image.size}")
        rgb = image.convert("RGB")
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_suffix(".part")
        rgb.save(temp, format="JPEG", quality=92, optimize=True)
        temp.replace(target)
    return row, segment, str(relative).replace("\\", "/"), encoded_page


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.input.open(encoding="utf-8") if line.strip()]
    items = []
    for row in rows:
        if not row.get("useful_answer") or not row.get("synthetic_queries"):
            continue
        for segment, url in enumerate(dict.fromkeys(row.get("image_urls") or []), 1):
            items.append((row, segment, url))
    downloaded = []
    failed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(download, item, args.data_root): item for item in items}
        for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
            try:
                downloaded.append(future.result())
            except Exception as exc:
                failed += 1
                if failed <= 30:
                    print(f"image_failed={type(exc).__name__}:{exc}", flush=True)
            if index % 100 == 0 or index == len(items):
                print(f"images={index}/{len(items)} ok={len(downloaded)} failed={failed}", flush=True)

    by_source_page = defaultdict(list)
    for row, segment, relative, encoded_page in downloaded:
        by_source_page[(row["page_url"], int(row["page_number"]))].append((row, segment, relative, encoded_page))
    source_pages = sorted(by_source_page, key=lambda key: (by_source_page[key][0][0]["grade"], by_source_page[key][0][0]["book_slug"], key[1]))
    neighbors = defaultdict(list)
    grouped = defaultdict(list)
    for key in source_pages:
        row = by_source_page[key][0][0]
        grouped[safe_slug(row)].append(key)
    for slug, keys in grouped.items():
        keys.sort(key=lambda key: key[1])
        for position, key in enumerate(keys):
            nearby = keys[max(0, position - 2):position] + keys[position + 1:position + 3]
            neighbors[key] = [
                f"{slug}:{by_source_page[other][0][3]}" for other in nearby
            ][:4]

    output_rows = []
    for key in source_pages:
        segments = sorted(by_source_page[key], key=lambda value: value[1])
        row, _segment, relative, encoded_page = segments[0]
        slug = safe_slug(row)
        page_id = f"{slug}:{encoded_page}"
        same_page_ids = [f"{slug}:{value[3]}" for value in segments]
        for query in row["synthetic_queries"]:
            output_rows.append({
                "query": str(query).strip(),
                "positive_page_id": page_id,
                "positive_image": relative,
                "positive_answer_text": row.get("answer_text", ""),
                "hard_negative_page_ids": neighbors[key],
                "same_source_page_ids": same_page_ids,
                "subject": subject_for(row),
                "grade": int(row["grade"]),
                "book_slug": slug,
                "source": "evvelcevap:qwen_visual_multi_segment",
                "source_url": row["page_url"],
            })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output:
        for row in output_rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    grades = defaultdict(int)
    for row in output_rows:
        grades[row["grade"]] += 1
    print(json.dumps({"input_pages": len(rows), "candidate_images": len(items), "downloaded_images": len(downloaded), "failed_images": failed, "pairs": len(output_rows), "grades": grades, "output": str(args.output)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
