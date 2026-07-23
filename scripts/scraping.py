import json
import re
import sys
import threading
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse

import requests
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.paths import BOOKS_DIR, ensure_data_dirs

BASE_URL = "https://www.odevjet.com"
OUTPUT_DIR = BOOKS_DIR
REQUEST_HEADERS = {"User-Agent": "Mozilla/5.0"}
MAX_WORKERS = 6
TIMEOUT = (10, 30)  # connect, read
SITEMAP_NS = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}

_thread_local = threading.local()


def log(message: str) -> None:
    print(message, flush=True)


def get_session() -> requests.Session:
    session = getattr(_thread_local, "session", None)
    if session is None:
        session = requests.Session()
        session.headers.update(REQUEST_HEADERS)
        _thread_local.session = session
    return session


def fetch_text(url: str) -> str:
    response = get_session().get(url, timeout=TIMEOUT)
    response.raise_for_status()
    response.encoding = response.apparent_encoding or "utf-8"
    return response.text


def get_book_urls() -> list[str]:
    xml_text = fetch_text(f"{BASE_URL}/kitap-sitemap.xml")
    root = ET.fromstring(xml_text)
    return [node.text.strip() for node in root.findall(".//sm:loc", SITEMAP_NS) if node.text]


def book_slug_from_url(book_url: str) -> str:
    return urlparse(book_url).path.strip("/").replace(".html", "")


def get_page_numbers(book_url: str) -> list[int]:
    html = fetch_text(book_url)
    page_numbers: list[int] = []

    for script in re.findall(
            r'<script type="application/ld\+json">(.*?)</script>',
            html,
            flags=re.DOTALL,
    ):
        try:
            data = json.loads(script)
        except json.JSONDecodeError:
            continue

        items = data if isinstance(data, list) else [data]
        for item in items:
            if item.get("@type") != "LearningResource":
                continue
            main_entity = item.get("mainEntity") or {}
            for element in main_entity.get("itemListElement") or []:
                page_url = (element.get("item") or {}).get("url") or ""
                match = re.search(r"-sayfa-(\d+)(?:\.html)?/?$", page_url)
                if match:
                    page_numbers.append(int(match.group(1)))

    return sorted(set(page_numbers))


def download_page(book_slug: str, page_number: int) -> str:
    output_file = OUTPUT_DIR / book_slug / f"{page_number:04d}.jpg"
    if output_file.exists() and output_file.stat().st_size > 0:
        return "already_exists"

    image_bytes = None
    for extension in ("jpg", "webp"):
        image_url = f"{BASE_URL}/download/sayfalar/{book_slug}/{page_number}.{extension}"
        try:
            response = get_session().get(image_url, timeout=TIMEOUT)
        except requests.RequestException:
            continue
        if response.status_code == 200 and len(response.content) > 1000:
            image_bytes = response.content
            break

    if not image_bytes:
        return "error"

    output_file.parent.mkdir(parents=True, exist_ok=True)
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image.save(output_file, "JPEG", quality=90)
    return "downloaded"


def download_book(book_slug: str, page_numbers: list[int]) -> tuple[int, int, int]:
    downloaded = skipped = errors = 0
    if not page_numbers:
        return downloaded, skipped, errors

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        tasks = [
            executor.submit(download_page, book_slug, page_number)
            for page_number in page_numbers
        ]
        for task in as_completed(tasks):
            try:
                result = task.result(timeout=TIMEOUT[0] + TIMEOUT[1] + 5)
            except Exception:
                errors += 1
                continue

            if result == "downloaded":
                downloaded += 1
            elif result == "already_exists":
                skipped += 1
            else:
                errors += 1

    return downloaded, skipped, errors


def main() -> None:
    ensure_data_dirs()
    log("Загружаю список книг с сайта...")
    book_urls = get_book_urls()
    log(f"Книг: {len(book_urls)}")

    total_downloaded = total_skipped = total_errors = 0

    for index, book_url in enumerate(book_urls, 1):
        book_slug = book_slug_from_url(book_url)
        try:
            page_numbers = get_page_numbers(book_url)
        except Exception as exc:
            log(f"[{index}/{len(book_urls)}] {book_slug}: не удалось получить список ({exc})")
            continue

        log(f"[{index}/{len(book_urls)}] {book_slug}: {len(page_numbers)} стр.")
        downloaded, skipped, errors = download_book(book_slug, page_numbers)
        total_downloaded += downloaded
        total_skipped += skipped
        total_errors += errors
        log(f"  -> скачано {downloaded}, уже было {skipped}, ошибок {errors}")

    log(f"Готово: скачано {total_downloaded}, уже было {total_skipped}, ошибок {total_errors}")
    log(f"Папка: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
