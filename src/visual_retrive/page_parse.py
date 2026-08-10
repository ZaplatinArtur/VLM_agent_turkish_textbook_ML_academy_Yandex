"""Parse an OdevJet kitap-sayfasi HTML page for answers (text and/or images)."""

from __future__ import annotations

import re
from html import unescape
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urljoin

from .http import BASE_URL

EMPTY_MARKERS = (
    "bu sayfada henüz çözüm bulunmamaktadır",
    "bu sayfada henuz cozum bulunmamaktadir",
)

ANSWER_IMAGE_RE = re.compile(
    r"https?://(?:www\.)?odevjet\.com/download/cevaplar/[^\"'\s>]+",
    flags=re.IGNORECASE,
)
PAGE_IMAGE_RE = re.compile(
    r"https?://(?:www\.)?odevjet\.com/download/sayfalar/[^\"'\s>]+",
    flags=re.IGNORECASE,
)


class _TextCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._skip_depth = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript"}:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"} and self._skip_depth:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        text = re.sub(r"\s+", " ", unescape(data)).strip()
        if text:
            self.parts.append(text)


def _class_list(attrs: list[tuple[str, str | None]]) -> list[str]:
    for key, value in attrs:
        if key == "class" and value:
            return value.split()
    return []


class _DivByClassExtractor(HTMLParser):
    """Extract inner HTML of the first div that has ``target_class``."""

    def __init__(self, target_class: str) -> None:
        super().__init__()
        self.target_class = target_class
        self.capture_depth = 0
        self.chunks: list[str] = []
        self.found = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        classes = _class_list(attrs)
        entering = (
            tag == "div"
            and self.target_class in classes
            and self.capture_depth == 0
            and not self.found
        )
        if self.capture_depth > 0 or entering:
            attr_text = "".join(
                f' {key}="{unescape(value)}"' if value is not None else f" {key}"
                for key, value in attrs
            )
            self.chunks.append(f"<{tag}{attr_text}>")
            self.capture_depth += 1
            return

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if self.capture_depth <= 0:
            return
        attr_text = "".join(
            f' {key}="{unescape(value)}"' if value is not None else f" {key}"
            for key, value in attrs
        )
        self.chunks.append(f"<{tag}{attr_text} />")

    def handle_endtag(self, tag: str) -> None:
        if self.capture_depth <= 0:
            return
        self.chunks.append(f"</{tag}>")
        self.capture_depth -= 1
        if self.capture_depth == 0:
            self.found = True

    def handle_data(self, data: str) -> None:
        if self.capture_depth > 0:
            self.chunks.append(data)

    def handle_entityref(self, name: str) -> None:
        if self.capture_depth > 0:
            self.chunks.append(f"&{name};")

    def handle_charref(self, name: str) -> None:
        if self.capture_depth > 0:
            self.chunks.append(f"&#{name};")

    @property
    def html(self) -> str:
        return "".join(self.chunks)


def extract_div_html(html: str, class_name: str) -> str:
    parser = _DivByClassExtractor(class_name)
    parser.feed(html)
    return parser.html


def html_to_text(html: str) -> str:
    collector = _TextCollector()
    collector.feed(html)
    # Keep paragraph-ish breaks when adjacent blocks were separate tags.
    return "\n".join(collector.parts).strip()


def _unique_urls(urls: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for url in urls:
        absolute = urljoin(BASE_URL + "/", url)
        if absolute in seen:
            continue
        seen.add(absolute)
        ordered.append(absolute)
    return ordered


def _normalize_for_marker(text: str) -> str:
    translated = text.casefold().translate(
        str.maketrans(
            {
                "ı": "i",
                "ğ": "g",
                "ü": "u",
                "ş": "s",
                "ö": "o",
                "ç": "c",
            }
        )
    )
    return re.sub(r"\s+", " ", translated).strip()


def parse_answer_page(html: str) -> dict[str, Any]:
    """Return structured answer payload from a page HTML document."""

    text_html = extract_div_html(html, "text-solution-content")
    answer_text = html_to_text(text_html) if text_html else ""
    no_solution_html = extract_div_html(html, "no-solution")
    marker_text = _normalize_for_marker(
        answer_text or html_to_text(no_solution_html)
    )
    marked_empty = any(marker in marker_text for marker in EMPTY_MARKERS) or bool(
        no_solution_html and not answer_text
    )
    if marked_empty:
        answer_text = ""

    answer_image_urls = _unique_urls(ANSWER_IMAGE_RE.findall(html))
    # Prefer cevap images scoped to the solution container when present.
    solution_html = extract_div_html(html, "solution-container")
    if solution_html:
        scoped = _unique_urls(ANSWER_IMAGE_RE.findall(solution_html))
        if scoped:
            answer_image_urls = scoped

    page_image_urls = _unique_urls(PAGE_IMAGE_RE.findall(html))

    kinds: list[str] = []
    if answer_text:
        kinds.append("text")
    if answer_image_urls:
        kinds.append("image")

    return {
        "has_solution": bool(kinds),
        "no_solution": not bool(kinds),
        "answer_kinds": kinds,
        "answer_text": answer_text,
        "answer_image_urls": answer_image_urls,
        "page_image_urls": page_image_urls,
    }
