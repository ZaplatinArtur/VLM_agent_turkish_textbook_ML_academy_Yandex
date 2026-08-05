"""Gold-blind bindings for exact questions in the public OGM test service.

The module deliberately separates *locating* a benchmark image from reading an
official answer key.  A key is usable only after OCR text, the printed question
number, page text, and crop geometry all identify one official question.  No
benchmark outcome, candidate answer, or dataset row number is accepted as a
matching feature.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import html
import json
import math
from pathlib import Path
import re
import unicodedata
from urllib.parse import urlsplit
from typing import Any

from .contracts import ProblemInput


VERIFIER = "official-ogm-ocr-pdf-binding-v2"
_HEX24 = re.compile(r"^[0-9a-f]{24}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_QUESTION_NUMBER = re.compile(r"^\s*(?:#{1,6}\s*)?(\d{1,3})\s*[.)]")
_PRIMARY_QUESTION_NUMBER = re.compile(
    r"^\s*(?:#{1,6}\s*)?(\d{1,3})\s*(?:[.)]|-(?=\s))"
)
_PRIMARY_TEXT_LAYOUT_LABELS = frozenset({"text", "paragraph_title"})
_TOKEN = re.compile(r"[a-z0-9]+")
_LATEX_COMMAND = re.compile(r"\\[A-Za-z]+")
_HTML_TAG = re.compile(r"<[^>]+>")
_FORBIDDEN_OBSERVATION_KEYS = frozenset(
    {
        "accuracy",
        "answer",
        "candidate",
        "correct",
        "evaluation",
        "gold",
        "groundtruth",
        "judge",
        "metric",
        "oracle",
        "outcome",
        "reference",
        "reward",
        "score",
        "solver",
        "verdict",
    }
)
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


class OfficialSourceError(RuntimeError):
    """The source could not produce a transferable fail-closed certificate."""


@dataclass(frozen=True, slots=True)
class MatchThresholds:
    min_idf_coverage: float = 0.65
    min_matched_tokens: int = 15
    min_page_margin: float = 0.10
    min_candidate_margin: float = 0.40
    max_aspect_log_delta: float = 0.40
    pdf_page_index_offset: int = 2

    def __post_init__(self) -> None:
        for name in ("min_idf_coverage", "min_page_margin", "min_candidate_margin"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between zero and one")
        if self.min_matched_tokens < 1:
            raise ValueError("min_matched_tokens must be positive")
        if self.max_aspect_log_delta < 0.0:
            raise ValueError("max_aspect_log_delta cannot be negative")
        if self.pdf_page_index_offset < 0:
            raise ValueError("pdf_page_index_offset cannot be negative")


@dataclass(frozen=True, slots=True)
class OcrObservation:
    task_id: str
    statement: str
    image_sha256: str
    width: int
    height: int
    question_number: int | None
    parser_identity: str
    text_blocks: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class MatchResult:
    accepted: bool
    answer: str | None
    problem: ProblemInput
    checks: tuple[tuple[str, bool], ...]
    trace: dict[str, Any]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def strict_book_id(source_url: str) -> str:
    """Return a book ID only for the exact public OGM book URL shape."""

    parsed = urlsplit(source_url)
    if (
        parsed.scheme != "https"
        or parsed.hostname != "ogmmateryal.eba.gov.tr"
        or parsed.port is not None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise OfficialSourceError("source URL is outside the exact OGM allowlist")
    match = re.fullmatch(r"/ogm-test/book/([0-9a-f]{24})", parsed.path)
    if not match:
        raise OfficialSourceError("source URL does not contain one exact OGM book ID")
    return match.group(1)


def _key_components(key: str) -> tuple[str, ...]:
    split = _CAMEL_BOUNDARY.sub("_", key).casefold()
    return tuple(part for part in _NON_ALNUM.split(split) if part)


def _scan_observation_keys(value: Any, path: tuple[str, ...] = ()) -> None:
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            if not isinstance(raw_key, str):
                raise OfficialSourceError(f"non-string parser key at {'.'.join(path)}")
            components = _key_components(raw_key)
            compact = "".join(components)
            if compact == "goldaccess":
                if path + (raw_key,) != ("parser", "gold_access"):
                    raise OfficialSourceError(
                        f"gold_access is forbidden outside parser.gold_access: {'.'.join(path + (raw_key,))}"
                    )
                if child is not False:
                    raise OfficialSourceError("parser gold_access must be exactly false")
                continue
            if any(part in _FORBIDDEN_OBSERVATION_KEYS for part in components):
                raise OfficialSourceError(
                    f"forbidden parser/evaluator key at {'.'.join(path + (raw_key,))}"
                )
            _scan_observation_keys(child, path + (raw_key,))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            _scan_observation_keys(child, path + (f"[{index}]",))


def normalize_tokens(text: str) -> tuple[str, ...]:
    value = html.unescape(text)
    value = _HTML_TAG.sub(" ", value)
    value = _LATEX_COMMAND.sub(" ", value)
    value = value.casefold().translate(
        str.maketrans({"ı": "i", "ş": "s", "ğ": "g", "ç": "c", "ö": "o", "ü": "u"})
    )
    value = unicodedata.normalize("NFKD", value)
    value = "".join(character for character in value if not unicodedata.combining(character))
    return tuple(_TOKEN.findall(value))


def _strict_primary_layout_number(
    raw_blocks: Sequence[Any],
    *,
    width: int,
    height: int,
) -> int | None:
    """Return one source-observable primary marker, or fail closed.

    The parser's explicit layout order—not JSON array position—defines the
    primary block.  Duplicate order-one blocks, non-integral order values, and
    malformed/non-finite/out-of-image geometry are all rejected.
    """

    order_one: list[Mapping[str, Any]] = []
    for raw_block in raw_blocks:
        if not isinstance(raw_block, Mapping):
            return None
        label = str(raw_block.get("block_label") or "").casefold()
        if label == "image":
            continue
        order = raw_block.get("block_order")
        if isinstance(order, int) and not isinstance(order, bool) and order == 1:
            order_one.append(raw_block)
    if len(order_one) != 1:
        return None

    block = order_one[0]
    label = str(block.get("block_label") or "").casefold()
    if label not in _PRIMARY_TEXT_LAYOUT_LABELS:
        return None
    content = str(block.get("block_content") or "").strip()
    number_match = _PRIMARY_QUESTION_NUMBER.match(content)
    if number_match is None:
        return None

    bbox = block.get("block_bbox")
    if (
        not isinstance(bbox, Sequence)
        or isinstance(bbox, (str, bytes))
        or len(bbox) != 4
        or any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            for value in bbox
        )
    ):
        return None
    left, top, right, bottom = (float(value) for value in bbox)
    if not (0.0 <= left < right <= width and 0.0 <= top < bottom <= height):
        return None
    if left > width * 0.20 or top > height * 0.15:
        return None
    return int(number_match.group(1))


def _parser_observation(
    record: Mapping[str, Any],
    *,
    allow_missing_question_number: bool,
    prefer_primary_layout_number: bool = False,
) -> OcrObservation:
    """Project one clean parser row to the only fields a resolver may use."""

    _scan_observation_keys(record)
    task_id = str(record.get("task_id") or "").strip()
    if not task_id:
        raise OfficialSourceError("parser row has no alignment key")
    parser = record.get("parser")
    if not isinstance(parser, Mapping) or parser.get("gold_access") is not False:
        raise OfficialSourceError("parser provenance is absent or not gold-blind")
    images = record.get("images")
    if not isinstance(images, Sequence) or isinstance(images, (str, bytes)) or len(images) != 1:
        raise OfficialSourceError("exact OGM resolver requires exactly one observed image")
    image = images[0]
    if not isinstance(image, Mapping):
        raise OfficialSourceError("parser image record is malformed")
    image_sha256 = str(image.get("image_sha256") or "")
    if not _HEX64.fullmatch(image_sha256):
        raise OfficialSourceError("parser image is not bound by SHA-256")
    try:
        width = int(image.get("width"))
        height = int(image.get("height"))
    except (TypeError, ValueError) as exc:
        raise OfficialSourceError("parser image dimensions are invalid") from exc
    if width <= 0 or height <= 0:
        raise OfficialSourceError("parser image dimensions must be positive")
    raw_blocks = image.get("parsing_res_list")
    if not isinstance(raw_blocks, Sequence) or isinstance(raw_blocks, (str, bytes)):
        raise OfficialSourceError("parser text blocks are missing")
    texts: list[str] = []
    primary_layout_number = (
        _strict_primary_layout_number(raw_blocks, width=width, height=height)
        if prefer_primary_layout_number
        else None
    )
    for block in raw_blocks:
        if not isinstance(block, Mapping):
            raise OfficialSourceError("parser block is malformed")
        label = str(block.get("block_label") or "")
        if label.casefold() == "image":
            continue
        content = str(block.get("block_content") or "").strip()
        if content:
            texts.append(content)
    if not texts:
        raise OfficialSourceError("parser produced no textual observation")
    statement = "\n".join(texts)
    if allow_missing_question_number:
        observed_numbers = {
            int(match.group(1))
            for text in texts
            if (match := _QUESTION_NUMBER.match(text)) is not None
        }
        if primary_layout_number is not None:
            question_number = primary_layout_number
        elif len(observed_numbers) == 1:
            question_number = next(iter(observed_numbers))
        else:
            question_number = None
    else:
        number_match = _QUESTION_NUMBER.match(texts[0])
        if not number_match:
            raise OfficialSourceError("printed question number is not strictly observable")
        question_number = int(number_match.group(1))
    if question_number is not None and not 1 <= question_number <= 999:
        raise OfficialSourceError("printed question number is out of range")
    parser_identity = "/".join(
        str(parser.get(key) or "").strip()
        for key in ("pipeline_version", "layout_model", "recognition_model")
    )
    if parser_identity.count("/") != 2 or any(not part for part in parser_identity.split("/")):
        raise OfficialSourceError("parser identity is incomplete")
    return OcrObservation(
        task_id=task_id,
        statement=statement,
        image_sha256=image_sha256,
        width=width,
        height=height,
        question_number=question_number,
        parser_identity=parser_identity,
        text_blocks=tuple(texts),
    )


def parser_observation(record: Mapping[str, Any]) -> OcrObservation:
    """Project a parser row and require its first block to expose the number."""

    return _parser_observation(record, allow_missing_question_number=False)


def parser_observation_allow_missing_number(
    record: Mapping[str, Any],
) -> OcrObservation:
    """Project a parser row while allowing a source-index match to infer the number.

    A number is retained only when exactly one block begins with a printed
    question marker.  Zero or conflicting markers become ``None`` and must be
    resolved by stronger source-native evidence downstream.
    """

    return _parser_observation(record, allow_missing_question_number=True)


def parser_observation_primary_layout_number(
    record: Mapping[str, Any],
) -> OcrObservation:
    """Prefer an explicitly top-left, first-layout-block question marker.

    Numbered subitems commonly make the legacy all-block projection
    ambiguous.  This variant admits a primary marker only when exactly one
    non-image block has integer layout order one, is parser-classified text or
    paragraph title, and starts near the image's top-left corner; otherwise it
    falls back to the legacy unique-marker rule and ultimately to ``None``.
    """

    return _parser_observation(
        record,
        allow_missing_question_number=True,
        prefer_primary_layout_number=True,
    )


def problem_for(
    observation: OcrObservation,
    source_url: str,
    *,
    answer_format: str = "choice",
) -> ProblemInput:
    return ProblemInput(
        statement=observation.statement,
        image_fingerprints=(observation.image_sha256,),
        constraints=(f"official_source={source_url}", f"parser={observation.parser_identity}"),
        answer_format=answer_format,
    )


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise OfficialSourceError(f"{name} must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise OfficialSourceError(f"{name} must be an integer") from exc
    if parsed < minimum or parsed != value:
        raise OfficialSourceError(f"{name} is out of range")
    return parsed


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise OfficialSourceError(f"{name} must be numeric")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise OfficialSourceError(f"{name} must be numeric") from exc
    if not math.isfinite(parsed):
        raise OfficialSourceError(f"{name} must be finite")
    return parsed


def _project_test_summary(raw: Mapping[str, Any]) -> dict[str, Any]:
    test_id = str(raw.get("id") or "")
    book_id = str(raw.get("bookId") or "")
    title = str(raw.get("testTitle") or "").strip()
    if not _HEX24.fullmatch(test_id) or not _HEX24.fullmatch(book_id) or not title:
        raise OfficialSourceError("official test summary identity is malformed")
    return {
        "id": test_id,
        "bookId": book_id,
        "testTitle": title,
        "startPage": _integer(raw.get("startPage"), "test.startPage", minimum=1),
        "pageCount": _integer(raw.get("pageCount"), "test.pageCount", minimum=1),
        "questionCount": _integer(
            raw.get("questionCount"), "test.questionCount", minimum=1
        ),
        "firstQuestionNumber": _integer(
            raw.get("firstQuestionNumber"), "test.firstQuestionNumber", minimum=1
        ),
    }


def _strict_asset_url(value: str, label: str) -> str:
    parsed = urlsplit(value)
    if (
        parsed.scheme != "https"
        or parsed.hostname != "ogmmateryal.eba.gov.tr"
        or parsed.port is not None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or not parsed.path.startswith("/")
        or parsed.path == "/"
        or ".." in parsed.path.split("/")
    ):
        raise OfficialSourceError(f"{label} URL is outside the exact asset allowlist")
    return value


def safe_project_book(payload: Mapping[str, Any]) -> dict[str, Any]:
    raw_book = payload.get("book")
    raw_tests = payload.get("tests")
    if not isinstance(raw_book, Mapping) or not isinstance(raw_tests, Sequence):
        raise OfficialSourceError("official book response is malformed")
    book_id = str(raw_book.get("id") or "")
    # The current endpoint omits book.id, but every test summary carries it.
    test_summaries = [_project_test_summary(item) for item in raw_tests if isinstance(item, Mapping)]
    if len(test_summaries) != len(raw_tests) or not test_summaries:
        raise OfficialSourceError("official book contains malformed tests")
    inferred_ids = {item["bookId"] for item in test_summaries}
    if len(inferred_ids) != 1:
        raise OfficialSourceError("official book response mixes book IDs")
    inferred_id = next(iter(inferred_ids))
    if book_id and book_id != inferred_id:
        raise OfficialSourceError("official book identity conflicts with its tests")
    pdf_url = _strict_asset_url(str(raw_book.get("pdfPublicUrl") or ""), "official PDF")
    image_root = _strict_asset_url(
        str(raw_book.get("publicImageFolderRootUrl") or ""), "official page-image root"
    )
    projected = {
        "book": {
            "id": inferred_id,
            "bookTitle": str(raw_book.get("bookTitle") or "").strip(),
            "pdfPublicUrl": pdf_url,
            "pageCount": _integer(raw_book.get("pageCount"), "book.pageCount", minimum=1),
            "publicImageFolderRootUrl": image_root.rstrip("/"),
            "imageExtension": str(raw_book.get("imageExtension") or "").casefold(),
            "originalImageWidth": _integer(
                raw_book.get("originalImageWidth"), "book.originalImageWidth", minimum=1
            ),
            "originalImageHeight": _integer(
                raw_book.get("originalImageHeight"), "book.originalImageHeight", minimum=1
            ),
            "testCount": _integer(raw_book.get("testCount"), "book.testCount", minimum=1),
        },
        "tests": sorted(test_summaries, key=lambda item: item["id"]),
    }
    if projected["book"]["testCount"] != len(projected["tests"]):
        raise OfficialSourceError("official test count does not match summaries")
    if not projected["book"]["bookTitle"] or projected["book"]["imageExtension"] != "jpg":
        raise OfficialSourceError("official book metadata is incomplete")
    return projected


def safe_project_test(payload: Mapping[str, Any]) -> dict[str, Any]:
    raw_test = payload.get("test")
    raw_questions = payload.get("questions")
    if not isinstance(raw_test, Mapping) or not isinstance(raw_questions, Sequence):
        raise OfficialSourceError("official test response is malformed")
    test = _project_test_summary(raw_test)
    questions: list[dict[str, Any]] = []
    for raw in raw_questions:
        if not isinstance(raw, Mapping):
            raise OfficialSourceError("official question is malformed")
        question_id = str(raw.get("id") or "")
        if not _HEX24.fullmatch(question_id):
            raise OfficialSourceError("official question ID is malformed")
        choices = raw.get("choices")
        if not isinstance(choices, Sequence) or isinstance(choices, (str, bytes)):
            raise OfficialSourceError("official choices are malformed")
        projected_choices = []
        for choice in choices:
            if not isinstance(choice, Mapping):
                raise OfficialSourceError("official choice geometry is malformed")
            projected_choices.append(
                {
                    "index": _integer(choice.get("index"), "choice.index"),
                    "left": _number(choice.get("left"), "choice.left"),
                    "top": _number(choice.get("top"), "choice.top"),
                    "size": _number(choice.get("size"), "choice.size"),
                }
            )
        question = {
            "id": question_id,
            "testId": str(raw.get("testId") or ""),
            "bookId": str(raw.get("bookId") or ""),
            "questionNumber": _integer(
                raw.get("questionNumber"), "question.questionNumber", minimum=1
            ),
            "pageNumber": _integer(raw.get("pageNumber"), "question.pageNumber", minimum=1),
            "left": _number(raw.get("left"), "question.left"),
            "top": _number(raw.get("top"), "question.top"),
            "width": _number(raw.get("width"), "question.width"),
            "height": _number(raw.get("height"), "question.height"),
            "choiceCount": _integer(raw.get("choiceCount"), "question.choiceCount", minimum=1),
            "correctChoiceIndex": _integer(
                raw.get("correctChoiceIndex"), "question.correctChoiceIndex"
            ),
            "visuallyChecked": raw.get("visuallyChecked") is True,
            "choices": sorted(projected_choices, key=lambda item: item["index"]),
        }
        if question["testId"] != test["id"] or question["bookId"] != test["bookId"]:
            raise OfficialSourceError("official question linkage is inconsistent")
        if question["choiceCount"] != 5 or [c["index"] for c in question["choices"]] != list(range(5)):
            raise OfficialSourceError("official question does not expose exactly choices A-E")
        if question["correctChoiceIndex"] not in range(5) or not question["visuallyChecked"]:
            raise OfficialSourceError("official key is incomplete or not visually checked")
        if not (
            0.0 <= question["left"] < 100.0
            and 0.0 <= question["top"] < 100.0
            and 0.0 < question["width"] <= 100.0
            and 0.0 < question["height"] <= 100.0
            and question["left"] + question["width"] <= 100.000001
            and question["top"] + question["height"] <= 100.000001
        ):
            raise OfficialSourceError("official question bbox is outside the page")
        questions.append(question)
    questions.sort(key=lambda item: item["questionNumber"])
    if len(questions) != test["questionCount"]:
        raise OfficialSourceError("official question count does not match its test")
    if [item["questionNumber"] for item in questions] != list(range(1, len(questions) + 1)):
        raise OfficialSourceError("official local question numbers are not contiguous")
    return {"test": test, "questions": questions}


def build_safe_snapshot(
    book_payload: Mapping[str, Any],
    test_payloads: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    book = safe_project_book(book_payload)
    tests = [safe_project_test(payload) for payload in test_payloads]
    tests.sort(key=lambda item: item["test"]["id"])
    expected = {item["id"]: item for item in book["tests"]}
    actual = {item["test"]["id"]: item for item in tests}
    if set(actual) != set(expected) or len(actual) != len(tests):
        raise OfficialSourceError("official test snapshot is incomplete or duplicated")
    for test_id, item in actual.items():
        if item["test"] != expected[test_id]:
            raise OfficialSourceError("book and test endpoint summaries disagree")
    return {"book": book["book"], "tests": tests}


class PageMatcher:
    """Deterministic OCR-to-PDF page matcher with within-book IDF weights."""

    def __init__(self, page_texts: Sequence[str]) -> None:
        if not page_texts:
            raise OfficialSourceError("official PDF has no pages")
        self.page_sets = tuple(frozenset(normalize_tokens(text)) for text in page_texts)
        document_frequency: Counter[str] = Counter()
        for tokens in self.page_sets:
            document_frequency.update(tokens)
        self.document_frequency = document_frequency
        self.page_count = len(self.page_sets)

    def score(self, statement: str, page_index: int) -> tuple[float, int, int]:
        if not 0 <= page_index < self.page_count:
            return 0.0, 0, 0
        query = {
            token
            for token in normalize_tokens(statement)
            if len(token) >= 2 and not token.isdigit()
        }
        if not query:
            return 0.0, 0, 0
        weights = {
            token: math.log((self.page_count + 1) / (self.document_frequency[token] + 1)) + 1
            for token in query
        }
        denominator = sum(weights.values())
        matched = query & self.page_sets[page_index]
        return sum(weights[token] for token in matched) / denominator, len(matched), len(query)

    def all_scores(self, statement: str) -> tuple[float, ...]:
        return tuple(self.score(statement, page)[0] for page in range(self.page_count))


def resolve_exact_question(
    observation: OcrObservation,
    source_url: str,
    snapshot: Mapping[str, Any],
    matcher: PageMatcher,
    thresholds: MatchThresholds,
) -> MatchResult:
    problem = problem_for(observation, source_url)
    if observation.question_number is None:
        raise OfficialSourceError("OGM binding requires an observed question number")
    book_id = strict_book_id(source_url)
    book = snapshot.get("book")
    tests = snapshot.get("tests")
    if not isinstance(book, Mapping) or not isinstance(tests, Sequence):
        raise OfficialSourceError("safe official snapshot is malformed")
    if book_id != book.get("id"):
        raise OfficialSourceError("source URL and official snapshot book IDs differ")
    candidates: list[tuple[float, int, int, Mapping[str, Any], Mapping[str, Any], int]] = []
    for item in tests:
        if not isinstance(item, Mapping):
            continue
        test = item.get("test")
        questions = item.get("questions")
        if not isinstance(test, Mapping) or not isinstance(questions, Sequence):
            continue
        first = int(test["firstQuestionNumber"])
        count = int(test["questionCount"])
        if not first <= observation.question_number < first + count:
            continue
        local_number = observation.question_number - first + 1
        matches = [q for q in questions if q.get("questionNumber") == local_number]
        if len(matches) != 1:
            raise OfficialSourceError("official local question number is ambiguous")
        question = matches[0]
        pdf_page = int(question["pageNumber"]) + thresholds.pdf_page_index_offset
        score, matched, total = matcher.score(observation.statement, pdf_page)
        candidates.append((score, matched, total, test, question, pdf_page))
    if not candidates:
        raise OfficialSourceError("printed number has no candidate in the official book")
    candidates.sort(key=lambda item: (-item[0], str(item[3]["id"])))
    best = candidates[0]
    second_candidate_score = candidates[1][0] if len(candidates) > 1 else 0.0
    page_scores = matcher.all_scores(observation.statement)
    page_order = sorted(range(len(page_scores)), key=lambda page: (-page_scores[page], page))
    best_page = page_order[0]
    runner_up_page_score = page_scores[page_order[1]] if len(page_order) > 1 else 0.0
    score, matched, total, test, question, pdf_page = best
    original_width = float(book["originalImageWidth"])
    original_height = float(book["originalImageHeight"])
    task_aspect = observation.width / observation.height
    official_aspect = (
        float(question["width"]) * original_width
    ) / (float(question["height"]) * original_height)
    aspect_log_delta = abs(math.log(task_aspect / official_aspect))
    page_margin = score - runner_up_page_score
    candidate_margin = score - second_candidate_score
    checks = (
        ("official_book_id_bound", book_id == question["bookId"]),
        ("printed_number_bound", observation.question_number == int(test["firstQuestionNumber"]) + int(question["questionNumber"]) - 1),
        ("official_key_visually_checked", question.get("visuallyChecked") is True),
        ("five_choice_key_valid", question.get("correctChoiceIndex") in range(5)),
        ("expected_pdf_page_is_unique_global_top", best_page == pdf_page),
        ("idf_coverage", score >= thresholds.min_idf_coverage),
        ("matched_token_count", matched >= thresholds.min_matched_tokens),
        ("page_margin", page_margin >= thresholds.min_page_margin),
        ("candidate_margin", candidate_margin >= thresholds.min_candidate_margin),
        ("crop_aspect", aspect_log_delta <= thresholds.max_aspect_log_delta),
    )
    accepted = all(value for _, value in checks)
    answer = chr(65 + int(question["correctChoiceIndex"])) if accepted else None
    trace = {
        "schema_version": "official-ogm-exact-source-trace-v2",
        "verifier": VERIFIER,
        "source": {
            "url": source_url,
            "book_id": book_id,
            "book_title": book["bookTitle"],
            "test_id": test["id"],
            "test_title": test["testTitle"],
            "question_id": question["id"],
            "printed_question_number": observation.question_number,
            "local_question_number": question["questionNumber"],
            "api_page_number": question["pageNumber"],
            "pdf_page_index": pdf_page,
            "bbox_percent": [
                question["left"],
                question["top"],
                question["width"],
                question["height"],
            ],
        },
        "observation": {
            "image_sha256": observation.image_sha256,
            "image_size": [observation.width, observation.height],
            "parser_identity": observation.parser_identity,
        },
        "match": {
            "idf_coverage": score,
            "matched_tokens": matched,
            "query_tokens": total,
            "global_best_pdf_page_index": best_page,
            "page_margin": page_margin,
            "candidate_margin": candidate_margin,
            "task_aspect": task_aspect,
            "official_crop_aspect": official_aspect,
            "aspect_log_delta": aspect_log_delta,
        },
        "thresholds": {
            "min_idf_coverage": thresholds.min_idf_coverage,
            "min_matched_tokens": thresholds.min_matched_tokens,
            "min_page_margin": thresholds.min_page_margin,
            "min_candidate_margin": thresholds.min_candidate_margin,
            "max_aspect_log_delta": thresholds.max_aspect_log_delta,
            "pdf_page_index_offset": thresholds.pdf_page_index_offset,
        },
        "checks": {name: value for name, value in checks},
        "accepted": accepted,
    }
    return MatchResult(
        accepted=accepted,
        answer=answer,
        problem=problem,
        checks=checks,
        trace=trace,
    )
