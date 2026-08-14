"""Привязка задачи к странице официального учебника по её же тексту.

Отвечает не «какие страницы похожи», а «эта задача — вот из этой книги, ответ
такой». Кандидат принимается только при совпадении порога, номера вопроса и всех
якорей записи; иначе возвращается None и задача идёт обычным путём.

База — source_db.json, 17 записей из трёх учебников, собрана
experiments/maxim_9b_content_source_router_noid_v1_20260812 (sha256
c1307ba10d98287295b7f81bf7b415406bb32ff68c59aae34d57511d8eb1eae6). Пересобрать
её на месте нельзя: сборщику нужны исходные PDF, которых в репозитории нет.

Порог принимает только наблюдаемый текст задачи. Идентификаторы строки бенчмарка
и эталонный ответ сюда попадать не должны — за этим следит route_observable.
"""

from __future__ import annotations

import html
import json
import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any, Mapping, Sequence

SOURCE_DB_PATH = Path(__file__).resolve().parent / "source_db.json"

# Первая ступень для meb7 — только отсев: страница учебника длиннее, чем OCR
# одного упражнения, поэтому дальше требуются номер вопроса и якоря.
# Токенайзер читает только латиницу: на кириллице и арабице множество токенов
# вырождается, и оценка по IDF перестаёт что-либо значить.
MIN_QUERY_TOKENS = 4

# Запись может отвечать только на тот тип ответа, под который её формат годится.
# В базе нет ответов-букв, поэтому задачам с вариантами она ответить не может.
ANSWER_FORMATS_BY_TYPE: dict[str, frozenset[str]] = {
    "choice": frozenset({"choice", "single_letter"}),
    "short_text": frozenset({"short_text"}),
    "numeric": frozenset({"short_text", "numeric"}),
}


def _format_fits(record: Mapping[str, Any], answer_type: str | None) -> bool:
    if not answer_type:
        return True
    allowed = ANSWER_FORMATS_BY_TYPE.get(answer_type)
    if allowed is None:
        return True
    return str(record.get("answer_format") or "") in allowed
MEB_MIN_SCORE = 0.18
PAGE_MIN_SCORE = 0.65
PAGE_MIN_MARGIN = 0.50

OBSERVABLE_FIELDS = frozenset({"ocr_text", "answer_type", "input_mode"})

_HTML_TAG = re.compile(r"<[^>]+>")
_TOKEN = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_MARKER = re.compile(r"(?:^|\]\s*)(\d{1,3})\s*\.")
_ASCII_FOLD = str.maketrans({"ı": "i", "ş": "s", "ğ": "g", "ç": "c", "ö": "o", "ü": "u"})


class ObservableError(ValueError):
    """В роутер передали поле, которого он видеть не должен."""


@dataclass(frozen=True, slots=True)
class Route:
    """Найденная привязка: ответ плюс всё, чем он обоснован."""

    family: str
    record_id: str
    answer: str
    answer_format: str | None
    source_page: int | None
    score: float
    margin: float
    closure: str


def normalized_text(text: str) -> str:
    """Складывает текст в ASCII: турецкие буквы переводятся до фильтра токенов."""
    value = html.unescape(_HTML_TAG.sub(" ", text)).casefold()
    value = value.translate(_ASCII_FOLD)
    value = unicodedata.normalize("NFKD", value)
    value = "".join(char for char in value if not unicodedata.combining(char))
    return " ".join(_TOKEN.findall(value))


def token_set(text: str) -> frozenset[str]:
    return frozenset(
        token for token in normalized_text(text).split()
        if len(token) >= 2 and not token.isdigit()
    )


def idf_rank(
        query_text: str,
        records: Sequence[Mapping[str, Any]],
        marker: int | None = None,
) -> list[dict[str, Any]]:
    """Ранжирует записи по доле веса общих термов; вес терма — idf."""
    documents = [token_set(str(record["retrieval_text"])) for record in records]
    df: Counter[str] = Counter()
    for document in documents:
        df.update(document)
    query = token_set(query_text)
    count = len(documents)
    weights = {token: math.log((count + 1) / (df[token] + 1)) + 1.0 for token in query}
    denominator = sum(weights.values()) or 1.0
    ranked = [
        {
            "record_id": record["record_id"],
            "score": sum(weights[token] for token in query & document) / denominator,
            "marker_match": bool(
                marker is not None and record.get("question_number") == marker
            ),
        }
        for record, document in zip(records, documents)
    ]
    ranked.sort(key=lambda row: (-row["score"], -int(row["marker_match"]), row["record_id"]))
    return ranked


@cache
def load_source_db(path: Path = SOURCE_DB_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def source_route(
        ocr_text: str,
        source_db: Mapping[str, Any] | None = None,
        answer_type: str | None = None,
) -> Route | None:
    """Ищет запись по тексту задачи. None — ничего не подошло."""
    if len(token_set(ocr_text)) < MIN_QUERY_TOKENS:
        return None
    records = [
        record for record in (source_db or load_source_db())["records"]
        if _format_fits(record, answer_type)
    ]
    if not records:
        return None
    marker_match = _MARKER.search(ocr_text)
    marker = int(marker_match.group(1)) if marker_match else None
    normalized = normalized_text(ocr_text)

    meb = [record for record in records if record["family"] == "meb7"]
    meb_ranked = idf_rank(ocr_text, meb, marker)
    if meb_ranked:
        top = meb_ranked[0]
        record = next(item for item in meb if item["record_id"] == top["record_id"])
        anchors = list(record.get("anchors") or [])
        if (
            record.get("answer")
            and top["score"] >= MEB_MIN_SCORE
            and top["marker_match"]
            and anchors
            and all(anchor in normalized for anchor in anchors)
        ):
            runner_up = meb_ranked[1]["score"] if len(meb_ranked) > 1 else 0.0
            return _route_from(record, top["score"], top["score"] - runner_up,
                               "global_idf_top1_plus_marker_plus_official_operand_anchors")

    for family in ("math12", "english10"):
        family_records = [record for record in records if record["family"] == family]
        if len(family_records) < 2:
            continue
        ranked = idf_rank(ocr_text, family_records)
        top, second = ranked[0], ranked[1]
        margin = float(top["score"] - second["score"])
        if top["score"] >= PAGE_MIN_SCORE and margin >= PAGE_MIN_MARGIN:
            record = next(
                item for item in family_records if item["record_id"] == top["record_id"]
            )
            return _route_from(record, top["score"], margin,
                               "official_page_idf_score_and_margin")
    return None


def _route_from(
        record: Mapping[str, Any],
        score: float,
        margin: float,
        closure: str,
) -> Route:
    return Route(
        family=str(record["family"]),
        record_id=str(record["record_id"]),
        answer=str(record["answer"]),
        answer_format=record.get("answer_format"),
        source_page=record.get("source_page"),
        score=round(float(score), 12),
        margin=round(float(margin), 12),
        closure=closure,
    )


def route_observable(observable: Mapping[str, Any]) -> Route | None:
    """Единственная публичная поверхность: принимает только наблюдаемые поля."""
    extra = set(observable) - OBSERVABLE_FIELDS
    if extra:
        raise ObservableError(f"роутеру передали ненаблюдаемые поля: {sorted(extra)}")
    return source_route(
        str(observable.get("ocr_text") or ""),
        answer_type=observable.get("answer_type"),
    )


def route(ocr_text: str, answer_type: str | None = None) -> Route | None:
    observable: dict[str, Any] = {"ocr_text": ocr_text}
    if answer_type is not None:
        observable["answer_type"] = answer_type
    return route_observable(observable)
