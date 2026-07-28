from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

from schemas.retrieve import RetrievedChunk


SEARCHABLE_KINDS = {"theory", "exercise", "worked_example"}
SOLUTION_KINDS = {"solution", "answer_key"}


def _clean(value: Any) -> str:
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in str(value or "").splitlines()]
    return "\n".join(line for line in lines if line).strip()


def _page(chunk: RetrievedChunk) -> int:
    value = chunk.metadata.get("source_page", chunk.metadata.get("page"))
    try:
        return int(value)
    except (TypeError, ValueError):
        return -1


def _unit_index(chunk: RetrievedChunk) -> int:
    try:
        return int(chunk.metadata.get("unit_index", 0))
    except (TypeError, ValueError):
        return 0


def _kind(chunk: RetrievedChunk) -> str:
    return str(chunk.metadata.get("unit_kind") or "other")


def _book(chunk: RetrievedChunk) -> str:
    return str(
        chunk.metadata.get("textbook")
        or chunk.metadata.get("book_id")
        or chunk.chunk_id.split(":", 1)[0]
    )


def _parent(chunk: RetrievedChunk) -> str:
    return str(chunk.metadata.get("parent_chunk_id") or chunk.chunk_id)


def _stable_card_id(source_ids: Iterable[str], retrieval_text: str) -> str:
    digest = hashlib.sha256()
    for source_id in source_ids:
        digest.update(source_id.encode("utf-8"))
        digest.update(b"\0")
    digest.update(retrieval_text.encode("utf-8"))
    return f"kb_{digest.hexdigest()[:24]}"


def _take_unique_text(chunks: Iterable[RetrievedChunk], limit: int) -> tuple[str, list[str]]:
    parts: list[str] = []
    source_ids: list[str] = []
    seen: set[str] = set()
    used = 0
    for chunk in chunks:
        text = _clean(chunk.text)
        key = re.sub(r"\W+", " ", text.casefold()).strip()
        if not text or key in seen or used >= limit:
            continue
        clipped = text[: max(0, limit - used)]
        if not clipped:
            continue
        parts.append(clipped)
        source_ids.append(chunk.chunk_id)
        seen.add(key)
        used += len(clipped)
    return "\n\n".join(parts), source_ids


class KnowledgeBaseBuilder:
    """Build searchable cards that preserve educational relations.

    Dense/lexical retrieval searches ``metadata.retrieval_text``.  The model
    receives the card body containing nearby theory, a similar exercise or
    worked example, and its linked solution when the source book has one.
    """

    def __init__(
        self,
        *,
        min_retrieval_chars: int = 70,
        theory_chars: int = 1_100,
        example_chars: int = 900,
        solution_chars: int = 1_300,
        max_card_chars: int = 3_600,
        context_page_radius: int = 2,
    ) -> None:
        self.min_retrieval_chars = min_retrieval_chars
        self.theory_chars = theory_chars
        self.example_chars = example_chars
        self.solution_chars = solution_chars
        self.max_card_chars = max_card_chars
        self.context_page_radius = context_page_radius

    @staticmethod
    def _ordered(chunks: Iterable[RetrievedChunk]) -> list[RetrievedChunk]:
        return sorted(chunks, key=lambda chunk: (_page(chunk), _unit_index(chunk), chunk.chunk_id))

    def _nearest_before(
        self,
        ordered: list[RetrievedChunk],
        position: int,
        kinds: set[str],
        limit: int,
    ) -> tuple[str, list[str]]:
        anchor_page = _page(ordered[position])
        candidates: list[RetrievedChunk] = []
        for candidate in reversed(ordered[:position]):
            if anchor_page >= 0 and _page(candidate) >= 0:
                if anchor_page - _page(candidate) > self.context_page_radius:
                    break
            if _kind(candidate) in kinds:
                candidates.append(candidate)
        candidates.reverse()
        return _take_unique_text(candidates, limit)

    @staticmethod
    def _same_parent_following_solutions(
        ordered: list[RetrievedChunk],
        position: int,
    ) -> list[RetrievedChunk]:
        source = ordered[position]
        output: list[RetrievedChunk] = []
        for candidate in ordered[position + 1 :]:
            if _parent(candidate) != _parent(source):
                break
            if _kind(candidate) in SOLUTION_KINDS:
                output.append(candidate)
            elif _kind(candidate) in {"exercise", "worked_example"}:
                break
        return output

    @staticmethod
    def _images(chunks: Iterable[RetrievedChunk]) -> list:
        images = []
        seen: set[str] = set()
        for chunk in chunks:
            for image in chunk.images:
                key = image.model_dump_json()
                if key not in seen:
                    seen.add(key)
                    images.append(image)
        return images[:3]

    def build(self, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        by_book: dict[str, list[RetrievedChunk]] = defaultdict(list)
        for chunk in chunks:
            if _clean(chunk.text):
                by_book[_book(chunk)].append(chunk)

        cards: list[RetrievedChunk] = []
        seen_retrieval_text: set[tuple[str, str]] = set()
        for book, source_chunks in sorted(by_book.items()):
            ordered = self._ordered(source_chunks)
            solutions_by_exercise: dict[str, list[RetrievedChunk]] = defaultdict(list)
            solution_link_method: dict[str, str] = {}
            exercises_by_number: dict[str, list[RetrievedChunk]] = defaultdict(list)
            last_exercise: RetrievedChunk | None = None
            for chunk in ordered:
                if _kind(chunk) == "exercise":
                    last_exercise = chunk
                    task_number = str(chunk.metadata.get("task_number") or "")
                    if task_number:
                        exercises_by_number[task_number].append(chunk)
                    continue
                exercise_id = str(chunk.metadata.get("exercise_id") or "")
                if exercise_id and _kind(chunk) in SOLUTION_KINDS:
                    solutions_by_exercise[exercise_id].append(chunk)
                    solution_link_method[chunk.chunk_id] = "explicit_exercise_id"
                    continue
                if _kind(chunk) not in SOLUTION_KINDS:
                    continue
                task_number = str(chunk.metadata.get("task_number") or "")
                numbered_candidates = exercises_by_number.get(task_number, [])
                numbered = next(
                    (
                        exercise
                        for exercise in reversed(numbered_candidates)
                        if _page(chunk) < 0
                        or _page(exercise) < 0
                        or 0 <= _page(chunk) - _page(exercise) <= 3
                    ),
                    None,
                )
                inferred = numbered
                method = "task_number"
                if inferred is None and last_exercise is not None:
                    page_distance = _page(chunk) - _page(last_exercise)
                    if _page(chunk) < 0 or _page(last_exercise) < 0 or 0 <= page_distance <= 1:
                        inferred = last_exercise
                        method = "nearest_exercise"
                if inferred is not None:
                    solutions_by_exercise[inferred.chunk_id].append(chunk)
                    solution_link_method[chunk.chunk_id] = method

            theory_parents_emitted: set[str] = set()
            for position, source in enumerate(ordered):
                source_kind = _kind(source)
                if source_kind not in SEARCHABLE_KINDS:
                    continue

                parent_id = _parent(source)
                main_chunks: list[RetrievedChunk]
                if source_kind == "theory":
                    if parent_id in theory_parents_emitted:
                        continue
                    theory_parents_emitted.add(parent_id)
                    main_chunks = [
                        candidate
                        for candidate in ordered
                        if _parent(candidate) == parent_id and _kind(candidate) == "theory"
                    ]
                else:
                    main_chunks = [source]

                main_text, main_ids = _take_unique_text(main_chunks, 1_800)
                section = _clean(source.metadata.get("section_title"))
                retrieval_text = "\n".join(part for part in (section, main_text) if part)
                if len(re.sub(r"\W+", "", retrieval_text)) < self.min_retrieval_chars:
                    continue
                dedupe_key = (book, re.sub(r"\W+", " ", retrieval_text.casefold()).strip())
                if dedupe_key in seen_retrieval_text:
                    continue
                seen_retrieval_text.add(dedupe_key)

                theory_text = ""
                theory_ids: list[str] = []
                example_text = ""
                example_ids: list[str] = []
                solution_text = ""
                solution_ids: list[str] = []

                if source_kind != "theory":
                    theory_text, theory_ids = self._nearest_before(
                        ordered, position, {"theory"}, self.theory_chars
                    )
                if source_kind == "exercise":
                    example_text, example_ids = self._nearest_before(
                        ordered, position, {"worked_example"}, self.example_chars
                    )
                    linked = solutions_by_exercise.get(source.chunk_id)
                    if not linked:
                        linked = self._same_parent_following_solutions(ordered, position)
                    solution_text, solution_ids = _take_unique_text(
                        linked, self.solution_chars
                    )
                elif source_kind == "worked_example":
                    linked = self._same_parent_following_solutions(ordered, position)
                    solution_text, solution_ids = _take_unique_text(
                        linked, self.solution_chars
                    )

                sections: list[tuple[str, str]] = []
                if theory_text:
                    sections.append(("THEORY", theory_text))
                if example_text:
                    sections.append(("WORKED EXAMPLE", example_text))
                sections.append(
                    (
                        "SIMILAR EXERCISE"
                        if source_kind == "exercise"
                        else source_kind.replace("_", " ").upper(),
                        main_text,
                    )
                )
                if solution_text:
                    sections.append(("SOLUTION", solution_text))
                card_text = "\n\n".join(
                    f"[{label}]\n{text}" for label, text in sections if text
                )[: self.max_card_chars]
                source_ids = list(
                    dict.fromkeys(main_ids + theory_ids + example_ids + solution_ids)
                )
                metadata = dict(source.metadata)
                metadata.update(
                    {
                        "knowledge_card": True,
                        "knowledge_kind": source_kind,
                        "unit_kind": "knowledge_card",
                        "retrieval_text": retrieval_text,
                        "source_chunk_ids": source_ids,
                        "theory_chunk_ids": theory_ids,
                        "example_chunk_ids": example_ids,
                        "exercise_chunk_id": (
                            source.chunk_id if source_kind == "exercise" else None
                        ),
                        "solution_chunk_ids": solution_ids,
                        "solution_link_methods": [
                            solution_link_method.get(chunk_id, "same_parent")
                            for chunk_id in solution_ids
                        ],
                        "has_theory": bool(theory_text),
                        "has_example": bool(example_text),
                        "has_solution": bool(solution_text),
                        "card_chars": len(card_text),
                    }
                )
                cards.append(
                    RetrievedChunk(
                        chunk_id=_stable_card_id(source_ids, retrieval_text),
                        text=card_text,
                        images=self._images(
                            [source]
                            + [chunk for chunk in ordered if chunk.chunk_id in source_ids]
                        ),
                        score=0.0,
                        metadata=metadata,
                    )
                )
        return cards
