from schemas.retrieve import RetrievedChunk

from .metadata import subjects_match


class Index:
    """Корпус в памяти: чанки из JSONL-хранилища парсера, доступ по id и фильтрам."""

    def __init__(self, chunks: list[RetrievedChunk] | None = None) -> None:
        self._chunks_by_id: dict[str, RetrievedChunk] = {}
        if chunks:
            self.add(chunks)

    def add(self, chunks: list[RetrievedChunk]) -> None:
        self._chunks_by_id.update({chunk.chunk_id: chunk for chunk in chunks})

    def get(
            self,
            subject: str | None = None,
            textbook: str | None = None,
            grade: int | str | None = None,
    ) -> list[RetrievedChunk]:
        """Возвращает чанки, опционально суженные по предмету и/или книге.

        В реальных данных metadata содержит `textbook` (и не всегда `subject`),
        поэтому сужение «в пределах книги» — основной рабочий фильтр.
        """
        chunks = list(self._chunks_by_id.values())
        if subject is not None:
            chunks = [
                chunk
                for chunk in chunks
                if subjects_match(chunk.metadata.get("subject"), subject)
            ]
        if textbook is not None:
            chunks = [c for c in chunks if c.metadata.get("textbook") == textbook]
        if grade is not None:
            chunks = [
                chunk
                for chunk in chunks
                if str(chunk.metadata.get("grade")) == str(grade)
            ]
        return chunks

    def get_by_id(self, chunk_id: str) -> RetrievedChunk | None:
        """Резолв чанка по id — для цитат и восстановления контента после реранка."""
        return self._chunks_by_id.get(chunk_id)
