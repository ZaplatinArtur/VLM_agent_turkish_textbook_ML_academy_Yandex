from ..schemas.retrieve import RetrievedChunk

from .rankers import Ranker


class RetrievalPipeline:
    def __init__(self, rankers: list[Ranker]) -> None:
        self.rankers = rankers

    def persist(self) -> None:
        """Просит ранкеры зафиксировать их индексы на диск (у кого есть чем)."""
        for ranker in self.rankers:
            persist = getattr(ranker, "persist", None)
            if callable(persist):
                persist()

    def run(
            self,
            query: str,
            k: int,
            subject: str | None = None,
    ) -> list[RetrievedChunk]:
        chunks: list[RetrievedChunk] = []
        for ranker in self.rankers:
            chunks = ranker.rank(query, chunks, subject=subject)
        return chunks[:k]
