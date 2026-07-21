from ..schemas.retrieve import RetrievedChunk

from .rankers import Ranker


class RetrievalPipeline:
    def __init__(self, rankers: list[Ranker]) -> None:
        self.rankers = rankers

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
