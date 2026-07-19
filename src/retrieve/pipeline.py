from schemas.retrieve import RetrievedChunk
from .rankers import Ranker


class RetrievalPipeline:
    def __init__(self, rankers: list[Ranker], chunks: list[RetrievedChunk]) -> None:
        self.rankers = rankers
        self.chunks = chunks

    def run(
        self,
        query: str,
        subject: str | None = None,
    ) -> list[RetrievedChunk]:

        chunks: list[RetrievedChunk] = self.chunks
        for ranker in self.rankers:
            chunks = ranker.rank(query, chunks, subject=subject)
        return chunks
