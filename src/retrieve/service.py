from schemas.retrieve import RetrievedChunk
from .parsing.service import get_retrieved_chuncks
from .pipeline import RetrievalPipeline


def get_pipeline(k: int) -> RetrievalPipeline:
    return RetrievalPipeline(
        rankers=[
            # TODO: Implement rankers
        ],
        chunks=get_retrieved_chuncks()
    )


def textbook_retrieve(
    query: str,
    k: int = 5,
    subject: str | None = None,
) -> list[RetrievedChunk]:

    return get_pipeline(k).run(query, subject=subject)
