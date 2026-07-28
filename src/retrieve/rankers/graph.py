from __future__ import annotations

from schemas.retrieve import RetrievedChunk

from ..graph import KnowledgeGraph


class GraphExpansionRanker:
    """Expand ranked task nodes through explicit educational relations."""

    def __init__(
        self,
        graph: KnowledgeGraph,
        *,
        max_exercise_candidates: int = 20,
        include_solutions: bool = False,
        anchor_first: bool = True,
    ) -> None:
        self.graph = graph
        self.max_exercise_candidates = max_exercise_candidates
        self.include_solutions = include_solutions
        self.anchor_first = anchor_first

    def rank(
        self,
        query: str,
        chunks: list[RetrievedChunk] | None = None,
        subject: str | None = None,
        grade: int | str | None = None,
    ) -> list[RetrievedChunk]:
        del query, subject, grade
        candidates = list(chunks or [])
        # The upstream reranker has already compared exercises, theory, and
        # worked examples. Preserve that relevance order: moving every
        # exercise ahead of a better-scoring theory block makes the graph
        # expansion override retrieval instead of enriching it.
        supported = {"exercise", "theory", "worked_example"}
        ordered = [
            candidate
            for candidate in candidates
            if str(candidate.metadata.get("unit_kind")) in supported
        ]
        bundles: list[RetrievedChunk] = []
        seen_anchors: set[str] = set()
        exercise_candidates = 0
        for similarity_rank, candidate in enumerate(ordered, 1):
            if candidate.chunk_id in seen_anchors:
                continue
            if str(candidate.metadata.get("unit_kind")) == "exercise":
                exercise_candidates += 1
                if exercise_candidates > self.max_exercise_candidates:
                    continue
            seen_anchors.add(candidate.chunk_id)
            bundle = self.graph.bundle(
                candidate.chunk_id,
                score=candidate.score,
                include_solutions=self.include_solutions,
                anchor_first=self.anchor_first,
            )
            metadata = dict(bundle.metadata)
            metadata.update(
                {
                    "similarity_anchor_id": candidate.chunk_id,
                    "similarity_rank": similarity_rank,
                    "similarity_score": float(candidate.score),
                }
            )
            bundles.append(bundle.model_copy(update={"metadata": metadata}))
        return bundles
