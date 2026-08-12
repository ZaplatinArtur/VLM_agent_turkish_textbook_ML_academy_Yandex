from __future__ import annotations

import math
import json
from pathlib import Path

import pytest

from experiments.maxim_9b_ykslop_teslov_retrieval_ablation_v1_20260812.teslov_retrieval import (
    GradeAwareTeslovCrossEncoder,
    OfflineRetrievalArms,
    RankedChunk,
    TheoryChunk,
    bm25_rank,
    grade_filter,
    reciprocal_rank_fusion,
    subject_filter,
)


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]


class StubScorer:
    def __init__(self, values: dict[str, float]) -> None:
        self.values = values
        self.seen: list[str] = []

    def score(self, query: str, documents: list[str]) -> list[float]:
        self.seen = list(documents)
        return [self.values[document] for document in documents]


def chunks() -> list[TheoryChunk]:
    return [
        TheoryChunk("g11-area", "üçgen alan taban yükseklik", 11, "matematik"),
        TheoryChunk("g12-area", "üçgen alan heron formülü", 12, "matematik"),
        TheoryChunk("g12-speed", "hız yol zaman formülü", "12", "fizik"),
        TheoryChunk("unknown", "üçgen alan", None, "matematik"),
    ]


def test_grade_filter_is_strict_and_normalizes_int_string() -> None:
    assert [chunk.chunk_id for chunk in grade_filter(chunks(), "12")] == [
        "g12-area",
        "g12-speed",
    ]
    assert [chunk.chunk_id for chunk in grade_filter(chunks(), 11)] == ["g11-area"]


def test_bm25_never_crosses_the_requested_grade() -> None:
    ranked = bm25_rank("ÜÇGEN ALAN", chunks(), grade=12)
    assert [item.chunk.chunk_id for item in ranked] == ["g12-area"]


def test_subject_filter_uses_exact_yks_mapping_and_never_crosses_books() -> None:
    assert [
        chunk.chunk_id for chunk in subject_filter(chunks(), "Matematik")
    ] == ["g11-area", "g12-area", "unknown"]
    assert subject_filter(chunks(), "Kimya") == []


def test_rrf_rewards_consensus_and_uses_deterministic_order() -> None:
    a, b, c = chunks()[:3]
    dense = [RankedChunk(a, 0.9), RankedChunk(b, 0.8)]
    lexical = [RankedChunk(b, 5.0), RankedChunk(c, 4.0)]
    fused = reciprocal_rank_fusion([dense, lexical])
    assert fused[0].chunk.chunk_id == "g12-area"
    assert math.isclose(fused[0].score, 1 / 62 + 1 / 61)


def test_cross_encoder_filters_grade_and_rescores_only_head() -> None:
    scorer = StubScorer({"üçgen alan heron formülü": 0.2, "hız yol zaman formülü": 0.9})
    candidates = [RankedChunk(chunk, 1.0 - index / 10) for index, chunk in enumerate(chunks())]
    reranked = GradeAwareTeslovCrossEncoder(scorer, top_n=2).rerank(
        "formül", candidates, grade="12"
    )
    assert [item.chunk.chunk_id for item in reranked] == ["g12-speed", "g12-area"]
    assert scorer.seen == ["üçgen alan heron formülü", "hız yol zaman formülü"]


def test_cross_encoder_rejects_bad_score_shape() -> None:
    class BadScorer:
        def score(self, query, documents):
            return [float("nan")]

    with pytest.raises(ValueError, match="one finite score"):
        GradeAwareTeslovCrossEncoder(BadScorer()).rerank(
            "q", [RankedChunk(chunks()[0], 1.0)]
        )


def test_offline_builder_passes_only_target_grade_to_dense_and_reranker() -> None:
    observed: list[str] = []

    def dense(query, eligible):
        observed.extend(chunk.chunk_id for chunk in eligible)
        return [RankedChunk(chunk, 1.0 / (index + 1)) for index, chunk in enumerate(eligible)]

    scorer = StubScorer({"üçgen alan heron formülü": 0.8, "hız yol zaman formülü": 0.1})
    arms = OfflineRetrievalArms(chunks(), dense, GradeAwareTeslovCrossEncoder(scorer))
    result = arms.teslov_rrf_cross_encoder(
        "üçgen alan", subject="Matematik", grade=12
    )
    assert observed == ["g12-area"]
    assert all(item.chunk.grade in (12, "12") for item in result)


def test_bm25_replays_all_185_frozen_dev_rankings_exactly() -> None:
    benchmark_path = (
        REPO_ROOT
        / "experiments/maxim_9b_ykslop_generic_content_pipeline_v5_20260811"
        / "frozen/benchmark_public_dev.jsonl"
    )
    plan_path = benchmark_path.parent / "dev_routing_plan_public.jsonl"
    corpus_path = (
        REPO_ROOT
        / "experiments/maxim_9b_ykslop_no_overlap_theory_v6_20260811"
        / "frozen/local_textbook_strict_theory_corpus.jsonl"
    )

    read_rows = lambda path: [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    public_rows = read_rows(benchmark_path)
    frozen_plan = {
        row["benchmark_id"]: row["retrieval"]
        for row in read_rows(plan_path)
        if row["arm"] == "local_textbook_theory_bm25"
    }
    corpus = [
        TheoryChunk(
            row["chunk_id"],
            row["text"],
            row.get("grade"),
            row.get("subject"),
        )
        for row in read_rows(corpus_path)
    ]

    mismatches = []
    for row in public_rows:
        query = row["question"] + "\n" + "\n".join(row["choices"].values())
        ranked = bm25_rank(query, corpus, subject=row["subject"])
        selected = []
        total_chars = 0
        for item in ranked:
            if total_chars + len(item.chunk.text) > 5200 and selected:
                continue
            selected.append(
                (item.chunk.chunk_id, round(item.score, 8))
            )
            total_chars += len(item.chunk.text)
            if len(selected) == 4:
                break
        expected = [
            (item["chunk_id"], item["score"])
            for item in frozen_plan[row["benchmark_id"]]
        ]
        if selected != expected:
            mismatches.append((row["benchmark_id"], selected, expected))

    assert mismatches == []
