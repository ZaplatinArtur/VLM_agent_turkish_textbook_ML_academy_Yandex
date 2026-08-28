import json
import math

import pytest

from retrieve.evaluation.compare import (
    calibrate_gate,
    iter_rankers,
    load_queries,
    score_stats,
    silver_metrics,
    write_qrels,
)
from retrieve.confidence import PROFILE_MIN_SCORE
from retrieve.pipelines import DEFAULT_PROFILE, PROFILES


class Node:
    """Минимальный ранкер-контейнер: у fusion ранкеры лежат в .rankers."""

    def __init__(self, name: str, rankers: list | None = None) -> None:
        self.name = name
        self.rankers = rankers or []


def test_silver_metrics_perfect_match():
    metrics = silver_metrics(["a", "b", "c"], ["a", "b", "c"], k=3)
    assert metrics["agree_at_k"] == 1.0
    assert metrics["recall_ref_at_k"] == 1.0
    assert metrics["ndcg_ref_at_k"] == pytest.approx(1.0)
    assert metrics["mrr_ref_at_k"] == 1.0
    assert metrics["top1_match"] == 1.0


def test_silver_metrics_counts_only_reference_head_for_agree():
    # Эталон длиннее k: "c" релевантен, но в голову эталона (k=2) не входит.
    metrics = silver_metrics(["x", "c"], ["a", "b", "c"], k=2)
    assert metrics["agree_at_k"] == 0.0        # {x, c} ∩ {a, b} = ∅
    assert metrics["recall_ref_at_k"] == 0.5   # {c} из min(k, |R|) = 2
    assert metrics["mrr_ref_at_k"] == 0.5      # первый релевантный на позиции 2
    assert metrics["top1_match"] == 0.0


def test_silver_metrics_ndcg_rewards_reference_order():
    # Веса линейные: a=3, b=2, c=1. Выдача b,a вместо a,b — nDCG ниже единицы.
    metrics = silver_metrics(["b", "a"], ["a", "b", "c"], k=2)
    ideal = 3 + 2 / math.log2(3)
    actual = 2 + 3 / math.log2(3)
    assert metrics["ndcg_ref_at_k"] == pytest.approx(actual / ideal)
    assert metrics["recall_ref_at_k"] == 1.0


def test_silver_metrics_without_reference_are_zero():
    assert silver_metrics(["a"], [], k=1)["recall_ref_at_k"] == 0.0
    assert silver_metrics([], ["a"], k=1)["recall_ref_at_k"] == 0.0


def test_score_stats_reads_top_and_margin():
    stats = score_stats([0.9, 0.7, 0.5], k=3)
    assert stats["score_top1"] == 0.9
    assert stats["score_at_k"] == 0.5
    assert stats["score_margin"] == pytest.approx(0.2)


def test_score_stats_clamps_k_to_available_scores():
    assert score_stats([0.9, 0.7], k=5)["score_at_k"] == 0.7
    assert score_stats([], k=5) == {"score_top1": 0.0, "score_at_k": 0.0, "score_margin": 0.0}


def test_iter_rankers_descends_into_fusion():
    inner = [Node("e5"), Node("m3"), Node("bm25")]
    pipeline = Node("pipeline", [Node("fusion", inner), Node("reranker")])
    names = [node.name for node in iter_rankers(pipeline)]
    assert names == ["fusion", "e5", "m3", "bm25", "reranker"]


def test_load_queries_skips_comments_and_blanks(tmp_path):
    path = tmp_path / "queries.txt"
    path.write_text("# комментарий\n\nüçgen alanı\n  kesirler  \n", encoding="utf-8")
    assert load_queries(path, limit=None) == ["üçgen alanı", "kesirler"]


def test_load_queries_respects_limit(tmp_path):
    path = tmp_path / "queries.txt"
    path.write_text("bir\niki\nüç\n", encoding="utf-8")
    assert load_queries(path, limit=2) == ["bir", "iki"]


def test_load_queries_missing_file_raises(tmp_path):
    with pytest.raises(SystemExit):
        load_queries(tmp_path / "yok.txt", limit=None)


def test_write_qrels_drops_candidates_below_threshold(tmp_path):
    path = tmp_path / "qrels.jsonl"
    written, skipped = write_qrels(
        path,
        ["soru bir", "soru iki"],
        {"soru bir": ["a", "b"], "soru iki": ["c"]},
        {"soru bir": {"a": 0.9, "b": 0.2}, "soru iki": {"c": 0.1}},
        min_score=0.5,
        origin="pool",
    )
    assert (written, skipped) == (1, 1)
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 1
    assert records[0]["query"] == "soru bir"
    assert records[0]["relevant_chunk_ids"] == ["a"]  # b отсеян порогом
    assert records[0]["silver"]["reference"] == "pool"


def test_calibrate_gate_finds_separating_threshold():
    report = calibrate_gate(in_domain=[0.80, 0.85, 0.90], out_domain=[0.40, 0.50, 0.60])
    assert 0.60 < report["threshold"] < 0.80
    assert report["accuracy"] == 1.0
    assert report["in_domain_kept"] == 1.0
    assert report["out_domain_leaked"] == 0.0
    assert report["separable"] is True


def test_calibrate_gate_reports_overlap_instead_of_pretending():
    # Классы пересекаются — идеального порога нет, разделимость False.
    report = calibrate_gate(in_domain=[0.50, 0.90], out_domain=[0.40, 0.80])
    assert report["separable"] is False
    assert report["accuracy"] < 1.0


def test_calibrate_gate_threshold_scales_with_the_encoder():
    # Тот же порядок «своих» и «чужих», но сдвинутая шкала (как у e5 vs MiniLM):
    # порог обязан переехать вместе со шкалой, а не остаться прежним.
    low = calibrate_gate([0.60, 0.65], [0.30, 0.35])["threshold"]
    high = calibrate_gate([0.90, 0.95], [0.84, 0.86])["threshold"]
    assert high > low


def test_calibrate_gate_without_data_returns_empty():
    assert calibrate_gate([], [0.1]) == {}
    assert calibrate_gate([0.1], []) == {}


def test_e5_small_profile_is_registered():
    assert "e5-small" in PROFILES


def test_default_profile_has_a_measured_gate_threshold():
    # Профиль по умолчанию не должен молча ехать на DEFAULT_MIN_SCORE:
    # у каждого энкодера своя шкала косинуса, чужой порог гейт ломает.
    assert DEFAULT_PROFILE in PROFILE_MIN_SCORE
