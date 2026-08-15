import pytest

from retrieve.calibrate_visual import load_queries, top_scores


class StubClient:
    """Отдаёт заранее заданный счёт на запрос; сети и карты не нужно."""

    def __init__(self, scores: dict[str, float], broken: set[str] | None = None):
        self.scores = scores
        self.broken = broken or set()
        self.seen: list[str] = []

    def search(self, query, *, top_k=5):
        self.seen.append(query)
        if query in self.broken:
            raise RuntimeError("index unavailable")
        score = self.scores.get(query)
        hits = [] if score is None else [{"page_id": "p1", "score": score}]
        return {"hits": hits}


def test_load_queries_skips_blanks_and_comments(tmp_path):
    path = tmp_path / "q.txt"
    path.write_text("üçgen alanı\n\n# комментарий\nkesirler\n", encoding="utf-8")
    assert load_queries(path) == ["üçgen alanı", "kesirler"]


def test_load_queries_honours_limit(tmp_path):
    path = tmp_path / "q.txt"
    path.write_text("a\nb\nc\n", encoding="utf-8")
    assert load_queries(path, 2) == ["a", "b"]


def test_empty_file_is_refused(tmp_path):
    path = tmp_path / "q.txt"
    path.write_text("\n# только комментарий\n", encoding="utf-8")
    with pytest.raises(ValueError, match="нет запросов"):
        load_queries(path)


def test_top_scores_collects_top_one():
    client = StubClient({"a": 18.4, "b": 4.2})
    assert top_scores(client, ["a", "b"]) == [18.4, 4.2]


def test_empty_output_is_skipped_not_counted_as_zero():
    # Ноль исказил бы распределение: пустая выдача — это отсутствие
    # наблюдения, а не низкий счёт.
    client = StubClient({"a": 18.4, "b": None})
    assert top_scores(client, ["a", "b"]) == [18.4]


def test_failing_query_does_not_abort_calibration(capsys):
    client = StubClient({"a": 18.4, "c": 12.0}, broken={"b"})
    assert top_scores(client, ["a", "b", "c"]) == [18.4, 12.0]
    assert "пропущен" in capsys.readouterr().err


def test_calibration_separates_own_from_foreign_queries():
    from retrieve.compare import calibrate_gate

    # Чужие запросы по MaxSim набирают заметно меньше своих — порог обязан
    # лечь между ними, а не остаться дефолтным 0.57 из другой шкалы.
    report = calibrate_gate([18.4, 16.9, 15.2], [7.1, 6.4, 5.8])
    assert report["separable"] is True
    assert 7.1 < report["threshold"] < 15.2
    assert report["out_domain_leaked"] == 0.0
