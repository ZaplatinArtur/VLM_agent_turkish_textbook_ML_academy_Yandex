import pytest

from retrieve.confidence import Relevance
from retrieve.gate import SemanticGate, gate_from_env
from schemas.retrieve import RetrievedChunk


class ScriptedBackend:
    """Отдаёт заготовленный ответ и запоминает промпт."""

    def __init__(self, answer: str | Exception) -> None:
        self.answer = answer
        self.prompts: list[str] = []

    def ask(self, system_prompt: str, user_prompt: str) -> str:
        self.prompts.append(user_prompt)
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer


def chunk(chunk_id: str, text: str, score: float = 0.9) -> RetrievedChunk:
    return RetrievedChunk(chunk_id=chunk_id, text=text, score=score, metadata={})


CHUNKS = [
    chunk("a", "Üçgenin alanı taban çarpı yükseklik bölü iki.", 0.95),
    chunk("b", "İçindekiler: 1. Ünite ... 2. Ünite ...", 0.90),
    chunk("c", "Örnek: tabanı 6 cm olan üçgenin alanını bulunuz.", 0.85),
]


def test_gate_keeps_only_the_listed_fragments():
    backend = ScriptedBackend('{"keep": [1, 3], "reason": "formula and worked example"}')
    kept, verdict = SemanticGate(backend).judge("üçgenin alanı formülü", CHUNKS)
    assert [c.chunk_id for c in kept] == ["a", "c"]
    assert verdict.relevance is Relevance.CONFIDENT
    assert verdict.top_score == 0.95
    assert "formula and worked example" in verdict.reason


def test_gate_hides_everything_when_nothing_fits():
    backend = ScriptedBackend('{"keep": [], "reason": "only contents pages"}')
    kept, verdict = SemanticGate(backend).judge("vergi beyannamesi", CHUNKS)
    assert kept == []
    assert verdict.relevance is Relevance.WEAK
    assert not verdict.is_useful


@pytest.mark.parametrize("answer", ["не JSON вовсе", "{}", '{"keep": "все"}', '{"keep": [1'])
def test_unreadable_answer_fails_closed(answer):
    kept, verdict = SemanticGate(ScriptedBackend(answer)).judge("q", CHUNKS)
    assert kept == []
    assert verdict.relevance is Relevance.ERROR
    assert not verdict.is_useful


def test_backend_failure_fails_closed():
    backend = ScriptedBackend(RuntimeError("connection refused"))
    kept, verdict = SemanticGate(backend).judge("q", CHUNKS)
    assert kept == []
    assert verdict.relevance is Relevance.ERROR
    assert "connection refused" in verdict.reason


def test_out_of_range_numbers_are_dropped():
    # Модель может выдумать номер; ссылаться на несуществующий чанк нельзя.
    backend = ScriptedBackend('{"keep": [2, 9, 0, -1, 2], "reason": "ok"}')
    kept, _ = SemanticGate(backend).judge("q", CHUNKS)
    assert [c.chunk_id for c in kept] == ["b"]


def test_json_wrapped_in_prose_is_still_parsed():
    backend = ScriptedBackend('Here you go:\n```json\n{"keep": [1], "reason": "ok"}\n```')
    kept, verdict = SemanticGate(backend).judge("q", CHUNKS)
    assert [c.chunk_id for c in kept] == ["a"]
    assert verdict.is_useful


def test_empty_results_skip_the_llm_call():
    backend = ScriptedBackend('{"keep": [1]}')
    kept, verdict = SemanticGate(backend).judge("q", [])
    assert kept == []
    assert verdict.relevance is Relevance.EMPTY
    assert backend.prompts == []


def test_prompt_carries_the_query_and_numbered_fragments():
    backend = ScriptedBackend('{"keep": [1], "reason": "ok"}')
    SemanticGate(backend).judge("üçgenin alanı formülü", CHUNKS)
    prompt = backend.prompts[0]
    assert "üçgenin alanı formülü" in prompt
    assert "[1] Üçgenin alanı" in prompt
    assert "[3] Örnek" in prompt


def test_gate_is_off_until_the_url_is_set(monkeypatch):
    monkeypatch.delenv("RETRIEVE_GATE_URL", raising=False)
    assert gate_from_env() is None
    monkeypatch.setenv("RETRIEVE_GATE_URL", "   ")
    assert gate_from_env() is None
