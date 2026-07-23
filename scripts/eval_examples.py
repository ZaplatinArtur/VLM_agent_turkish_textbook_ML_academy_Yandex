"""Прогон примеров на реальных данных.

Запуск:  PYTHONIOENCODING=utf-8 python scripts/eval_examples.py
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from src.paths import CHUNKS_JSONL_DIR
from src.retrieve.confidence import Relevance, assess_relevance
from src.retrieve.embedders import SentenceTransformerEmbedder
from src.retrieve.index import Index
from src.retrieve.rankers import DenseRanker
from src.schemas.retrieve import RetrievedChunk

EXAMPLES_FILE = Path(__file__).resolve().parent / "retrieval_examples.json"


def load_book(book: str) -> list[RetrievedChunk]:
    path = CHUNKS_JSONL_DIR / f"{book}.jsonl"
    with path.open(encoding="utf-8") as f:
        return [RetrievedChunk.model_validate_json(line) for line in f if line.strip()]


def main() -> int:
    spec = json.loads(EXAMPLES_FILE.read_text(encoding="utf-8"))
    book, examples = spec["book"], spec["examples"]
    ranker = DenseRanker(embedder=SentenceTransformerEmbedder(), index=Index(load_book(book)))

    print(f"Книга: {book}\n")
    failures = 0
    for ex in examples:
        results = ranker.rank(ex["query"])[:3]
        verdict = assess_relevance(results)
        top = results[0] if results else None

        print(f"Q: {ex['query']}")
        if top is not None:
            page = top.metadata.get("page")
            print(f"   top: score={top.score:.3f} page={page}  {top.text[:70]!r}")
        print(f"   verdict: {verdict.relevance.value} ({verdict.reason})")

        # Проверки: on_topic → уверенно + ожидаемая подстрока в топ-3;
        #           off_topic → детектор НЕ должен пометить как confident.
        ok = True
        if ex.get("on_topic"):
            if verdict.relevance is not Relevance.CONFIDENT:
                ok = False
            sub = ex.get("expect_substring")
            if sub and not any(sub.lower() in r.text.lower() for r in results):
                ok = False
        else:
            if verdict.relevance is Relevance.CONFIDENT:
                ok = False
        print(f"   -> {'OK' if ok else 'FAIL'}\n")
        failures += 0 if ok else 1

    print(f"Итого: {len(examples) - failures}/{len(examples)} примеров прошли.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
