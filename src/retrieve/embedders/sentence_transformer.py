from __future__ import annotations

from .base import AsymmetricTextEmbedder, SymmetricTextEmbedder

MINILM_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
E5_SMALL_MODEL = "intfloat/multilingual-e5-small"
E5_BASE_MODEL = "intfloat/multilingual-e5-base"
M3_MODEL = "BAAI/bge-m3"
QWEN3_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"

E5_QUERY_PREFIX = "query: "
E5_PASSAGE_PREFIX = "passage: "

# Qwen3-Embedding обучен принимать инструкцию задачи вместе с запросом; документ
# идёт без неё. Формат из карточки модели, двоеточие без пробела — как там.
QWEN3_TASK = (
    "Given a query from a Turkish school task, retrieve the textbook page that answers it"
)
QWEN3_QUERY_PREFIX = f"Instruct: {QWEN3_TASK}\nQuery:"

class SentenceTransformerBackend:
    def __init__(
            self,
            model_name: str,
            batch_size: int = 32,
            normalize: bool = False,
            max_seq_length: int | None = None,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize = normalize
        self.max_seq_length = max_seq_length
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
            if self.max_seq_length is not None:
                self._model.max_seq_length = self.max_seq_length
        return self._model

    def encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
        )
        return vectors.tolist()

class PlainEmbedder(SentenceTransformerBackend, SymmetricTextEmbedder):
    """Текст кодируется как есть, без префиксов: MiniLM, bge-m3."""

class E5Embedder(SentenceTransformerBackend, AsymmetricTextEmbedder):
    """Семейство multilingual-e5: обучено с префиксами query:/passage"""

    query_prefix = E5_QUERY_PREFIX
    passage_prefix = E5_PASSAGE_PREFIX

class Qwen3Embedder(SentenceTransformerBackend, AsymmetricTextEmbedder):
    """Qwen3-Embedding: инструкция только к запросу, документ как есть.

    Длина входа 512, а не родные 32k: на 32 ГБ карты матрица внимания даёт OOM
    уже при батче 8. Тот же лимит у e5 обрезает 9.8% страниц корпуса.
    """

    query_prefix = QWEN3_QUERY_PREFIX

    def __init__(self, model_name: str, batch_size: int = 32,
                 normalize: bool = False, max_seq_length: int | None = 512) -> None:
        super().__init__(model_name, batch_size, normalize, max_seq_length)
