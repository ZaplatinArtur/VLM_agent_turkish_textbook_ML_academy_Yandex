from __future__ import annotations

from .base import AsymmetricTextEmbedder, SymmetricTextEmbedder

DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
E5_MODEL = "intfloat/multilingual-e5-base"
M3_MODEL = "BAAI/bge-m3"

E5_QUERY_PREFIX = "query: "
E5_PASSAGE_PREFIX = "passage: "

class SentenceTransformerBackend:
    def __init__(
            self,
            model_name: str,
            batch_size: int = 32,
            normalize: bool = False,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize = normalize
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
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

class SentenceTransformerEmbedder(SentenceTransformerBackend, SymmetricTextEmbedder):
    def __init__(self, model_name: str = DEFAULT_MODEL, **kwargs) -> None:
        super().__init__(model_name=model_name, **kwargs)

class M3Embedder(SentenceTransformerEmbedder):
    def __init__(self, model_name: str = M3_MODEL, **kwargs) -> None:
        super().__init__(model_name=model_name, **kwargs)


class E5Embedder(SentenceTransformerBackend, AsymmetricTextEmbedder):
    query_prefix = E5_QUERY_PREFIX
    passage_prefix = E5_PASSAGE_PREFIX

    def __init__(self, model_name: str = E5_MODEL, **kwargs) -> None:
        super().__init__(model_name=model_name, **kwargs)
