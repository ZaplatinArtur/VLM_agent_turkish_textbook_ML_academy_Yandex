"""Персистентность индекса FAISS: снимок на диск + манифест с фиксацией состава.

Без снимка эта сборка повторяется при каждом старте процесса. Манифест фиксирует, по какому корпусу
и каким эмбеддером построен снимок, чтобы при запуске решить: загрузить готовое
с диска (состав совпал) или пересобрать (книги добавились/сменился эмбеддер).
"""

import hashlib
import json
import time
from collections.abc import Iterable
from pathlib import Path

from .vector_store import FaissVectorStore

MANIFEST_FILE = "manifest.json"


def corpus_fingerprint(chunk_ids: list[str], embedder_name: str) -> str:
    """Отпечаток корпуса: хэш от эмбеддера и отсортированного набора chunk_id."""
    digest = hashlib.sha256()
    digest.update(embedder_name.encode("utf-8"))
    for chunk_id in sorted(chunk_ids):
        digest.update(b"\x00")
        digest.update(chunk_id.encode("utf-8"))
    return digest.hexdigest()


def save_index(
        directory: Path | str,
        store: FaissVectorStore,
        embedder_name: str,
        book_ids: Iterable[str] | None = None,
) -> dict:
    """Сохраняет снимок стора и манифест. Возвращает записанный манифест."""
    directory = Path(directory)
    store.save(directory)
    # chunk_id = "<book-slug>:<page>", книга — часть до первого двоеточия.
    # Legacy page chunks use "<book-slug>:<page>". Educational graph nodes
    # have opaque "edu_*" ids, so their textbook ids must come from metadata.
    books = (
        {str(book_id) for book_id in book_ids}
        if book_ids is not None
        else {chunk_id.split(":", 1)[0] for chunk_id in store.chunk_ids}
    )
    manifest = {
        "n_vectors": store.size,
        "n_chunks": len(store.chunk_ids),
        "n_books": len(books),
        "index_type": store.index_type,
        "embedder": embedder_name,
        "corpus_hash": corpus_fingerprint(store.chunk_ids, embedder_name),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (directory / MANIFEST_FILE).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def load_manifest(directory: Path | str) -> dict | None:
    path = Path(directory) / MANIFEST_FILE
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_index(
        directory: Path | str,
        chunk_ids: list[str],
        embedder_name: str,
) -> FaissVectorStore | None:
    """Загружает снимок, только если он построен по ТОМУ ЖЕ корпусу и эмбеддеру.

    Иначе (манифеста нет, состав книг изменился, сменился эмбеддер) возвращает
    None — вызывающий код пересоберёт индекс и перезапишет снимок.
    """
    manifest = load_manifest(directory)
    if manifest is None:
        return None
    expected = corpus_fingerprint(chunk_ids, embedder_name)
    if manifest.get("corpus_hash") != expected:
        return None
    return FaissVectorStore.load(directory)
