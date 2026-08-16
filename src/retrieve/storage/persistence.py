"""Персистентность индекса FAISS: снимок на диск + манифест состава.

Без снимка индекс пересобирается при каждом старте процесса. Манифест помнит, по
какому корпусу и каким эмбеддером снимок построен, — по нему и решается, грузить
готовое или пересобирать (добавились книги, сменился эмбеддер).
"""

import hashlib
import json
import os
import secrets
import stat
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .vector_store import (
    CHUNK_IDS_FILE,
    INDEX_FILE,
    FaissVectorStore,
    IndexKind,
    resolve_index_kind,
)

MANIFEST_FILE = "manifest.json"
MANIFEST_SCHEMA_V2 = "textbook-faiss-index-v2"
BUILD_LOCK_FILE = ".strict-build.lock"
_ARTIFACT_FILES = (INDEX_FILE, CHUNK_IDS_FILE)


class IndexValidationError(RuntimeError):
    """An existing strict snapshot failed validation and must not be overwritten."""


class StrictBuildLock:
    """Exclusive, fail-closed ownership marker for one strict index build."""

    def __init__(self, directory: Path, token: str) -> None:
        self.directory = directory
        self.token = token

    @property
    def path(self) -> Path:
        return self.directory / BUILD_LOCK_FILE

    def assert_owned(self) -> None:
        details = _strict_lstat(self.path, "build lock")
        if details is None or not stat.S_ISREG(details.st_mode):
            raise IndexValidationError("strict build lock is missing or not regular")
        try:
            actual = self.path.read_text(encoding="ascii")
        except (OSError, UnicodeError) as exc:
            raise IndexValidationError("strict build lock could not be read") from exc
        if actual != self.token:
            raise IndexValidationError("strict build lock ownership changed")

    def release(self) -> None:
        self.assert_owned()
        try:
            self.path.unlink()
        except OSError as exc:
            raise IndexValidationError("strict build lock could not be removed") from exc


def _strict_failure(required: bool, message: str) -> None:
    if required:
        raise IndexValidationError(message)


def _strict_lstat(path: Path, label: str):
    try:
        details = path.lstat()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise IndexValidationError(f"strict {label} could not be inspected") from exc
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    file_attributes = getattr(details, "st_file_attributes", 0)
    if stat.S_ISLNK(details.st_mode) or (
        reparse_flag and file_attributes & reparse_flag
    ):
        raise IndexValidationError(
            f"strict {label} must not be a symlink or reparse point"
        )
    return details


def _validate_strict_paths(directory: Path) -> bool:
    details = _strict_lstat(directory, "index directory")
    if details is None:
        return False
    if not stat.S_ISDIR(details.st_mode):
        raise IndexValidationError("strict index directory path is not a directory")
    for filename in (MANIFEST_FILE, *_ARTIFACT_FILES):
        artifact_details = _strict_lstat(
            directory / filename,
            f"index artifact {filename!r}",
        )
        if artifact_details is not None and not stat.S_ISREG(
            artifact_details.st_mode
        ):
            raise IndexValidationError(
                f"strict index artifact {filename!r} is not a regular file"
            )
    return True


def _strict_directory_entries(directory: Path) -> set[str]:
    try:
        return {entry.name for entry in directory.iterdir()}
    except OSError as exc:
        raise IndexValidationError(
            "strict index directory could not be enumerated"
        ) from exc


def _validate_strict_entries(
    directory: Path,
    active_build_lock: StrictBuildLock | None = None,
) -> set[str]:
    entries = _strict_directory_entries(directory)
    allowed = {MANIFEST_FILE, *_ARTIFACT_FILES}
    if BUILD_LOCK_FILE in entries:
        if active_build_lock is None:
            raise IndexValidationError("strict index build lock already exists")
        if active_build_lock.directory.resolve(strict=False) != directory.resolve(
            strict=False
        ):
            raise IndexValidationError("strict build lock belongs to another directory")
        active_build_lock.assert_owned()
        allowed.add(BUILD_LOCK_FILE)
    unknown = sorted(entries - allowed)
    if unknown:
        raise IndexValidationError(
            "strict index directory contains unrecognized entries: "
            + ", ".join(unknown)
        )
    return entries


def acquire_strict_build_lock(directory: Path | str) -> StrictBuildLock:
    directory = Path(directory)
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise IndexValidationError("strict index directory could not be created") from exc
    _validate_strict_paths(directory)
    if _strict_directory_entries(directory):
        raise IndexValidationError(
            "strict index directory is no longer clean before build lock acquisition"
        )
    token = f"pid={os.getpid()};token={secrets.token_hex(16)}"
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    flags |= getattr(os, "O_BINARY", 0)
    try:
        descriptor = os.open(str(directory / BUILD_LOCK_FILE), flags, 0o600)
    except FileExistsError as exc:
        raise IndexValidationError("strict index build lock already exists") from exc
    except OSError as exc:
        raise IndexValidationError("strict index build lock could not be created") from exc
    try:
        os.write(descriptor, token.encode("ascii"))
        os.fsync(descriptor)
    except OSError as exc:
        raise IndexValidationError("strict index build lock could not be written") from exc
    finally:
        os.close(descriptor)
    lock = StrictBuildLock(directory, token)
    lock.assert_owned()
    entries = _strict_directory_entries(directory)
    if entries != {BUILD_LOCK_FILE}:
        raise IndexValidationError(
            "strict index directory changed during build lock acquisition"
        )
    return lock


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _canonical_provenance(
    provenance: Mapping[str, object] | None,
) -> dict[str, object] | None:
    if provenance is None:
        return None
    if not isinstance(provenance, Mapping) or not provenance:
        raise ValueError("embedder provenance must be a non-empty mapping")
    # A JSON round trip rejects opaque/mutable values and produces plain types.
    canonical = json.loads(_canonical_json(dict(provenance)).decode("utf-8"))
    if not isinstance(canonical, dict):
        raise ValueError("embedder provenance must serialize to an object")
    return canonical


def provenance_fingerprint(provenance: Mapping[str, object]) -> str:
    canonical = _canonical_provenance(provenance)
    assert canonical is not None
    return hashlib.sha256(_canonical_json(canonical)).hexdigest()


def _resolved_provenance_index_kind(
    provenance: Mapping[str, object],
    size_hint: int,
) -> IndexKind:
    raw = provenance.get("faiss_index_kind")
    try:
        requested = IndexKind(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "strict provenance requires faiss_index_kind auto/flat/hnsw"
        ) from exc
    return resolve_index_kind(requested, size_hint)


def _retrieval_projection_row(chunk: Any) -> dict[str, object]:
    chunk_id = str(getattr(chunk, "chunk_id", ""))
    if not chunk_id:
        raise ValueError("content-bound corpus has a missing chunk_id")
    raw_metadata = getattr(chunk, "metadata", {}) or {}
    if not isinstance(raw_metadata, Mapping):
        raise ValueError(f"chunk {chunk_id!r} metadata must be a mapping")
    retrieval_text = str(
        raw_metadata.get("retrieval_text") or getattr(chunk, "text", "")
    )
    return {
        "chunk_id": chunk_id,
        "grade": (
            None
            if raw_metadata.get("grade") is None
            else str(raw_metadata.get("grade"))
        ),
        "retrieval_text_sha256": hashlib.sha256(
            retrieval_text.encode("utf-8")
        ).hexdigest(),
        "subject": (
            None
            if raw_metadata.get("subject") is None
            else str(raw_metadata.get("subject"))
        ),
        "textbook": (
            None
            if raw_metadata.get("textbook") is None
            else str(raw_metadata.get("textbook"))
        ),
    }


def retrieval_chunk_projection_sha256(chunk: Any) -> str:
    return hashlib.sha256(_canonical_json(_retrieval_projection_row(chunk))).hexdigest()


def retrieval_corpus_projection_sha256(chunks: Sequence[Any]) -> str:
    """Bind a strict dense index to retrieval text and routing metadata."""
    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    for chunk in chunks:
        row = _retrieval_projection_row(chunk)
        chunk_id = str(row["chunk_id"])
        if chunk_id in seen:
            raise ValueError(
                f"content-bound corpus has a duplicate chunk_id: {chunk_id!r}"
            )
        seen.add(chunk_id)
        rows.append(row)
    rows.sort(key=lambda row: str(row["chunk_id"]))
    return hashlib.sha256(_canonical_json(rows)).hexdigest()


def corpus_fingerprint(
    chunk_ids: list[str],
    embedder_name: str,
    *,
    embedder_provenance: Mapping[str, object] | None = None,
    corpus_projection_sha256: str | None = None,
) -> str:
    """Отпечаток корпуса: хэш от эмбеддера и отсортированного набора chunk_id."""
    digest = hashlib.sha256()
    digest.update(embedder_name.encode("utf-8"))
    for chunk_id in sorted(chunk_ids):
        digest.update(b"\x00")
        digest.update(chunk_id.encode("utf-8"))
    if embedder_provenance is not None:
        digest.update(b"\x00embedder-provenance\x00")
        digest.update(
            _canonical_json(_canonical_provenance(embedder_provenance))
        )
    if corpus_projection_sha256 is not None:
        if len(corpus_projection_sha256) != 64:
            raise ValueError("corpus projection SHA-256 must be a 64-character hex digest")
        try:
            bytes.fromhex(corpus_projection_sha256)
        except ValueError as exc:
            raise ValueError("corpus projection SHA-256 is not hexadecimal") from exc
        digest.update(b"\x00corpus-projection-sha256\x00")
        digest.update(corpus_projection_sha256.encode("ascii"))
    return digest.hexdigest()


def _file_integrity(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def _artifacts_match(directory: Path, artifacts: object) -> bool:
    if not isinstance(artifacts, Mapping) or set(artifacts) != set(_ARTIFACT_FILES):
        return False
    for filename in _ARTIFACT_FILES:
        expected = artifacts.get(filename)
        path = directory / filename
        if not isinstance(expected, Mapping) or not path.is_file():
            return False
        try:
            actual = _file_integrity(path)
        except OSError:
            return False
        if expected.get("bytes") != actual["bytes"]:
            return False
        if expected.get("sha256") != actual["sha256"]:
            return False
    return True


def save_index(
        directory: Path | str,
        store: FaissVectorStore,
        embedder_name: str,
        book_ids: Iterable[str] | None = None,
        *,
        embedder_provenance: Mapping[str, object] | None = None,
        corpus_projection_sha256: str | None = None,
        strict_build_lock: StrictBuildLock | None = None,
) -> dict:
    """Сохраняет снимок стора и манифест. Возвращает записанный манифест."""
    directory = Path(directory)
    canonical_provenance = _canonical_provenance(embedder_provenance)
    internal_build_lock = False
    if canonical_provenance is not None:
        _validate_strict_paths(directory)
    if canonical_provenance is not None and store.size != len(store.chunk_ids):
        raise ValueError(
            "strict FAISS store vector count does not match its chunk ID count"
        )
    provenance_dimension = (
        canonical_provenance.get("embedding_dimension")
        if canonical_provenance is not None
        else None
    )
    if provenance_dimension is not None and provenance_dimension != store.dimension:
        raise ValueError(
            "FAISS dimension does not match the embedder provenance: "
            f"{store.dimension} != {provenance_dimension}"
        )
    if canonical_provenance is not None:
        expected_resolved_kind = _resolved_provenance_index_kind(
            canonical_provenance,
            len(store.chunk_ids),
        )
        if store.index_kind is not expected_resolved_kind:
            raise ValueError(
                "FAISS index kind does not match strict provenance: "
                f"{store.index_kind.value} != {expected_resolved_kind.value}"
            )
    if canonical_provenance is not None:
        if strict_build_lock is None:
            strict_build_lock = acquire_strict_build_lock(directory)
            internal_build_lock = True
        else:
            strict_build_lock.assert_owned()
            _validate_strict_entries(directory, strict_build_lock)
    store.save(directory)
    # chunk_id = "<book-slug>:<page>", книга — часть до первого двоеточия.
    # Legacy page chunks use "<book-slug>:<page>". Educational graph nodes
    # have opaque "edu_*" ids, so their textbook ids must come from metadata.
    books = (
        {str(book_id) for book_id in book_ids}
        if book_ids is not None
        else {chunk_id.split(":", 1)[0] for chunk_id in store.chunk_ids}
    )
    manifest: dict[str, object] = {
        "artifacts": {
            filename: _file_integrity(directory / filename)
            for filename in _ARTIFACT_FILES
        },
        "n_vectors": store.size,
        "n_chunks": len(store.chunk_ids),
        "n_books": len(books),
        "index_type": store.index_type,
        "resolved_index_kind": store.index_kind.value,
        "embedder": embedder_name,
        "embedding_dimension": store.dimension,
        "corpus_hash": corpus_fingerprint(
            store.chunk_ids,
            embedder_name,
            embedder_provenance=canonical_provenance,
            corpus_projection_sha256=corpus_projection_sha256,
        ),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "schema_version": MANIFEST_SCHEMA_V2,
    }
    if canonical_provenance is not None:
        manifest["embedder_provenance"] = canonical_provenance
        manifest["embedder_provenance_sha256"] = provenance_fingerprint(
            canonical_provenance
        )
    if corpus_projection_sha256 is not None:
        manifest["corpus_projection_sha256"] = corpus_projection_sha256
    (directory / MANIFEST_FILE).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if internal_build_lock:
        validated = load_index(
            directory,
            store.chunk_ids,
            embedder_name,
            embedder_provenance=canonical_provenance,
            corpus_projection_sha256=corpus_projection_sha256,
            require_strict_manifest=True,
            active_build_lock=strict_build_lock,
        )
        if validated is None:
            raise IndexValidationError(
                "strict index could not be revalidated after persistence"
            )
        strict_build_lock.release()
    return manifest


def load_manifest(directory: Path | str) -> dict | None:
    path = Path(directory) / MANIFEST_FILE
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def load_index(
        directory: Path | str,
        chunk_ids: list[str],
        embedder_name: str,
        *,
        embedder_provenance: Mapping[str, object] | None = None,
        corpus_projection_sha256: str | None = None,
        require_strict_manifest: bool = False,
        active_build_lock: StrictBuildLock | None = None,
) -> FaissVectorStore | None:
    """Загружает снимок, только если он построен по ТОМУ ЖЕ корпусу и эмбеддеру.

    Иначе (манифеста нет, состав книг изменился, сменился эмбеддер) возвращает
    None — вызывающий код пересоберёт индекс и перезапишет снимок.
    """
    directory = Path(directory)
    strict_directory_exists = (
        _validate_strict_paths(directory) if require_strict_manifest else False
    )
    strict_entries = (
        _validate_strict_entries(directory, active_build_lock)
        if require_strict_manifest and strict_directory_exists
        else set()
    )
    manifest = load_manifest(directory)
    if manifest is None:
        if require_strict_manifest:
            if not strict_directory_exists:
                return None
            existing = sorted(strict_entries)
            _strict_failure(
                bool(existing),
                "strict index directory is not clean or has a malformed manifest: "
                + ", ".join(existing),
            )
            return None
        return None
    canonical_provenance = _canonical_provenance(embedder_provenance)
    expected_resolved_kind: IndexKind | None = None
    if require_strict_manifest:
        if canonical_provenance is None or corpus_projection_sha256 is None:
            raise ValueError(
                "strict index loading requires embedder provenance and corpus projection"
            )
        if manifest.get("schema_version") != MANIFEST_SCHEMA_V2:
            _strict_failure(True, "strict index manifest schema is missing or invalid")
            return None
        if manifest.get("embedder_provenance") != canonical_provenance:
            _strict_failure(True, "strict index embedder provenance does not match")
            return None
        if manifest.get("embedder_provenance_sha256") != provenance_fingerprint(
            canonical_provenance
        ):
            _strict_failure(True, "strict index provenance hash does not match")
            return None
        if manifest.get("corpus_projection_sha256") != corpus_projection_sha256:
            _strict_failure(True, "strict index corpus projection does not match")
            return None
        if manifest.get("embedding_dimension") != canonical_provenance.get(
            "embedding_dimension"
        ):
            _strict_failure(True, "strict index embedding dimension does not match")
            return None
        expected_resolved_kind = _resolved_provenance_index_kind(
            canonical_provenance,
            len(chunk_ids),
        )
        if manifest.get("resolved_index_kind") != expected_resolved_kind.value:
            _strict_failure(True, "strict index resolved FAISS kind does not match")
            return None
    expected = corpus_fingerprint(
        chunk_ids,
        embedder_name,
        embedder_provenance=canonical_provenance,
        corpus_projection_sha256=corpus_projection_sha256,
    )
    if manifest.get("corpus_hash") != expected:
        _strict_failure(
            require_strict_manifest,
            "strict index corpus fingerprint does not match",
        )
        return None
    artifacts = manifest.get("artifacts")
    if artifacts is not None:
        if not _artifacts_match(directory, artifacts):
            _strict_failure(
                require_strict_manifest,
                "strict index artifact hash or byte length does not match",
            )
            return None
    elif require_strict_manifest:
        _strict_failure(True, "strict index artifact integrity records are missing")
        return None
    if require_strict_manifest:
        if manifest.get("n_chunks") != len(chunk_ids):
            _strict_failure(True, "strict index chunk count does not match")
            return None
        if manifest.get("n_vectors") != len(chunk_ids):
            _strict_failure(True, "strict index vector count does not match")
            return None
    try:
        store = FaissVectorStore.load(directory)
    except Exception as exc:
        if require_strict_manifest:
            raise IndexValidationError(
                "strict FAISS snapshot could not be deserialized"
            ) from exc
        return None
    if store is None:
        _strict_failure(
            require_strict_manifest,
            "strict FAISS snapshot artifacts are missing",
        )
        return None
    if require_strict_manifest and sorted(store.chunk_ids) != sorted(chunk_ids):
        _strict_failure(True, "strict FAISS chunk IDs do not match the corpus")
        return None
    if require_strict_manifest and store.size != len(chunk_ids):
        _strict_failure(True, "strict FAISS vector count does not match the corpus")
        return None
    if require_strict_manifest and manifest.get("index_type") != store.index_type:
        _strict_failure(True, "strict manifest FAISS type does not match artifact")
        return None
    if require_strict_manifest and store.index_kind is not expected_resolved_kind:
        _strict_failure(True, "strict FAISS artifact kind does not match provenance")
        return None
    if (
        require_strict_manifest
        and store.dimension != canonical_provenance.get("embedding_dimension")
    ):
        _strict_failure(True, "strict FAISS vector dimension does not match")
        return None
    return store
