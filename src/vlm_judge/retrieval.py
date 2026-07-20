from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import time
from contextlib import closing
from pathlib import Path
from typing import Any, Iterable


INDEX_SCHEMA_VERSION = "bm25-fts5-v1"
_TOKEN_PATTERN = re.compile(r"[^\W_]+", re.UNICODE)


def _query_tokens(query: str, *, maximum: int = 64) -> list[str]:
    seen: set[str] = set()
    tokens: list[str] = []
    for token in _TOKEN_PATTERN.findall(query.casefold()):
        if len(token) < 2 and not token.isdigit():
            continue
        if token in seen:
            continue
        seen.add(token)
        tokens.append(token)
        if len(tokens) >= maximum:
            break
    return tokens


def build_match_query(query: str, *, mode: str = "or") -> tuple[str, list[str]]:
    tokens = _query_tokens(query)
    if not tokens:
        raise ValueError("query contains no searchable tokens")
    if mode not in {"or", "and"}:
        raise ValueError("mode must be 'or' or 'and'")
    operator = " OR " if mode == "or" else " AND "
    escaped = [f'"{token.replace(chr(34), chr(34) * 2)}"' for token in tokens]
    return operator.join(escaped), tokens


def _create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE index_meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE VIRTUAL TABLE chunks_fts USING fts5(
            chunk_id UNINDEXED,
            page_id UNINDEXED,
            subject UNINDEXED,
            grade UNINDEXED,
            book_id UNINDEXED,
            page_number UNINDEXED,
            source_url UNINDEXED,
            parent_page_hash UNINDEXED,
            text,
            metadata_json UNINDEXED,
            tokenize='unicode61 remove_diacritics 2'
        );
        """
    )


def build_bm25_index(
    chunks_path: Path,
    index_path: Path,
    *,
    batch_size: int = 1000,
) -> dict[str, Any]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if not chunks_path.is_file():
        raise FileNotFoundError(chunks_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = index_path.with_name(index_path.name + ".tmp")
    if temporary.exists():
        temporary.unlink()

    started = time.perf_counter()
    source_digest = hashlib.sha256()
    source_records = 0
    indexed = 0
    skipped_non_text = 0
    skipped_empty = 0
    parse_errors: list[dict[str, Any]] = []
    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(temporary)
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute("PRAGMA temp_store=MEMORY")
        connection.execute("PRAGMA cache_size=-200000")
        _create_schema(connection)
        batch: list[tuple[str, ...]] = []

        with chunks_path.open("rb") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                source_digest.update(raw_line)
                if not raw_line.strip():
                    continue
                source_records += 1
                try:
                    record = json.loads(raw_line.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    if len(parse_errors) < 100:
                        parse_errors.append({"line": line_number, "error": str(exc)})
                    continue
                if record.get("kind") != "text":
                    skipped_non_text += 1
                    continue
                text = str(record.get("text") or "").strip()
                if not text:
                    skipped_empty += 1
                    continue
                metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
                batch.append(
                    (
                        str(record.get("chunk_id") or ""),
                        str(record.get("page_id") or ""),
                        str(metadata.get("subject") or ""),
                        str(metadata.get("grade") or ""),
                        str(metadata.get("book_id") or ""),
                        str(metadata.get("page_number") or ""),
                        str(metadata.get("source_url") or ""),
                        str(metadata.get("parent_page_hash") or ""),
                        text,
                        json.dumps(metadata, ensure_ascii=False, separators=(",", ":")),
                    )
                )
                if len(batch) >= batch_size:
                    connection.executemany(
                        "INSERT INTO chunks_fts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        batch,
                    )
                    indexed += len(batch)
                    batch.clear()
                    connection.commit()
        if batch:
            connection.executemany(
                "INSERT INTO chunks_fts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                batch,
            )
            indexed += len(batch)
        metadata = {
            "schema_version": INDEX_SCHEMA_VERSION,
            "source_path": str(chunks_path.resolve()),
            "source_sha256": source_digest.hexdigest(),
            "source_records": str(source_records),
            "indexed_text_chunks": str(indexed),
            "created_unix": str(int(time.time())),
        }
        connection.executemany("INSERT INTO index_meta(key, value) VALUES (?, ?)", metadata.items())
        connection.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('optimize')")
        connection.commit()
        actual_count = int(connection.execute("SELECT count(*) FROM chunks_fts").fetchone()[0])
        if actual_count != indexed:
            raise RuntimeError(f"index verification failed: expected {indexed}, found {actual_count}")
        connection.close()
        connection = None
        os.replace(temporary, index_path)
    except Exception:
        if connection is not None:
            connection.close()
        if temporary.exists():
            temporary.unlink()
        raise

    return {
        "schema_version": INDEX_SCHEMA_VERSION,
        "source_records": source_records,
        "indexed_text_chunks": indexed,
        "skipped_non_text_chunks": skipped_non_text,
        "skipped_empty_text_chunks": skipped_empty,
        "parse_errors": len(parse_errors),
        "parse_error_examples": parse_errors,
        "source_sha256": source_digest.hexdigest(),
        "index_path": str(index_path),
        "index_bytes": index_path.stat().st_size,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }


def _open_read_only(index_path: Path) -> sqlite3.Connection:
    if not index_path.is_file():
        raise FileNotFoundError(index_path)
    connection = sqlite3.connect(f"file:{index_path.resolve().as_posix()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def index_info(index_path: Path) -> dict[str, Any]:
    with closing(_open_read_only(index_path)) as connection:
        metadata = {
            str(row["key"]): str(row["value"])
            for row in connection.execute("SELECT key, value FROM index_meta")
        }
        count = int(connection.execute("SELECT count(*) FROM chunks_fts").fetchone()[0])
    return {**metadata, "indexed_text_chunks": count, "index_path": str(index_path)}


def get_chunk(index_path: Path, chunk_id: str) -> dict[str, Any] | None:
    with closing(_open_read_only(index_path)) as connection:
        row = connection.execute(
            """
            SELECT chunk_id, page_id, subject, grade, book_id, page_number,
                   source_url, parent_page_hash, text, metadata_json
            FROM chunks_fts WHERE chunk_id = ? LIMIT 1
            """,
            (chunk_id,),
        ).fetchone()
    return _row_to_hit(row, rank=None, raw_score=None) if row is not None else None


def _row_to_hit(
    row: sqlite3.Row,
    *,
    rank: int | None,
    raw_score: float | None,
) -> dict[str, Any]:
    metadata = json.loads(row["metadata_json"]) if row["metadata_json"] else {}
    result = {
        "chunk_id": row["chunk_id"],
        "page_id": row["page_id"],
        "text": row["text"],
        "subject": row["subject"] or None,
        "grade": row["grade"] or None,
        "book_id": row["book_id"] or None,
        "page_number": row["page_number"] or None,
        "source_url": row["source_url"] or None,
        "parent_page_hash": row["parent_page_hash"] or None,
        "metadata": metadata,
    }
    if rank is not None:
        result["rank"] = rank
    if raw_score is not None:
        result["bm25_raw"] = raw_score
        result["lexical_score"] = -raw_score
    return result


def search_bm25(
    index_path: Path,
    query: str,
    *,
    top_k: int = 10,
    subject: str | None = None,
    grade: str | int | None = None,
    mode: str = "or",
    low_information_weight: float = 0.25,
) -> dict[str, Any]:
    if not 1 <= top_k <= 100:
        raise ValueError("top_k must be between 1 and 100")
    if not 0.0 <= low_information_weight <= 1.0:
        raise ValueError("low_information_weight must be between 0 and 1")
    match_query, tokens = build_match_query(query, mode=mode)
    clauses = ["chunks_fts MATCH ?"]
    parameters: list[Any] = [match_query]
    if subject:
        clauses.append("subject = ?")
        parameters.append(str(subject))
    if grade not in (None, ""):
        clauses.append("grade = ?")
        parameters.append(str(grade))
    candidate_limit = min(1000, max(top_k * 10, top_k))
    parameters.append(candidate_limit)
    sql = f"""
        SELECT chunk_id, page_id, subject, grade, book_id, page_number,
               source_url, parent_page_hash, text, metadata_json,
               bm25(chunks_fts) AS raw_score
        FROM chunks_fts
        WHERE {' AND '.join(clauses)}
        ORDER BY raw_score ASC
        LIMIT ?
    """
    started = time.perf_counter()
    with closing(_open_read_only(index_path)) as connection:
        rows = list(connection.execute(sql, parameters))
    adjusted_rows: list[tuple[float, float, sqlite3.Row]] = []
    for row in rows:
        raw_score = float(row["raw_score"])
        metadata = json.loads(row["metadata_json"]) if row["metadata_json"] else {}
        adjusted_score = (
            raw_score * low_information_weight
            if metadata.get("index_policy") == "downweight"
            else raw_score
        )
        adjusted_rows.append((adjusted_score, raw_score, row))
    adjusted_rows.sort(key=lambda value: value[0])
    hits = []
    for rank, (adjusted_score, raw_score, row) in enumerate(adjusted_rows[:top_k], start=1):
        hit = _row_to_hit(row, rank=rank, raw_score=adjusted_score)
        hit["bm25_raw_unadjusted"] = raw_score
        hit["index_policy"] = hit["metadata"].get("index_policy", "normal")
        hits.append(hit)
    return {
        "query": query,
        "query_tokens": tokens,
        "mode": mode,
        "filters": {"subject": subject, "grade": grade},
        "low_information_weight": low_information_weight,
        "top_k": top_k,
        "returned": len(hits),
        "latency_ms": round((time.perf_counter() - started) * 1000, 3),
        "hits": hits,
    }


def iter_text_chunks(index_path: Path) -> Iterable[dict[str, Any]]:
    with closing(_open_read_only(index_path)) as connection:
        cursor = connection.execute(
            """
            SELECT chunk_id, page_id, subject, grade, book_id, page_number,
                   source_url, parent_page_hash, text, metadata_json
            FROM chunks_fts ORDER BY rowid
            """
        )
        for row in cursor:
            yield _row_to_hit(row, rank=None, raw_score=None)
