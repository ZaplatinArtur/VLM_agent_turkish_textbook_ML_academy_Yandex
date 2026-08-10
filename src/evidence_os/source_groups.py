"""Evaluator-only source-family grouping for leakage-resistant validation.

Nothing in this module should be passed to an inference policy.  Source URLs
can correlate strongly with benchmark answers; they exist here solely to keep
documents/source families together in validation folds and aggregate reports.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from hashlib import sha256
import math
import posixpath
import re
from types import MappingProxyType
from typing import Any
from urllib.parse import parse_qsl, quote, unquote, urlencode, urlsplit


MISSING_SOURCE_FAMILY = "source:missing"

_TRACKING_QUERY_KEYS = frozenset(
    {
        "fbclid",
        "gclid",
        "mc_cid",
        "mc_eid",
        "nosw",
        "ref",
        "ref_src",
        "source",
        "spm",
    }
)
_WHITESPACE = re.compile(r"\s+")
_SLUG_CHARS = re.compile(r"[^\w.-]+", re.UNICODE)


def _is_tracking_key(key: str) -> bool:
    folded = key.casefold()
    return folded in _TRACKING_QUERY_KEYS or folded.startswith("utm_")


def _normalized_host(host: str) -> str:
    folded = host.rstrip(".").casefold()
    if folded.startswith("www."):
        folded = folded[4:]
    try:
        return folded.encode("idna").decode("ascii")
    except UnicodeError:
        return folded


def _normalized_path(path: str) -> str:
    decoded = unquote(path or "/")
    collapsed = re.sub(r"/{2,}", "/", decoded)
    normalized = posixpath.normpath(collapsed)
    if not normalized.startswith("/"):
        normalized = "/" + normalized
    if normalized != "/":
        normalized = normalized.rstrip("/")
    return quote(normalized, safe="/:@-._~")


def _safe_slug(value: str, *, fallback: str) -> str:
    compact = _WHITESPACE.sub(" ", unquote(value)).strip().casefold()
    slug = _SLUG_CHARS.sub("-", compact).strip("-._")
    return (slug[:80] or fallback)


def _youtube_video_id(host: str, path: str, query: list[tuple[str, str]]) -> str | None:
    segments = [segment for segment in path.split("/") if segment]
    if host == "youtu.be" and segments:
        return segments[0]
    if host in {"youtube.com", "m.youtube.com"}:
        if segments and segments[0] in {"embed", "shorts", "live"} and len(segments) > 1:
            return segments[1]
        for key, value in query:
            if key.casefold() == "v" and value:
                return value
    return None


def normalize_source_family(source: Any) -> str:
    """Return a conservative document-level family for a source reference.

    The scheme and known tracking parameters are ignored, but document paths
    and all unknown query parameters are retained.  Yandex document viewer and
    YouTube links receive stable document/video identities rather than being
    collapsed to one large host-wide group.
    """

    if source is None:
        return MISSING_SOURCE_FAMILY
    raw = _WHITESPACE.sub(" ", str(source)).strip()
    if not raw:
        return MISSING_SOURCE_FAMILY

    parse_target = raw
    if "://" not in parse_target and re.match(r"^[\w.-]+\.[a-z]{2,}(?:/|$)", parse_target, re.I):
        parse_target = "https://" + parse_target
    parsed = urlsplit(parse_target)
    if not parsed.hostname:
        normalized_text = raw.casefold().replace("\\", "/")
        return f"source:text:{normalized_text}"

    host = _normalized_host(parsed.hostname)
    port = parsed.port
    if port and not ((parsed.scheme.casefold() == "http" and port == 80) or (parsed.scheme.casefold() == "https" and port == 443)):
        host = f"{host}:{port}"
    path = _normalized_path(parsed.path)
    query = [(key, value) for key, value in parse_qsl(parsed.query, keep_blank_values=True) if not _is_tracking_key(key)]

    video_id = _youtube_video_id(host, unquote(path), query)
    if video_id:
        return f"source:youtube:{quote(video_id, safe='-._~')}"

    if host == "docs.yandex.ru" and unquote(path).startswith("/docs/view"):
        values: dict[str, list[str]] = defaultdict(list)
        for key, value in query:
            values[key.casefold()].append(value)
        identity_material = "\x1f".join(sorted(values.get("url", ())))
        display_name = next(iter(values.get("name", ())), "document")
        if not identity_material:
            # Keep all non-tracking parameters if the expected share URL is
            # absent; never collapse distinct unknown documents by name alone.
            identity_material = urlencode(sorted(query), doseq=True)
        identity = sha256(identity_material.encode("utf-8")).hexdigest()[:16]
        return f"source:docs.yandex.ru:{_safe_slug(display_name, fallback='document')}:{identity}"

    normalized_query = urlencode(sorted(query, key=lambda item: (item[0].casefold(), item[1])), doseq=True)
    suffix = f"?{normalized_query}" if normalized_query else ""
    return f"source:url:{host}{path}{suffix}"


@dataclass(frozen=True, slots=True)
class SourceGroupIndex:
    """Opaque evaluator map from task ID to normalized source family."""

    _family_by_task_id: Mapping[str, str] = field(repr=False)

    @classmethod
    def from_records(
        cls,
        records: Iterable[Mapping[str, Any]],
        *,
        task_id_key: str = "task_id",
        source_key: str = "source",
    ) -> "SourceGroupIndex":
        families: dict[str, str] = {}
        for row_number, record in enumerate(records, start=1):
            task_id = str(record.get(task_id_key) or "").strip()
            if not task_id:
                raise ValueError(f"source metadata row {row_number} has no task ID")
            if task_id in families:
                raise ValueError(f"duplicate source metadata task ID {task_id!r}")
            families[task_id] = normalize_source_family(record.get(source_key))
        return cls(_family_by_task_id=MappingProxyType(families))

    def family_for(self, task_id: str, *, require_known: bool = True) -> str:
        family = self._family_by_task_id.get(task_id)
        if family is None:
            if require_known:
                raise KeyError(f"no source metadata for task ID {task_id!r}")
            return MISSING_SOURCE_FAMILY
        return family

    def group_task_ids(
        self,
        task_ids: Iterable[str],
        *,
        require_known: bool = True,
    ) -> Mapping[str, tuple[str, ...]]:
        groups: dict[str, list[str]] = defaultdict(list)
        for task_id in task_ids:
            groups[self.family_for(task_id, require_known=require_known)].append(task_id)
        return MappingProxyType({family: tuple(ids) for family, ids in sorted(groups.items())})


@dataclass(frozen=True, slots=True)
class SourceGroupSummary:
    family: str
    count: int
    mean: float


def summarize_by_source(
    task_ids: Sequence[str],
    values: Sequence[float | int | bool],
    source_index: SourceGroupIndex,
    *,
    require_known: bool = True,
) -> tuple[SourceGroupSummary, ...]:
    """Aggregate numeric evaluator outcomes without exposing them to policy."""

    if len(task_ids) != len(values):
        raise ValueError("task_ids and values must have equal length")
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("task_ids must be unique")
    buckets: dict[str, list[float]] = defaultdict(list)
    for task_id, raw_value in zip(task_ids, values, strict=True):
        value = float(raw_value)
        if not math.isfinite(value):
            raise ValueError(f"non-finite evaluator value for task ID {task_id!r}")
        family = source_index.family_for(task_id, require_known=require_known)
        buckets[family].append(value)
    return tuple(
        SourceGroupSummary(family=family, count=len(bucket), mean=sum(bucket) / len(bucket))
        for family, bucket in sorted(buckets.items())
    )


def assign_group_folds(
    task_ids: Sequence[str],
    source_index: SourceGroupIndex,
    *,
    n_folds: int = 5,
    seed: str = "evidence-os-v1",
    require_known: bool = True,
) -> Mapping[str, int]:
    """Assign whole source families to deterministic, size-balanced folds."""

    if n_folds < 2:
        raise ValueError("n_folds must be at least 2")
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("task_ids must be unique")
    groups = source_index.group_task_ids(task_ids, require_known=require_known)
    if len(groups) < n_folds:
        raise ValueError(
            f"need at least {n_folds} source families, found {len(groups)}"
        )

    def tie_break(family: str) -> str:
        return sha256(f"{seed}\x1f{family}".encode("utf-8")).hexdigest()

    ordered_groups = sorted(groups.items(), key=lambda item: (-len(item[1]), tie_break(item[0])))
    fold_sizes = [0] * n_folds
    assignment: dict[str, int] = {}
    for family, family_task_ids in ordered_groups:
        minimum_size = min(fold_sizes)
        available = [index for index, size in enumerate(fold_sizes) if size == minimum_size]
        chosen = available[int(tie_break(family), 16) % len(available)]
        for task_id in family_task_ids:
            assignment[task_id] = chosen
        fold_sizes[chosen] += len(family_task_ids)
    return MappingProxyType(assignment)
