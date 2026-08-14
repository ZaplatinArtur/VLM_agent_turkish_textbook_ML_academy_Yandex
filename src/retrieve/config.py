from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

from packaging.specifiers import SpecifierSet
from packaging.version import InvalidVersion, Version

from .embedders.sentence_transformer import BGE_M3_SEMANTIC_SMOKE_CONTRACT

BGE_M3_MODEL_ID = "BAAI/bge-m3"
BGE_M3_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
BGE_M3_LICENSE = "mit"
BGE_M3_TASK_CONTRACT = "symmetric_retrieval_text_v1"
BGE_M3_EMBEDDING_DIMENSION = 1024
BGE_M3_RUNTIME_DISTRIBUTIONS = (
    "sentence-transformers",
    "transformers",
    "torch",
    "tokenizers",
    "faiss-cpu",
    "numpy",
)
BGE_M3_SENTENCE_TRANSFORMERS_RANGE = ">=3.0,<5.4"

_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSE_VALUES = frozenset({"0", "false", "no", "off"})
_CANDIDATE_MODES = frozenset({"union", "fallback"})
_distribution_version = metadata.version
_sentence_transformers_specifier = SpecifierSet(
    BGE_M3_SENTENCE_TRANSFORMERS_RANGE
)


def resolve_bge_runtime_versions() -> dict[str, str]:
    """Return installed runtime versions that can change embedding semantics."""
    versions: dict[str, str] = {}
    for distribution in BGE_M3_RUNTIME_DISTRIBUTIONS:
        try:
            version = _distribution_version(distribution)
        except metadata.PackageNotFoundError as exc:
            raise RuntimeError(
                "enabled BGE-M3 retrieval requires installed distribution "
                f"{distribution!r}"
            ) from exc
        if not isinstance(version, str) or not version.strip():
            raise RuntimeError(
                f"installed distribution {distribution!r} has no usable version"
            )
        version = version.strip()
        if distribution == "sentence-transformers":
            try:
                parsed = Version(version)
            except InvalidVersion as exc:
                raise RuntimeError(
                    "installed sentence-transformers version is invalid: "
                    f"{version!r}"
                ) from exc
            if parsed not in _sentence_transformers_specifier:
                raise RuntimeError(
                    "enabled BGE-M3 retrieval requires sentence-transformers"
                    f"{BGE_M3_SENTENCE_TRANSFORMERS_RANGE}; installed {version}"
                )
        versions[distribution] = version
    return versions


def _env_value(environ: Mapping[str, str], name: str, default: str) -> str:
    value = environ.get(name, default)
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    return value.strip()


def _env_bool(environ: Mapping[str, str], name: str, default: bool) -> bool:
    value = _env_value(environ, name, "true" if default else "false").casefold()
    if value in _TRUE_VALUES:
        return True
    if value in _FALSE_VALUES:
        return False
    raise ValueError(
        f"{name} must be one of: 1/0, true/false, yes/no, on/off"
    )


def _env_int(
    environ: Mapping[str, str],
    name: str,
    default: int,
    *,
    minimum: int,
    maximum: int,
) -> int:
    raw = _env_value(environ, name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be in [{minimum}, {maximum}]")
    return value


@dataclass(frozen=True, slots=True)
class BgeM3Config:
    """Fail-closed runtime contract for the optional BGE-M3 candidate arm."""

    # Включён по умолчанию: с кросс-энкодером даёт +6.4 пп hit@1 на реальном
    # эталоне и +12.7 на синтетике против MiniLM. Модель должна лежать локально
    # (allow_download остаётся False) — иначе загрузка падает с понятной ошибкой.
    enabled: bool = True
    model_id: str = BGE_M3_MODEL_ID
    revision: str = BGE_M3_REVISION
    license: str = BGE_M3_LICENSE
    task_contract: str = BGE_M3_TASK_CONTRACT
    embedding_dimension: int = BGE_M3_EMBEDDING_DIMENSION
    candidate_mode: str = "union"
    primary_candidate_k: int = 32
    semantic_candidate_k: int = 32
    fallback_min_candidates: int = 5
    batch_size: int = 2
    max_length: int = 1024
    allow_download: bool = False
    cache_dir: Path | None = None
    index_dir: Path | None = None
    device: str = "cuda"
    faiss_index_kind: str = "auto"

    def __post_init__(self) -> None:
        string_fields = {
            "model_id": self.model_id,
            "revision": self.revision,
            "license": self.license,
            "task_contract": self.task_contract,
            "candidate_mode": self.candidate_mode,
            "device": self.device,
            "faiss_index_kind": self.faiss_index_kind,
        }
        if any(not isinstance(value, str) for value in string_fields.values()):
            raise ValueError("BGE-M3 identity and candidate mode must be strings")
        if type(self.enabled) is not bool or type(self.allow_download) is not bool:
            raise ValueError("BGE-M3 enabled and allow_download must be booleans")
        integer_fields = {
            "embedding_dimension": self.embedding_dimension,
            "primary_candidate_k": self.primary_candidate_k,
            "semantic_candidate_k": self.semantic_candidate_k,
            "fallback_min_candidates": self.fallback_min_candidates,
            "batch_size": self.batch_size,
            "max_length": self.max_length,
        }
        if any(type(value) is not int for value in integer_fields.values()):
            raise ValueError("BGE-M3 numeric configuration must use integers")
        if self.model_id != BGE_M3_MODEL_ID:
            raise ValueError(
                f"BGE-M3 model_id must remain pinned to {BGE_M3_MODEL_ID!r}"
            )
        if self.revision != BGE_M3_REVISION:
            raise ValueError(
                f"BGE-M3 revision must remain pinned to {BGE_M3_REVISION!r}"
            )
        if self.license.casefold() != BGE_M3_LICENSE:
            raise ValueError(
                f"BGE-M3 license must remain pinned to {BGE_M3_LICENSE!r}"
            )
        if self.task_contract != BGE_M3_TASK_CONTRACT:
            raise ValueError(
                "BGE-M3 task contract must remain pinned to "
                f"{BGE_M3_TASK_CONTRACT!r}"
            )
        if self.embedding_dimension != BGE_M3_EMBEDDING_DIMENSION:
            raise ValueError(
                "BGE-M3 embedding dimension must remain pinned to "
                f"{BGE_M3_EMBEDDING_DIMENSION}"
            )
        if self.candidate_mode not in _CANDIDATE_MODES:
            raise ValueError(
                "BGE-M3 candidate_mode must be 'union' or 'fallback'"
            )
        if self.faiss_index_kind not in {"auto", "flat", "hnsw"}:
            raise ValueError(
                "BGE-M3 faiss_index_kind must be 'auto', 'flat', or 'hnsw'"
            )
        if not 1 <= self.primary_candidate_k <= 10_000:
            raise ValueError("BGE-M3 primary_candidate_k must be in [1, 10000]")
        if not 1 <= self.semantic_candidate_k <= 10_000:
            raise ValueError("BGE-M3 semantic_candidate_k must be in [1, 10000]")
        if not 1 <= self.fallback_min_candidates <= 10_000:
            raise ValueError(
                "BGE-M3 fallback_min_candidates must be in [1, 10000]"
            )
        if not 1 <= self.batch_size <= 256:
            raise ValueError("BGE-M3 batch_size must be in [1, 256]")
        if not 1 <= self.max_length <= 8192:
            raise ValueError("BGE-M3 max_length must be in [1, 8192]")

    @classmethod
    def from_env(
        cls,
        environ: Mapping[str, str] | None = None,
    ) -> "BgeM3Config":
        values = os.environ if environ is None else environ
        enabled = _env_bool(values, "MLA_BGE_M3_ENABLED", True)
        if not enabled:
            # Disabled means a genuinely inert optional arm: stale BGE-only
            # settings must not break or alter the legacy retrieval path.
            return cls(enabled=False)
        model_id = _env_value(values, "MLA_BGE_M3_MODEL_ID", BGE_M3_MODEL_ID)
        revision = _env_value(values, "MLA_BGE_M3_REVISION", BGE_M3_REVISION)
        license_id = _env_value(values, "MLA_BGE_M3_LICENSE", BGE_M3_LICENSE)
        task_contract = _env_value(
            values,
            "MLA_BGE_M3_TASK_CONTRACT",
            BGE_M3_TASK_CONTRACT,
        )
        candidate_mode = _env_value(
            values,
            "MLA_BGE_M3_CANDIDATE_MODE",
            "union",
        ).casefold()
        cache_dir_value = _env_value(values, "MLA_BGE_M3_CACHE_DIR", "")
        index_dir_value = _env_value(values, "MLA_BGE_M3_INDEX_DIR", "")
        device_value = _env_value(values, "MLA_BGE_M3_DEVICE", "cuda") or "cuda"
        faiss_index_kind = _env_value(
            values,
            "MLA_FAISS_INDEX_KIND",
            "auto",
        ).casefold()
        return cls(
            enabled=True,
            model_id=model_id,
            revision=revision,
            license=license_id.casefold(),
            task_contract=task_contract,
            embedding_dimension=_env_int(
                values,
                "MLA_BGE_M3_EMBEDDING_DIMENSION",
                BGE_M3_EMBEDDING_DIMENSION,
                minimum=BGE_M3_EMBEDDING_DIMENSION,
                maximum=BGE_M3_EMBEDDING_DIMENSION,
            ),
            candidate_mode=candidate_mode,
            primary_candidate_k=_env_int(
                values,
                "MLA_BGE_M3_PRIMARY_CANDIDATE_K",
                32,
                minimum=1,
                maximum=10_000,
            ),
            semantic_candidate_k=_env_int(
                values,
                "MLA_BGE_M3_SEMANTIC_CANDIDATE_K",
                32,
                minimum=1,
                maximum=10_000,
            ),
            fallback_min_candidates=_env_int(
                values,
                "MLA_BGE_M3_FALLBACK_MIN_CANDIDATES",
                5,
                minimum=1,
                maximum=10_000,
            ),
            batch_size=_env_int(
                values,
                "MLA_BGE_M3_BATCH_SIZE",
                2,
                minimum=1,
                maximum=256,
            ),
            max_length=_env_int(
                values,
                "MLA_BGE_M3_MAX_LENGTH",
                1024,
                minimum=1,
                maximum=8192,
            ),
            allow_download=_env_bool(
                values,
                "MLA_BGE_M3_ALLOW_DOWNLOAD",
                False,
            ),
            cache_dir=(
                Path(cache_dir_value).expanduser() if cache_dir_value else None
            ),
            index_dir=(
                Path(index_dir_value).expanduser() if index_dir_value else None
            ),
            device=device_value,
            faiss_index_kind=faiss_index_kind,
        )

    @property
    def local_files_only(self) -> bool:
        return not self.allow_download

    def embedder_provenance(
        self,
        runtime_versions: Mapping[str, str] | None = None,
    ) -> dict[str, object]:
        if runtime_versions is not None and not isinstance(runtime_versions, Mapping):
            raise ValueError("BGE-M3 runtime provenance must be a mapping")
        resolved_runtime = dict(
            resolve_bge_runtime_versions()
            if runtime_versions is None
            else runtime_versions
        )
        if set(resolved_runtime) != set(BGE_M3_RUNTIME_DISTRIBUTIONS) or any(
            not isinstance(version, str) or not version.strip()
            for version in resolved_runtime.values()
        ):
            raise ValueError(
                "BGE-M3 runtime provenance must contain non-empty versions for "
                + ", ".join(BGE_M3_RUNTIME_DISTRIBUTIONS)
            )
        return {
            "backend": "sentence-transformers",
            "batch_size": self.batch_size,
            "device": self.device,
            "embedding_dimension": BGE_M3_EMBEDDING_DIMENSION,
            "encode_normalize_embeddings": False,
            "faiss_index_kind": self.faiss_index_kind,
            "license": BGE_M3_LICENSE,
            "max_length": self.max_length,
            "model_id": BGE_M3_MODEL_ID,
            "revision": BGE_M3_REVISION,
            "runtime_packages": {
                distribution: resolved_runtime[distribution].strip()
                for distribution in BGE_M3_RUNTIME_DISTRIBUTIONS
            },
            "semantic_smoke_contract": BGE_M3_SEMANTIC_SMOKE_CONTRACT,
            "task_contract": BGE_M3_TASK_CONTRACT,
            "trust_remote_code": False,
            "vector_store_normalization": "l2",
        }
