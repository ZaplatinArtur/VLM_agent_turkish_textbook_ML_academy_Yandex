from importlib import metadata
from pathlib import Path

import pytest

from retrieve import config as config_module
from retrieve.config import (
    BGE_M3_EMBEDDING_DIMENSION,
    BGE_M3_LICENSE,
    BGE_M3_MODEL_ID,
    BGE_M3_REVISION,
    BGE_M3_RUNTIME_DISTRIBUTIONS,
    BGE_M3_SENTENCE_TRANSFORMERS_RANGE,
    BGE_M3_TASK_CONTRACT,
    BgeM3Config,
    resolve_bge_runtime_versions,
)
from retrieve.embedders.sentence_transformer import BGE_M3_SEMANTIC_SMOKE_CONTRACT


RUNTIME_VERSIONS = {
    "sentence-transformers": "5.3.0",
    "transformers": "4.57.1",
    "torch": "2.8.0",
    "tokenizers": "0.22.1",
    "faiss-cpu": "1.15.0",
    "numpy": "2.3.0",
}


def test_bge_m3_is_disabled_by_default_but_identity_is_immutable():
    config = BgeM3Config.from_env({})

    assert config.enabled is False
    assert config.model_id == BGE_M3_MODEL_ID
    assert config.revision == BGE_M3_REVISION
    assert config.license == BGE_M3_LICENSE
    assert config.task_contract == BGE_M3_TASK_CONTRACT
    assert config.embedding_dimension == BGE_M3_EMBEDDING_DIMENSION
    assert config.local_files_only is True
    assert config.candidate_mode == "union"
    assert config.device == "cpu"
    assert config.faiss_index_kind == "auto"


def test_bge_m3_env_contract_is_explicit_and_typed(tmp_path):
    config = BgeM3Config.from_env(
        {
            "MLA_BGE_M3_ENABLED": "yes",
            "MLA_BGE_M3_MODEL_ID": BGE_M3_MODEL_ID,
            "MLA_BGE_M3_REVISION": BGE_M3_REVISION,
            "MLA_BGE_M3_LICENSE": "MIT",
            "MLA_BGE_M3_TASK_CONTRACT": BGE_M3_TASK_CONTRACT,
            "MLA_BGE_M3_EMBEDDING_DIMENSION": "1024",
            "MLA_BGE_M3_CANDIDATE_MODE": "fallback",
            "MLA_BGE_M3_PRIMARY_CANDIDATE_K": "24",
            "MLA_BGE_M3_SEMANTIC_CANDIDATE_K": "16",
            "MLA_BGE_M3_FALLBACK_MIN_CANDIDATES": "3",
            "MLA_BGE_M3_BATCH_SIZE": "4",
            "MLA_BGE_M3_MAX_LENGTH": "512",
            "MLA_BGE_M3_ALLOW_DOWNLOAD": "true",
            "MLA_BGE_M3_CACHE_DIR": str(tmp_path / "models"),
            "MLA_BGE_M3_INDEX_DIR": str(tmp_path / "index"),
            "MLA_BGE_M3_DEVICE": "cpu",
            "MLA_FAISS_INDEX_KIND": "hnsw",
        }
    )

    assert config.enabled is True
    assert config.candidate_mode == "fallback"
    assert config.primary_candidate_k == 24
    assert config.semantic_candidate_k == 16
    assert config.fallback_min_candidates == 3
    assert config.batch_size == 4
    assert config.max_length == 512
    assert config.local_files_only is False
    assert config.cache_dir == Path(tmp_path / "models")
    assert config.index_dir == Path(tmp_path / "index")
    assert config.device == "cpu"
    assert config.faiss_index_kind == "hnsw"


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("MLA_BGE_M3_ENABLED", "sometimes", "must be one of"),
        ("MLA_BGE_M3_MODEL_ID", "other/model", "model_id"),
        ("MLA_BGE_M3_REVISION", "main", "revision"),
        ("MLA_BGE_M3_LICENSE", "apache-2.0", "license"),
        ("MLA_BGE_M3_TASK_CONTRACT", "other-task", "task contract"),
        ("MLA_BGE_M3_EMBEDDING_DIMENSION", "768", "must be in"),
        ("MLA_BGE_M3_CANDIDATE_MODE", "replace", "candidate_mode"),
        ("MLA_FAISS_INDEX_KIND", "ivf", "faiss_index_kind"),
        ("MLA_BGE_M3_SEMANTIC_CANDIDATE_K", "0", "must be in"),
        ("MLA_BGE_M3_MAX_LENGTH", "not-an-int", "must be an integer"),
    ],
)
def test_bge_m3_env_tamper_and_invalid_values_fail_closed(name, value, message):
    environ = {name: value}
    if name != "MLA_BGE_M3_ENABLED":
        environ["MLA_BGE_M3_ENABLED"] = "true"
    with pytest.raises(ValueError, match=message):
        BgeM3Config.from_env(environ)


def test_disabled_bge_ignores_stale_arm_specific_environment():
    config = BgeM3Config.from_env(
        {
            "MLA_BGE_M3_ENABLED": "false",
            "MLA_BGE_M3_MODEL_ID": "stale/model",
            "MLA_BGE_M3_REVISION": "main",
            "MLA_BGE_M3_EMBEDDING_DIMENSION": "768",
            "MLA_BGE_M3_ALLOW_DOWNLOAD": "sometimes",
        }
    )

    assert config == BgeM3Config(enabled=False)


def test_bge_m3_provenance_contains_every_embedding_identity_field():
    config = BgeM3Config()
    assert config.embedder_provenance(RUNTIME_VERSIONS) == {
        "backend": "sentence-transformers",
        "batch_size": 2,
        "device": "cpu",
        "embedding_dimension": BGE_M3_EMBEDDING_DIMENSION,
        "encode_normalize_embeddings": False,
        "faiss_index_kind": "auto",
        "license": BGE_M3_LICENSE,
        "max_length": 1024,
        "model_id": BGE_M3_MODEL_ID,
        "revision": BGE_M3_REVISION,
        "runtime_packages": RUNTIME_VERSIONS,
        "semantic_smoke_contract": BGE_M3_SEMANTIC_SMOKE_CONTRACT,
        "task_contract": BGE_M3_TASK_CONTRACT,
        "trust_remote_code": False,
        "vector_store_normalization": "l2",
    }


def test_bge_m3_runtime_versions_are_resolved_and_missing_packages_fail_closed(
    monkeypatch,
):
    monkeypatch.setattr(
        config_module,
        "_distribution_version",
        lambda name: RUNTIME_VERSIONS[name],
    )
    assert resolve_bge_runtime_versions() == RUNTIME_VERSIONS

    def missing_torch(name: str) -> str:
        if name == "torch":
            raise metadata.PackageNotFoundError(name)
        return RUNTIME_VERSIONS[name]

    monkeypatch.setattr(config_module, "_distribution_version", missing_torch)
    with pytest.raises(RuntimeError, match="torch"):
        resolve_bge_runtime_versions()


@pytest.mark.parametrize(
    "installed",
    ["2.7.0", "5.4.0rc1", "5.4.0", "5.6.1", "not-a-version"],
)
def test_bge_m3_rejects_unsupported_sentence_transformers_runtime(
    monkeypatch,
    installed,
):
    versions = {**RUNTIME_VERSIONS, "sentence-transformers": installed}
    monkeypatch.setattr(
        config_module,
        "_distribution_version",
        lambda name: versions[name],
    )

    with pytest.raises(RuntimeError, match="sentence-transformers"):
        resolve_bge_runtime_versions()

    assert BGE_M3_SENTENCE_TRANSFORMERS_RANGE == ">=3.0,<5.4"


def test_bge_m3_runtime_provenance_requires_exact_package_set():
    assert set(BGE_M3_RUNTIME_DISTRIBUTIONS) == set(RUNTIME_VERSIONS)
    with pytest.raises(ValueError, match="runtime provenance"):
        BgeM3Config().embedder_provenance({"torch": "2.8.0"})


@pytest.mark.parametrize(
    "kwargs",
    [
        {"enabled": "false"},
        {"allow_download": "false"},
        {"batch_size": 2.5},
        {"semantic_candidate_k": True},
        {"faiss_index_kind": "ivf"},
    ],
)
def test_bge_m3_direct_config_rejects_type_coercion(kwargs):
    with pytest.raises(ValueError, match="booleans|integers|faiss_index_kind"):
        BgeM3Config(**kwargs)
