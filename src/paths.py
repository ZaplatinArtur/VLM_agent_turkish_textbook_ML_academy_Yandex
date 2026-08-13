import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = Path(os.environ.get("MLA_DATA_DIR", PROJECT_ROOT / "data")).expanduser()
CORPUS_DIR = DATA_DIR / "corpus"
EVAL_DIR = DATA_DIR / "eval"
CACHE_ROOT = DATA_DIR / "cache"

BOOKS_DIR = CORPUS_DIR / "books"
VISUAL_RETRIVE_DIR = Path(
    os.environ.get("MLA_VISUAL_RETRIVE_DIR", DATA_DIR / "visual_retrive")
).expanduser()
CHUNKS_JSONL_DIR = Path(
    os.environ.get("MLA_CHUNKS_JSONL_DIR", CORPUS_DIR / "chunks" / "jsonl")
).expanduser()
TESSDATA_DIR = CORPUS_DIR / "tessdata"
INDEX_DIR = Path(
    os.environ.get("MLA_INDEX_DIR", CACHE_ROOT / "index")
).expanduser()
MODELS_DIR = DATA_DIR / "models"
RERANKER_ADAPTER_DIR = MODELS_DIR / "bge-reranker-v2" / "chosen"


def ensure_data_dirs() -> None:
    BOOKS_DIR.mkdir(parents=True, exist_ok=True)
    VISUAL_RETRIVE_DIR.mkdir(parents=True, exist_ok=True)
    CHUNKS_JSONL_DIR.mkdir(parents=True, exist_ok=True)
    TESSDATA_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)


def to_data_relpath(path: Path | str) -> str:
    """Path relative to data/, POSIX-style (e.g. books/slug/0001.jpg)."""
    resolved = Path(path).resolve()
    return resolved.relative_to(DATA_DIR.resolve()).as_posix()


def resolve_data_path(path: Path | str) -> Path:
    """Resolve a path stored relative to data/ (absolute paths pass through)."""
    p = Path(path)
    if p.is_absolute():
        return p
    return (DATA_DIR / p).resolve()
