import hashlib
import pickle
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"
BOOKS_RAW_DIR = DATA_DIR / "books" / "raw"
CACHE_DIR = DATA_DIR / "cache"


class Cache:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self.root / f"{digest}.pkl"

    def get(self, key: str) -> Any | None:
        path = self._path(key)
        if not path.exists():
            return None
        with path.open("rb") as f:
            return pickle.load(f)

    def set(self, key: str, obj: Any) -> None:
        path = self._path(key)
        with path.open("wb") as f:
            pickle.dump(obj, f)

    def values(self) -> list[Any]:
        items: list[Any] = []
        for path in sorted(self.root.glob("*.pkl")):
            with path.open("rb") as f:
                items.append(pickle.load(f))
        return items
