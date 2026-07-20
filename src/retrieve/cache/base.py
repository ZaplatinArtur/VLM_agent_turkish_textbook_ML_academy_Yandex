import hashlib
import pickle
from pathlib import Path
from typing import Any

from paths import DATA_DIR

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
        with path.open("rb") as file:
            return pickle.load(file)

    def set(self, key: str, value: Any) -> None:
        path = self._path(key)
        with path.open("wb") as file:
            pickle.dump(value, file)
