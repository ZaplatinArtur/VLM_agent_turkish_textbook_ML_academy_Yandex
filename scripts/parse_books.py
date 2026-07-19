import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from paths import BOOKS_DIR, ensure_data_dirs
from retrieve.parsing.factory import get_parser


def main() -> None:
    ensure_data_dirs()
    book_dirs = (
        sorted(path for path in BOOKS_DIR.iterdir() if path.is_dir())
        if BOOKS_DIR.exists()
        else []
    )
    print(f"Books: {len(book_dirs)}", flush=True)
    print(f"Books dir: {BOOKS_DIR.resolve()}", flush=True)
    get_parser().parse_books(book_dirs)


if __name__ == "__main__":
    main()
