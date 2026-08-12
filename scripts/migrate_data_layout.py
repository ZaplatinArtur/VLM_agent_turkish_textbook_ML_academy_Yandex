"""Переезд data/ на раскладку corpus / eval / cache.

Раньше всё лежало плоско в data/. Теперь каталог разделён по происхождению:
corpus — источник, eval — то, чем меряем, cache — производное (пересобираемо).
Сам data/ в .gitignore, поэтому вместе с кодом раскладка не приезжает — на
каждой машине переезд нужно выполнить один раз этим скриптом.

Заодно правятся пути внутри чанков: ссылка на скан страницы хранится
относительно data/ (books/<книга>/0001.jpg) и после переезда книг обязана стать
corpus/books/..., иначе paths.resolve_data_path ведёт в никуда.

    python scripts/migrate_data_layout.py            # только показать план
    python scripts/migrate_data_layout.py --apply
    python scripts/migrate_data_layout.py --apply --data-root /path/to/other/data

Повторный запуск безопасен: уже переехавшее пропускается.
"""

import argparse
import json
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# старое место -> новое, относительно data/
MOVES = [
    ("books", "corpus/books"),
    ("chunks", "corpus/chunks"),
    ("tessdata", "corpus/tessdata"),
    ("validation.jsonl", "eval/validation.jsonl"),
    ("validation.meta.jsonl", "eval/validation.meta.jsonl"),
    ("validation_sheet.csv", "eval/validation_sheet.csv"),
    ("tasks.sample.jsonl", "eval/tasks.sample.jsonl"),
    ("qrels_smoke.jsonl", "eval/qrels_smoke.jsonl"),
]
# data/cache и data/images остаются на месте: cache и так производный,
# а на data/images завязаны sheet.py и vlm_judge.validation_archive.

IMAGE_PREFIX_OLD = "books/"
IMAGE_PREFIX_NEW = "corpus/books/"


def plan_moves(data_root: Path) -> list[tuple[Path, Path]]:
    moves = []
    for old, new in MOVES:
        source, target = data_root / old, data_root / new
        if not source.exists():
            continue
        if target.exists():
            print(f"  ! {old} и {new} существуют оба — разберись вручную, пропускаю")
            continue
        moves.append((source, target))
    return moves


def rewrite_chunk_images(chunks_dir: Path, apply: bool) -> tuple[int, int]:
    """Меняет префикс books/ на corpus/books/ в images[].data. Идемпотентно:
    уже переписанный путь начинается с corpus/ и под условие не попадает."""
    files = touched = 0
    for path in sorted(chunks_dir.glob("*.jsonl")):
        records, dirty = [], False
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            for image in record.get("images") or []:
                data = str(image.get("data") or "")
                if data.startswith(IMAGE_PREFIX_OLD):
                    image["data"] = IMAGE_PREFIX_NEW + data[len(IMAGE_PREFIX_OLD):]
                    dirty = True
                    touched += 1
            records.append(record)
        if not dirty:
            continue
        files += 1
        if apply:
            path.write_text(
                "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
                encoding="utf-8",
            )
    return files, touched


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--apply", action="store_true", help="без него — только план")
    args = parser.parse_args(argv)

    data_root = args.data_root.resolve()
    if not data_root.is_dir():
        print(f"Нет каталога: {data_root}")
        return 1
    print(f"data: {data_root}\nрежим: {'ПЕРЕНОС' if args.apply else 'план (добавь --apply)'}\n")

    moves = plan_moves(data_root)
    if moves:
        for source, target in moves:
            print(f"  {source.relative_to(data_root)}  ->  {target.relative_to(data_root)}")
            if args.apply:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(source), str(target))
    else:
        print("  переносить нечего — раскладка уже новая")

    chunks_dir = data_root / "corpus" / "chunks" / "jsonl"
    if not chunks_dir.is_dir():
        chunks_dir = data_root / "chunks" / "jsonl"
    if chunks_dir.is_dir():
        files, touched = rewrite_chunk_images(chunks_dir, args.apply)
        print(f"\nссылки чанк->страница: файлов с правками {files}, ссылок {touched}"
              if files else "\nссылки чанк->страница: править нечего")
    else:
        print("\nчанков не нашёл — пропускаю правку ссылок")

    if not args.apply:
        print("\nЭто был план. Повтори с --apply.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
