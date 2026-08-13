"""Охват роутера по официальным источникам: на скольких задачах он срабатывает.

Модель не вызывается, карта не нужна — считаются секунды. Полезно перед дорогим
прогоном, чтобы знать, какую долю каскад вообще затрагивает.

    python scripts/check_source_router.py --tasks data/eval/tasks.jsonl
    python scripts/check_source_router.py --tasks <файл> --field ocr_text --show 5
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from source_router import route  # noqa: E402


def read_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Охват роутера по официальным источникам")
    parser.add_argument("--tasks", required=True, type=Path, help="JSONL с задачами")
    parser.add_argument("--field", default="question",
                        help="поле с текстом задачи (question | ocr_text | ...)")
    parser.add_argument("--show", type=int, default=0, help="показать N сработавших")
    args = parser.parse_args(argv)

    rows = read_rows(args.tasks)
    if not rows:
        raise SystemExit(f"пусто: {args.tasks}")
    if args.field not in rows[0]:
        raise SystemExit(
            f"в строках нет поля {args.field!r}; есть: {sorted(rows[0])[:12]}"
        )

    hits = []
    for row in rows:
        found = route(str(row.get(args.field) or ""))
        if found is not None:
            hits.append((row, found))

    families = Counter(found.family for _, found in hits)
    total = len(rows)
    print(f"задач: {total}")
    print(f"сработал: {len(hits)} ({len(hits) / total:.1%})")
    for family, count in sorted(families.items()):
        print(f"  {family:<12} {count}")
    print(f"abstain: {total - len(hits)}")

    for row, found in hits[: args.show]:
        task_id = row.get("task_id") or row.get("query_id") or "?"
        print(f"\n[{task_id}] {found.family} {found.record_id}")
        print(f"  score {found.score:.4f} margin {found.margin:.4f}")
        print(f"  ответ: {found.answer[:100]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
