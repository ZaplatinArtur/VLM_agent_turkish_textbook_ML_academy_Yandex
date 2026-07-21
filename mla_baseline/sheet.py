"""Конвертер выборки валидации (Google Sheets CSV) в Task JSONL.

Каждая строка таблицы — задача-скриншот (колонка Visual) с эталонным ответом.
На выходе:
  - tasks JSONL по командному контракту Task (картинки — file_path в data/images/)
  - meta JSONL с полями таблицы, которых нет в контракте (Type, Class,
    Question format и т.п.) — для срезов метрик в eval

Запуск:
  python -m mla_baseline.sheet --csv data/validation_sheet.csv
  python -m mla_baseline.sheet --sheet-id 15VJ_gVErnAy2fJLT-JBUO5WvSsBNthhRQyVHti-RVhc
"""

import argparse
import csv
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

from .config import get_settings
from .contracts import ImageRef, Task

# Question type (таблица) -> answer_type (контракт). "precise" уточняется по
# виду эталонного ответа: число -> numeric, иначе short_text.
_QTYPE_MAP = {
    "single-choice question": "choice",
    "multiple-choice": "choice",
    "open question (precise answer)": "short_text",
    "open question (arbitrary answer)": "free_form",
    "fill in the blanks": "short_text",
    "match the items": "short_text",
    "order the items": "short_text",
    "true-false": "short_text",
}


def _is_number(s: str) -> bool:
    try:
        float(s.replace(",", ".").strip())
        return True
    except ValueError:
        return False


def _answer_type(qtype: str, answer: str) -> str:
    mapped = _QTYPE_MAP.get(qtype.strip(), "free_form")
    if mapped == "short_text" and _is_number(answer):
        return "numeric"
    return mapped


def _grade(cls: str) -> int | None:
    cls = cls.strip()
    return int(cls) if cls.isdigit() else None


def _mime(url: str) -> str:
    ext = Path(urllib.parse.urlparse(url).path).suffix.lower()
    return {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}.get(
        ext, "image/png"
    )


def _download(url: str, dest: Path, retries: int = 5, timeout: float = 60.0) -> bool:
    """True — файл на месте (скачан сейчас или раньше)."""
    if dest.exists() and dest.stat().st_size > 0:
        return True
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "mla-baseline/0.1"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = resp.read()
            dest.write_bytes(data)
            return True
        except Exception as exc:  # сеть нестабильная — ретраим всё
            last_err = exc
            time.sleep(min(2 * attempt, 10))
    print(f"  !! не скачалось {url}: {last_err}", file=sys.stderr)
    return False


def fetch_sheet_csv(sheet_id: str, gid: int, dest: Path) -> Path:
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"
    if not _download(url, dest):
        raise RuntimeError("не удалось скачать таблицу")
    return dest


def _get(row: list[str], idx: int) -> str:
    return row[idx].strip() if idx < len(row) and row[idx] else ""


def _fetch_image(url: str, dest: Path, archive: dict[str, Path] | None,
                 download: bool) -> bool:
    """Кладёт картинку в dest: сперва из локального архива, иначе скачивает."""
    import shutil
    import urllib.parse as up

    if dest.exists() and dest.stat().st_size > 0:
        return True
    if archive is not None:
        name = Path(up.unquote(up.urlparse(url).path)).name
        src = archive.get(name)
        if src is not None:
            shutil.copy2(src, dest)
            return True
    if download:
        return _download(url, dest)
    return False


# Колонки Sheet1 (позиции — у хвостовых колонок нет заголовков):
#  0 Type | 1 Subject | 2 Class | 3 Source | 4 Visual | 5 Correct answer
#  6 Input format | 7 Question format | 8 Question type | 9 handwritten samples
# 11 URL картинки вопроса (для строк, где Visual = id вида q5001)
# 12 URL картинки ответа  (для строк, где Correct answer = id вида a5001)
def convert(csv_path: Path, tasks_out: Path, meta_out: Path,
            images_dir: Path, download: bool,
            archive_dir: Path | None = None) -> None:
    settings = get_settings()
    images_abs = settings.data_root / images_dir
    images_abs.mkdir(parents=True, exist_ok=True)

    archive: dict[str, Path] | None = None
    if archive_dir is not None:
        archive = {p.name: p for p in archive_dir.rglob("*") if p.is_file()}
        print(f"Локальный архив: {len(archive)} файлов")

    all_rows = list(csv.reader(csv_path.open(encoding="utf-8-sig")))
    rows = all_rows[1:]
    print(f"Строк в таблице: {len(rows)}")

    tasks_out.parent.mkdir(parents=True, exist_ok=True)
    n_ok = n_noimg = n_nourl = 0
    with tasks_out.open("w", encoding="utf-8") as ft, \
         meta_out.open("w", encoding="utf-8") as fm:
        for i, row in enumerate(rows, start=2):  # 2 = номер строки в таблице
            task_id = f"val_{i:04d}"
            visual = _get(row, 4)
            answer = _get(row, 5)
            qtype = _get(row, 8)
            if not visual or not answer:
                continue

            # URL картинки вопроса: либо прямо в Visual, либо в колонке 11
            url = visual if visual.startswith("http") else _get(row, 11)
            answer_is_image = bool(re.fullmatch(r"a\d+\w*", answer))
            answer_image_url = _get(row, 12) if answer_is_image else ""
            if not url.startswith("http"):
                n_nourl += 1
                continue

            mime = _mime(url)
            ext = ".jpg" if mime == "image/jpeg" else ".png"
            rel = images_dir / f"{task_id}{ext}"
            if not _fetch_image(url, settings.data_root / rel, archive, download):
                n_noimg += 1
                continue

            task = Task(
                task_id=task_id,
                subject=_get(row, 1),
                grade=_grade(_get(row, 2)),
                question="(soru görselde)",
                question_images=[ImageRef(
                    image_id=f"{task_id}_img",
                    format="file_path",
                    data=str(rel).replace("\\", "/"),
                    mime_type=mime,
                )],
                reference_answer=answer,
                answer_type=_answer_type(qtype, answer),
            )
            ft.write(task.model_dump_json() + "\n")
            fm.write(json.dumps({
                "task_id": task_id,
                "type": _get(row, 0),
                "class": _get(row, 2),
                "input_format": _get(row, 6),
                "question_format": _get(row, 7),
                "question_type": qtype,
                "source": _get(row, 3),
                "visual_url": url,
                # ответ дан картинкой (id a5xxx) — авто-оценке не подлежит
                "answer_is_url": answer.startswith("http") or answer_is_image,
                "answer_image_url": answer_image_url,
            }, ensure_ascii=False) + "\n")
            n_ok += 1

    print(f"Готово: {n_ok} задач -> {tasks_out}, meta -> {meta_out}")
    if n_nourl or n_noimg:
        print(f"Пропущено: без URL картинки: {n_nourl}, картинка недоступна: {n_noimg}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--csv", type=Path, help="локальный CSV-экспорт таблицы")
    src.add_argument("--sheet-id", help="id Google Sheets (скачать самим)")
    ap.add_argument("--gid", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("data/validation.jsonl"))
    ap.add_argument("--meta", type=Path, default=Path("data/validation.meta.jsonl"))
    ap.add_argument("--images-dir", type=Path, default=Path("images"),
                    help="куда класть картинки, относительно MLA_DATA_ROOT")
    ap.add_argument("--no-download", action="store_true",
                    help="не скачивать картинки (только сгенерировать JSONL)")
    ap.add_argument("--archive-dir", type=Path, default=None,
                    help="папка с уже скачанными картинками (имена = basename URL)")
    args = ap.parse_args()

    csv_path = args.csv
    if csv_path is None:
        csv_path = Path("data/validation_sheet.csv")
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        fetch_sheet_csv(args.sheet_id, args.gid, csv_path)
        print(f"Таблица скачана: {csv_path}")

    convert(csv_path, args.out, args.meta, args.images_dir,
            not args.no_download, archive_dir=args.archive_dir)


if __name__ == "__main__":
    main()
