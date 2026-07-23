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


def _find_by_id(archive: dict[str, Path] | None, image_id: str) -> Path | None:
    """Ищет в архиве файл по голому id (q0014 -> q0014.png/.jpg/.jpeg)."""
    if archive is None:
        return None
    for ext in (".png", ".jpg", ".jpeg", ".webp"):
        p = archive.get(image_id + ext)
        if p is not None:
            return p
    return None


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
            archive_dir: Path | None = None,
            sheet2_csv: Path | None = None) -> None:
    settings = get_settings()
    images_abs = settings.data_root / images_dir
    images_abs.mkdir(parents=True, exist_ok=True)

    archive: dict[str, Path] | None = None
    if archive_dir is not None:
        archive = {p.name: p for p in archive_dir.rglob("*") if p.is_file()}
        print(f"Локальный архив: {len(archive)} файлов")

    # Sheet2 как справочник id -> URL: имя файла в ссылке = q-идентификатор
    # из Visual первого листа (q0014 -> .../q0014.png). Фолбэк для скачивания,
    # когда файла нет в архиве.
    id_url_map: dict[str, str] = {}
    if sheet2_csv is not None and sheet2_csv.exists():
        for r2 in list(csv.reader(sheet2_csv.open(encoding="utf-8-sig")))[1:]:
            u = _get(r2, 4)
            if u.startswith("http"):
                name = Path(urllib.parse.unquote(urllib.parse.urlparse(u).path)).name
                m = re.match(r"^(q\d+\w*)\.", name)
                if m:
                    id_url_map[m.group(1).lower()] = u
        if id_url_map:
            print(f"Sheet2: справочник id->URL, записей: {len(id_url_map)}")

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

            answer_is_image = bool(re.fullmatch(r"a\d+\w*", answer))
            answer_image_url = _get(row, 12) if answer_is_image else ""

            # Картинки вопроса, по приоритету:
            #  1) Visual = id(ы) вида q0014 / "q5005a q5005b" -> файлы в архиве
            #  2) Visual = прямой URL  3) URL в колонке 11
            import shutil
            refs: list[tuple[Path, Path]] = []  # (куда, откуда-архив) для id-пути
            url = ""
            if not visual.startswith("http") and re.fullmatch(r"[qQ]\d+\w*( +\w+)*", visual):
                ids = visual.split()
                found = [_find_by_id(archive, i) for i in ids]
                if all(found):
                    for k, src in enumerate(found):
                        suffix = f"_{k + 1}" if len(found) > 1 else ""
                        refs.append((images_dir / f"{task_id}{suffix}{src.suffix}", src))
                    url = f"archive:{visual}"
                elif download and all(i.lower() in id_url_map for i in ids):
                    # файлов нет, но Sheet2 знает URL — качаем по id
                    ok_dl = True
                    for k, i in enumerate(ids):
                        u = id_url_map[i.lower()]
                        suffix = f"_{k + 1}" if len(ids) > 1 else ""
                        ext = ".jpg" if _mime(u) == "image/jpeg" else ".png"
                        rel = images_dir / f"{task_id}{suffix}{ext}"
                        if not _download(u, settings.data_root / rel):
                            ok_dl = False
                            break
                        refs.append((rel, settings.data_root / rel))
                    if not ok_dl:
                        refs.clear()
                    else:
                        url = id_url_map[ids[0].lower()]
            if not refs:
                url = visual if visual.startswith("http") else _get(row, 11)
                if not url.startswith("http"):
                    n_nourl += 1
                    continue

            if refs:  # id-путь: просто копируем из архива
                for rel, src in refs:
                    dst = settings.data_root / rel
                    if not (dst.exists() and dst.stat().st_size > 0):
                        shutil.copy2(src, dst)
                mime = _mime(refs[0][1].name)
            else:
                mime = _mime(url)
                ext = ".jpg" if mime == "image/jpeg" else ".png"
                rel = images_dir / f"{task_id}{ext}"
                if not _fetch_image(url, settings.data_root / rel, archive, download):
                    n_noimg += 1
                    continue
                refs = [(rel, rel)]

            task = Task(
                task_id=task_id,
                subject=_get(row, 1),
                grade=_grade(_get(row, 2)),
                question="(soru görselde)",
                question_images=[ImageRef(
                    image_id=f"{task_id}_img{k + 1}",
                    format="file_path",
                    data=str(r).replace("\\", "/"),
                    mime_type=_mime(r.name),
                ) for k, (r, _src) in enumerate(refs)],
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

    # Sheet2: самостоятельные задачи (question = URL скриншота). Дедуп по
    # содержимому картинок: строки Sheet1 (q0xxx) могут дублировать Sheet2.
    if sheet2_csv is not None and sheet2_csv.exists():
        import hashlib
        seen = set()
        for p in images_abs.iterdir():
            if p.is_file():
                seen.add(hashlib.md5(p.read_bytes()).hexdigest())
        n2_ok = n2_dup = n2_noimg = 0
        rows2 = list(csv.reader(sheet2_csv.open(encoding="utf-8-sig")))[1:]
        with tasks_out.open("a", encoding="utf-8") as ft, \
             meta_out.open("a", encoding="utf-8") as fm:
            for i, row in enumerate(rows2, start=2):
                url = _get(row, 4)
                answer = _get(row, 5)
                qtype = _get(row, 8)
                if not url.startswith("http") or not answer:
                    continue
                task_id = f"val2_{i:04d}"
                mime = _mime(url)
                ext = ".jpg" if mime == "image/jpeg" else ".png"
                rel = images_dir / f"{task_id}{ext}"
                dst = settings.data_root / rel
                if not _fetch_image(url, dst, archive, download):
                    n2_noimg += 1
                    continue
                h = hashlib.md5(dst.read_bytes()).hexdigest()
                if h in seen:
                    dst.unlink()
                    n2_dup += 1
                    continue
                seen.add(h)
                task = Task(
                    task_id=task_id,
                    subject=_get(row, 1),
                    grade=_grade(_get(row, 2)),
                    question="(soru görselde)",
                    question_images=[ImageRef(
                        image_id=f"{task_id}_img", format="file_path",
                        data=str(rel).replace("\\", "/"), mime_type=mime)],
                    reference_answer=answer,
                    answer_type=_answer_type(qtype, answer),
                )
                ft.write(task.model_dump_json() + "\n")
                fm.write(json.dumps({
                    "task_id": task_id, "type": _get(row, 0),
                    "class": _get(row, 2), "input_format": _get(row, 6),
                    "question_format": _get(row, 7), "question_type": qtype,
                    "source": _get(row, 3), "visual_url": url,
                    "answer_is_url": answer.startswith("http"),
                    "answer_image_url": "", "sheet": 2,
                }, ensure_ascii=False) + "\n")
                n2_ok += 1
        print(f"Sheet2: добавлено {n2_ok}, дублей с Sheet1: {n2_dup}, "
              f"без картинки: {n2_noimg}")

    print(f"Готово: {n_ok} задач (Sheet1) -> {tasks_out}, meta -> {meta_out}")
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
                    help="папка с уже скачанными картинками (basename URL или <id>.png)")
    ap.add_argument("--sheet2-csv", type=Path, default=None,
                    help="CSV второго листа (question=URL) — добавить как задачи")
    args = ap.parse_args()

    csv_path = args.csv
    if csv_path is None:
        csv_path = Path("data/validation_sheet.csv")
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        fetch_sheet_csv(args.sheet_id, args.gid, csv_path)
        print(f"Таблица скачана: {csv_path}")

    convert(csv_path, args.out, args.meta, args.images_dir,
            not args.no_download, archive_dir=args.archive_dir,
            sheet2_csv=args.sheet2_csv)


if __name__ == "__main__":
    main()
