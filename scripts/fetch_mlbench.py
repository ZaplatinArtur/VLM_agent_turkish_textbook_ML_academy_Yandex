# -*- coding: utf-8 -*-
"""Скачивание мультиязычных школьных бенчмарков и конвертация в наш Task-формат.

Каждый бенчмарк превращается в JSONL, совместимый с data/validation.jsonl
(контракт mla_baseline.contracts.Task), так что весь конвейер — runner, судья,
отчёты — работает без изменений. Картинки складываются рядом файлами.

    python scripts/fetch_mlbench.py --list
    python scripts/fetch_mlbench.py tumlu                       # все языки
    python scripts/fetch_mlbench.py tumlu --langs kazakh,uzbek
    python scripts/fetch_mlbench.py exams-v --langs Turkish --limit 200
    python scripts/fetch_mlbench.py kaleidoscope --langs hi,bn --limit 300

Выход: data/mlbench/<бенчмарк>/<язык>.jsonl (+ images/).

Реестр ниже — источник правды о том, что мы берём и почему. Мультимодальные
бенчмарки помечены; у EXAMS-V вопрос — это СКРИНШОТ задания (наш сценарий
«ленивый школьник» один в один), у Kaleidoscope — текст + опциональная картинка.
"""

import argparse
import io
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "data" / "mlbench"

REGISTRY = {
    "tumlu": {
        "hf": "jafarisbarov/TUMLU-mini",
        "multimodal": False,
        "languages": ["azerbaijani", "crimean-tatar", "karakalpak", "kazakh",
                      "kyrgyz", "tatar", "turkish", "uyghur", "uzbek"],
        "note": "тюркские языки, школьные вопросы от носителей; ближе всего к "
                "нашему домену — проверка переноса на родственные языки",
    },
    "exams-v": {
        "hf": "Rocktim/EXAMS-V",
        "multimodal": True,
        # фактический состав test-сплита (сверен по datasets-server 10.08):
        # в данных на два языка больше, чем в статье
        "languages": ["Arabic", "Bulgarian", "Chinese", "Croatian", "English",
                      "French", "German", "Hungarian", "Italian", "Polish",
                      "Serbian", "Slovakian", "Spanish"],
        "note": "вопрос = скриншот экзаменационного задания (как наши задачи); "
                "20 дисциплин, Apache 2.0",
    },
    "mgsm": {
        "hf": "juletxara/mgsm",
        "multimodal": False,
        "languages": ["en", "es", "fr", "de", "ru", "zh", "ja", "th", "sw",
                      "bn", "te"],
        "note": "открытые числовые ответы (не multiple choice): 250 школьных "
                "матзадач, вручную переведённых на 10 языков; exact-match "
                "без судьи — закрывает дыру открытых вопросов",
    },
    "kaleidoscope": {
        "hf": "CohereLabs/kaleidoscope",
        "multimodal": True,
        "languages": ["ar", "bn", "de", "en", "es", "fa", "fr", "hi", "hr",
                      "hu", "lt", "ne", "nl", "pt", "ru", "sr", "te", "uk"],
        "note": "in-language экзамены, 55% вопросов требуют картинку; "
                "лицензия указана per-item — фильтровать при публикации",
    },
}

CHOICES = "ABCDEFGH"


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.casefold()).strip("-")


def write_rows(rows, bench: str, lang: str) -> Path:
    out_dir = OUT_ROOT / bench
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{slug(lang)}.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path


def save_image(img, bench: str, lang: str, task_id: str) -> dict:
    img_dir = OUT_ROOT / bench / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    name = f"{task_id}.png"
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    img.save(img_dir / name, format="PNG")
    rel = f"data/mlbench/{bench}/images/{name}"
    return {"image_id": f"{task_id}_img1", "format": "file_path",
            "data": rel, "mime_type": "image/png", "caption": None}


def fetch_tumlu(langs, limit):
    from datasets import load_dataset
    for lang in langs:
        ds = load_dataset(REGISTRY["tumlu"]["hf"], lang, split="test")
        rows = []
        for i, r in enumerate(ds):
            if limit and i >= limit:
                break
            choices = r["choices"]
            if isinstance(choices, str):
                choices = json.loads(choices.replace("'", '"')) \
                    if choices.startswith("[") else [choices]
            opts = "\n".join(f"{CHOICES[j]}) {c}" for j, c in enumerate(choices))
            rows.append({
                "task_id": f"tumlu_{slug(lang)}_{i:04d}",
                "subject": r.get("subject") or "unknown",
                "grade": None,
                "question": f"{r['question']}\n\n{opts}",
                "question_images": [],
                "reference_answer": str(r["answer"]).strip(),
                "answer_type": "choice",
                "reference_solution": r.get("CoT") or None,
            })
        path = write_rows(rows, "tumlu", lang)
        print(f"tumlu/{lang}: {len(rows)} задач -> {path.relative_to(ROOT)}")


def fetch_exams_v(langs, limit):
    """Тест-сплит EXAMS-V. Стриминг только при --limit (проба); полный набор
    качается parquet'ом целиком — так на порядок быстрее."""
    from datasets import load_dataset
    ds = load_dataset(REGISTRY["exams-v"]["hf"], split="test",
                      streaming=bool(limit))
    want = {l.casefold() for l in langs} if langs else None
    buckets: dict[str, list] = {}
    counts: dict[str, int] = {}
    for r in ds:
        lang = r["language"]
        if want and lang.casefold() not in want:
            continue
        if limit and counts.get(lang, 0) >= limit:
            if want and all(counts.get(l, 0) >= limit for l in want):
                break
            continue
        i = counts.get(lang, 0)
        counts[lang] = i + 1
        tid = f"examsv_{slug(lang)}_{i:04d}"
        img_ref = save_image(r["image"], "exams-v", lang, tid)
        buckets.setdefault(lang, []).append({
            "task_id": tid,
            "subject": r.get("subject_grouped") or r.get("subject") or "unknown",
            "grade": int(r["grade"]) if str(r.get("grade") or "").isdigit() else None,
            "question": "(question in the image)",
            "question_images": [img_ref],
            "reference_answer": str(r["answer_key"]).strip(),
            "answer_type": "choice",
            "reference_solution": None,
        })
    for lang, rows in buckets.items():
        path = write_rows(rows, "exams-v", lang)
        print(f"exams-v/{lang}: {len(rows)} задач -> {path.relative_to(ROOT)}")


def fetch_mgsm(langs, limit):
    from datasets import load_dataset
    for lang in langs:
        ds = load_dataset(REGISTRY["mgsm"]["hf"], lang, split="test")
        rows = []
        for i, r in enumerate(ds):
            if limit and i >= limit:
                break
            rows.append({
                "task_id": f"mgsm_{lang}_{i:04d}",
                "subject": "Maths",
                "grade": None,
                "question": r["question"],
                "question_images": [],
                "reference_answer": str(r["answer_number"]),
                "answer_type": "numeric",
                "reference_solution": r.get("answer") or None,
            })
        path = write_rows(rows, "mgsm", lang)
        print(f"mgsm/{lang}: {len(rows)} задач -> {path.relative_to(ROOT)}")


def fetch_kaleidoscope(langs, limit):
    from datasets import load_dataset
    from PIL import Image
    ds = load_dataset(REGISTRY["kaleidoscope"]["hf"], split="train", streaming=True)
    want = {l.casefold() for l in langs} if langs else None
    buckets: dict[str, list] = {}
    counts: dict[str, int] = {}
    for r in ds:
        lang = r["language"]
        if want and lang.casefold() not in want:
            continue
        if limit and counts.get(lang, 0) >= limit:
            if want and all(counts.get(l, 0) >= limit for l in want):
                break
            continue
        i = counts.get(lang, 0)
        counts[lang] = i + 1
        tid = f"kal_{slug(lang)}_{i:04d}"
        images = []
        raw_img = r.get("image_bytes") or None
        if raw_img:
            images.append(save_image(Image.open(io.BytesIO(raw_img)),
                                     "kaleidoscope", lang, tid))
        opts = "\n".join(f"{CHOICES[j]}) {c}" for j, c in enumerate(r["options"]))
        buckets.setdefault(lang, []).append({
            "task_id": tid,
            "subject": r.get("category_en") or "unknown",
            "grade": None,
            "question": f"{r['question']}\n\n{opts}",
            "question_images": images,
            "reference_answer": CHOICES[int(r["answer"])],
            "answer_type": "choice",
            "reference_solution": None,
            # метаданные для фильтрации при публикации
            "_license": r.get("license"),
            "_needs_image": bool(r.get("image_png")),
            "_source": r.get("source"),
        })
    for lang, rows in buckets.items():
        path = write_rows(rows, "kaleidoscope", lang)
        print(f"kaleidoscope/{lang}: {len(rows)} задач -> {path.relative_to(ROOT)}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("bench", nargs="?", choices=sorted(REGISTRY))
    parser.add_argument("--langs", default="", help="через запятую; пусто = все")
    parser.add_argument("--limit", type=int, default=0, help="задач на язык")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    if args.list or not args.bench:
        for name, spec in REGISTRY.items():
            mm = "мультимодальный" if spec["multimodal"] else "текстовый"
            print(f"{name:<14} {mm:<15} {spec['hf']}")
            print(f"{'':14} языки: {', '.join(spec['languages'])}")
            print(f"{'':14} {spec['note']}\n")
        return 0

    langs = [l.strip() for l in args.langs.split(",") if l.strip()] \
        or REGISTRY[args.bench]["languages"]
    if args.bench == "tumlu":
        fetch_tumlu(langs, args.limit)
    elif args.bench == "mgsm":
        fetch_mgsm(langs, args.limit)
    elif args.bench == "exams-v":
        fetch_exams_v(langs, args.limit)
    elif args.bench == "kaleidoscope":
        fetch_kaleidoscope(langs, args.limit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
